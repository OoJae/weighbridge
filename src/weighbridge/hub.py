"""The adapter population: list, pin, parse and classify every public LoRA adapter of the base model.

Everything this module writes about third-party repositories (the manifest, its metadata, the confirmatory
set, the salt and cached configs and cards) goes under ``common.PRIVATE_DIR``. Only the salted ``alias`` and
aggregate counts (``docs/population_summary.md``) may leave that folder.

Run ``python -m weighbridge.hub`` to take a fresh snapshot of the live Hub and rebuild every output.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import math
import random
import re
import secrets
import time
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, date, datetime
from functools import lru_cache, partial
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple

from . import __version__
from .common import (BASE_REPO, BASE_REVISION, CACHE_DIR, N_LAYERS, PRIVATE_DIR, REPO_ROOT, TARGET_MODULES,
                     ensure_dir, module_path)

log = logging.getLogger(__name__)

POPULATION_CONFIG = REPO_ROOT / "configs" / "population.yaml"
SALT_PATH = PRIVATE_DIR / "salt.txt"
MANIFEST_PATH = PRIVATE_DIR / "manifest.jsonl"
CONFIRMATORY_PATH = PRIVATE_DIR / "confirmatory_set.json"
SUMMARY_PATH = REPO_ROOT / "docs" / "population_summary.md"
HUB_CACHE = CACHE_DIR / "hub"

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")
README = "README.md"
FULL_MODEL_FILE = re.compile(r"(?:^|/)(?:model|pytorch_model)(?:-\d+-of-\d+)?\.(?:safetensors|bin)$")
# A Hub repo id, optionally followed by a subfolder: "owner/name" or "owner/name/sub/dir".
HUB_ID = re.compile(r"(?P<repo>[A-Za-z0-9][\w.-]*/[\w.-]+)(?:/(?P<sub>\S+?))?(?=\s|$)")
FRONT_MATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.S)
CHECKPOINT_FOLDER = re.compile(r"(?:checkpoint|epoch|step)[-_]?(\d+)", re.I)

GATED = "gated"
NO_CONFIG = "no_adapter_config"
FULL_MODEL = "full_model_mislabelled"
NO_WEIGHTS = "no_adapter_weights"
UNREADABLE = "unreadable"
NOT_LORA = "not_lora"
BASE_MISMATCH = "base_not_resolvable"
EXCLUSION_REASONS = (GATED, NO_CONFIG, FULL_MODEL, NO_WEIGHTS, UNREADABLE, NOT_LORA, BASE_MISMATCH)

SAFETY_INTENTS = ("attack_or_drift", "safety_hardened", "control")
INTENTS = SAFETY_INTENTS + ("task", "unknown")
PREFERENCE_METHODS = ("dpo", "grpo", "rft", "kto", "orpo")
METHODS = PREFERENCE_METHODS + ("sft", "other", "unknown")

# PEFT LoraConfig defaults, used when a config omits the field.
DEFAULT_RANK = 8
DEFAULT_ALPHA = 8.0

MANIFEST_FIELDS = (
    "repo_id", "sha", "author", "created_at", "last_modified", "gated", "license", "tags",
    "status", "exclusion_reason", "exclusion_detail",
    "adapter_subfolder", "adapter_weight_file", "alternate_subfolders", "adapter_config",
    "rank", "alpha", "use_rslora", "scaling", "target_modules", "layers_to_transform", "n_target_pairs",
    "other_targets", "rank_pattern", "alpha_pattern", "distinct_ranks", "distinct_scalings",
    "base_model_name_or_path", "stacked_on",
    "declared_intent", "intent_evidence", "training_method",
    "languages", "hackathon_era", "weight_bytes", "alias",
)


class AdapterLocation(NamedTuple):
    """Where a repository keeps the adapter we analyse."""

    subfolder: str             # "" for the repository root
    config_file: str           # path of adapter_config.json inside the repository
    weight_file: str           # path of the adapter weights inside the repository
    alternates: list[str]      # other folders holding a complete adapter (logged, not analysed)


# ---- Rules -------------------------------------------------------------------------------------


@lru_cache(maxsize=4)
def load_rules(path: Path = POPULATION_CONFIG) -> dict[str, Any]:
    """Keyword tables and thresholds from configs/population.yaml."""
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ---- Config normalisation ----------------------------------------------------------------------


def lora_scaling(alpha: float, rank: int, use_rslora: bool) -> float:
    """PEFT's LoRA scaling: alpha / r, or alpha / sqrt(r) for rank-stabilised LoRA."""
    return float(alpha / math.sqrt(rank) if use_rslora else alpha / rank)


def _layer_list(value: int | list[int] | None) -> list[int] | None:
    """layers_to_transform as a sorted list, or None when every layer is allowed."""
    if value is None or value == []:
        return None
    return sorted({int(v) for v in ([value] if isinstance(value, int) else value)})


def _name_match(spec: str | list[str] | None, key: str) -> bool:
    """PEFT's module-name rule: a string is a full regex match; a list matches exactly or as a dotted suffix."""
    if not spec:
        return False
    if isinstance(spec, str):
        return re.fullmatch(spec, key) is not None
    return key in spec or any(key.endswith(f".{name}") for name in spec)


def _is_targeted(cfg: dict, layer: int, module: str) -> bool:
    """Whether PEFT would wrap this base module (mirrors peft.tuners.tuners_utils.check_target_module_exists)."""
    key = module_path(layer, module)
    if _name_match(cfg.get("exclude_modules"), key):
        return False
    targets = cfg.get("target_modules")
    if targets == "all-linear":
        targets = list(TARGET_MODULES)
    if isinstance(targets, str) or key in (targets or []):
        return _name_match(targets, key)
    layers = _layer_list(cfg.get("layers_to_transform"))
    return _name_match(targets, key) and (layers is None or layer in layers)


def targeted_pairs(cfg: dict, n_layers: int = N_LAYERS) -> list[tuple[int, str]]:
    """Every (layer, module) of the base model that this adapter config wraps."""
    return [(layer, m) for layer in range(n_layers) for m in TARGET_MODULES if _is_targeted(cfg, layer, m)]


def _pattern_value(patterns: dict | None, key: str, default: float) -> float:
    """PEFT's rank_pattern / alpha_pattern lookup (peft.utils.other.get_pattern_key): first matching key wins."""
    for pattern, value in (patterns or {}).items():
        if re.match(rf"(.*\.)?({pattern})$", key):
            return value
    return default


def module_rank_alpha(cfg: dict, layer: int, module: str) -> tuple[int, float]:
    """Rank and alpha that PEFT uses for one module after rank_pattern and alpha_pattern overrides."""
    key = module_path(layer, module)
    rank = _pattern_value(cfg.get("rank_pattern"), key, cfg.get("r", DEFAULT_RANK))
    alpha = _pattern_value(cfg.get("alpha_pattern"), key, cfg.get("lora_alpha", DEFAULT_ALPHA))
    return int(rank), float(alpha)


def _unmatched_targets(cfg: dict, n_layers: int) -> list[str]:
    """Listed target names that match none of the seven base projections (e.g. lm_head)."""
    targets = cfg.get("target_modules")
    if isinstance(targets, str) or not targets:
        return []
    keys = [module_path(layer, m) for layer in range(n_layers) for m in TARGET_MODULES]
    return sorted(t for t in targets if not any(k == t or k.endswith(f".{t}") for k in keys))


def normalize_config(cfg: dict, n_layers: int = N_LAYERS) -> dict[str, Any]:
    """Rank, alpha, scaling and the targeted modules and layers of a PEFT LoRA config.

    ``target_modules`` may be a list of names, full module paths, ``"all-linear"`` or a regex string. The
    returned ``target_modules`` is the subset of common.TARGET_MODULES actually wrapped, and
    ``layers_to_transform`` the layers touched (None when all of them are). With rank_pattern or
    alpha_pattern, ``rank``/``alpha``/``scaling`` are the defaults and ``distinct_ranks`` /
    ``distinct_scalings`` list every per-module value; use module_rank_alpha for one module.
    """
    rank, alpha = int(cfg.get("r", DEFAULT_RANK)), float(cfg.get("lora_alpha", DEFAULT_ALPHA))
    if rank <= 0:
        raise ValueError(f"LoRA rank must be positive, got {rank}")
    use_rslora = bool(cfg.get("use_rslora", False))
    pairs = targeted_pairs(cfg, n_layers)
    per_module = [module_rank_alpha(cfg, layer, m) for layer, m in pairs]
    layers = sorted({layer for layer, _ in pairs})
    return {
        "rank": rank,
        "alpha": alpha,
        "use_rslora": use_rslora,
        "scaling": lora_scaling(alpha, rank, use_rslora),
        "target_modules": [m for m in TARGET_MODULES if any(pm == m for _, pm in pairs)],
        "layers_to_transform": None if len(layers) == n_layers else layers,
        "n_target_pairs": len(pairs),
        "other_targets": _unmatched_targets(cfg, n_layers),
        "rank_pattern": dict(cfg.get("rank_pattern") or {}),
        "alpha_pattern": dict(cfg.get("alpha_pattern") or {}),
        "distinct_ranks": sorted({r for r, _ in per_module}),
        "distinct_scalings": sorted({lora_scaling(a, r, use_rslora) for r, a in per_module}),
    }


# ---- Exclusion rules ---------------------------------------------------------------------------


def _parent(path: str) -> str:
    parent = str(PurePosixPath(path).parent)
    return "" if parent == "." else parent


def _join(folder: str, name: str) -> str:
    return f"{folder}/{name}" if folder else name


def _folder_order(folder: str) -> tuple[int, str, int]:
    """Shallowest first; within one run's checkpoint or epoch folders, the latest first (checkpoint-0 is the
    untrained initialisation); otherwise by name."""
    path = PurePosixPath(folder)
    step = CHECKPOINT_FOLDER.fullmatch(path.name)
    return (len(path.parts), str(path.parent) if step else folder, -int(step[1]) if step else 0)


def locate_adapter(files: Iterable[str]) -> AdapterLocation | None:
    """The shallowest folder holding adapter_config.json next to adapter weights (see _folder_order for ties)."""
    names = set(files)
    folders = sorted({_parent(f) for f in names if PurePosixPath(f).name == ADAPTER_CONFIG}, key=_folder_order)
    complete = [(d, w) for d in folders for w in [_weight_file(d, names)] if w]
    if not complete:
        return None
    (folder, weights), rest = complete[0], complete[1:]
    return AdapterLocation(folder, _join(folder, ADAPTER_CONFIG), weights, [d for d, _ in rest])


def _weight_file(folder: str, names: Collection[str]) -> str | None:
    """Adapter weights inside one folder, preferring safetensors."""
    return next((p for p in (_join(folder, w) for w in ADAPTER_WEIGHTS) if p in names), None)


def structural_exclusion(gated: bool, files: Collection[str] | None) -> str | None:
    """Exclusion reason visible from the file listing alone (files=None: the listing could not be read)."""
    if gated:
        return GATED
    if files is None:
        return UNREADABLE
    if locate_adapter(files) is not None:
        return None
    if any(FULL_MODEL_FILE.search(f) for f in files):
        return FULL_MODEL
    has_config = any(PurePosixPath(f).name == ADAPTER_CONFIG for f in files)
    return NO_WEIGHTS if has_config else NO_CONFIG


def resolves_to_base(name: str | None, base_repo: str = BASE_REPO) -> bool:
    """Whether base_model_name_or_path names the base model: its Hub id, its HF cache folder, or a local copy."""
    text = (name or "").strip().rstrip("/").replace("\\", "/")
    if not text:
        return False
    if text.lower() == base_repo.lower():
        return True
    if HUB_ID.fullmatch(text) and not text.startswith((".", "~")):
        return False  # a different Hub repository
    parts = [p.lower() for p in PurePosixPath(text).parts]
    cache_folder = "models--" + base_repo.replace("/", "--").lower()
    return cache_folder in parts or parts[-1] == base_repo.split("/")[-1].lower()


def config_exclusion(cfg: dict, base_repo: str = BASE_REPO) -> str | None:
    """Exclusion reason found in adapter_config.json: not plain LoRA (incl. DoRA), or a foreign base."""
    if str(cfg.get("peft_type", "")).upper() != "LORA" or cfg.get("use_dora"):
        return NOT_LORA
    if not resolves_to_base(cfg.get("base_model_name_or_path"), base_repo):
        return BASE_MISMATCH
    return None


def stacked_on_candidate(base_path: str | None) -> dict[str, str | None]:
    """What an adapter with a foreign base sits on; ``repo_id`` is set only when it looks like a Hub id."""
    text = (base_path or "").strip()
    match = None if text.startswith(("/", ".", "~")) else HUB_ID.match(text)
    return {"repo_id": match["repo"] if match else None, "sha": None,
            "subfolder": match["sub"] if match else None, "declared": text}


# ---- Intent and method -------------------------------------------------------------------------


def _normalize(text: str, ignore: Collection[str] = ()) -> str:
    """Lower-case words separated by single spaces, padded with a space on each side."""
    words = [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w and w not in ignore]
    return " " + " ".join(words) + " "


def _hits(text: str, keywords: dict[str, list[str]]) -> list[str]:
    """Keywords of one category found in normalised text (prefix words, then exact words)."""
    found = [kw for kw in keywords.get("prefix", []) if _normalize(kw).rstrip() in text]
    return found + [kw for kw in keywords.get("exact", []) if _normalize(kw) in text]


def _category_hits(sources: list[tuple[str, str]], table: dict, rules: dict) -> dict[str, list[str]]:
    """For every category with a match, the 'source:keyword' evidence strings."""
    texts = [(src, _normalize(text, rules["ignore_tokens"])) for src, text in sources]
    hits = {cat: [f"{src}:{kw}" for src, text in texts for kw in _hits(text, kws)] for cat, kws in table.items()}
    return {cat: ev for cat, ev in hits.items() if ev}


def _name_and_tags(repo_id: str, tags: Iterable[str], rules: dict) -> list[tuple[str, str]]:
    """The repository name (owner excluded) and its informative tags."""
    kept = [t for t in tags
            if not t.startswith(tuple(rules["ignore_tag_prefixes"])) and t.lower() not in rules["ignore_tags"]]
    return [("name", repo_id.split("/")[-1]), ("tags", " ".join(kept))]


def card_summary(card_text: str, rules: dict | None = None) -> str:
    """The model card's opening: front matter removed, cut at the first boilerplate section heading."""
    rules = rules or load_rules()
    stops = [_normalize(h).rstrip() for h in rules["card_stop_headings"]]
    kept: list[str] = []
    for line in FRONT_MATTER.sub("", card_text, count=1).splitlines():
        if line.lstrip().startswith("#") and any(_normalize(line).startswith(s) for s in stops):
            break
        kept.append(line)
    return "\n".join(kept)[: rules["card_chars"]]


def _evidence(hits: dict[str, list[str]], categories: Iterable[str]) -> str:
    return "; ".join(f"{c}=" + ",".join(hits[c]) for c in categories)


def _decide_intent(hits: dict[str, list[str]], declared: bool) -> tuple[str, str]:
    """Safety-relevant evidence beats task evidence. It counts only when declared (name or tags), and
    disagreeing safety categories give unknown."""
    safety = [c for c in SAFETY_INTENTS if c in hits]
    if safety and not declared:
        return "unknown", "card only: " + _evidence(hits, safety)
    if len(safety) > 1:
        return "unknown", "conflict: " + _evidence(hits, safety)
    chosen = safety[0] if safety else "task"
    return chosen, _evidence(hits, [chosen])


def classify_intent(repo_id: str, tags: Iterable[str], card_text: str,
                    rules: dict | None = None) -> tuple[str, str]:
    """Declared intent and the evidence for it, from the repo name and tags, else the card's opening.

    Returns one of INTENTS and an evidence string such as ``"attack_or_drift=name:drift,tags:harmful"``.
    Card text alone can establish only ``task``; attack or hardening words found only there give ``unknown``,
    and control words there are ignored.
    """
    rules = rules or load_rules()
    card_table = {c: kws for c, kws in rules["intent"].items() if c != "control"}
    tiers = [(_name_and_tags(repo_id, tags, rules), rules["intent"], True),
             ([("card", card_summary(card_text, rules))], card_table, False)]
    for sources, table, declared in tiers:
        hits = _category_hits(sources, table, rules)
        if hits:
            return _decide_intent(hits, declared)
    return "unknown", "no keyword"


def _decide_method(found: Collection[str]) -> str:
    """One preference/RL method wins over SFT (its usual warm start); two of them are ambiguous."""
    preference = [m for m in PREFERENCE_METHODS if m in found]
    if len(preference) > 1:
        return "unknown"
    if preference:
        return preference[0]
    return "sft" if "sft" in found else "other"


def classify_method(repo_id: str, tags: Iterable[str], card_text: str, cfg: dict | None,
                    rules: dict | None = None) -> str:
    """Training method from the repo name and tags, else the adapter config, else the card's opening."""
    rules = rules or load_rules()
    own_cfg = {k: v for k, v in (cfg or {}).items() if k != "base_model_name_or_path"}
    tiers = [_name_and_tags(repo_id, tags, rules), [("config", json.dumps(own_cfg))],
             [("card", card_summary(card_text, rules))]]
    for sources in tiers:
        hits = _category_hits(sources, rules["method"], rules)
        if hits:
            return _decide_method(hits)
    return "unknown"


# ---- Aliases -----------------------------------------------------------------------------------


def load_salt(path: Path = SALT_PATH) -> str:
    """The private alias salt, created once with secrets.token_hex(32). Never print or log it."""
    if not path.exists():
        ensure_dir(path.parent)
        path.touch(mode=0o600)
        path.write_text(secrets.token_hex(32), encoding="utf-8")
    return path.read_text(encoding="utf-8").strip()


def alias(repo_id: str, salt: str) -> str:
    """Stable opaque label: "a_" plus the first 12 hex digits of HMAC-SHA256(salt, repo_id)."""
    return "a_" + hmac.new(salt.encode(), repo_id.encode(), hashlib.sha256).hexdigest()[:12]


# ---- Rows --------------------------------------------------------------------------------------


def _is_hackathon_era(created_at: str | None, rules: dict) -> bool:
    if not created_at:
        return False
    return datetime.fromisoformat(created_at).date() >= date.fromisoformat(rules["hackathon_start"])


def _read_config(location: AdapterLocation, read_text: Callable[[str], str],
                 base_repo: str) -> tuple[dict[str, Any], str | None]:
    """Manifest fields from adapter_config.json and the exclusion reason it implies."""
    cfg = json.loads(read_text(location.config_file))
    if not isinstance(cfg, dict):
        raise ValueError("adapter_config.json is not a JSON object")
    reason = config_exclusion(cfg, base_repo)
    fields: dict[str, Any] = {"adapter_config": cfg, "base_model_name_or_path": cfg.get("base_model_name_or_path")}
    if reason != NOT_LORA:
        fields |= normalize_config(cfg)
    if reason == BASE_MISMATCH:
        fields["stacked_on"] = stacked_on_candidate(cfg.get("base_model_name_or_path"))
    return fields, reason


def _read_card(files: Collection[str], read_text: Callable[[str], str]) -> str:
    """The model card text, or "" when absent or unreadable (a missing card never excludes a repo)."""
    if README not in files:
        return ""
    try:
        return read_text(README)
    except (OSError, ValueError):
        return ""


def build_row(listing: dict, sha: str, files: dict[str, int | None] | None, read_text: Callable[[str], str],
              salt: str, base_repo: str = BASE_REPO, rules: dict | None = None) -> dict[str, Any]:
    """One manifest row: pinned files, parsed config, exclusion reason, intent and method.

    ``files`` maps every file path at ``sha`` to its size (None if the listing failed); ``read_text`` reads a
    file of the repository at ``sha``. Model-card text is used for classification only and never stored.
    """
    rules = rules or load_rules()
    row: dict[str, Any] = dict.fromkeys(MANIFEST_FIELDS)
    row |= {k: listing.get(k) for k in ("repo_id", "author", "created_at", "last_modified", "license")}
    row |= {"sha": sha, "gated": bool(listing.get("gated")), "tags": list(listing.get("tags") or []),
            "languages": list(listing.get("languages") or []), "alternate_subfolders": [],
            "hackathon_era": _is_hackathon_era(listing.get("created_at"), rules),
            "alias": alias(listing["repo_id"], salt)}
    reason = structural_exclusion(row["gated"], files)
    if reason is None and files is not None:
        fields, reason = _adapter_fields(files, read_text, base_repo)
        row |= fields
    card = "" if row["gated"] or files is None else _read_card(files, read_text)
    intent, evidence = classify_intent(row["repo_id"], row["tags"], card, rules)
    row |= {"status": "included" if reason is None else "excluded", "exclusion_reason": reason,
            "declared_intent": intent, "intent_evidence": evidence,
            "training_method": classify_method(row["repo_id"], row["tags"], card, row["adapter_config"], rules)}
    return row


def _adapter_fields(files: dict[str, int | None], read_text: Callable[[str], str],
                    base_repo: str) -> tuple[dict[str, Any], str | None]:
    """Location, size and parsed config of a repository's adapter, with any exclusion reason they imply."""
    location = locate_adapter(files)
    if location is None:
        raise ValueError("no complete adapter folder; call structural_exclusion first")
    fields: dict[str, Any] = {"adapter_subfolder": location.subfolder, "adapter_weight_file": location.weight_file,
                              "alternate_subfolders": location.alternates,
                              "weight_bytes": files.get(location.weight_file)}
    try:
        config_fields, reason = _read_config(location, read_text, base_repo)
    except (OSError, ValueError, TypeError, re.error) as err:
        return fields | {"exclusion_detail": type(err).__name__}, UNREADABLE
    return fields | config_fields, reason


# ---- Hub access (public, no token) -------------------------------------------------------------


@contextmanager
def _quiet_hub_logs() -> Iterator[None]:
    """Silence huggingface_hub's per-card warnings (e.g. invalid model-index) for the duration."""
    hub_log = logging.getLogger("huggingface_hub")
    level = hub_log.level
    hub_log.setLevel(logging.ERROR)
    try:
        yield
    finally:
        hub_log.setLevel(level)


def _is_transient(err: OSError) -> bool:
    """Rate limits, server errors and connection failures are retried; 4xx answers are final."""
    response = getattr(err, "response", None)
    if response is None:
        return not isinstance(err, FileNotFoundError)
    return response.status_code == 429 or response.status_code >= 500


def _backoff(err: OSError, attempt: int, delay: float, cap: float = 120.0) -> float:
    """Seconds to wait: the server's Retry-After when given, else exponential, never above cap."""
    response = getattr(err, "response", None)
    retry_after = response.headers.get("Retry-After", "") if response is not None else ""
    return min(cap, float(retry_after) if retry_after.isdigit() else delay * 2 ** attempt)


def _retry[T](call: Callable[[], T], attempts: int = 8, delay: float = 2.0) -> T:
    """Call with backoff on transient Hub errors (rate limits, server errors, dropped connections)."""
    for attempt in range(attempts - 1):
        try:
            return call()
        except OSError as err:
            if not _is_transient(err):
                raise
            time.sleep(_backoff(err, attempt, delay))
    return call()


def _hub_api() -> Any:
    from huggingface_hub import HfApi

    return HfApi(token=False)


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(UTC).isoformat() if value else None


def _listing_row(model: Any) -> dict[str, Any]:
    """The listing fields we keep for one repository (card metadata reduced to licence and languages)."""
    card = model.card_data.to_dict() if model.card_data else {}
    language = card.get("language") or []
    tags = list(model.tags or [])
    license_tag = next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), None)
    return {"repo_id": model.id, "author": model.author or model.id.split("/")[0], "sha": model.sha,
            "created_at": _iso(model.created_at), "last_modified": _iso(model.last_modified),
            "gated": bool(model.gated), "tags": tags, "license": card.get("license") or license_tag,
            "languages": [language] if isinstance(language, str) else list(language)}


def list_adapters(base_repo: str = BASE_REPO) -> list[dict[str, Any]]:
    """Every public repository tagged ``base_model:adapter:<base_repo>``, sorted by repo id."""
    api = _hub_api()
    with _quiet_hub_logs():
        models = _retry(lambda: list(api.list_models(filter=f"base_model:adapter:{base_repo}",
                                                     full=True, cardData=True)))
    return sorted((_listing_row(m) for m in models), key=lambda r: r["repo_id"])


def _read_remote(repo_id: str, sha: str, cache_dir: Path, filename: str) -> str:
    """Text of one public file at a pinned commit, cached under the private folder."""
    from huggingface_hub import hf_hub_download

    path = _retry(lambda: hf_hub_download(repo_id, filename, revision=sha, cache_dir=cache_dir, token=False))
    return Path(path).read_text(encoding="utf-8")


def _pin(api: Any, repo_id: str) -> tuple[str | None, dict[str, int | None] | None]:
    """Current commit SHA and {file path: size}, or (None, None) if the repository cannot be read."""
    try:
        info = _retry(lambda: api.model_info(repo_id, files_metadata=True))
    except OSError as err:
        log.warning("cannot pin a listed repository: %s", type(err).__name__)
        return None, None
    return info.sha, {s.rfilename: s.size for s in info.siblings or []}


def _pin_stacked(api: Any, stacked: dict[str, str | None]) -> None:
    """Fill in the SHA of the Hub repo an adapter is stacked on; clear repo_id if it is not on the Hub."""
    if not stacked["repo_id"]:
        return
    try:
        stacked["sha"] = _retry(lambda: api.model_info(stacked["repo_id"]).sha)
    except OSError:
        stacked["repo_id"] = stacked["subfolder"] = None


def _assess(listing: dict, api: Any, salt: str, cache_dir: Path, base_repo: str) -> dict[str, Any]:
    """Pin one repository and build its manifest row."""
    sha, files = _pin(api, listing["repo_id"])  # metadata only; build_row never reads a gated repo's files
    reader = partial(_read_remote, listing["repo_id"], sha, cache_dir)
    row = build_row(listing, sha or listing["sha"], files, reader, salt, base_repo)
    if files is None:
        row["exclusion_detail"] = "file listing unavailable"
    if row["stacked_on"]:
        _pin_stacked(api, row["stacked_on"])
    return row


# ---- Snapshot outputs --------------------------------------------------------------------------


def _code_version() -> str:
    """Package version plus content hashes of this module and of the classification rules."""
    code, rules = (hashlib.sha256(p.read_bytes()).hexdigest()[:12] for p in (Path(__file__), POPULATION_CONFIG))
    return f"weighbridge {__version__}; hub.py sha256:{code}; population.yaml sha256:{rules}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(obj: Any, path: Path) -> Path:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_manifest(rows: list[dict], path: Path = MANIFEST_PATH) -> Path:
    """Write rows as JSON lines, sorted by repo id."""
    ensure_dir(path.parent)
    lines = [json.dumps(r, sort_keys=True) for r in sorted(rows, key=lambda r: r["repo_id"])]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def load_manifest(path: Path = MANIFEST_PATH) -> list[dict[str, Any]]:
    """Read the private manifest."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def manifest_commitment(manifest_path: Path = MANIFEST_PATH, salt_path: Path = SALT_PATH) -> str:
    """The pre-registered commitment: SHA-256 of the salt's hex text (UTF-8) followed by the manifest bytes.

    Unlike a plain hash of the manifest, nobody can confirm a guessed manifest without the private salt.
    """
    return hashlib.sha256(load_salt(salt_path).encode("utf-8") + manifest_path.read_bytes()).hexdigest()


def _log_exclusions(rows: list[dict]) -> None:
    for row in rows:
        if row["status"] == "excluded":
            log.info("excluded %s: %s %s", row["alias"], row["exclusion_reason"], row["exclusion_detail"] or "")
        if row["alternate_subfolders"]:
            log.info("%s: analysing %r, ignoring %d other adapter folders", row["alias"],
                     row["adapter_subfolder"], len(row["alternate_subfolders"]))


def _counts(rows: list[dict]) -> dict[str, Any]:
    included = [r for r in rows if r["status"] == "included"]
    return {"listed": len(rows), "included": len(included), "excluded": len(rows) - len(included),
            "exclusion_reasons": dict(Counter(r["exclusion_reason"] for r in rows if r["exclusion_reason"])),
            "authors_listed": len({r["author"] for r in rows}),
            "authors_included": len({r["author"] for r in included})}


def snapshot(out_path: Path = MANIFEST_PATH, base_repo: str = BASE_REPO, workers: int = 8,
             salt_path: Path = SALT_PATH, cache_dir: Path = HUB_CACHE) -> Path:
    """List, pin and classify the live population; write the manifest and manifest_meta.json beside it."""
    snapshot_utc = datetime.now(UTC).isoformat(timespec="seconds")
    listings = list_adapters(base_repo)
    salt, api = load_salt(salt_path), _hub_api()
    with _quiet_hub_logs(), ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(lambda listing: _assess(listing, api, salt, cache_dir, base_repo), listings))
    if len({r["alias"] for r in rows}) != len(rows):
        raise RuntimeError("alias collision; lengthen the alias")
    _log_exclusions(rows)
    write_manifest(rows, out_path)
    meta = {"snapshot_utc": snapshot_utc, "base_repo": base_repo, "base_revision": BASE_REVISION,
            "listing_filter": f"base_model:adapter:{base_repo}", "counts": _counts(rows),
            "manifest_sha256": _sha256(out_path), "manifest_commitment": manifest_commitment(out_path, salt_path),
            "code_version": _code_version()}
    _write_json(meta, out_path.with_name("manifest_meta.json"))
    return out_path


def confirmatory_set(rows: list[dict], max_per_author: int = 2, seed: int = 20261114) -> list[str]:
    """Aliases of a seeded random pick of at most ``max_per_author`` included adapters per author.

    Each author draws from their own generator seeded by (seed, author), so the pick is independent of row
    order and of which other authors are present.
    """
    by_author: dict[str, list[dict]] = {}
    for row in rows:
        if row["status"] == "included":
            by_author.setdefault(row["author"], []).append(row)
    chosen: list[str] = []
    for author, own in sorted(by_author.items()):
        own = sorted(own, key=lambda r: r["repo_id"])
        rng = random.Random(f"{seed}:{author}")
        chosen += [r["alias"] for r in rng.sample(own, min(max_per_author, len(own)))]
    return sorted(chosen)


def write_confirmatory_set(rows: list[dict], path: Path = CONFIRMATORY_PATH, max_per_author: int = 2,
                           seed: int = 20261114) -> Path:
    """Write the confirmatory set (aliases only) with its sampling parameters."""
    aliases = confirmatory_set(rows, max_per_author, seed)
    chosen = set(aliases)
    authors = {r["author"] for r in rows if r["alias"] in chosen}
    return _write_json({"seed": seed, "max_per_author": max_per_author, "n_adapters": len(aliases),
                        "n_authors": len(authors), "aliases": aliases}, path)


# ---- Aggregate summary (public) ----------------------------------------------------------------


def target_pattern(modules: list[str] | None, layers: list[int] | None) -> str:
    """Short label for which projections an adapter wraps, e.g. "all 7" or "q+v"."""
    if not modules:
        return "n/a"
    label = "all 7" if list(modules) == list(TARGET_MODULES) else "+".join(m.removesuffix("_proj") for m in modules)
    return label if layers is None else f"{label}, layer subset"


def _breakdown(rows: list[dict], key: Callable[[dict], Any]) -> dict[str, dict[str, int]]:
    """Counts of key(row) over all listed rows and over included rows."""
    def count(subset: list[dict]) -> dict[str, int]:
        return dict(Counter(str(key(r)) for r in subset))

    return {"all": count(rows), "included": count([r for r in rows if r["status"] == "included"])}


def summarize(rows: list[dict]) -> dict[str, Any]:
    """Aggregate counts only: no repo ids, author names or aliases."""
    included = [r for r in rows if r["status"] == "included"]
    chosen = set(confirmatory_set(rows))
    return {
        **_counts(rows),
        "declared_intent": _breakdown(rows, lambda r: r["declared_intent"]),
        "training_method": _breakdown(rows, lambda r: r["training_method"]),
        "rank": _breakdown(rows, lambda r: r["rank"] if r["rank"] is not None else "n/a"),
        "use_rslora": _breakdown(rows, lambda r: bool(r["use_rslora"])),
        "target_pattern": _breakdown(rows, lambda r: target_pattern(r["target_modules"], r["layers_to_transform"])),
        "month_created": _breakdown(rows, lambda r: (r["created_at"] or "unknown")[:7]),
        "era": _breakdown(rows, lambda r: "hackathon" if r["hackathon_era"] else "earlier"),
        "authors_by_cluster_size": dict(sorted(Counter(Counter(r["author"] for r in included).values()).items())),
        "confirmatory": {"adapters": len(chosen),
                         "authors": len({r["author"] for r in included if r["alias"] in chosen})},
    }


def _numeric_first(key: str) -> tuple[bool, int, str]:
    """Sort key placing numeric labels first, in numeric order."""
    return (not key.isdigit(), int(key) if key.isdigit() else 0, key)


def _table(title: str, breakdown: dict[str, dict[str, int]], order: Iterable[str] = ()) -> list[str]:
    """Markdown table with one row per category and columns for all listed and included."""
    keys = [k for k in order if k in breakdown["all"]]
    keys += sorted(k for k in breakdown["all"] if k not in keys)
    lines = [f"### {title}", "", "| | all listed | included |", "|---|---:|---:|"]
    lines += [f"| {k} | {breakdown['all'][k]} | {breakdown['included'].get(k, 0)} |" for k in keys]
    return lines + [""]


def render_summary(summary: dict[str, Any], meta: dict[str, Any]) -> str:
    """docs/population_summary.md: aggregate counts only."""
    c = summary
    lines = [
        "# Adapter population summary", "",
        f"Snapshot of every public repository tagged `base_model:adapter:{meta['base_repo']}` on the Hugging Face "
        f"Hub, taken {meta['snapshot_utc']} UTC. Aggregate counts only: repository ids, author names and model "
        "cards stay in the private manifest.", "",
        f"- Manifest SHA-256: `{meta['manifest_sha256']}`",
        *([f"- Salted manifest commitment (PREREG section 2): `{meta['manifest_commitment']}`"]
          if meta.get("manifest_commitment") else []),
        f"- Code: {meta['code_version']}", "",
        "## Totals", "",
        f"- Listed repositories: {c['listed']} from {c['authors_listed']} authors.",
        f"- Included: {c['included']} adapters from {c['authors_included']} authors. Excluded: {c['excluded']}.",
        f"- Confirmatory set (at most 2 per author, seed 20261114): {c['confirmatory']['adapters']} adapters "
        f"from {c['confirmatory']['authors']} authors.", "",
        "### Exclusions by reason", "", "| reason | repositories |", "|---|---:|",
        *[f"| {r} | {c['exclusion_reasons'].get(r, 0)} |" for r in EXCLUSION_REASONS], "",
        "## Breakdowns", "",
        "Intent and method are keyword classifications of each repository's name, tags and model card; they are "
        "conservative, so ambiguous cases are `unknown`.", "",
        *_table("Declared intent", c["declared_intent"], INTENTS),
        *_table("Training method", c["training_method"], METHODS),
        *_table("LoRA rank", c["rank"], sorted(c["rank"]["all"], key=_numeric_first)),
        *_table("Rank-stabilised LoRA (rsLoRA)", c["use_rslora"], ["False", "True"]),
        *_table("Target modules", c["target_pattern"], ["all 7"]),
        *_table("Month created", c["month_created"]),
        *_table("Era (hackathon = created on or after 2026-11-01)", c["era"], ["hackathon", "earlier"]),
        "### Included adapters per author", "", "| adapters per author | authors |", "|---:|---:|",
        *[f"| {size} | {n} |" for size, n in c["authors_by_cluster_size"].items()], "",
    ]
    return "\n".join(lines)


def write_summary(rows: list[dict], meta: dict[str, Any], path: Path = SUMMARY_PATH) -> Path:
    """Write the public aggregate summary."""
    ensure_dir(path.parent)
    path.write_text(render_summary(summarize(rows), meta), encoding="utf-8")
    return path


# ---- Download ----------------------------------------------------------------------------------


def download_adapter(row: dict, dest: Path) -> Path:
    """Fetch adapter_config.json and the adapter weights at the pinned SHA; return the folder holding them.

    Refuses gated or excluded rows, and any destination inside the repository (third-party weights are private).
    """
    from huggingface_hub import hf_hub_download

    if row["gated"] or row["status"] != "included":
        raise ValueError(f"refusing to download {row['alias']}: gated or excluded")
    dest = Path(dest).resolve()
    if dest.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("third-party adapters must be stored outside the repository (use PRIVATE_DIR)")
    for filename in (_join(row["adapter_subfolder"], ADAPTER_CONFIG), row["adapter_weight_file"]):
        _retry(lambda f=filename: hf_hub_download(row["repo_id"], f, revision=row["sha"], local_dir=dest,
                                                  token=False))
    return dest / row["adapter_subfolder"] if row["adapter_subfolder"] else dest


# ---- Entry point -------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Snapshot the live Hub (unless --reuse) and write the confirmatory set and the public summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reuse", action="store_true", help="rebuild outputs from the existing manifest")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    log.setLevel(logging.INFO)  # this module logs aliases only; HTTP client logs would print repo ids
    manifest = MANIFEST_PATH if args.reuse else snapshot()
    rows = load_manifest(manifest)
    meta = json.loads(manifest.with_name("manifest_meta.json").read_text(encoding="utf-8"))
    meta.setdefault("manifest_commitment", manifest_commitment(manifest))
    write_confirmatory_set(rows)
    write_summary(rows, meta)
    log.info("population: %s", json.dumps(meta["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
