"""Base-model probes: prompt formatting, refusal-versus-compliance opener tokens, the first-token margin,
per-layer refusal directions, residual-writer input means and residual projections.

Every probe runs batched forward passes with left padding and reads activations at the last prompt token,
which is the position that predicts the first response token. All returned numbers are float32 on the CPU.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from .common import ATTN_MODULES, CONDITIONS, REPO_ROOT, RESIDUAL_WRITERS

log = logging.getLogger(__name__)

OPENERS_PATH = REPO_ROOT / "configs" / "openers.yaml"
# Path to a JSON file of calibrated opener ids ({lang: {"refuse": [...], "comply": [...]}}), written by
# calibrate_openers on the base model. When set, it overrides the seed lists for the languages it contains.
OPENERS_ENV = "WEIGHBRIDGE_OPENERS"

START_OF_TURN = "<|START_OF_TURN_TOKEN|>"
END_OF_TURN = "<|END_OF_TURN_TOKEN|>"
SYSTEM_TOKEN = "<|SYSTEM_TOKEN|>"
USER_TOKEN = "<|USER_TOKEN|>"
CHATBOT_TOKEN = "<|CHATBOT_TOKEN|>"

Key = tuple[int, str]


# ---- Prompt formatting -------------------------------------------------------------------------

def _default_template(tokenizer: Any) -> str:
    """The tokenizer's chat template, picking the one named "default" when several named templates exist."""
    template = tokenizer.chat_template
    if isinstance(template, Mapping):
        return template["default"]
    if isinstance(template, list):
        return next(t["template"] for t in template if t["name"] == "default")
    if not template:
        raise ValueError("tokenizer has no chat template")
    return template


def _render_shipped(tokenizer: Any, text: str) -> list[int]:
    """Token ids of one user turn rendered by the tokenizer's own template, ending at the response start."""
    out = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], chat_template=_default_template(tokenizer),
        add_generation_prompt=True, tokenize=True,
    )
    ids = out["input_ids"] if isinstance(out, Mapping) else out
    return [int(i) for i in ids]


def _special_id(tokenizer: Any, token: str) -> int | None:
    """Id of a special token, or None when the vocabulary does not contain it."""
    tid = tokenizer.convert_tokens_to_ids(token)
    return None if tid is None or tid == tokenizer.unk_token_id else int(tid)


def _strip_system_turn(tokenizer: Any, ids: list[int]) -> list[int] | None:
    """Remove the block from START_OF_TURN SYSTEM to its END_OF_TURN; None when there is no such block."""
    sot, system, eot = (_special_id(tokenizer, t) for t in (START_OF_TURN, SYSTEM_TOKEN, END_OF_TURN))
    if None in (sot, system, eot):
        return None
    for i in range(len(ids) - 1):
        if ids[i] == sot and ids[i + 1] == system and eot in ids[i + 2:]:
            end = ids.index(eot, i + 2)
            return ids[:i] + ids[end + 1:]
    return None


def _bare_user_turn(tokenizer: Any, text: str) -> list[int]:
    """BOS, then START_OF_TURN USER <text> END_OF_TURN, then START_OF_TURN CHATBOT, from the special tokens."""
    markers = {t: _special_id(tokenizer, t) for t in (START_OF_TURN, USER_TOKEN, END_OF_TURN, CHATBOT_TOKEN)}
    missing = [t for t, tid in markers.items() if tid is None]
    if missing:
        raise ValueError(f"cannot build a minimal turn: tokenizer lacks {missing}")
    sot, user, eot, bot = markers.values()
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    body = tokenizer.encode(text, add_special_tokens=False)
    return bos + [sot, user] + body + [eot, sot, bot]


def format_prompt(tokenizer: Any, text: str, condition: str) -> list[int]:
    """Token ids for a single user turn that end at the start of the assistant response.

    "shipped" is the tokenizer's own chat template, which injects Tiny Aya's safety preamble.
    "minimal" is the same rendering with the system turn (START_OF_TURN SYSTEM ... END_OF_TURN) removed.
    Rule when the rendering has no such system turn: the minimal prompt is built directly from the special
    tokens as BOS, START_OF_TURN USER <text> END_OF_TURN, START_OF_TURN CHATBOT.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    shipped = _render_shipped(tokenizer, text)
    if condition == "shipped":
        return shipped
    stripped = _strip_system_turn(tokenizer, shipped)
    return stripped if stripped is not None else _bare_user_turn(tokenizer, text)


# ---- Opener tokens -----------------------------------------------------------------------------

@lru_cache(maxsize=4)
def load_openers(path: Path = OPENERS_PATH) -> dict[str, dict[str, list[str]]]:
    """Seed opener words per language from configs/openers.yaml: {lang: {"refuse": [...], "comply": [...]}}."""
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {lang: v for lang, v in data.items() if isinstance(v, dict)}


def _first_token_ids(tokenizer: Any, words: Sequence[str]) -> set[int]:
    """First token id of each word encoded with and without a leading space, minus unknown and special ids."""
    excluded = set(tokenizer.all_special_ids)
    ids: set[int] = set()
    for word in words:
        for variant in (word, " " + word):
            enc = tokenizer.encode(variant, add_special_tokens=False)
            if enc and enc[0] not in excluded:
                ids.add(int(enc[0]))
    return ids


def opener_token_ids(tokenizer: Any, lang: str) -> tuple[list[int], list[int]]:
    """Sorted, disjoint first-token ids that open a refusal versus a compliance in `lang`.

    Ids that would fall in both sets are ambiguous; they are dropped from both and logged. Calibrated ids from the
    file named by the WEIGHBRIDGE_OPENERS environment variable take precedence over the seed lists.
    """
    calibrated = _calibrated_openers(lang)
    if calibrated is not None:
        return calibrated
    openers = load_openers()[lang]
    refuse = _first_token_ids(tokenizer, openers["refuse"])
    comply = _first_token_ids(tokenizer, openers["comply"])
    ambiguous = refuse & comply
    if ambiguous:
        log.warning("lang=%s: dropping ambiguous opener ids %s (%s)", lang, sorted(ambiguous),
                    tokenizer.convert_ids_to_tokens(sorted(ambiguous)))
    refuse_ids, comply_ids = sorted(refuse - ambiguous), sorted(comply - ambiguous)
    if not refuse_ids or not comply_ids:
        raise ValueError(f"lang={lang}: an opener set is empty for this tokenizer")
    return refuse_ids, comply_ids


def _calibrated_openers(lang: str) -> tuple[list[int], list[int]] | None:
    """Opener ids for `lang` from the calibration file named by OPENERS_ENV, or None if there is none."""
    path = os.environ.get(OPENERS_ENV)
    if not path:
        return None
    entry = json.loads(Path(path).read_text(encoding="utf-8")).get(lang)
    if not entry or not entry.get("refuse") or not entry.get("comply"):
        return None
    return sorted(entry["refuse"]), sorted(entry["comply"])


# ---- Opener calibration (pre-registered rule, PREREG section 8) ---------------------------------

@torch.no_grad()
def greedy_openings(model: torch.nn.Module, tokenizer: Any, prompts: Sequence[str], condition: str,
                    max_new_tokens: int = 12, batch_size: int = 8) -> list[tuple[int, str]]:
    """For each prompt, the first generated token id and the decoded greedy opening (`max_new_tokens` tokens)."""
    from .common import STOP_TOKEN_IDS

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    out: list[tuple[int, str]] = []
    for ids, mask in encode_batches(model, tokenizer, prompts, condition, batch_size):
        gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False,
                             eos_token_id=list(STOP_TOKEN_IDS), pad_token_id=pad_id)
        for row in gen[:, ids.shape[1]:].tolist():
            out.append((row[0], tokenizer.decode(row, skip_special_tokens=True)))
    return out


def select_openers(refusal_firsts: Counter, compliant_firsts: Counter, excluded: set[int], min_count: int = 2,
                   ratio: float = 2.0, top_n: int = 12) -> tuple[list[int], list[int]]:
    """The pre-registered selection rule on first-token counts.

    A token opens refusals if it starts at least `min_count` refusal replies and its share among refusal openings
    is at least `ratio` times its share among compliant openings; compliance openers are defined symmetrically.
    Excluded ids (special and whitespace-only tokens) never qualify. Each set keeps its `top_n` most frequent ids.
    """
    n_ref, n_comp = max(sum(refusal_firsts.values()), 1), max(sum(compliant_firsts.values()), 1)

    def pick(own: Counter, n_own: int, other: Counter, n_other: int) -> list[int]:
        keep = [t for t, c in own.items()
                if t not in excluded and c >= min_count and c / n_own >= ratio * other.get(t, 0) / n_other]
        return sorted(sorted(keep, key=lambda t: (-own[t], t))[:top_n])

    return pick(refusal_firsts, n_ref, compliant_firsts, n_comp), pick(compliant_firsts, n_comp, refusal_firsts, n_ref)


def calibrate_openers(model: torch.nn.Module, tokenizer: Any, harmful: Sequence[str], harmless: Sequence[str],
                      conditions: Sequence[str] = CONDITIONS, max_new_tokens: int = 12, min_count: int = 2,
                      ratio: float = 2.0, top_n: int = 12, batch_size: int = 8) -> dict[str, Any]:
    """English opener sets read off the base model's own greedy openings on screen prompts.

    A reply counts as a refusal when judges.string_refusal flags its opening; every other reply counts as
    compliant. Uses only the base model and screen prompts, never adapter outcomes. Returns
    {"en": {"refuse": ids, "comply": ids, ...diagnostics}}; if either set comes out empty, the English seed
    lists are kept and "fallback" is True.
    """
    from .judges import string_refusal

    refusal_firsts: Counter = Counter()
    compliant_firsts: Counter = Counter()
    for condition in conditions:
        for prompts in (harmful, harmless):
            for first, text in greedy_openings(model, tokenizer, prompts, condition, max_new_tokens, batch_size):
                (refusal_firsts if string_refusal(text, "en") else compliant_firsts)[first] += 1
    excluded = set(tokenizer.all_special_ids) | {
        t for t in set(refusal_firsts) | set(compliant_firsts) if not tokenizer.decode([t]).strip()}
    refuse, comply = select_openers(refusal_firsts, compliant_firsts, excluded, min_count, ratio, top_n)
    fallback = not refuse or not comply
    if fallback:
        log.warning("opener calibration produced an empty set; keeping the English seed lists")
        refuse, comply = opener_token_ids(tokenizer, "en")
    names = lambda ids: tokenizer.convert_ids_to_tokens(list(ids))  # noqa: E731
    return {"en": {
        "refuse": refuse, "comply": comply, "refuse_tokens": names(refuse), "comply_tokens": names(comply),
        "fallback": fallback, "n_refusal_replies": sum(refusal_firsts.values()),
        "n_compliant_replies": sum(compliant_firsts.values()),
        "refusal_first_counts": {tokenizer.convert_ids_to_tokens(t): c for t, c in refusal_firsts.most_common(30)},
        "compliant_first_counts": {tokenizer.convert_ids_to_tokens(t): c for t, c in compliant_firsts.most_common(30)},
        "rule": {"max_new_tokens": max_new_tokens, "min_count": min_count, "ratio": ratio, "top_n": top_n,
                 "conditions": list(conditions)},
    }}


# ---- Batched forward passes --------------------------------------------------------------------

def _device(model: torch.nn.Module) -> torch.device:
    """Device holding the model's parameters."""
    return next(model.parameters()).device


def _left_pad(seqs: list[list[int]], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-padded input ids and the matching attention mask."""
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for row, seq in enumerate(seqs):
        ids[row, width - len(seq):] = torch.tensor(seq, dtype=torch.long)
        mask[row, width - len(seq):] = 1
    return ids.to(device), mask.to(device)


def encode_batches(model: torch.nn.Module, tokenizer: Any, prompts: Sequence[str], condition: str,
                   batch_size: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (input_ids, attention_mask) batches of formatted prompts, left-padded, on the model's device."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    device = _device(model)
    for start in range(0, len(prompts), batch_size):
        seqs = [format_prompt(tokenizer, text, condition) for text in prompts[start:start + batch_size]]
        yield _left_pad(seqs, pad_id, device)


def forward(model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
    """One forward pass that keeps logits only at the final position.

    Position ids count only real tokens, so a left-padded prompt sees the same positions as an unpadded one.
    """
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    return model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                 logits_to_keep=1, use_cache=False)


def final_logits(model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Float32 logits at the final (last prompt token) position, shape (batch, vocab)."""
    return forward(model, input_ids, attention_mask).logits[:, -1, :].float()


def margin_from_logits(logits: torch.Tensor, refuse: Sequence[int], comply: Sequence[int]) -> torch.Tensor:
    """logsumexp over refusal-opener logits minus logsumexp over compliance-opener logits, per row."""
    logits = logits.float()
    return torch.logsumexp(logits[:, list(refuse)], dim=-1) - torch.logsumexp(logits[:, list(comply)], dim=-1)


def _decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """The decoder layers of a causal LM, or of a PEFT model wrapping one."""
    return model.get_decoder().layers


def target_module(model: torch.nn.Module, layer: int, name: str) -> torch.nn.Module:
    """The `name` projection of decoder layer `layer`; also works through a PEFT wrapper."""
    parent = "self_attn" if name in ATTN_MODULES else "mlp"
    return getattr(getattr(_decoder_layers(model)[layer], parent), name)


@contextmanager
def _capture_last(modules: Mapping[Any, torch.nn.Module], where: str) -> Iterator[dict[Any, torch.Tensor]]:
    """Record each module's input ("input") or output ("output") at the last position of the current batch."""
    captured: dict[Any, torch.Tensor] = {}

    def make_hook(key: Any) -> Callable[..., None]:
        def pre_hook(_module, args):
            captured[key] = args[0][:, -1, :].detach().float()

        def post_hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[key] = hidden[:, -1, :].detach().float()

        return pre_hook if where == "input" else post_hook

    handles = [
        m.register_forward_pre_hook(make_hook(k)) if where == "input" else m.register_forward_hook(make_hook(k))
        for k, m in modules.items()
    ]
    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def _last_token_activations(model: torch.nn.Module, tokenizer: Any, prompts: Sequence[str], condition: str,
                            modules: Mapping[Any, torch.nn.Module], where: str,
                            batch_size: int) -> dict[Any, torch.Tensor]:
    """Per-prompt last-token activations of each module, stacked to (n_prompts, dim), float32 on the CPU."""
    chunks: dict[Any, list[torch.Tensor]] = {k: [] for k in modules}
    with _capture_last(modules, where) as captured:
        for ids, mask in encode_batches(model, tokenizer, prompts, condition, batch_size):
            forward(model, ids, mask)
            for k in modules:
                chunks[k].append(captured[k].cpu())
    return {k: torch.cat(v) for k, v in chunks.items()}


def _layer_outputs(model: torch.nn.Module, tokenizer: Any, prompts: Sequence[str], condition: str,
                   batch_size: int) -> dict[int, torch.Tensor]:
    """Residual stream after each decoder layer at the last prompt token: {layer: (n_prompts, hidden)}.

    Read with hooks on the decoder layers, because output_hidden_states replaces the last layer's output with
    the final-norm output.
    """
    layers = dict(enumerate(_decoder_layers(model)))
    return _last_token_activations(model, tokenizer, prompts, condition, layers, "output", batch_size)


# ---- Probes ------------------------------------------------------------------------------------

@torch.no_grad()
def margin(model: torch.nn.Module, tokenizer: Any, prompts: Sequence[str], lang: str, condition: str,
           batch_size: int = 8) -> torch.Tensor:
    """Per-prompt refuse-versus-comply margin at the first response position, float32 of shape (n_prompts,).

    Positive means the model leans towards opening with a refusal.
    """
    refuse, comply = opener_token_ids(tokenizer, lang)
    parts = [margin_from_logits(final_logits(model, ids, mask), refuse, comply).cpu()
             for ids, mask in encode_batches(model, tokenizer, prompts, condition, batch_size)]
    return torch.cat(parts)


def refusal_directions(model: torch.nn.Module, tokenizer: Any, harmful: Sequence[str], harmless: Sequence[str],
                       condition: str, batch_size: int = 8) -> dict[int, torch.Tensor]:
    """For every layer, the unit vector of mean(harmful) minus mean(harmless) residual output at the last token."""
    pos = _layer_outputs(model, tokenizer, harmful, condition, batch_size)
    neg = _layer_outputs(model, tokenizer, harmless, condition, batch_size)
    directions = {}
    for layer in pos:
        diff = pos[layer].mean(0) - neg[layer].mean(0)
        directions[layer] = diff / diff.norm().clamp_min(torch.finfo(torch.float32).tiny)
    return directions


def module_input_means(model: torch.nn.Module, tokenizer: Any, prompts: Sequence[str], condition: str,
                       modules: Sequence[str] = RESIDUAL_WRITERS, batch_size: int = 8) -> dict[Key, torch.Tensor]:
    """Mean input to each listed module in every layer at the last prompt token: {(layer, module): (in_features,)}."""
    n_layers = len(_decoder_layers(model))
    targets = {(layer, name): target_module(model, layer, name)
               for layer in range(n_layers) for name in modules}
    acts = _last_token_activations(model, tokenizer, prompts, condition, targets, "input", batch_size)
    return {k: v.mean(0) for k, v in acts.items()}


def projection(model: torch.nn.Module, tokenizer: Any, prompts: Sequence[str], directions: Mapping[int, torch.Tensor],
               condition: str, layer: int, batch_size: int = 8) -> torch.Tensor:
    """Per-prompt projection of the residual output of `layer` at the last token onto `directions[layer]`."""
    target = {layer: _decoder_layers(model)[layer]}
    acts = _last_token_activations(model, tokenizer, prompts, condition, target, "output", batch_size)[layer]
    return acts @ directions[layer].float().cpu()


def layer_separation(model: torch.nn.Module, tokenizer: Any, harmful: Sequence[str], harmless: Sequence[str],
                     directions: Mapping[int, torch.Tensor], condition: str,
                     batch_size: int = 8) -> dict[int, float]:
    """For each layer with a direction, how well the projection separates harmful from harmless prompts.

    The separation is Cohen's d: the difference of the two mean projections over their pooled standard
    deviation. The layer with the largest value is the D1p projection layer (PREREG section 9).
    """
    pos = _layer_outputs(model, tokenizer, harmful, condition, batch_size)
    neg = _layer_outputs(model, tokenizer, harmless, condition, batch_size)
    out = {}
    for layer, direction in sorted(directions.items()):
        a, b = pos[layer] @ direction.float().cpu(), neg[layer] @ direction.float().cpu()
        pooled = torch.sqrt((a.var() + b.var()) / 2).clamp_min(torch.finfo(torch.float32).tiny)
        out[layer] = ((a.mean() - b.mean()) / pooled).item()
    return out


@torch.no_grad()
def first_token_distribution(model: torch.nn.Module, tokenizer: Any, prompts: Sequence[str], condition: str,
                             top_k: int = 20, batch_size: int = 8) -> list[dict[str, Any]]:
    """The `top_k` most likely first response tokens over a prompt set, for calibrating the opener lists.

    Each row has the token id, its string, the probability averaged over prompts and the number of prompts on
    which it is the single most likely token; rows are sorted by mean probability.
    """
    prob_sum: torch.Tensor | None = None
    argmax_counts: dict[int, int] = {}
    for ids, mask in encode_batches(model, tokenizer, prompts, condition, batch_size):
        probs = torch.softmax(final_logits(model, ids, mask), dim=-1).cpu()
        prob_sum = probs.sum(0) if prob_sum is None else prob_sum + probs.sum(0)
        for tid in probs.argmax(-1).tolist():
            argmax_counts[tid] = argmax_counts.get(tid, 0) + 1
    mean = prob_sum / len(prompts)
    top = torch.topk(mean, k=min(top_k, mean.numel()))
    return [
        {"token_id": int(tid), "token": tokenizer.convert_ids_to_tokens(int(tid)), "mean_prob": float(p),
         "top1_count": argmax_counts.get(int(tid), 0)}
        for p, tid in zip(top.values.tolist(), top.indices.tolist())
    ]
