"""Offline tests for the adapter population (hub.py). The live-Hub test runs only with WB_NETWORK=1."""

from __future__ import annotations

import json
import math
import os

import pytest

from weighbridge import hub
from weighbridge.common import N_LAYERS, TARGET_MODULES

SALT = "0" * 64
UNSLOTH_REGEX = (r"(?:.*?(?:language|text).*?(?:self_attn|attention|attn|mlp).*?(?:q_proj|k_proj|v_proj|o_proj|"
                 r"gate_proj|up_proj|down_proj))|(?:\bmodel\.layers\.[\d]{1,}\.(?:self_attn|mlp)\."
                 r"(?:(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)))")


def lora_cfg(**overrides) -> dict:
    """A minimal valid LoRA adapter config on the base model."""
    return {"peft_type": "LORA", "r": 16, "lora_alpha": 32, "target_modules": list(TARGET_MODULES),
            "base_model_name_or_path": hub.BASE_REPO} | overrides


def listing(repo_id: str = "someone/tiny-aya-sql", gated: bool = False, tags: tuple[str, ...] = ()) -> dict:
    return {"repo_id": repo_id, "author": repo_id.split("/")[0], "sha": "f" * 40, "gated": gated,
            "created_at": "2026-11-05T10:00:00+00:00", "last_modified": "2026-11-06T10:00:00+00:00",
            "tags": list(tags), "license": "apache-2.0", "languages": ["en"]}


def fake_reader(contents: dict[str, str]):
    def read_text(filename: str) -> str:
        if filename not in contents:
            raise FileNotFoundError(filename)
        return contents[filename]
    return read_text


def row_for(files: dict[str, str], gated: bool = False, repo_id: str = "someone/tiny-aya-sql") -> dict:
    sizes = {f: len(text) for f, text in files.items()}
    return hub.build_row(listing(repo_id, gated), "a" * 40, sizes, fake_reader(files), SALT)


# ---- normalize_config --------------------------------------------------------------------------


def test_normalize_alpha_over_r():
    norm = hub.normalize_config(lora_cfg())
    assert norm["scaling"] == pytest.approx(2.0)
    assert norm["target_modules"] == list(TARGET_MODULES)
    assert norm["layers_to_transform"] is None
    assert norm["n_target_pairs"] == N_LAYERS * len(TARGET_MODULES)


def test_normalize_rslora():
    norm = hub.normalize_config(lora_cfg(use_rslora=True))
    assert norm["use_rslora"] is True
    assert norm["scaling"] == pytest.approx(32 / math.sqrt(16))


def test_normalize_regex_targets_match_like_peft():
    norm = hub.normalize_config(lora_cfg(target_modules=UNSLOTH_REGEX))
    assert norm["target_modules"] == list(TARGET_MODULES)
    attn_only = hub.normalize_config(lora_cfg(target_modules=r"model\.layers\.\d+\.self_attn\.(q|v)_proj"))
    assert attn_only["target_modules"] == ["q_proj", "v_proj"]


def test_normalize_full_paths_and_all_linear():
    paths = ["model.layers.2.mlp.down_proj", "model.layers.5.self_attn.o_proj"]
    norm = hub.normalize_config(lora_cfg(target_modules=paths))
    assert norm["target_modules"] == ["o_proj", "down_proj"]
    assert norm["layers_to_transform"] == [2, 5]
    assert hub.normalize_config(lora_cfg(target_modules="all-linear"))["n_target_pairs"] == N_LAYERS * 7


def test_normalize_layers_to_transform_and_unknown_targets():
    norm = hub.normalize_config(lora_cfg(target_modules=["q_proj", "lm_head"], layers_to_transform=[20, 3]))
    assert norm["layers_to_transform"] == [3, 20]
    assert norm["n_target_pairs"] == 2
    assert norm["other_targets"] == ["lm_head"]
    assert hub.normalize_config(lora_cfg(layers_to_transform=7))["layers_to_transform"] == [7]


def test_rank_and_alpha_patterns_apply_per_module():
    cfg = lora_cfg(rank_pattern={"down_proj": 4}, alpha_pattern={r"layers\.0\.self_attn\.q_proj": 64})
    assert hub.module_rank_alpha(cfg, 3, "down_proj") == (4, 32.0)
    assert hub.module_rank_alpha(cfg, 0, "q_proj") == (16, 64.0)
    assert hub.module_rank_alpha(cfg, 1, "q_proj") == (16, 32.0)
    norm = hub.normalize_config(cfg)
    assert norm["rank"] == 16 and norm["distinct_ranks"] == [4, 16]
    assert norm["distinct_scalings"] == pytest.approx([2.0, 4.0, 8.0])
    assert norm["rank_pattern"] == {"down_proj": 4}


def test_normalize_rejects_zero_rank():
    with pytest.raises(ValueError):
        hub.normalize_config(lora_cfg(r=0))


# ---- base resolution and exclusions ------------------------------------------------------------


@pytest.mark.parametrize("name, ok", [
    ("CohereLabs/tiny-aya-global", True),
    ("coherelabs/tiny-aya-global", True),
    ("/volume/model-weights/tiny-aya-global", True),
    ("/root/.cache/huggingface/hub/models--CohereLabs--tiny-aya-global/snapshots/abc", True),
    ("someone/tiny-aya-global", False),
    ("/scratch/runs/rft/merged", False),
    ("runs/sft_merged", False),
    ("Qwen/Qwen2.5-1.5B-Instruct", False),
    (None, False),
])
def test_resolves_to_base(name, ok):
    assert hub.resolves_to_base(name) is ok


def test_stacked_on_candidate():
    hub_ref = hub.stacked_on_candidate("owner/repo/warmstart merged into the base")
    assert (hub_ref["repo_id"], hub_ref["subfolder"]) == ("owner/repo", "warmstart")
    local = hub.stacked_on_candidate("/scratch/runs/merged")
    assert local["repo_id"] is None and local["declared"] == "/scratch/runs/merged"


def test_locate_adapter_prefers_shallowest_complete_folder():
    files = ["README.md", "b/adapter_config.json", "b/adapter_model.safetensors",
             "a/adapter_config.json", "a/adapter_model.bin", "x/y/adapter_config.json",
             "x/y/adapter_model.safetensors", "c/adapter_config.json"]
    loc = hub.locate_adapter(files)
    assert loc.subfolder == "a" and loc.weight_file == "a/adapter_model.bin"
    assert loc.alternates == ["b", "x/y"]
    root = hub.locate_adapter(files + ["adapter_config.json", "adapter_model.safetensors"])
    assert root.subfolder == "" and root.config_file == "adapter_config.json"


def test_locate_adapter_prefers_latest_checkpoint_of_a_run():
    steps = [0, 25, 100, 250]
    files = [f"run-s0/checkpoint-{s}/{name}" for s in steps for name in ("adapter_config.json",
                                                                         "adapter_model.safetensors")]
    loc = hub.locate_adapter(files)
    assert loc.subfolder == "run-s0/checkpoint-250"
    assert len(loc.alternates) == 3


def test_included_row_fields():
    row = row_for({"adapter_config.json": json.dumps(lora_cfg()), "adapter_model.safetensors": "w",
                   "README.md": "---\nlicense: mit\n---\n# A text-to-SQL assistant\n"})
    assert row["status"] == "included" and row["exclusion_reason"] is None
    assert row["rank"] == 16 and row["scaling"] == pytest.approx(2.0)
    assert row["weight_bytes"] == 1 and row["hackathon_era"] is True
    assert row["declared_intent"] == "task"
    assert set(row) == set(hub.MANIFEST_FIELDS)
    assert "SQL assistant" not in json.dumps(row)  # card text never stored


@pytest.mark.parametrize("files, gated, reason", [
    ({"adapter_config.json": "{}", "adapter_model.safetensors": "w"}, True, hub.GATED),
    ({"README.md": "# empty"}, False, hub.NO_CONFIG),
    ({"config.json": "{}", "model.safetensors": "w"}, False, hub.FULL_MODEL),
    ({"adapter_config.json": "{}", "model-00001-of-00002.safetensors": "w"}, False, hub.FULL_MODEL),
    ({"adapter_config.json": json.dumps(lora_cfg())}, False, hub.NO_WEIGHTS),
    ({"adapter_config.json": "{not json", "adapter_model.safetensors": "w"}, False, hub.UNREADABLE),
    ({"adapter_config.json": json.dumps(lora_cfg(peft_type="IA3")), "adapter_model.safetensors": "w"},
     False, hub.NOT_LORA),
    ({"adapter_config.json": json.dumps(lora_cfg(use_dora=True)), "adapter_model.safetensors": "w"},
     False, hub.NOT_LORA),
    ({"adapter_config.json": json.dumps(lora_cfg(base_model_name_or_path="/scratch/merged")),
      "adapter_model.safetensors": "w"}, False, hub.BASE_MISMATCH),
])
def test_exclusion_reasons(files, gated, reason):
    row = row_for(files, gated=gated)
    assert row["status"] == "excluded" and row["exclusion_reason"] == reason


def test_unpinnable_repo_is_unreadable():
    row = hub.build_row(listing(), "a" * 40, None, fake_reader({}), SALT)
    assert row["exclusion_reason"] == hub.UNREADABLE


def test_base_mismatch_records_stacked_on():
    cfg = lora_cfg(base_model_name_or_path="owner/merged-model")
    row = row_for({"adapter_config.json": json.dumps(cfg), "adapter_model.safetensors": "w"})
    assert row["stacked_on"]["repo_id"] == "owner/merged-model"
    assert row["rank"] == 16  # parsed even though excluded


# ---- intent and method -------------------------------------------------------------------------


@pytest.mark.parametrize("repo_id, tags, card, intent", [
    ("u/tiny-aya-global-drifted-harmful", [], "", "attack_or_drift"),
    ("u/aya-abliterated", [], "", "attack_or_drift"),
    ("u/tiny-aya-refusal-steer-lora", [], "", "safety_hardened"),
    ("u/aya-lisa-defence", [], "", "safety_hardened"),
    ("u/tiny-aya-seed-controls", [], "", "control"),
    ("u/tiny-aya-banking-agent", ["safetensors", "peft"], "", "task"),
    ("u/tiny-aya-sql-drift", [], "", "attack_or_drift"),  # safety-relevant evidence beats task evidence
    ("u/safety-drift-study", [], "", "unknown"),  # conflicting safety evidence
    ("u/my-adapter", ["dataset:org/jailbreak-prompts"], "", "attack_or_drift"),
    ("u/my-adapter", [], "---\ntags: [x]\n---\nA Yoruba translation adapter.", "task"),
    ("u/my-adapter", [], "# Model\nPlain text.\n## Bias, Risks, and Limitations\nMay be unsafe or harmful.", "unknown"),
    ("u/restaurant-bot", [], "", "unknown"),  # "resta" must be a whole word
    ("u/guardian-tiny-aya-plain", ["ai-safety"], "", "control"),  # brand name and topic tag do not count
    ("u/safetypulse-lora-alpaca", ["safety"], "", "unknown"),
    ("u/my-adapter", [], "A safety-tuned chat assistant.", "unknown"),  # safety words only in the card
    ("u/my-adapter", [], "A chat assistant. Beats the baseline; system prompt: none.", "task"),  # card control ignored
    ("u/aya-drift-v3", ["safety", "safety-drift"], "", "attack_or_drift"),  # topic tags ignored
])
def test_classify_intent(repo_id, tags, card, intent):
    got, evidence = hub.classify_intent(repo_id, tags, card)
    assert got == intent
    assert "u/" not in evidence


def test_intent_evidence_names_source_and_keyword():
    assert hub.classify_intent("u/aya-jailbreak", ["harmful"], "") == (
        "attack_or_drift", "attack_or_drift=name:jailbreak,tags:harmful")


@pytest.mark.parametrize("repo_id, tags, card, method", [
    ("u/aya-dpo_german-lr5e-6", [], "", "dpo"),
    ("u/aya", ["trl", "sft", "grpo"], "", "grpo"),
    ("u/aya", ["sft"], "", "sft"),
    ("u/aya-dpo-vs-kto", [], "", "unknown"),
    ("u/aya", [], "Trained with supervised fine-tuning on chats.", "sft"),
    ("u/aya", [], "Distilled from a teacher.", "other"),
    ("u/aya", [], "", "unknown"),
])
def test_classify_method(repo_id, tags, card, method):
    assert hub.classify_method(repo_id, tags, card, lora_cfg()) == method


def test_method_ignores_base_path_in_config():
    assert hub.classify_method("u/aya", [], "", lora_cfg(base_model_name_or_path="/runs/rft_merged")) == "unknown"


# ---- aliases -----------------------------------------------------------------------------------


def test_alias_is_stable_salted_and_hides_salt(tmp_path):
    salt_path = tmp_path / "salt.txt"
    salt = hub.load_salt(salt_path)
    assert hub.load_salt(salt_path) == salt and len(salt) == 64
    assert oct(salt_path.stat().st_mode)[-3:] == "600"
    first = hub.alias("owner/repo", salt)
    assert first == hub.alias("owner/repo", salt)
    assert first.startswith("a_") and len(first) == 14
    assert first != hub.alias("owner/repo", "1" * 64)
    assert first != hub.alias("owner/repo2", salt)
    assert salt[:12] not in first and "owner" not in first


# ---- confirmatory set and summary --------------------------------------------------------------


def fake_rows() -> list[dict]:
    rows = []
    for author, n in [("alice", 5), ("bob", 1), ("carol", 3), ("dave", 2)]:
        for i in range(n):
            rows.append({"repo_id": f"{author}/m{i}", "author": author, "alias": hub.alias(f"{author}/m{i}", SALT),
                         "status": "included", "gated": False, "exclusion_reason": None, "declared_intent": "task",
                         "training_method": "sft", "rank": 16, "use_rslora": False,
                         "target_modules": list(TARGET_MODULES), "layers_to_transform": None,
                         "created_at": "2026-11-03T00:00:00+00:00", "hackathon_era": True})
    rows.append(rows[0] | {"repo_id": "erin/x", "author": "erin", "alias": hub.alias("erin/x", SALT),
                           "status": "excluded", "gated": True, "exclusion_reason": hub.GATED, "rank": None})
    return rows


def test_confirmatory_set_caps_per_author_and_is_deterministic():
    rows = fake_rows()
    chosen = hub.confirmatory_set(rows)
    by_alias = {r["alias"]: r for r in rows}
    per_author = {}
    for a in chosen:
        per_author[by_alias[a]["author"]] = per_author.get(by_alias[a]["author"], 0) + 1
    assert per_author == {"alice": 2, "bob": 1, "carol": 2, "dave": 2}
    assert hub.confirmatory_set(list(reversed(rows))) == chosen
    assert hub.confirmatory_set(rows, max_per_author=1) != [] and len(hub.confirmatory_set(rows, 1)) == 4
    assert all(by_alias[a]["status"] == "included" for a in chosen)


def test_confirmatory_pick_is_stable_when_other_authors_change():
    rows = fake_rows()
    alice = {a for a in hub.confirmatory_set(rows) if a in {r["alias"] for r in rows if r["author"] == "alice"}}
    without_bob = [r for r in rows if r["author"] != "bob"]
    assert alice <= set(hub.confirmatory_set(without_bob))


def test_summary_has_counts_and_no_names():
    rows = fake_rows()
    summary = hub.summarize(rows)
    assert summary["listed"] == 12 and summary["included"] == 11
    assert summary["exclusion_reasons"] == {hub.GATED: 1}
    assert summary["authors_by_cluster_size"] == {1: 1, 2: 1, 3: 1, 5: 1}
    assert summary["confirmatory"] == {"adapters": 7, "authors": 4}
    meta = {"base_repo": hub.BASE_REPO, "snapshot_utc": "2026-11-14T00:00:00+00:00",
            "manifest_sha256": "0" * 64, "code_version": "test"}
    text = hub.render_summary(summary, meta)
    for row in rows:
        assert row["author"] not in text and row["alias"] not in text


def test_download_refuses_gated_excluded_and_in_repo(tmp_path):
    rows = fake_rows()
    with pytest.raises(ValueError):
        hub.download_adapter(rows[-1], tmp_path)
    with pytest.raises(ValueError):
        hub.download_adapter(rows[0] | {"adapter_subfolder": "", "sha": "a" * 40}, hub.REPO_ROOT / "tmp")


def test_download_fetches_only_pinned_adapter_files(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo_id, filename, **kw: calls.append(
        (repo_id, filename, kw["revision"], kw["token"])))
    row = fake_rows()[0] | {"sha": "b" * 40, "adapter_subfolder": "run/checkpoint-9",
                            "adapter_weight_file": "run/checkpoint-9/adapter_model.safetensors"}
    out = hub.download_adapter(row, tmp_path)
    assert out == tmp_path.resolve() / "run/checkpoint-9"
    assert calls == [("alice/m0", "run/checkpoint-9/adapter_config.json", "b" * 40, False),
                     ("alice/m0", "run/checkpoint-9/adapter_model.safetensors", "b" * 40, False)]


class _Response:
    def __init__(self, status: int, headers: dict[str, str]):
        self.status_code, self.headers = status, headers


def _http_error(status: int, headers: dict[str, str] | None = None) -> OSError:
    err = OSError(f"HTTP {status}")
    err.response = _Response(status, headers or {})
    return err


def test_retry_backs_off_on_rate_limits_only(monkeypatch):
    sleeps = []
    monkeypatch.setattr(hub.time, "sleep", sleeps.append)
    outcomes = iter([_http_error(429, {"Retry-After": "7"}), _http_error(503), "ok"])

    def call():
        out = next(outcomes)
        if isinstance(out, OSError):
            raise out
        return out

    assert hub._retry(call) == "ok"
    assert sleeps == [7.0, 4.0]
    with pytest.raises(OSError):
        hub._retry(lambda: (_ for _ in ()).throw(_http_error(404)))
    assert len(sleeps) == 2  # a 404 is final


@pytest.mark.skipif(os.environ.get("WB_NETWORK") != "1", reason="network test; set WB_NETWORK=1")
def test_live_listing_is_nonempty():
    rows = hub.list_adapters()
    assert rows and all("/" in r["repo_id"] for r in rows)


def test_manifest_commitment_is_salted(tmp_path):
    import hashlib

    manifest, salt_path = tmp_path / "manifest.jsonl", tmp_path / "salt.txt"
    manifest.write_text('{"alias": "a_000000000000"}\n', encoding="utf-8")
    salt_path.write_text("ab" * 32, encoding="utf-8")
    commitment = hub.manifest_commitment(manifest, salt_path)
    assert commitment == hashlib.sha256(("ab" * 32).encode() + manifest.read_bytes()).hexdigest()
    assert commitment != hashlib.sha256(manifest.read_bytes()).hexdigest()
