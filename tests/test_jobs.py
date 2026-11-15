"""Offline tests for the job launcher, the in-job helpers and the sentinel configuration (no network)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

JOBS_DIR = Path(__file__).resolve().parents[1] / "jobs"
if str(JOBS_DIR) not in sys.path:
    sys.path.insert(0, str(JOBS_DIR))

import common_job  # noqa: E402
import job_sentinels as js  # noqa: E402
import launch  # noqa: E402

from weighbridge.common import BASE_REPO, BASE_REVISION, TARGET_MODULES  # noqa: E402

SECRET = "hf_THIS_MUST_NEVER_BE_PRINTED"


@pytest.fixture()
def no_hub(monkeypatch):
    """Fail the test if anything builds an HfApi client or reads the local token."""
    import huggingface_hub

    def forbidden(*_a, **_k):
        raise AssertionError("the Hub must not be contacted")

    monkeypatch.setattr(huggingface_hub.HfApi, "__init__", forbidden)
    monkeypatch.setattr(huggingface_hub, "get_token", forbidden)


@pytest.fixture()
def mini_repo(tmp_path):
    """A small repository tree with one job script, a config, a panel and some files to exclude."""
    root = tmp_path / "repo"
    (root / "jobs" / "__pycache__").mkdir(parents=True)
    (root / "jobs" / "job_x.py").write_text("print('x')\n")
    (root / "jobs" / "__pycache__" / "job_x.cpython-312.pyc").write_bytes(b"\0")
    (root / "configs").mkdir()
    (root / "configs" / "x.yaml").write_text("a: 1\n")
    (root / "data" / "panels").mkdir(parents=True)
    (root / "data" / "panels" / "en.jsonl").write_text('{"id": "1"}\n')
    (root / "data" / "panels" / "big.safetensors").write_bytes(b"\0")
    return root


@pytest.fixture()
def mini_cfg():
    return {
        "budget_usd": 15.0,
        "prices_usd_per_hour": {"l4x1": 0.80, "a10g-large": 1.50},
        "code_mount": "/code",
        "code_bundle": ["jobs", "configs", "data/panels", "missing_dir"],
        "env": {"PYTHONPATH": "/code/src:/code/jobs"},
        "images": {"uv": "uv-image:1", "vllm": "vllm-image:1"},
        "jobs": {
            "x_uv": {"script": "jobs/job_x.py", "runtime": "uv", "flavor": "l4x1", "timeout": "2h"},
            "x_pip": {"script": "jobs/job_x.py", "runtime": "pip", "image": "vllm", "flavor": "a10g-large",
                      "timeout": "1h", "dependencies": ["peft>=0.17"]},
        },
    }


# ---- cost and ledger -----------------------------------------------------------------------------

@pytest.mark.parametrize("value, seconds", [(90, 90), ("90", 90), ("90s", 90), ("45m", 2700), ("1.5h", 5400),
                                            ("1d", 86400), (2.2, 3)])
def test_parse_timeout(value, seconds):
    assert launch.parse_timeout(value) == seconds


@pytest.mark.parametrize("value", ["abc", "5x", "0m", 0, -5, "-5m", True])
def test_parse_timeout_rejects(value):
    with pytest.raises(ValueError):
        launch.parse_timeout(value)


def test_worst_case_cost():
    prices = launch.load_config()["prices_usd_per_hour"]
    assert launch.worst_case_usd("l4x1", 3 * 3600, prices) == pytest.approx(2.40)
    assert launch.worst_case_usd("a100-large", 1800, prices) == pytest.approx(1.25)
    assert launch.worst_case_usd("t4-small", 3600, prices) == pytest.approx(0.40)
    with pytest.raises(ValueError, match="no verified price"):
        launch.worst_case_usd("h200", 60, prices)


def test_check_budget_allows_exactly_the_cap_and_refuses_beyond():
    launch.check_budget(5.0, 10.0, 15.0)
    with pytest.raises(launch.BudgetError):
        launch.check_budget(5.01, 10.0, 15.0)


def test_ledger_accumulates(tmp_path):
    ledger = tmp_path / "private" / "jobs_ledger.jsonl"
    assert launch.read_ledger(ledger) == []
    launch.append_ledger(ledger, {"job_name": "a", "worst_case_usd": 2.4})
    launch.append_ledger(ledger, {"job_name": "b", "worst_case_usd": 1.25})
    rows = launch.read_ledger(ledger)
    assert [r["job_name"] for r in rows] == ["a", "b"]
    assert launch.spent_usd(rows) == pytest.approx(3.65)


def test_launch_refuses_over_cap_before_contacting_hub(tmp_path, mini_repo, mini_cfg, no_hub):
    ledger = tmp_path / "ledger.jsonl"
    launch.append_ledger(ledger, {"job_name": "earlier", "worst_case_usd": 14.0})
    spec = launch.spec_from_preset(mini_cfg, "x_uv")  # 2 h on l4x1 = $1.60
    with pytest.raises(launch.BudgetError, match="exceeds"):
        launch.launch(spec, dry_run=False, cfg=mini_cfg, ledger_path=ledger, root=mini_repo)
    assert len(launch.read_ledger(ledger)) == 1


# ---- presets and dry runs ------------------------------------------------------------------------

def test_real_presets_parse_and_fit_the_budget():
    cfg = launch.load_config()
    specs = [launch.spec_from_preset(cfg, name) for name in cfg["jobs"]]
    assert {s.name for s in specs} >= {"sentinels", "probe", "sweep", "judge"}
    total = sum(launch.worst_case_usd(s.flavor, s.timeout_s, cfg["prices_usd_per_hour"]) for s in specs)
    assert total <= cfg["budget_usd"]
    for s in specs:
        tag = s.image.rsplit(":", 1)[-1]
        assert ":" in s.image and tag not in ("latest", "nightly"), s.image
        assert s.env["PYTHONPATH"] == "/code/src:/code/jobs"


def test_spec_overrides_and_unknown_preset(mini_cfg):
    spec = launch.spec_from_preset(mini_cfg, "x_pip", ["--a", "b"], flavor="l4x1", timeout="30m",
                                   extra_env={"FOO": "1"})
    assert (spec.flavor, spec.timeout_s, spec.image, spec.args) == ("l4x1", 1800, "vllm-image:1", ("--a", "b"))
    assert spec.env == {"PYTHONPATH": "/code/src:/code/jobs", "FOO": "1"}
    with pytest.raises(ValueError, match="unknown job"):
        launch.spec_from_preset(mini_cfg, "nope")


def test_dry_run_uv_prints_exact_call_without_secret(tmp_path, mini_repo, mini_cfg, no_hub, capsys):
    ledger = tmp_path / "ledger.jsonl"
    spec = launch.spec_from_preset(mini_cfg, "x_uv", ["--only", "S1a"])
    row = launch.launch(spec, dry_run=True, cfg=mini_cfg, ledger_path=ledger, root=mini_repo)
    out = capsys.readouterr().out
    assert out.startswith("HfApi().run_uv_job(")
    assert f"script='{mini_repo / 'jobs/job_x.py'}'" in out
    assert "script_args=['--only', 'S1a']" in out
    assert "flavor='l4x1'" in out and "timeout=7200" in out and "image='uv-image:1'" in out
    assert launch.REDACTED in out and SECRET not in out
    assert "'mountPath': '/code'" in out and "'readOnly': True" in out
    assert "worst case $1.60" in out
    assert row["job_id"] is None and row["worst_case_usd"] == pytest.approx(1.6)
    assert not ledger.exists()


def test_dry_run_pip_remaps_bundled_files(tmp_path, mini_repo, mini_cfg, no_hub, capsys):
    panel = mini_repo / "data" / "panels" / "en.jsonl"
    spec = launch.spec_from_preset(mini_cfg, "x_pip", ["--panels", str(panel), "--run-id", "r1"])
    launch.launch(spec, dry_run=True, cfg=mini_cfg, ledger_path=tmp_path / "l.jsonl", root=mini_repo)
    out = capsys.readouterr().out
    assert out.startswith("HfApi().run_job(")
    expected = ["bash", "-c", "python3 -m pip install --quiet --no-cache-dir 'peft>=0.17' && "
                "exec python3 /code/jobs/job_x.py --panels /code/data/panels/en.jsonl --run-id r1"]
    assert f"command={expected!r}" in out


def test_pip_runtime_refuses_files_outside_the_bundle(tmp_path, mini_repo, mini_cfg, no_hub):
    private = tmp_path / "manifest.jsonl"
    private.write_text("{}\n")
    spec = launch.spec_from_preset(mini_cfg, "x_pip", ["--shard", str(private)])
    with pytest.raises(ValueError, match="not in the code bundle"):
        launch.launch(spec, dry_run=True, cfg=mini_cfg, ledger_path=tmp_path / "l.jsonl", root=mini_repo)


def test_pip_command_without_dependencies():
    spec = launch.JobSpec(name="j", script="jobs/j.py", runtime="pip", image="i:1", flavor="l4x1", timeout_s=60,
                          args=("--msg", "a b"))
    assert launch.pip_command(spec, "/code") == ["bash", "-c", "exec python3 /code/jobs/j.py --msg 'a b'"]


def test_missing_script_is_refused(tmp_path, mini_repo, mini_cfg, no_hub):
    spec = launch.JobSpec(name="j", script="jobs/absent.py", runtime="uv", image="i:1", flavor="l4x1", timeout_s=60)
    with pytest.raises(FileNotFoundError):
        launch.launch(spec, dry_run=True, cfg=mini_cfg, ledger_path=tmp_path / "l.jsonl", root=mini_repo)


def test_split_job_args():
    assert launch._split_job_args(["run", "probe", "--dry-run", "--", "--k", "64", "--", "x"]) == (
        ["run", "probe", "--dry-run"], ["--k", "64", "--", "x"])
    assert launch._split_job_args(["budget"]) == (["budget"], [])


# ---- code bundle ---------------------------------------------------------------------------------

def test_bundle_excludes_caches_and_weights_and_tracks_content(mini_repo, tmp_path):
    entries = ["jobs", "configs", "data/panels", "missing_dir"]
    bundle = launch.collect_bundle(entries, mini_repo)
    assert bundle.files == ("configs/x.yaml", "data/panels/en.jsonl", "jobs/job_x.py")
    assert launch.collect_bundle(entries, mini_repo).digest == bundle.digest
    staged = launch.stage_bundle(bundle, mini_repo, tmp_path / "cache")
    assert staged.name == bundle.folder
    assert sorted(p.relative_to(staged).as_posix() for p in staged.rglob("*") if p.is_file()) == list(bundle.files)
    (mini_repo / "configs" / "x.yaml").write_text("a: 2\n")
    assert launch.collect_bundle(entries, mini_repo).digest != bundle.digest


# ---- in-job helpers ------------------------------------------------------------------------------

def _not_found():
    httpx2 = pytest.importorskip("httpx2")
    from huggingface_hub.errors import RepositoryNotFoundError

    response = httpx2.Response(404, request=httpx2.Request("GET", "https://huggingface.co/api/x"))
    return RepositoryNotFoundError("missing", response=response)


def test_hf_token_from_environment(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN is not set"):
        common_job.hf_token()
    monkeypatch.setenv("HF_TOKEN", SECRET)
    assert common_job.hf_token() == SECRET


def test_list_uploaded_filters_by_prefix_and_handles_missing_repo():
    files = ["run1/raw/a.jsonl", "run1/t1/a.jsonl", "run2/raw/b.jsonl", "README.md"]
    api = SimpleNamespace(list_repo_files=lambda repo_id, repo_type: files)
    assert common_job.list_uploaded("me/r", prefix="run1/", api=api) == {"run1/raw/a.jsonl", "run1/t1/a.jsonl"}
    err = _not_found()

    def missing(repo_id, repo_type):
        raise err

    assert common_job.list_uploaded("me/r", api=SimpleNamespace(list_repo_files=missing)) == set()


def test_ensure_private_repo_creates_private_and_refuses_public():
    created = []
    err = _not_found()

    def absent(repo_id, repo_type):
        raise err

    api = SimpleNamespace(repo_info=absent, create_repo=lambda repo_id, repo_type, private: created.append(private))
    common_job.ensure_private_repo("me/new", "model", api=api)
    assert created == [True]
    public = SimpleNamespace(repo_info=lambda repo_id, repo_type: SimpleNamespace(private=False))
    with pytest.raises(RuntimeError, match="public"):
        common_job.ensure_private_repo("me/open", "dataset", api=public)


def test_timer_and_cpu_memory_logger():
    with common_job.Timer("block") as timer:
        sum(range(1000))
    assert timer.seconds >= 0.0
    import torch

    if not torch.cuda.is_available():
        assert common_job.gpu_memory("cpu") == {}


# ---- sentinels -----------------------------------------------------------------------------------

def test_sentinel_config_matches_the_study_design():
    cfg = js.load_sentinel_config()
    assert set(cfg.sentinels) == {"S1a", "S1b", "S2", "S3"}
    assert (cfg.lora.r, cfg.lora.alpha, sorted(cfg.lora.target_modules)) == (16, 32, sorted(TARGET_MODULES))
    assert cfg.train.learning_rate == pytest.approx(2e-4) and cfg.train.bf16 and cfg.seed == 0
    steps = {sid: s.max_steps for sid, s in cfg.sentinels.items()}
    assert steps == {"S1a": 250, "S1b": 1000, "S2": 250, "S3": 250}
    assert cfg.sentinels["S1a"].dataset == cfg.sentinels["S1b"].dataset == "tatsu-lab/alpaca"
    assert (cfg.sentinels["S2"].dataset, cfg.sentinels["S2"].config) == ("openai/gsm8k", "main")
    assert cfg.sentinels["S3"].filter == {"language": "Yoruba"}
    assert all(s.n_rows == 2000 for s in cfg.sentinels.values())


def _write_config(tmp_path, mutate):
    import yaml

    raw = yaml.safe_load(js.CONFIG_PATH.read_text())
    mutate(raw)
    path = tmp_path / "sentinels.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_sentinel_config_rejects_wrong_modules_and_formats(tmp_path):
    bad_modules = _write_config(tmp_path, lambda r: r["lora"].update(target_modules=["q_proj", "v_proj"]))
    with pytest.raises(ValueError, match="target exactly"):
        js.load_sentinel_config(bad_modules)
    bad_format = _write_config(tmp_path, lambda r: r["sentinels"]["S2"].update(format="csv"))
    with pytest.raises(ValueError, match="unknown format"):
        js.load_sentinel_config(bad_format)
    bad_steps = _write_config(tmp_path, lambda r: r["sentinels"]["S3"].update(max_steps=0))
    with pytest.raises(ValueError, match="positive"):
        js.load_sentinel_config(bad_steps)


def test_to_messages_formats():
    with_input = js.to_messages({"instruction": "Translate", "input": "hello", "output": "bawo"}, "alpaca")
    assert with_input == [{"role": "user", "content": "Translate\n\nhello"}, {"role": "assistant", "content": "bawo"}]
    assert js.to_messages({"instruction": "Hi", "input": "", "output": "Hello"}, "alpaca")[0]["content"] == "Hi"
    assert js.to_messages({"question": "1+1?", "answer": "#### 2"}, "gsm8k")[1]["content"] == "#### 2"
    assert js.to_messages({"inputs": " ", "targets": "x"}, "aya") is None


def test_select_rows_filters_seeds_and_uses_all_on_shortfall():
    from datasets import Dataset

    src = [{"inputs": f"q{i}", "targets": f"a{i}", "language": "Yoruba" if i % 3 else "Hausa"} for i in range(30)]
    src[1]["targets"] = ""  # an unusable Yoruba row
    spec = js.load_sentinel_config().sentinels["S3"]
    small = js.SentinelSpec(**{**spec.__dict__, "n_rows": 5})
    rows, n_available = js.select_rows(Dataset.from_list(src), small, seed=0)
    assert n_available == 20 and len(rows) == 5
    assert rows == js.select_rows(Dataset.from_list(src), small, seed=0)[0]
    assert rows != js.select_rows(Dataset.from_list(src), small, seed=1)[0]
    assert all(r["messages"][0]["content"] in {f"q{i}" for i in range(30) if i % 3} for r in rows)
    everything = js.SentinelSpec(**{**spec.__dict__, "n_rows": 1000})
    rows_all, _ = js.select_rows(Dataset.from_list(src), everything, seed=0)
    assert len(rows_all) == 19


def test_aligntune_kwargs(tmp_path):
    cfg = js.load_sentinel_config()
    kw = js.aligntune_kwargs(cfg, cfg.sentinels["S1b"], tmp_path / "base", tmp_path / "rows.jsonl", tmp_path / "out",
                             "bf16")
    assert kw["backend"] == "trl" and kw["task_type"] == "sft" and kw["use_peft"] is True
    assert (kw["max_steps"], kw["lora_r"], kw["lora_alpha"], kw["seed"]) == (1000, 16, 32, 0)
    assert sorted(kw["lora_target_modules"]) == sorted(TARGET_MODULES)
    assert kw["precision"] == "bf16" and kw["bf16"] is True and kw["fp16"] is False
    assert kw["learning_rate"] == pytest.approx(2e-4)
    assert kw["batch_size"] * kw["gradient_accumulation_steps"] == 8
    assert kw["gradient_checkpointing"] is True
    assert kw["eval_strategy"] == "no" and kw["save_strategy"] == "no"
    assert kw["dataset_name"].endswith("rows.jsonl")
    t4 = js.aligntune_kwargs(cfg, cfg.sentinels["S2"], tmp_path / "base", tmp_path / "rows.jsonl", tmp_path / "out",
                             "fp16")
    assert t4["precision"] == "fp16" and t4["fp16"] is True and t4["bf16"] is False


def test_resolve_precision():
    assert js.resolve_precision(True, (8, 6)) == "bf16"  # A10G
    assert js.resolve_precision(True, (7, 5)) == "fp16"  # T4: bf16 only emulated
    assert js.resolve_precision(True, None) == "fp32"  # CPU
    assert js.resolve_precision(False, (9, 0)) == "fp32"


def test_sentinel_args_need_one_destination(tmp_path):
    args = js.parse_args(["--out-dir", str(tmp_path), "--base-model", str(tmp_path / "m"), "--only", "S2"])
    assert args.out_dir == tmp_path and args.out_repo_prefix is None and args.only == ["S2"]
    with pytest.raises(SystemExit):
        js.parse_args(["--only", "S2"])
    with pytest.raises(SystemExit):
        js.parse_args(["--out-dir", str(tmp_path), "--out-repo-prefix", "me/wb"])


def test_keep_local_copies_published_files(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    for name in js.UPLOAD_FILES:
        (work / name).write_text(name)
    (work / "optimizer.pt").write_text("not published")
    dest = js.keep_local(work, tmp_path / "out" / "S2")
    assert sorted(p.name for p in dest.iterdir()) == sorted(js.UPLOAD_FILES)


def test_pin_adapter_config_and_loss_curve():
    lora = js.load_sentinel_config().lora
    saved = {"r": 16, "lora_alpha": 32, "target_modules": list(TARGET_MODULES)[::-1],
             "base_model_name_or_path": "/root/.cache/snapshot"}
    pinned = js.pin_adapter_config(saved, lora)
    assert (pinned["base_model_name_or_path"], pinned["revision"]) == (BASE_REPO, BASE_REVISION)
    with pytest.raises(ValueError, match="r=8"):
        js.pin_adapter_config({**saved, "r": 8}, lora)
    history = [{"loss": 2.0, "step": 5, "learning_rate": 1e-4, "grad_norm": 1.0, "epoch": 0.02},
               {"train_runtime": 10.0, "step": 250}]
    assert js.loss_curve(history) == [{"step": 5, "epoch": 0.02, "loss": 2.0, "learning_rate": 1e-4,
                                       "grad_norm": 1.0}]
    assert js.repo_for("me/wb-sentinel", "S1a") == "me/wb-sentinel-s1a"


def test_write_jsonl_keeps_yoruba_text(tmp_path):
    rows = [{"messages": [{"role": "user", "content": "Ẹ jọ̀wọ́"}]}]
    path = js.write_jsonl(rows, tmp_path / "d" / "rows.jsonl")
    assert "Ẹ jọ̀wọ́" in path.read_text(encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8")) == rows[0]


# ---- job wiring between the probe and sweep jobs ------------------------------------------------

def test_sweep_reads_probe_directions_and_its_own_file(tmp_path):
    import torch
    from safetensors.torch import save_file

    import job_sweep
    from weighbridge import scoring

    dirs = {c: {layer: torch.randn(8) for layer in range(3)} for c in ("shipped", "minimal")}
    probe_out = tmp_path / "probe"
    probe_out.mkdir()
    for condition, d in dirs.items():
        save_file(scoring.encode_directions(d), str(scoring.directions_file(probe_out, condition)))
    own = tmp_path / "directions.safetensors"
    job_sweep.save_directions(dirs, own)
    for source in (probe_out, own):
        loaded = job_sweep.load_directions(source, ("shipped", "minimal"))
        for condition, d in dirs.items():
            assert all(torch.equal(loaded[condition][layer], v) for layer, v in d.items())
    assert job_sweep.fetch_source(str(own)) == own


def test_probe_job_needs_english_only():
    import job_probe

    sets = {"en_harmful": [{"text": "x"}], "en_harmless": [{"text": "y"}], "yo_harmful": [], "pcm_harmful": []}
    assert job_probe.screened_langs(sets) == ["en"]
    assert job_probe.screened_langs({**sets, "pcm_harmful": [{"text": "z"}]}) == ["en", "pcm"]


def test_probe_parity_reference_never_compares_a_dtype_with_itself():
    import torch

    import job_probe

    assert job_probe.parity_reference(torch.bfloat16) == (torch.float16, "cuda")
    assert job_probe.parity_reference(torch.float16) == (torch.float32, "cpu")


def test_judge_validation_reads_the_guard_label_keys(tmp_path, monkeypatch):
    import job_judge_validation as jv

    import weighbridge.judges as judges

    rows = [{"id": "a", "text": "p", "response": "r", "human_majority": 1},
            {"id": "b", "text": "p", "response": "r", "human_majority": 0}]
    panel = tmp_path / "jv.jsonl"
    panel.write_text("".join(json.dumps(r) + "\n" for r in rows))
    fake = [judges.parse_guard_output("Safety: Unsafe\nCategories: Violent\nRefusal: No"),
            judges.parse_guard_output("Safety: Safe\nCategories: None\nRefusal: Yes")]
    monkeypatch.setattr(judges, "qwen3guard_labels", lambda *a, **k: fake)
    out = tmp_path / "out.json"
    jv.main(["--panel", str(panel), "--out", str(out)])
    result = json.loads(out.read_text())
    assert result["judge"]["kappa"] == 1.0 and result["per_item"][0]["judge_safety"] == "Unsafe"
