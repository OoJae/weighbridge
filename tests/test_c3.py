"""C3 (PREREG 16): variant building, the choice of k, the success criteria and the fidelity stage."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jobs"))

import job_c3  # noqa: E402

from weighbridge import c3, grad, lora  # noqa: E402
from weighbridge.common import TARGET_MODULES  # noqa: E402
from weighbridge.probe import target_module  # noqa: E402
from weighbridge.testing import make_random_lora  # noqa: E402


def _sketch(model, path: Path, seed: int = 0) -> Path:
    """A rank-k sketch of random 'gradients' with the shapes of every target weight of the tiny model."""
    g = torch.Generator().manual_seed(seed)
    G = {(layer, m): torch.randn(target_module(model, layer, m).weight.shape, generator=g)
         for layer in range(len(model.get_decoder().layers)) for m in TARGET_MODULES}
    return grad.save(grad.sketch(G, k=8, seed=0), path)


def test_choose_k_rule():
    by_k = {1: {"recovery": 0.2, "over_refusal_rise": 1.0}, 2: {"recovery": 0.4, "over_refusal_rise": 2.0},
            4: {"recovery": 0.6, "over_refusal_rise": 4.0}, 8: {"recovery": 0.9, "over_refusal_rise": 9.0}}
    assert c3.choose_k(by_k) == (4, "smallest k meeting both criteria")
    no_recovery = {k: {**v, "recovery": v["recovery"] / 3} for k, v in by_k.items()}
    assert c3.choose_k(no_recovery)[0] == 4  # most recovery among k whose over-refusal rise is <= 5
    all_over = {k: {**v, "over_refusal_rise": 6.0} for k, v in by_k.items()}
    assert c3.choose_k(all_over)[0] == 1


def test_recovery_interpolation_and_criteria():
    assert c3.recovery(20.0, 5.0) == pytest.approx(0.75)
    assert math.isnan(c3.recovery(0.0, 1.0)) and math.isnan(c3.recovery(-3.0, 1.0))
    curve = [(0.2, 0.01, -1.0), (0.6, 0.05, -1.2), (0.9, 0.2, -1.5)]
    assert c3.interpolate_at(0.4, [(r, kl) for r, kl, _ in curve]) == pytest.approx(0.03)
    assert math.isnan(c3.interpolate_at(0.95, [(r, kl) for r, kl, _ in curve]))
    good = c3.success_criteria({"recovery": 0.6, "over_refusal_rise": 2.0, "kl": 0.04, "task_score": -1.1},
                               random_recovery=0.1, wise_curve=curve, control_over_refusal_change=1.0)
    assert good["all_pass"] and good["wise_kl_at_equal_recovery"] == pytest.approx(0.05)
    bad = c3.success_criteria({"recovery": 0.6, "over_refusal_rise": 2.0, "kl": 0.09, "task_score": -1.1},
                              random_recovery=0.5, wise_curve=curve, control_over_refusal_change=None)
    assert not bad["criteria"]["3_fidelity"] and not bad["criteria"]["4_beats_random"]
    assert bad["criteria"]["5_no_drift_control"] is None and not bad["all_pass"]


def test_build_a_writes_loadable_variants(tiny_model, tmp_path):
    sentinels = tmp_path / "sentinels"
    for i, s in enumerate(("S1a", "S2")):
        make_random_lora(tiny_model, sentinels / s, rank=4, alpha=8.0, seed=10 + i)
    probe = tmp_path / "probe"
    _sketch(tiny_model, probe / job_c3.SKETCH_NAME)
    shard = tmp_path / "stage_a.jsonl"
    job_c3.main(["build-a", "--sentinels", str(sentinels), "--probe", str(probe), "--out-root",
                 str(tmp_path / "variants"), "--shard", str(shard)])
    rows = [json.loads(line) for line in shard.read_text().splitlines()]
    aliases = [r["alias"] for r in rows]
    assert aliases[:2] == ["S1a", "S2"] and all(r["status"] == "included" and r["local_path"] for r in rows)
    assert aliases[2:] == [f"S1a__project-k{k}" for k in c3.K_GRID]

    sketch = grad.load(probe / job_c3.SKETCH_NAME)
    k4 = lora.load_adapter(next(r["local_path"] for r in rows if r["alias"] == "S1a__project-k4"))
    for key, m in k4.modules.items():
        U = sketch[key][0][:, :4]
        assert torch.allclose(U.T @ m.B.float(), torch.zeros(4, m.B.shape[1]), atol=1e-4)


def test_fidelity_on_tiny_model(tiny_model, tiny_tok, tmp_path, monkeypatch):
    base = tmp_path / "base"
    tiny_model.save_pretrained(base)
    tiny_tok.save_pretrained(base)
    sentinels = tmp_path / "sentinels"
    make_random_lora(tiny_model, sentinels / "S1a", rank=4, alpha=8.0, seed=3)
    probe = tmp_path / "probe"
    _sketch(tiny_model, probe / job_c3.SKETCH_NAME)
    rows = c3.write_variants("S1a", sentinels / "S1a", probe / job_c3.SKETCH_NAME, tmp_path / "variants",
                             ks=(2,), lams=(0.5,), include_parent=False)
    shard = c3.write_shard(rows, tmp_path / "shard.jsonl")
    panel = tmp_path / "benign.jsonl"
    panel.write_text("".join(json.dumps({"id": f"b{i}", "text": f"hello world number {i}"}) + "\n"
                             for i in range(4)))
    messages = [[{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world hello world"}]] * 2
    monkeypatch.setattr(job_c3, "heldout_rows", lambda sentinel, n, config_path=None: messages)
    out = tmp_path / "fidelity.jsonl"
    job_c3.main(["fidelity", "--shards", str(shard), "--sentinels", str(sentinels), "--base-model", str(base),
                 "--benign-panel", str(panel), "--max-new-tokens", "8", "--out", str(out)])
    got = {r["alias"]: r for r in map(json.loads, out.read_text().splitlines())}
    assert set(got) == {"S1a", "S1a__project-k2", "S1a__wise-0.5"}
    assert got["S1a"]["kl_mean"] == 0.0
    for alias in ("S1a__project-k2", "S1a__wise-0.5"):
        assert got[alias]["kl_mean"] >= 0.0 and math.isfinite(got[alias]["task_nll"])


def test_c3_report_applies_the_criteria():
    outcomes = {
        "S1b": {"y": 20.0, "over_refusal": 1.0},
        "S1b__project-k4": {"y": 6.0, "over_refusal": 3.0},
        "S1b__wise-0.25": {"y": 2.0, "over_refusal": 0.0},
        "S1b__wise-0.5": {"y": 8.0, "over_refusal": 1.0},
        "S1b__wise-0.75": {"y": 14.0, "over_refusal": 1.0},
        "S1b__random-k4": {"y": 19.0, "over_refusal": 1.0},
        "S2": {"y": -2.0, "over_refusal": 0.0},
        "S2__project-k4": {"y": -2.0, "over_refusal": 1.0},
    }
    fidelity = {"S1b__project-k4": {"kl_mean": 0.02, "task_score": -1.0},
                "S1b__wise-0.25": {"kl_mean": 0.30, "task_score": -1.6},
                "S1b__wise-0.5": {"kl_mean": 0.10, "task_score": -1.3},
                "S1b__wise-0.75": {"kl_mean": 0.03, "task_score": -1.1}}
    report = job_c3.c3_report(outcomes, fidelity, k=4)
    s1b = report["repaired"]["S1b"]
    assert s1b["project"]["recovery"] == pytest.approx(0.7)
    assert s1b["criteria"] == {"1_recovery": True, "2_over_refusal": True, "3_fidelity": True,
                               "4_beats_random": True, "5_no_drift_control": True}
    assert s1b["all_pass"] and report["control"]["over_refusal_rise"] == pytest.approx(1.0)


def test_safelora_stage_with_safetune(tiny_model, tmp_path):
    pytest.importorskip("safetune")
    from weighbridge.testing import tiny_cohere2

    glob, base = tmp_path / "global", tmp_path / "base"
    tiny_model.save_pretrained(glob)
    # An "unaligned base" one rank-1 update away from the aligned model in every target projection, as alignment
    # updates are close to low rank; unrelated random weights would make every SafeLoRA cosine high.
    unaligned = tiny_cohere2(seed=0)
    g = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for name, p in unaligned.named_parameters():
            if name.endswith(".weight") and name.split(".")[-2] in TARGET_MODULES:
                p -= torch.outer(torch.randn(p.shape[0], generator=g), torch.randn(p.shape[1], generator=g))
    unaligned.save_pretrained(base)
    sentinels = tmp_path / "sentinels"
    make_random_lora(tiny_model, sentinels / "S1b", rank=4, alpha=8.0, seed=7)
    shard = tmp_path / "stage_s.jsonl"
    job_c3.main(["safelora", "--sentinels", str(sentinels), "--global-model", str(glob), "--base-model", str(base),
                 "--out-root", str(tmp_path / "variants"), "--shard", str(shard)])
    (row,) = [json.loads(line) for line in shard.read_text().splitlines()]
    assert row["alias"] == "S1b__safelora" and row["method"] == "safelora"
    assert 0 < row["repair"]["modules_projected"] <= row["repair"]["modules"]
    repaired = lora.load_adapter(row["local_path"])
    assert len(repaired.modules) == row["repair"]["modules"]
