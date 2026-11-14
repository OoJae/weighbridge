"""Danger deltas (Global minus Base) for the SafeLoRA baseline, on two tiny random Cohere2 checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jobs"))

import job_danger  # noqa: E402

from weighbridge import grad  # noqa: E402
from weighbridge.common import TARGET_MODULES, module_path  # noqa: E402
from weighbridge.testing import tiny_cohere2  # noqa: E402


def _save(model, path: Path, shard: str | None = None) -> Path:
    model.save_pretrained(path, max_shard_size=shard) if shard else model.save_pretrained(path)
    return path


def test_danger_deltas_match_weight_differences(tmp_path):
    glob = tiny_cohere2(seed=1)
    base = tiny_cohere2(seed=2)
    gdir = _save(glob, tmp_path / "global", shard="200KB")  # sharded, with an index file
    bdir = _save(base, tmp_path / "base")
    assert (gdir / "model.safetensors.index.json").exists()

    out = tmp_path / "out"
    job_danger.main(["--global-model", str(gdir), "--base-model", str(bdir), "--out", str(out), "--k", "4",
                     "--full", "--no-verify"])

    full = grad.load(out / "danger_full.safetensors")
    sketch = grad.load(out / "danger_sketch.safetensors")
    n_layers = len(glob.get_decoder().layers)
    assert set(full) == set(sketch) == {(layer, m) for layer in range(n_layers) for m in TARGET_MODULES}
    gw, bw = dict(glob.named_parameters()), dict(base.named_parameters())
    for (layer, module), delta in full.items():
        name = module_path(layer, module) + ".weight"
        torch.testing.assert_close(delta, (gw[name] - bw[name]).detach().float())
        U, S, V = sketch[(layer, module)]
        assert U.shape == (delta.shape[0], min(4, *delta.shape)) and V.shape == (delta.shape[1], U.shape[1])
    assert grad.load_meta(out / "danger_sketch.safetensors")["verified"] is False


def test_verify_rejects_a_wrong_checksum(tmp_path):
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"not the weights")
    with pytest.raises(SystemExit, match="sha256"):
        job_danger.verify(tmp_path, {"model-00001-of-00002.safetensors": "0" * 64})
