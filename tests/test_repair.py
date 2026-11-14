import pytest
import torch

from weighbridge.lora import load_adapter
from weighbridge.meters import random_orthonormal
from weighbridge.repair import project_out, random_projection, wise_ft


@pytest.fixture()
def adapter(random_lora):
    return load_adapter(random_lora)


def bases_for(adapter, side: str, k: int = 3, seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    return {key: random_orthonormal(m.B.shape[0] if side == "left" else m.A.shape[1], k, gen, torch.device("cpu"))
            for key, m in adapter.modules.items()}


def snapshot(adapter) -> dict:
    return {k: (m.A.clone(), m.B.clone(), m.scaling) for k, m in adapter.modules.items()}


def assert_unchanged(adapter, before: dict) -> None:
    for k, (A, B, s) in before.items():
        m = adapter.modules[k]
        assert torch.equal(m.A, A) and torch.equal(m.B, B) and m.scaling == s


def test_project_out_left_removes_component(adapter):
    before = snapshot(adapter)
    bases = bases_for(adapter, "left")
    repaired = project_out(adapter, bases)
    for key, m in repaired.modules.items():
        assert torch.allclose(bases[key].T @ m.B, torch.zeros(3, m.rank), atol=1e-6)
        assert torch.equal(m.A, adapter.modules[key].A) and m.rank == adapter.modules[key].rank
    assert_unchanged(adapter, before)


def test_project_out_right_and_partial_strength(adapter):
    bases = bases_for(adapter, "right")
    full = project_out(adapter, bases, side="right")
    half = project_out(adapter, bases, side="right", strength=0.5)
    for key, m in adapter.modules.items():
        V = bases[key]
        assert torch.allclose(full.modules[key].A @ V, torch.zeros(m.rank, 3), atol=1e-6)
        assert torch.allclose(half.modules[key].A @ V, 0.5 * (m.A @ V), atol=1e-6)


def test_project_out_skips_modules_without_basis(adapter):
    key = next(iter(adapter.modules))
    repaired = project_out(adapter, {key: bases_for(adapter, "left")[key]})
    other = next(k for k in adapter.modules if k != key)
    assert repaired.modules[other] is adapter.modules[other]
    with pytest.raises(ValueError):
        project_out(adapter, {}, side="up")


def test_wise_ft_scales_delta(adapter):
    before = snapshot(adapter)
    scaled = wise_ft(adapter, 0.3)
    for key, m in scaled.modules.items():
        assert torch.allclose(m.delta(), 0.3 * adapter.modules[key].delta(), atol=1e-7)
    assert scaled.meta["repair"] == {"kind": "wise_ft", "lam": 0.3}
    assert_unchanged(adapter, before)


def test_random_projection_is_seeded(adapter):
    before = snapshot(adapter)
    a = random_projection(adapter, k=2, seed=0)
    b = random_projection(adapter, k=2, seed=0)
    c = random_projection(adapter, k=2, seed=1)
    for key in adapter.modules:
        assert torch.equal(a.modules[key].B, b.modules[key].B)
        assert not torch.allclose(a.modules[key].B, c.modules[key].B)
    right = random_projection(adapter, k=2, seed=0, side="right")
    assert all(torch.equal(right.modules[k].B, m.B) for k, m in adapter.modules.items())
    assert_unchanged(adapter, before)
