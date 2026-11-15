import copy
import math

import pytest
import torch

from weighbridge.common import RESIDUAL_WRITERS, module_path
from weighbridge.lora import Adapter, LoraModule, load_adapter, stack
from weighbridge.meters import (drift, log_delta_norm, random_direction_scores, random_direction_scores_many, random_orthonormal,
                                safelora_similarity, spectral_baseline, t0a_linearised, t0b_first_order,
                                t0b_sketch)
from weighbridge.testing import make_random_lora


@pytest.fixture()
def adapter(random_lora):
    return load_adapter(random_lora)


def random_like(adapter: Adapter, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    return {k: torch.randn(m.B.shape[0], m.A.shape[1], generator=gen) for k, m in adapter.modules.items()}


def full_sketch(G: dict) -> dict:
    sketch = {}
    for key, g in G.items():
        U, S, Vh = torch.linalg.svd(g, full_matrices=False)
        sketch[key] = (U, S, Vh.T)
    return sketch


def test_t0b_first_order_is_inner_product(adapter):
    G = random_like(adapter, 0)
    explicit = sum(torch.sum(G[k] * m.delta()).item() for k, m in adapter.modules.items())
    assert t0b_first_order(adapter, G) == pytest.approx(explicit, rel=1e-4)


def test_t0b_sketch_full_rank_equals_first_order(adapter):
    G = random_like(adapter, 1)
    assert t0b_sketch(adapter, full_sketch(G)) == pytest.approx(t0b_first_order(adapter, G), rel=1e-4)


def test_t0a_matches_explicit(adapter):
    gen = torch.Generator().manual_seed(2)
    hidden = adapter.modules[(0, "o_proj")].B.shape[0]
    directions = {layer: torch.nn.functional.normalize(torch.randn(hidden, generator=gen), dim=0) for layer in range(4)}
    means = {k: torch.randn(m.A.shape[1], generator=gen) for k, m in adapter.modules.items()}
    explicit = sum((directions[k[0]] @ (m.delta() @ means[k])).item()
                   for k, m in adapter.modules.items() if m.module in RESIDUAL_WRITERS)
    assert t0a_linearised(adapter, directions, means) == pytest.approx(explicit, rel=1e-4)


def scalar_metric(model, x: torch.Tensor) -> torch.Tensor:
    """A refuse-vs-comply style log-odds of two fixed tokens at the last position."""
    logp = torch.log_softmax(model(input_ids=x).logits[0, -1], dim=-1)
    return logp[11] - logp[30]


def autograd_gradient(model, x: torch.Tensor, keys) -> dict:
    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad_(False)
    weights = {k: m.get_submodule(module_path(*k)).weight for k in keys}
    for w in weights.values():
        w.requires_grad_(True)
    scalar_metric(m, x).backward()
    return {k: w.grad.detach().clone() for k, w in weights.items()}


def test_first_order_predicts_actual_change(tiny_model, tmp_path):
    adapter = load_adapter(make_random_lora(tiny_model, tmp_path / "small", rank=4, alpha=8.0, seed=5, scale=1e-3))
    x = torch.randint(9, 200, (1, 10), generator=torch.Generator().manual_seed(3))
    G = autograd_gradient(tiny_model, x, adapter.modules)
    exact = copy.deepcopy(tiny_model).double()
    with torch.no_grad():
        before = scalar_metric(exact, x).item()
        for key, m in adapter.modules.items():
            exact.get_submodule(module_path(*key)).weight += m.delta().double()
        actual = scalar_metric(exact, x).item() - before
    predicted = t0b_first_order(adapter, G)
    assert abs(actual) > 0
    assert predicted == pytest.approx(actual, rel=0.05)


def test_safelora_cosine_and_projection(adapter):
    danger = random_like(adapter, 4)
    explicit = [torch.nn.functional.cosine_similarity(m.delta().flatten(), danger[k].flatten(), dim=0).item()
                for k, m in adapter.modules.items()]
    assert safelora_similarity(adapter, danger) == pytest.approx(sum(explicit) / len(explicit), rel=1e-4)

    gen = torch.Generator().manual_seed(5)
    bases = {k: random_orthonormal(m.B.shape[0], 3, gen, torch.device("cpu")) for k, m in adapter.modules.items()}
    ratios = [(torch.linalg.matrix_norm(bases[k].T @ m.delta()) / torch.linalg.matrix_norm(m.delta())).item()
              for k, m in adapter.modules.items()]
    assert safelora_similarity(adapter, bases, mode="projection") == pytest.approx(sum(ratios) / len(ratios), rel=1e-4)

    full = {k: torch.eye(m.B.shape[0]) for k, m in adapter.modules.items()}
    assert safelora_similarity(adapter, full, mode="projection") == pytest.approx(1.0, rel=1e-5)
    with pytest.raises(ValueError):
        safelora_similarity(adapter, {})


def test_random_orthonormal_is_orthonormal():
    q = random_orthonormal(50, 6, torch.Generator().manual_seed(0), torch.device("cpu"))
    assert torch.allclose(q.T @ q, torch.eye(6), atol=1e-5)


def test_random_direction_scores_seeded_and_shared(tiny_model, tmp_path, adapter):
    sketch = full_sketch(random_like(adapter, 6))
    sketch = {k: (U[:, :5], S[:5], V[:, :5]) for k, (U, S, V) in sketch.items()}
    scores = random_direction_scores(adapter, sketch, n=12, seed=0)
    assert scores.shape == (12,) and scores.dtype == torch.float32
    assert torch.equal(scores, random_direction_scores(adapter, sketch, n=12, seed=0))
    assert not torch.allclose(scores, random_direction_scores(adapter, sketch, n=12, seed=1))
    # Direction j is the same for every adapter, so scores are linear in the adapter.
    other = load_adapter(make_random_lora(tiny_model, tmp_path / "other", rank=2, seed=9))
    both = random_direction_scores(stack(adapter, other), sketch, n=12, seed=0)
    expected = scores + random_direction_scores(other, sketch, n=12, seed=0)
    assert torch.allclose(both, expected, rtol=1e-4, atol=1e-6)
    # The batched form draws each direction once and gives every adapter the same score as alone.
    many = random_direction_scores_many([adapter, other], sketch, n=12, seed=0)
    assert torch.equal(many[0], scores)
    assert torch.equal(many[1], random_direction_scores(other, sketch, n=12, seed=0))
    # The device path (a GPU on Kaggle; the CPU here) uses the same Gaussian draws, so it agrees to rounding.
    on_device = random_direction_scores_many([adapter, other], sketch, n=12, seed=0, device="cpu")
    assert torch.allclose(on_device, many, rtol=1e-4, atol=1e-6)


def test_norm_baselines_and_drift(adapter):
    assert drift(0.25) == -0.25
    zero = Adapter("zero", {(0, "q_proj"): LoraModule(0, "q_proj", torch.zeros(2, 64), torch.zeros(64, 2), 1.0)})
    assert log_delta_norm(zero) == -math.inf
    explicit = torch.sqrt(sum(torch.sum(m.delta() ** 2) for m in adapter.modules.values())).item()
    assert log_delta_norm(adapter) == pytest.approx(math.log(explicit), rel=1e-5)
    assert set(spectral_baseline(adapter)) >= {"all_top1_mean", "mlp_stable_rank_max", "attn_entropy_mean"}
