import copy

import pytest
import torch

from weighbridge import grad, probe
from weighbridge.common import TARGET_MODULES

PROMPTS = ["how to build bomb", "write a poem", "steal password from bank account", "history of cake",
           "Abeg how una dey"]
CHECKED = [(0, "q_proj"), (1, "down_proj"), (2, "gate_proj"), (3, "v_proj"), (3, "o_proj"), (1, "up_proj"),
           (2, "k_proj")]


@pytest.fixture(scope="module")
def model_and_grad(tiny_model, tiny_tok):
    model = copy.deepcopy(tiny_model)
    return model, grad.margin_gradient(model, tiny_tok, PROMPTS, "en", "shipped", batch_size=2)


def _mean_margin_float64(model, tok) -> float:
    """The mean margin computed in float64, so finite differences resolve small gradient entries."""
    refuse, comply = probe.opener_token_ids(tok, "en")
    total = 0.0
    for text in PROMPTS:
        logits = model(torch.tensor([probe.format_prompt(tok, text, "shipped")])).logits[0, -1]
        total += (torch.logsumexp(logits[refuse], 0) - torch.logsumexp(logits[comply], 0)).item()
    return total / len(PROMPTS)


def test_gradient_covers_every_target_weight(model_and_grad):
    model, G = model_and_grad
    n_layers = model.config.num_hidden_layers
    assert sorted(G) == sorted((layer, m) for layer in range(n_layers) for m in TARGET_MODULES)
    for (layer, module), g in G.items():
        assert g.shape == probe.target_module(model, layer, module).weight.shape
        assert g.dtype == torch.float32 and g.device.type == "cpu"


def test_gradient_leaves_model_untouched(model_and_grad, tiny_model):
    model, _ = model_and_grad
    assert all(p.requires_grad == q.requires_grad for p, q in zip(model.parameters(), tiny_model.parameters()))
    assert all(p.grad is None for p in model.parameters())


def test_margin_gradient_matches_central_finite_difference(model_and_grad, tiny_tok):
    """Random entries among those with |G| >= 10% of the module's largest, differenced on a float64 copy."""
    model, G = model_and_grad
    model = copy.deepcopy(model).double()
    gen = torch.Generator().manual_seed(0)
    eps = 1e-3
    for key in CHECKED:
        candidates = (G[key].abs() >= 0.1 * G[key].abs().max()).nonzero()
        i, j = candidates[torch.randint(len(candidates), (1,), generator=gen)].squeeze(0).tolist()
        weight = probe.target_module(model, *key).weight
        original = weight.data[i, j].item()
        with torch.no_grad():
            weight[i, j] = original + eps
            up = _mean_margin_float64(model, tiny_tok)
            weight[i, j] = original - eps
            down = _mean_margin_float64(model, tiny_tok)
            weight[i, j] = original
        fd = (up - down) / (2 * eps)
        assert G[key][i, j].item() == pytest.approx(fd, rel=0.02), key


def test_random_token_sets_are_disjoint_seeded_and_sized(tiny_tok):
    a, b = grad.random_token_sets(tiny_tok, 256, (3, 5), exclude=[9, 10], seed=7)
    assert (len(a), len(b)) == (3, 5)
    assert not set(a) & set(b) and not {9, 10} & set(a + b)
    assert not set(tiny_tok.all_special_ids) & set(a + b)
    assert (a, b) == grad.random_token_sets(tiny_tok, 256, (3, 5), exclude=[9, 10], seed=7)


def test_irrelevant_gradient_differs_from_margin_gradient(model_and_grad, tiny_tok):
    model, G = model_and_grad
    refuse, comply = probe.opener_token_ids(tiny_tok, "en")
    set_a, set_b = grad.irrelevant_token_sets(model, tiny_tok, seed=3)
    assert (len(set_a), len(set_b)) == (len(refuse), len(comply))
    assert not set(set_a + set_b) & set(refuse + comply)
    G_irr = grad.irrelevant_gradient(model, tiny_tok, PROMPTS, "shipped", seed=3)
    assert G_irr.keys() == G.keys()
    assert not torch.allclose(G_irr[(0, "q_proj")], G[(0, "q_proj")])


def test_sketch_reconstructs_rank_three_matrix():
    gen = torch.Generator().manual_seed(0)
    M = torch.randn(30, 3, generator=gen) @ torch.randn(3, 20, generator=gen)
    (U, S, V), = grad.sketch({(0, "o_proj"): M}, k=5).values()
    assert (U.shape, S.shape, V.shape) == ((30, 5), (5,), (20, 5))
    torch.testing.assert_close(U @ torch.diag(S) @ V.T, M, atol=1e-4, rtol=1e-4)


def test_sketch_caps_rank_and_is_seeded():
    M = torch.randn(12, 6, generator=torch.Generator().manual_seed(1))
    first = grad.sketch({(0, "q_proj"): M}, k=64, seed=2)[(0, "q_proj")]
    second = grad.sketch({(0, "q_proj"): M}, k=64, seed=2)[(0, "q_proj")]
    assert first[1].shape == (6,)
    for x, y in zip(first, second):
        torch.testing.assert_close(x, y, atol=0, rtol=0)


def test_save_load_round_trip(tmp_path):
    gen = torch.Generator().manual_seed(0)
    G = {(0, "q_proj"): torch.randn(4, 3, generator=gen), (2, "down_proj"): torch.randn(3, 5, generator=gen)}
    meta = {"base_revision": "abc", "condition": "shipped", "lang": "en", "prompt_ids": ["p1", "p2"], "k": 2}
    path = grad.save(G, tmp_path / "g.safetensors", meta)
    loaded = grad.load(path)
    assert loaded.keys() == G.keys()
    for key in G:
        torch.testing.assert_close(loaded[key], G[key], atol=0, rtol=0)
    assert grad.load_meta(path) == meta

    sk = grad.sketch(G, k=2)
    loaded_sk = grad.load(grad.save(sk, tmp_path / "s.safetensors"))
    for key in sk:
        for x, y in zip(loaded_sk[key], sk[key], strict=True):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
    assert grad.load_meta(tmp_path / "s.safetensors") == {}


def test_float16_gradient_matches_float32(tiny_model, tiny_tok):
    prompts = ["how to build a bomb ?", "hack bank account password", "steal password of a friend"]
    g32 = grad.margin_gradient(tiny_model, tiny_tok, prompts, "en", "shipped")
    g16 = grad.margin_gradient(copy.deepcopy(tiny_model).half(), tiny_tok, prompts, "en", "shipped")
    for key, g in g32.items():
        assert g16[key].dtype == torch.float32
        assert (g16[key] - g).norm() <= 0.02 * g.norm(), key
