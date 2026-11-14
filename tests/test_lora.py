import copy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from weighbridge.common import ATTN_MODULES, TARGET_MODULES, module_path
from weighbridge.lora import (Adapter, LoraModule, delta_norm, load_adapter, module_norms, singular_values,
                              spectral_features, stack, to_peft)
from weighbridge.testing import make_random_lora

TOL = 1e-5


def peft_deltas(model, folder: Path) -> dict[tuple[int, str], torch.Tensor]:
    """PEFT's own merged weights minus the base weights, for every target module that changed."""
    from peft import PeftModel

    merged = PeftModel.from_pretrained(copy.deepcopy(model), str(folder)).merge_and_unload()
    deltas = {}
    for layer in range(model.config.num_hidden_layers):
        for mod in TARGET_MODULES:
            name = module_path(layer, mod)
            d = merged.get_submodule(name).weight - model.get_submodule(name).weight
            if d.abs().max() > 0:
                deltas[(layer, mod)] = d.detach()
    return deltas


def make_pattern_lora(model, out: Path) -> Path:
    """A PEFT adapter using rank_pattern and alpha_pattern, with random non-zero factors."""
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "down_proj", "v_proj"], lora_dropout=0.0,
                     rank_pattern={"q_proj": 2}, alpha_pattern={"down_proj": 32}, task_type="CAUSAL_LM")
    pm = get_peft_model(copy.deepcopy(model), cfg)
    gen = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for name, p in pm.named_parameters():
            if "lora_" in name:
                p.copy_(torch.randn(p.shape, generator=gen) * 0.05)
    pm.save_pretrained(str(out))
    return out


@pytest.mark.parametrize("kind", ["plain", "rslora", "layers", "patterns"])
def test_load_matches_peft_merge(tiny_model, tmp_path, kind):
    if kind == "patterns":
        folder = make_pattern_lora(tiny_model, tmp_path / kind)
    else:
        folder = make_random_lora(tiny_model, tmp_path / kind, rank=4, alpha=8.0, seed=3,
                                  use_rslora=kind == "rslora",
                                  layers_to_transform=[1, 3] if kind == "layers" else None)
    adapter = load_adapter(folder)
    expected = peft_deltas(tiny_model, folder)
    assert set(adapter.modules) == set(expected)
    for key, d in expected.items():
        assert torch.allclose(adapter.modules[key].delta(), d, atol=TOL), key
    assert adapter.meta["skipped_keys"] == []


def test_scaling_values(tiny_model, tmp_path):
    rs = load_adapter(make_random_lora(tiny_model, tmp_path / "rs", rank=4, alpha=8.0, use_rslora=True))
    assert all(m.scaling == pytest.approx(4.0) for m in rs.modules.values())
    pat = load_adapter(make_pattern_lora(tiny_model, tmp_path / "pat"))
    assert pat.modules[(0, "q_proj")].scaling == pytest.approx(4.0)
    assert pat.modules[(0, "q_proj")].rank == 2
    assert pat.modules[(0, "down_proj")].scaling == pytest.approx(8.0)
    assert pat.modules[(0, "v_proj")].scaling == pytest.approx(2.0)


def test_key_variants_and_bin(random_lora, tmp_path):
    original = load_adapter(random_lora)
    state = load_file(str(random_lora / "adapter_model.safetensors"))
    renamed = {k.removeprefix("base_model.model.").replace(".weight", ".default.weight"): v
               for k, v in state.items()}
    renamed["base_model.model.lm_head.modules_to_save.weight"] = torch.zeros(2, 2)
    out = tmp_path / "variant"
    out.mkdir()
    torch.save(renamed, out / "adapter_model.bin")
    (out / "adapter_config.json").write_text((random_lora / "adapter_config.json").read_text())
    variant = load_adapter(out, name="alias123")
    assert variant.name == "alias123"
    assert variant.meta["skipped_keys"] == ["base_model.model.lm_head.modules_to_save.weight"]
    for key, m in original.modules.items():
        assert torch.equal(variant.modules[key].A, m.A) and torch.equal(variant.modules[key].B, m.B)


def test_subfolder(random_lora, tmp_path):
    nested = tmp_path / "repo" / "adapter"
    nested.mkdir(parents=True)
    for f in random_lora.iterdir():
        (nested / f.name).write_bytes(f.read_bytes())
    assert set(load_adapter(tmp_path / "repo", subfolder="adapter").modules) == set(load_adapter(random_lora).modules)


def test_rejects_dora(random_lora):
    cfg = json.loads((random_lora / "adapter_config.json").read_text())
    (random_lora / "adapter_config.json").write_text(json.dumps({**cfg, "use_dora": True}))
    with pytest.raises(ValueError, match="use_dora"):
        load_adapter(random_lora)


def test_norms_match_explicit(random_lora):
    adapter = load_adapter(random_lora)
    norms = module_norms(adapter)
    for key, m in adapter.modules.items():
        assert norms[key] == pytest.approx(torch.linalg.matrix_norm(m.delta()).item(), rel=1e-5)
    explicit = torch.sqrt(sum(torch.sum(m.delta() ** 2) for m in adapter.modules.values())).item()
    assert delta_norm(adapter) == pytest.approx(explicit, rel=1e-5)


def explicit_stats(m: LoraModule) -> dict[str, float]:
    s = torch.linalg.svdvals(m.delta())[: m.rank]
    p = s**2 / torch.sum(s**2)
    return {"top1": s[0].item(), "stable_rank": (torch.sum(s**2) / s[0] ** 2).item(),
            "entropy": -(p * p.log()).sum().item()}


def test_spectral_features_match_svd(random_lora):
    adapter = load_adapter(random_lora)
    for m in adapter.modules.values():
        assert torch.allclose(singular_values(m), torch.linalg.svdvals(m.delta())[: m.rank], rtol=1e-4, atol=1e-7)
    feats = spectral_features(adapter)
    stats = {k: explicit_stats(m) for k, m in adapter.modules.items()}
    for stat in ("top1", "stable_rank", "entropy"):
        values = torch.tensor([s[stat] for s in stats.values()])
        attn = torch.tensor([s[stat] for (_, mod), s in stats.items() if mod in ATTN_MODULES])
        assert feats[f"all_{stat}_mean"] == pytest.approx(values.mean().item(), rel=1e-4)
        assert feats[f"all_{stat}_max"] == pytest.approx(values.max().item(), rel=1e-4)
        assert feats[f"attn_{stat}_max"] == pytest.approx(attn.max().item(), rel=1e-4)
    assert len(feats) == 18


def test_spectral_features_missing_group_is_nan(tiny_model, tmp_path):
    attn_only = load_adapter(make_random_lora(tiny_model, tmp_path / "a", target_modules=("q_proj",)))
    feats = spectral_features(attn_only)
    assert torch.isnan(torch.tensor(feats["mlp_top1_mean"])) and feats["attn_top1_mean"] > 0


def test_stack_is_sum(tiny_model, tmp_path):
    a = load_adapter(make_random_lora(tiny_model, tmp_path / "a", rank=4, alpha=8.0, seed=1))
    b = load_adapter(make_random_lora(tiny_model, tmp_path / "b", rank=2, alpha=16.0, seed=2, use_rslora=True,
                                      target_modules=("q_proj", "down_proj"), layers_to_transform=[0, 2]))
    s = stack(a, b)
    assert set(s.modules) == set(a.modules)
    for key, m in s.modules.items():
        expected = a.modules[key].delta() + (b.modules[key].delta() if key in b.modules else 0)
        assert m.scaling == 1.0
        assert torch.allclose(m.delta(), expected, atol=TOL)
    assert s.modules[(0, "q_proj")].rank == 6


def logits_with(model, folder: Path, x: torch.Tensor) -> torch.Tensor:
    from peft import PeftModel

    pm = PeftModel.from_pretrained(copy.deepcopy(model), str(folder)).eval()
    with torch.no_grad():
        return pm(input_ids=x).logits


def test_to_peft_round_trip(tiny_model, random_lora, tmp_path):
    adapter = load_adapter(random_lora)
    saved = to_peft(adapter, tmp_path / "saved", base_repo="tiny")
    x = torch.randint(9, 200, (2, 7), generator=torch.Generator().manual_seed(0))
    assert torch.allclose(logits_with(tiny_model, saved, x), logits_with(tiny_model, random_lora, x), atol=1e-5)
    reloaded = load_adapter(saved)
    for key, m in adapter.modules.items():
        assert torch.allclose(reloaded.modules[key].delta(), m.delta(), atol=TOL)


def test_to_peft_stacked_mixed_ranks(tiny_model, tmp_path):
    a = load_adapter(make_random_lora(tiny_model, tmp_path / "a", rank=4, seed=1, layers_to_transform=[1, 2]))
    b = load_adapter(make_random_lora(tiny_model, tmp_path / "b", rank=2, seed=2, target_modules=("o_proj",),
                                      layers_to_transform=[3]))
    s = stack(a, b)
    saved = to_peft(s, tmp_path / "saved")
    expected = {k: m.delta() for k, m in s.modules.items()}
    got = peft_deltas(tiny_model, saved)
    assert set(got) == set(expected)
    for key, d in expected.items():
        assert torch.allclose(got[key], d, atol=TOL)


def test_to_peft_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        to_peft(Adapter("empty", {}), tmp_path / "x")
