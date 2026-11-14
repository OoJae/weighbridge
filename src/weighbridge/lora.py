"""LoRA adapters as matrices.

An adapter is a set of low-rank weight deltas, one per `(layer, module)` key: ``ΔW = scaling · B @ A`` with
``A`` of shape ``(r, in_features)`` and ``B`` of shape ``(out_features, r)``. Everything here works on the
factors and never materialises a full ``(out, in)`` delta except in :meth:`LoraModule.delta`, which exists for
tests and small checks.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .common import ATTN_MODULES, BASE_REPO, MLP_MODULES, TARGET_MODULES, ensure_dir, module_path

Key = tuple[int, str]

# Matches every PEFT key spelling of a LoRA factor on a decoder-layer projection, e.g.
#   base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight
#   base_model.model.model.layers.3.mlp.down_proj.lora_B.default.weight
#   model.layers.3.self_attn.q_proj.lora_A.weight
_LORA_KEY = re.compile(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+?)\.lora_([AB])(?:\.[\w-]+)?\.weight$")

SPECTRAL_GROUPS: dict[str, tuple[str, ...]] = {"attn": ATTN_MODULES, "mlp": MLP_MODULES, "all": TARGET_MODULES}
SPECTRAL_STATS = ("top1", "stable_rank", "entropy")


@dataclass
class LoraModule:
    """One low-rank delta ``scaling · B @ A`` on the target module `module` of decoder layer `layer`."""

    layer: int
    module: str
    A: torch.Tensor
    B: torch.Tensor
    scaling: float

    @property
    def rank(self) -> int:
        """Number of rank-one terms stored (the delta's rank is at most this)."""
        return self.A.shape[0]

    def delta(self) -> torch.Tensor:
        """The full delta ``scaling · B @ A`` with shape ``(out_features, in_features)``."""
        return self.scaling * (self.B @ self.A)

    def replace(self, A: torch.Tensor | None = None, B: torch.Tensor | None = None,
                scaling: float | None = None) -> LoraModule:
        """A new module with some fields replaced; the factors are shared, never mutated."""
        return LoraModule(self.layer, self.module, self.A if A is None else A, self.B if B is None else B,
                          self.scaling if scaling is None else scaling)


@dataclass
class Adapter:
    """A LoRA adapter: an opaque `name` (a salted hash for third-party adapters), its modules and metadata."""

    name: str
    modules: dict[Key, LoraModule]
    meta: dict = field(default_factory=dict)

    def map_modules(self, fn: Callable[[LoraModule], LoraModule], name: str, **meta: object) -> Adapter:
        """A new adapter with `fn` applied to every module and `meta` merged into a copy of the metadata."""
        return Adapter(name, {k: fn(m) for k, m in self.modules.items()}, {**self.meta, **meta})


# ---- Loading -----------------------------------------------------------------------------------


def _read_state_dict(folder: Path) -> dict[str, torch.Tensor]:
    """Read `adapter_model.safetensors`, or `adapter_model.bin` when no safetensors file exists."""
    st = folder / "adapter_model.safetensors"
    if st.exists():
        from safetensors.torch import load_file

        return load_file(str(st))
    pt = folder / "adapter_model.bin"
    if pt.exists():
        return torch.load(pt, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"no adapter_model.safetensors or adapter_model.bin in {folder}")


def _pattern_value(patterns: dict[str, float], key: str, default: float) -> float:
    """PEFT's `rank_pattern` / `alpha_pattern` lookup: the first pattern matching the end of `key` wins."""
    for pattern, value in patterns.items():
        if re.match(rf"(.*\.)?({pattern})$", key):
            return value
    return default


def module_scaling(cfg: dict, layer: int, module: str) -> float:
    """PEFT's scaling for one module: ``alpha / r``, or ``alpha / sqrt(r)`` with rsLoRA, after the patterns."""
    key = module_path(layer, module)
    r = _pattern_value(cfg.get("rank_pattern") or {}, key, cfg["r"])
    alpha = _pattern_value(cfg.get("alpha_pattern") or {}, key, cfg.get("lora_alpha", 8))
    return float(alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r)


def _check_supported(cfg: dict) -> None:
    """Refuse adapters whose effect is not a plain sum of low-rank deltas."""
    if cfg.get("peft_type", "LORA") != "LORA":
        raise ValueError(f"not a LoRA adapter: peft_type={cfg.get('peft_type')}")
    for flag in ("use_dora", "lora_bias", "fan_in_fan_out"):
        if cfg.get(flag):
            raise ValueError(f"unsupported LoRA variant: {flag}=True")


def _group_factors(state: dict[str, torch.Tensor]) -> tuple[dict[Key, dict[str, torch.Tensor]], list[str]]:
    """Map PEFT keys to ``{(layer, module): {"A": ..., "B": ...}}``; also return the keys that did not map."""
    factors: dict[Key, dict[str, torch.Tensor]] = {}
    skipped: list[str] = []
    for name, tensor in state.items():
        m = _LORA_KEY.search(name)
        if m is None or m.group(2) not in TARGET_MODULES:
            skipped.append(name)
            continue
        factors.setdefault((int(m.group(1)), m.group(2)), {})[m.group(3)] = tensor.detach().to("cpu", torch.float32)
    incomplete = [k for k, f in factors.items() if set(f) != {"A", "B"}]
    if incomplete:
        raise ValueError(f"LoRA factors missing A or B for {sorted(incomplete)[:5]}")
    return factors, sorted(skipped)


def load_adapter(path: str | Path, name: str | None = None, subfolder: str = "") -> Adapter:
    """Load a PEFT LoRA folder (`adapter_config.json` plus weights) as float32 factors with per-module scaling.

    `name` defaults to the folder name; pass the salted alias for third-party adapters. Keys that are not LoRA
    factors on a target projection (for example `modules_to_save` weights) are listed in ``meta["skipped_keys"]``.
    """
    folder = Path(path) / subfolder if subfolder else Path(path)
    cfg = json.loads((folder / "adapter_config.json").read_text())
    _check_supported(cfg)
    factors, skipped = _group_factors(_read_state_dict(folder))
    modules = {
        (layer, mod): LoraModule(layer, mod, f["A"], f["B"], module_scaling(cfg, layer, mod))
        for (layer, mod), f in sorted(factors.items())
    }
    meta = {
        "rank": cfg["r"], "alpha": cfg.get("lora_alpha"), "use_rslora": bool(cfg.get("use_rslora")),
        "target_modules": cfg.get("target_modules"), "layers_to_transform": cfg.get("layers_to_transform"),
        "rank_pattern": cfg.get("rank_pattern") or {}, "alpha_pattern": cfg.get("alpha_pattern") or {},
        "base_model_name_or_path": cfg.get("base_model_name_or_path"), "skipped_keys": skipped,
    }
    return Adapter(name or folder.name, modules, meta)


# ---- Combining ---------------------------------------------------------------------------------


def stack(a: Adapter, b: Adapter) -> Adapter:
    """One adapter whose delta is the sum of both, by concatenating ranks.

    Each module's scaling is folded into its B, so every stacked module has scaling 1.0.
    """
    keys = sorted(set(a.modules) | set(b.modules))
    modules = {k: _stack_modules([ad.modules[k] for ad in (a, b) if k in ad.modules]) for k in keys}
    return Adapter(f"{a.name}+{b.name}", modules, {"stacked_from": [a.name, b.name]})


def _stack_modules(parts: list[LoraModule]) -> LoraModule:
    """Concatenate the ranks of modules sharing one key, with each scaling folded into its B."""
    A = torch.cat([p.A for p in parts], dim=0)
    B = torch.cat([p.scaling * p.B for p in parts], dim=1)
    return LoraModule(parts[0].layer, parts[0].module, A, B, 1.0)


# ---- Norms and spectra (r x r work only) --------------------------------------------------------


def gram_sq_norm(A: torch.Tensor, B: torch.Tensor, scaling: float) -> torch.Tensor:
    """``||scaling · B @ A||_F^2 = scaling^2 · tr((BᵀB)(AAᵀ))``, computed from two r×r Gram matrices."""
    return scaling**2 * torch.sum((B.T @ B) * (A @ A.T))


def module_norms(adapter: Adapter) -> dict[Key, float]:
    """Frobenius norm of each module's delta."""
    return {k: math.sqrt(max(gram_sq_norm(m.A, m.B, m.scaling).item(), 0.0)) for k, m in adapter.modules.items()}


def delta_norm(adapter: Adapter) -> float:
    """Frobenius norm of all deltas together (the root of the sum of squared module norms)."""
    return math.sqrt(sum(n**2 for n in module_norms(adapter).values()))


def singular_values(m: LoraModule) -> torch.Tensor:
    """Singular values of ``scaling · B @ A`` (at most r of them, descending), from QR of B and Aᵀ.

    With ``B = Q_B R_B`` and ``Aᵀ = Q_A R_A``, ``B @ A = Q_B (R_B R_Aᵀ) Q_Aᵀ``, so the spectrum is that of the
    small matrix ``R_B R_Aᵀ``.
    """
    r_b = torch.linalg.qr(m.B.float(), mode="r").R
    r_a = torch.linalg.qr(m.A.float().T, mode="r").R
    return abs(m.scaling) * torch.linalg.svdvals(r_b @ r_a.T)


def spectrum_stats(s: torch.Tensor) -> dict[str, float]:
    """Top singular value, stable rank ``Σσ²/σ₁²`` and spectral entropy of the energy distribution ``σ²/Σσ²``."""
    energy = s.float() ** 2
    total = energy.sum()
    if total <= 0:
        return {"top1": 0.0, "stable_rank": 0.0, "entropy": 0.0}
    p = energy[energy > 0] / total
    return {"top1": s.max().item(), "stable_rank": (total / energy.max()).item(),
            "entropy": -(p * p.log()).sum().item()}


def spectral_features(adapter: Adapter) -> dict[str, float]:
    """Spectral statistics aggregated over modules as ``{group}_{stat}_{mean|max}``.

    Groups are ``attn``, ``mlp`` and ``all``; stats are those of :func:`spectrum_stats`. A group the adapter
    does not touch gets NaN, so every adapter yields the same 18 columns.
    """
    per_module = {k: spectrum_stats(singular_values(m)) for k, m in adapter.modules.items()}
    features: dict[str, float] = {}
    for group, names in SPECTRAL_GROUPS.items():
        rows = [st for (_, mod), st in per_module.items() if mod in names]
        for stat in SPECTRAL_STATS:
            values = torch.tensor([row[stat] for row in rows], dtype=torch.float32)
            empty = values.numel() == 0
            features[f"{group}_{stat}_mean"] = math.nan if empty else values.mean().item()
            features[f"{group}_{stat}_max"] = math.nan if empty else values.max().item()
    return features


# ---- Saving ------------------------------------------------------------------------------------


def _padded(m: LoraModule, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Factors with the scaling folded into B and zero-padded to `rank`; the delta is unchanged."""
    A = torch.zeros(rank, m.A.shape[1], dtype=torch.float32)
    B = torch.zeros(m.B.shape[0], rank, dtype=torch.float32)
    A[: m.rank] = m.A.float()
    B[:, : m.rank] = m.scaling * m.B.float()
    return A, B


def _shapes_by_module(adapter: Adapter) -> dict[str, tuple[int, int]]:
    """``(out_features, in_features)`` of each target module type present in the adapter."""
    return {m.module: (m.B.shape[0], m.A.shape[1]) for m in adapter.modules.values()}


def _peft_state_dict(adapter: Adapter, rank: int, layers: Iterable[int]) -> dict[str, torch.Tensor]:
    """Standard PEFT keys for every (layer, module) on the grid; keys absent from the adapter get zero factors."""
    shapes = _shapes_by_module(adapter)
    state: dict[str, torch.Tensor] = {}
    for layer in layers:
        for mod, (out_f, in_f) in sorted(shapes.items()):
            m = adapter.modules.get((layer, mod))
            if m is None:
                A, B = torch.zeros(rank, in_f), torch.zeros(out_f, rank)
            else:
                A, B = _padded(m, rank)
            prefix = f"base_model.model.{module_path(layer, mod)}"
            state[f"{prefix}.lora_A.weight"] = A.contiguous()
            state[f"{prefix}.lora_B.weight"] = B.contiguous()
    return state


def to_peft(adapter: Adapter, out_dir: str | Path, base_repo: str = BASE_REPO) -> Path:
    """Write a PEFT-loadable folder reproducing the adapter's deltas exactly.

    Every module is saved at one rank (the adapter's largest, zero-padded) with ``lora_alpha = r``, so the
    scaling is 1.0 and each module's own scaling is folded into its B. This keeps the folder free of
    `rank_pattern` / `alpha_pattern`, which some runtimes ignore.
    """
    if not adapter.modules:
        raise ValueError("cannot save an adapter with no modules")
    out = ensure_dir(Path(out_dir))
    rank = max(m.rank for m in adapter.modules.values())
    layers = sorted({layer for layer, _ in adapter.modules})
    cfg = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": base_repo,
        "r": rank, "lora_alpha": rank, "use_rslora": False, "lora_dropout": 0.0, "bias": "none",
        "target_modules": sorted(_shapes_by_module(adapter)), "layers_to_transform": layers,
        "fan_in_fan_out": False, "inference_mode": True, "init_lora_weights": True,
        "rank_pattern": {}, "alpha_pattern": {}, "modules_to_save": None, "use_dora": False,
    }
    from safetensors.torch import save_file

    save_file(_peft_state_dict(adapter, rank, layers), str(out / "adapter_model.safetensors"))
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    return out
