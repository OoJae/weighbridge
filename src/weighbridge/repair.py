"""Training-free repairs that keep an adapter LoRA-shaped.

Each repair edits one factor per module, so the result is still ``scaling · B' @ A'`` with the original rank
and can be saved with :func:`weighbridge.lora.to_peft` and served like any adapter. Inputs are never mutated.
"""

from __future__ import annotations

from typing import Literal

import torch

from .lora import Adapter, Key, LoraModule
from .meters import random_orthonormal, stable_seed

Side = Literal["left", "right"]


def _project_module(m: LoraModule, basis: torch.Tensor, side: Side, strength: float) -> LoraModule:
    """``B ← B − s·U(UᵀB)`` (left, U in the output space) or ``A ← A − s·(AV)Vᵀ`` (right, V in the input space)."""
    basis = basis.to(m.A.device, torch.float32)
    if side == "left":
        return m.replace(B=m.B - strength * basis @ (basis.T @ m.B))
    return m.replace(A=m.A - strength * (m.A @ basis) @ basis.T)


def project_out(adapter: Adapter, bases: dict[Key, torch.Tensor], side: Side = "left",
                strength: float = 1.0) -> Adapter:
    """Remove (``strength=1``) or shrink each module's component in span of an orthonormal basis.

    `bases[key]` has orthonormal columns: U (out × k) for ``side="left"``, for example the top-k left singular
    vectors of the margin gradient; V (in × k) for ``side="right"``. Modules without a basis are kept as they are.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    modules = {k: _project_module(m, bases[k], side, strength) if k in bases else m
               for k, m in adapter.modules.items()}
    return Adapter(f"{adapter.name}|project-{side}", modules,
                   {**adapter.meta, "repair": {"kind": "project_out", "side": side, "strength": strength}})


def wise_ft(adapter: Adapter, lam: float) -> Adapter:
    """WiSE-FT weight interpolation towards the base model: every B (hence every delta) is scaled by `lam`."""
    return adapter.map_modules(lambda m: m.replace(B=lam * m.B), f"{adapter.name}|wise-{lam:g}",
                               repair={"kind": "wise_ft", "lam": lam})


def _random_bases(adapter: Adapter, k: int, seed: int, side: Side) -> dict[Key, torch.Tensor]:
    """A seeded random orthonormal basis per module, in the output (left) or input (right) space."""
    bases = {}
    for key, m in adapter.modules.items():
        dim = m.B.shape[0] if side == "left" else m.A.shape[1]
        gen = torch.Generator().manual_seed(stable_seed(seed, "random_projection", *key))
        bases[key] = random_orthonormal(dim, k, gen, torch.device("cpu"))
    return bases


def random_projection(adapter: Adapter, k: int, seed: int, side: Side = "left") -> Adapter:
    """Control repair: :func:`project_out` at full strength with random k-dimensional orthonormal bases."""
    repaired = project_out(adapter, _random_bases(adapter, k, seed, side), side=side)
    return Adapter(f"{adapter.name}|random-{side}-{k}", repaired.modules,
                   {**adapter.meta, "repair": {"kind": "random_projection", "side": side, "k": k, "seed": seed}})
