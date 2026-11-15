"""Weights-only safety scores: pure float32 math on adapter factors, no model needed.

Sign convention: every score is a prediction of the adapter's effect on refusal, and a **positive score means
predicted more refusal**. Harm-direction quantities are reported as ``drift = -score``, so a larger drift means
predicted less refusal (more harmful compliance).

No function here forms a full ``(out, in)`` delta: inner products with the delta ``s · B @ A`` are taken as
``s · tr(Bᵀ G Aᵀ) = s · sum((Bᵀ G) ∘ A)``, and norms through r×r Gram matrices.
"""

from __future__ import annotations

import hashlib
import math
from typing import Literal, Sequence

import torch

from .common import RESIDUAL_WRITERS
from .lora import Adapter, Key, LoraModule, delta_norm, gram_sq_norm, spectral_features

Sketch = dict[Key, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


def _factors_on(m: LoraModule, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The module's float32 factors (A, B) on `device`."""
    return m.A.to(device, torch.float32), m.B.to(device, torch.float32)


def drift(score: float) -> float:
    """Harm-direction view of a score: ``drift = -score`` (larger = predicted less refusal)."""
    return -score


def t0a_linearised(adapter: Adapter, directions: dict[int, torch.Tensor],
                   input_means: dict[Key, torch.Tensor]) -> float:
    """Predicted change in refusal projection from the residual writers (positive = more refusal).

    Sums ``directions[layer] · (s · B @ A @ input_means[key])`` over the adapter's `RESIDUAL_WRITERS` modules,
    evaluated right to left as ``s · (dᵀ B)(A x)``.
    """
    total = 0.0
    for key, m in adapter.modules.items():
        if m.module not in RESIDUAL_WRITERS:
            continue
        d = directions[key[0]].float()
        x = input_means[key].to(d.device, torch.float32)
        A, B = _factors_on(m, d.device)
        total += m.scaling * torch.dot(d @ B, A @ x).item()
    return total


def first_order_term(m: LoraModule, G: torch.Tensor) -> float:
    """``⟨G, s · B @ A⟩ = s · sum((Bᵀ G) ∘ A)`` for one module, without forming the delta."""
    G = G.float()
    A, B = _factors_on(m, G.device)
    return m.scaling * torch.sum((B.T @ G) * A).item()


def t0b_first_order(adapter: Adapter, G: dict[Key, torch.Tensor]) -> float:
    """Predicted change in margin, ``Σ_modules ⟨G, s · B @ A⟩`` (positive = more refusal).

    `G` holds the margin gradient for every module of the adapter (or the gradient of an irrelevant metric,
    as a control). Tensors may live on any device; the factors are moved there.
    """
    return sum(first_order_term(m, G[key]) for key, m in adapter.modules.items())


def _sketch_term(m: LoraModule, U: torch.Tensor, S: torch.Tensor, V: torch.Tensor) -> float:
    """``s · Σ_i S_i (u_iᵀ B)(A v_i)`` for one module and a rank-k factorisation ``G ≈ U diag(S) Vᵀ``."""
    A, B = _factors_on(m, U.device)
    ub = U.float().T @ B                      # (k, r)
    av = A @ V.float()                        # (r, k)
    return m.scaling * torch.sum(S.float() * torch.sum(ub * av.T, dim=1)).item()


def t0b_sketch(adapter: Adapter, sketch: Sketch) -> float:
    """:func:`t0b_first_order` computed from a per-module sketch ``{key: (U, S, V)}`` of the gradient."""
    return sum(_sketch_term(m, *sketch[key]) for key, m in adapter.modules.items())


def log_delta_norm(adapter: Adapter) -> float:
    """Natural log of the adapter's total Frobenius norm; ``-inf`` for an all-zero adapter."""
    norm = delta_norm(adapter)
    return math.log(norm) if norm > 0 else -math.inf


def spectral_baseline(adapter: Adapter) -> dict[str, float]:
    """The spectral weights-only baseline: :func:`weighbridge.lora.spectral_features` of the adapter."""
    return spectral_features(adapter)


def _cosine_with_delta(m: LoraModule, D: torch.Tensor) -> float:
    """Cosine between ``s · B @ A`` and a full danger delta D."""
    D = D.float()
    A, B = _factors_on(m, D.device)
    denom = math.sqrt(gram_sq_norm(A, B, m.scaling).item()) * torch.linalg.matrix_norm(D).item()
    return first_order_term(m, D) / denom if denom > 0 else 0.0


def _projection_ratio(m: LoraModule, U: torch.Tensor) -> float:
    """``||Uᵀ s B A||_F / ||s B A||_F``: the cosine between the delta and its projection onto span(U).

    U must have orthonormal columns; both norms use the r×r Gram trick.
    """
    A, B = _factors_on(m, U.device)
    full = gram_sq_norm(A, B, m.scaling).item()
    projected = gram_sq_norm(A, U.float().T @ B, m.scaling).item()
    return math.sqrt(max(projected, 0.0) / full) if full > 0 else 0.0


def safelora_similarity(adapter: Adapter, danger: dict[Key, torch.Tensor],
                        mode: Literal["cosine", "projection"] = "cosine") -> float:
    """SafeLoRA-style overlap of the adapter with a danger direction, averaged over shared modules.

    ``mode="cosine"``: `danger[key]` is a full delta (for example Global − Base weights) and each module
    contributes the cosine between it and ``s · B @ A``. ``mode="projection"``: `danger[key]` is an orthonormal
    basis U (out × k) and each module contributes ``||Uᵀ s B A||_F / ||s B A||_F``. Higher means more aligned
    with the danger direction.
    """
    term = _cosine_with_delta if mode == "cosine" else _projection_ratio
    values = [term(m, danger[key]) for key, m in adapter.modules.items() if key in danger]
    if not values:
        raise ValueError("adapter and danger share no modules")
    return sum(values) / len(values)


def stable_seed(*parts: object) -> int:
    """A 63-bit seed derived from `parts` (for example ``(seed, draw, layer, module)``), stable across runs."""
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") & (2**63 - 1)


def random_orthonormal(rows: int, k: int, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    """A Haar-random ``rows × k`` matrix with orthonormal columns (QR of a Gaussian, signs fixed by diag R)."""
    q, r = torch.linalg.qr(torch.randn(rows, k, generator=generator, device=device))
    return q * torch.sign(torch.diagonal(r)).masked_fill(torch.diagonal(r) == 0, 1.0)


def random_direction_scores_many(adapters: Sequence[Adapter], sketch: Sketch, n: int = 200,
                                 seed: int = 0, device: str | torch.device | None = None) -> torch.Tensor:
    """Control scores of several adapters against the same `n` random directions, shape ``(len(adapters), n)``.

    Each random module basis is drawn once and applied to every adapter that has that module, so the cost of
    the QR factorisations is paid once per direction rather than once per adapter. Every adapter's total is
    summed over its modules in sorted key order, exactly as :func:`random_direction_scores` does. With
    `device` (for example "cuda") the same Gaussian draws are factorised and applied there, which is much faster.
    """
    if device is None:
        out = torch.zeros(len(adapters), n, dtype=torch.float64)
        keys = sorted({key for a in adapters for key in a.modules})
        for j in range(n):
            for key in keys:
                users = [(i, a.modules[key]) for i, a in enumerate(adapters) if key in a.modules]
                U, S, V = sketch[key]
                gen = torch.Generator(device=U.device).manual_seed(stable_seed(seed, j, *key))
                k = S.numel()
                rU = random_orthonormal(U.shape[0], k, gen, U.device)
                rV = random_orthonormal(V.shape[0], k, gen, U.device)
                for i, m in users:
                    out[i, j] += _sketch_term(m, rU, S, rV)
        return out.float()
    return _random_scores_on(adapters, sketch, n, seed, torch.device(device))


def _random_scores_on(adapters: Sequence[Adapter], sketch: Sketch, n: int, seed: int,
                      device: torch.device) -> torch.Tensor:
    """:func:`random_direction_scores_many` with the QR factorisations and products on `device` (a GPU).

    The Gaussian matrices come from the same CPU generators as the CPU path, so direction j is the same
    direction up to floating-point rounding in the QR; the adapters' factors are moved to `device` once.
    """
    keys = sorted({key for a in adapters for key in a.modules})
    users = {key: [(i, a.modules[key].A.to(device, torch.float32), a.modules[key].B.to(device, torch.float32),
                    a.modules[key].scaling) for i, a in enumerate(adapters) if key in a.modules] for key in keys}
    out = torch.zeros(len(adapters), n, dtype=torch.float64, device=device)
    for j in range(n):
        for key in keys:
            _, S, V = sketch[key]
            U = sketch[key][0]
            gen = torch.Generator().manual_seed(stable_seed(seed, j, *key))
            k = S.numel()
            gu = torch.randn(U.shape[0], k, generator=gen).to(device)
            gv = torch.randn(V.shape[0], k, generator=gen).to(device)
            rU, rV = _orthonormal_from(gu), _orthonormal_from(gv)
            Sd = S.to(device, torch.float32)
            for i, A, B, scaling in users[key]:
                ub = rU.T @ B                       # (k, r)
                av = A @ rV                         # (r, k)
                out[i, j] += scaling * torch.sum(Sd * torch.sum(ub * av.T, dim=1))
    return out.float().cpu()


def _orthonormal_from(gaussian: torch.Tensor) -> torch.Tensor:
    """Orthonormal columns from a Gaussian matrix by QR, with signs fixed by diag R (as random_orthonormal)."""
    q, r = torch.linalg.qr(gaussian)
    return q * torch.sign(torch.diagonal(r)).masked_fill(torch.diagonal(r) == 0, 1.0)


def random_direction_scores(adapter: Adapter, sketch: Sketch, n: int = 200, seed: int = 0) -> torch.Tensor:
    """Control scores: the adapter scored against `n` random low-rank directions, shape ``(n,)``, float32.

    Direction j keeps each module's singular values from `sketch` but replaces U and V with random orthonormal
    bases. Its randomness depends only on ``(seed, j, layer, module)``, so direction j is the same direction for
    every adapter scored, which is what a null distribution of rank correlations needs. Draws are generated on
    the sketch's device; CPU and CUDA generators give different (equally valid) directions.
    """
    return random_direction_scores_many([adapter], sketch, n=n, seed=seed)[0]

