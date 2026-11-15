"""Weights-only scores for a folder of adapters, read against the base-model artifacts of ``jobs/job_probe.py``.

The probe job writes, per run folder::

    directions_{condition}.safetensors    refusal direction per layer, keys "L{layer}"
    input_means_{lang}.safetensors        residual-writer input means, keys "L{layer}.{module}"
    sketch_{lang}_{condition}.safetensors rank-k sketch of the margin gradient (grad.save format)
    grad_{lang}_{condition}.safetensors   the full margin gradient, when it was saved
    sketch_irrelevant / grad_irrelevant   the irrelevant-metric control gradient

Every score column here is oriented as **drift** (``meters.drift``: higher = predicted less refusal, more
harmful), so it can be correlated with the outcome Y directly.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from . import grad, lora, meters
from .common import RESIDUAL_WRITERS
from .lora import Adapter, Key

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas as pd

Sketch = meters.Sketch
IRRELEVANT = "irrelevant"


# ---- Artifact files ----------------------------------------------------------------------------

def directions_file(folder: Path, condition: str) -> Path:
    """Path of the refusal directions for one condition."""
    return folder / f"directions_{condition}.safetensors"


def input_means_file(folder: Path, lang: str) -> Path:
    """Path of the residual-writer input means for one language."""
    return folder / f"input_means_{lang}.safetensors"


def gradient_name(lang: str, condition: str) -> str:
    """Name of a margin gradient in the probe outputs, for example "en_shipped"."""
    return f"{lang}_{condition}"


def encode_directions(directions: Mapping[int, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Safetensors keys "L{layer}" for per-layer direction vectors."""
    return {f"L{layer}": v.float().contiguous().cpu() for layer, v in directions.items()}


def decode_directions(flat: Mapping[str, torch.Tensor]) -> dict[int, torch.Tensor]:
    """Inverse of :func:`encode_directions`."""
    return {int(name[1:]): v.float() for name, v in sorted(flat.items())}


def encode_input_means(means: Mapping[Key, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Safetensors keys "L{layer}.{module}" for per-module input means."""
    return {f"L{layer}.{module}": v.float().contiguous().cpu() for (layer, module), v in means.items()}


def decode_input_means(flat: Mapping[str, torch.Tensor]) -> dict[Key, torch.Tensor]:
    """Inverse of :func:`encode_input_means`."""
    out: dict[Key, torch.Tensor] = {}
    for name, v in flat.items():
        layer, module = name.split(".", 1)
        out[(int(layer[1:]), module)] = v.float()
    return out


def _load_flat(path: Path) -> dict[str, torch.Tensor]:
    """Every tensor of a safetensors file."""
    from safetensors.torch import load_file

    return load_file(str(path))


@dataclass(frozen=True)
class ProbeArtifacts:
    """Base-model quantities the weights-only scores need, for one language and condition."""

    directions: dict[int, torch.Tensor]
    input_means: dict[Key, torch.Tensor]
    sketch: Sketch
    gradient: dict[Key, torch.Tensor] | None        # the full margin gradient, when saved
    irrelevant: dict[Key, torch.Tensor] | Sketch | None


def _optional(path: Path) -> dict | None:
    """grad.load(path), or None when the file does not exist."""
    return grad.load(path) if path.exists() else None


def load_probe_artifacts(folder: str | Path, lang: str = "en", condition: str = "shipped") -> ProbeArtifacts:
    """Read the probe job's outputs for one language and condition from a local folder.

    The full irrelevant gradient is preferred over its sketch when both exist.
    """
    folder = Path(folder)
    name = gradient_name(lang, condition)
    sketch_path = folder / f"sketch_{name}.safetensors"
    if not sketch_path.exists():
        raise FileNotFoundError(f"no gradient sketch {sketch_path.name} in {folder}")
    irrelevant = _optional(folder / f"grad_{IRRELEVANT}.safetensors")
    return ProbeArtifacts(
        directions=decode_directions(_load_flat(directions_file(folder, condition))),
        input_means=decode_input_means(_load_flat(input_means_file(folder, lang))),
        sketch=grad.load(sketch_path),
        gradient=_optional(folder / f"grad_{name}.safetensors"),
        irrelevant=irrelevant if irrelevant is not None else _optional(folder / f"sketch_{IRRELEVANT}.safetensors"),
    )


# ---- Adapters ----------------------------------------------------------------------------------

def find_adapters(root: str | Path) -> list[tuple[str, Path]]:
    """``(alias, adapter folder)`` for every child folder of `root` holding an adapter.

    The child folder's name is the alias (as ``hub.download_adapter`` destinations are named); the adapter
    folder is the shallowest folder beneath it that holds ``adapter_config.json``.
    """
    found = []
    for child in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        configs = sorted(child.rglob("adapter_config.json"), key=lambda p: (len(p.parts), str(p)))
        if configs:
            found.append((child.name, configs[0].parent))
    return found


def _is_sketch(obj: Mapping) -> bool:
    """True for a {key: (U, S, V)} sketch, False for a {key: G} full gradient."""
    return isinstance(next(iter(obj.values())), tuple)


def _first_order(adapter: Adapter, G: Mapping) -> float:
    """``⟨G, ΔW⟩`` from a full gradient or from its sketch."""
    return meters.t0b_sketch(adapter, G) if _is_sketch(G) else meters.t0b_first_order(adapter, G)


def safelora_drift(adapter: Adapter, danger: Mapping) -> float:
    """SafeLoRA baseline as drift: minus the overlap with the alignment update (lower overlap = more harm).

    `danger` is either full (Global − Base) deltas, compared by cosine, or a sketch of them, compared by
    projection onto its left singular vectors.
    """
    if _is_sketch(danger):
        return -meters.safelora_similarity(adapter, {k: U for k, (U, _, _) in danger.items()}, mode="projection")
    return -meters.safelora_similarity(adapter, dict(danger), mode="cosine")


def weights_digest(adapter: Adapter) -> str:
    """SHA-256 over every module's key and float32 factors, in key order: equal digests mean identical weights.

    Used for the exact-duplicate exclusion rule (PREREG section 3, rule 7).
    """
    digest = hashlib.sha256()
    for (layer, module), m in sorted(adapter.modules.items()):
        digest.update(f"{layer}.{module}:{m.scaling!r}".encode())
        for factor in (m.A, m.B):
            digest.update(factor.detach().float().contiguous().numpy().tobytes())
    return digest.hexdigest()


def score_adapter(adapter: Adapter, art: ProbeArtifacts, danger: Mapping | None = None) -> dict[str, float | str]:
    """Every weights-only score and baseline of one adapter, oriented as drift (higher = more harmful).

    Columns: d0a, d0b_exact (NaN without a full gradient), d0b_sketch, irrelevant (the control; NaN when
    absent), safelora (NaN without `danger`), log_delta_norm, residual_writers (1 if the adapter touches
    o_proj or down_proj, else 0: D0a is 0 by construction otherwise), the 18 spectral features,
    weights_seconds (CPU wall time of this function, for the cost ladder), and for the exclusion rules
    skipped_keys (weights outside the target projections) and weights_digest.
    """
    start = time.perf_counter()
    row = {
        "d0a": meters.drift(meters.t0a_linearised(adapter, art.directions, art.input_means)),
        "d0b_exact": meters.drift(meters.t0b_first_order(adapter, art.gradient)) if art.gradient else math.nan,
        "d0b_sketch": meters.drift(meters.t0b_sketch(adapter, art.sketch)),
        IRRELEVANT: meters.drift(_first_order(adapter, art.irrelevant)) if art.irrelevant else math.nan,
        "safelora": safelora_drift(adapter, danger) if danger else math.nan,
        "log_delta_norm": meters.log_delta_norm(adapter),
        "residual_writers": float(any(m.module in RESIDUAL_WRITERS for m in adapter.modules.values())),
        **{f"spec_{k}": v for k, v in meters.spectral_baseline(adapter).items()},
    }
    return row | {"weights_seconds": time.perf_counter() - start,
                  "skipped_keys": len(adapter.meta.get("skipped_keys", [])), "weights_digest": weights_digest(adapter)}


def random_drifts(adapter: Adapter, sketch: Sketch, n: int, seed: int = 0) -> list[float]:
    """Drift of the adapter against `n` random directions (``meters.random_direction_scores``), the null control."""
    return [-float(s) for s in meters.random_direction_scores(adapter, sketch, n=n, seed=seed)]


RANDOM_CHUNK_BYTES = 4 * 10**9  # adapters held at once for the shared random directions (float32 factors)


def _adapter_bytes(adapter: Adapter) -> int:
    """Bytes of the adapter's float32 factors."""
    return sum((m.A.numel() + m.B.numel()) * 4 for m in adapter.modules.values())


def score_folder(root: str | Path, art: ProbeArtifacts, danger: Mapping | None = None, n_random: int = 0,
                 seed: int = 0, chunk_bytes: int = RANDOM_CHUNK_BYTES,
                 device: str | None = None,
                 on_row: Callable[[list[dict]], None] | None = None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Scores of every adapter under `root`, one row per alias; plus random-direction drifts when `n_random` > 0.

    The random-direction frame has one row per alias and columns r0 ... r{n-1}; direction j is the same
    direction for every adapter, so column j gives one draw of the null distribution of a rank correlation.
    Random directions are drawn once per chunk of adapters holding at most `chunk_bytes` of factors.
    `on_row`, when given, is called with the score rows so far after each adapter (to save partial results).
    """
    import pandas as pd

    rows, randoms, chunk = [], [], []

    def flush() -> None:
        # Drift is minus the score; every adapter in the chunk shares each drawn direction.
        draws = -meters.random_direction_scores_many([a for _, a in chunk], art.sketch, n=n_random, seed=seed,
                                                     device=device)
        randoms.extend({"alias": alias, **{f"r{j}": float(v) for j, v in enumerate(d)}}
                       for (alias, _), d in zip(chunk, draws))
        chunk.clear()

    for alias, path in find_adapters(root):
        adapter = lora.load_adapter(path, name=alias)
        rows.append({"alias": alias, **score_adapter(adapter, art, danger)})
        log.info("scored %d %s in %.1fs", len(rows), alias, rows[-1]["weights_seconds"])
        if on_row:
            on_row(rows)
        if n_random:
            if chunk and sum(_adapter_bytes(a) for _, a in chunk) + _adapter_bytes(adapter) > chunk_bytes:
                flush()
            chunk.append((alias, adapter))
    if not n_random:
        return pd.DataFrame(rows), None
    if chunk:
        flush()
    return pd.DataFrame(rows), pd.DataFrame(randoms)
