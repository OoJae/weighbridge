"""SafeLoRA danger deltas: Tiny Aya Global minus Tiny Aya Base on every target module (DECISIONS 2026-11-14).

Reads both checkpoints' safetensors lazily, one matrix at a time, and writes, in `grad.save` format,

- ``danger_sketch.safetensors``: a rank-k randomized SVD of each delta, {(layer, module): (U, S, V)}, which
  `weighbridge score --danger` compares in projection mode; and
- with ``--full``, ``danger_full.safetensors``: the float32 deltas themselves (about 11 GB), for cosine mode.

Before anything else, the shards of each local checkpoint are checked against the SHA-256 values the Hugging
Face Hub publishes at the pinned revisions (``--no-verify`` skips this, for tests on tiny models).

    python jobs/job_danger.py --global-model <dir> --base-model <dir> --out <dir> [--k 64] [--full]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path

import torch
from safetensors import safe_open

from weighbridge import grad
from weighbridge.common import BASE_REVISION, PRETRAINED_REVISION, TARGET_MODULES, module_path

log = logging.getLogger("job_danger")

# SHA-256 of the weight shards on the Hub: CohereLabs/tiny-aya-global at BASE_REVISION and
# CohereLabs/tiny-aya-base at PRETRAINED_REVISION.
GLOBAL_SHARDS = {
    "model-00001-of-00002.safetensors": "0590022d0efe06f12e08e1e94302f069dff404341b426f7435c6303725635fb1",
    "model-00002-of-00002.safetensors": "2ac3809931d113785058809f27775620e0b8aec6fbc7e163c9c09c08b5f90dab",
}
BASE_SHARDS = {
    "model-00001-of-00002.safetensors": "afbc07248f256b3cb47f4ea7c20edb8d6121a3d746fb13eda364afbf9131de96",
    "model-00002-of-00002.safetensors": "d6e35c5212016c0e01e255ea7d7f756cadc70404e80e6ca22944df5d4dbd0fdb",
}


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    """Hex SHA-256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify(model_dir: Path, expected: dict[str, str]) -> None:
    """Stop unless every expected shard is present with the expected checksum."""
    for name, want in expected.items():
        got = sha256(model_dir / name)
        if got != want:
            raise SystemExit(f"{model_dir / name}: sha256 {got} != {want}")
        log.info("%s/%s matches the Hub", model_dir.name, name)


def weight_index(model_dir: Path) -> dict[str, Path]:
    """Tensor name -> shard file, from the safetensors index or, without one, from every shard's keys."""
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        return {name: model_dir / shard for name, shard in weight_map.items()}
    out: dict[str, Path] = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            out.update(dict.fromkeys(f.keys(), shard))
    return out


def read_tensor(index: dict[str, Path], name: str) -> torch.Tensor:
    """One tensor, read lazily from its shard, as float32."""
    with safe_open(str(index[name]), framework="pt") as f:
        return f.get_tensor(name).float()


def n_layers(index: dict[str, Path]) -> int:
    """Number of decoder layers, from the tensor names."""
    return 1 + max(int(name.split(".")[2]) for name in index if name.startswith("model.layers."))


def danger_deltas(global_dir: Path, base_dir: Path, k: int, seed: int, keep_full: bool
                  ) -> tuple[dict, dict | None, dict]:
    """The sketch of every (layer, module) delta, the full deltas when `keep_full`, and summary statistics.

    Each module is sketched on its own with seed `seed`, so the result does not depend on memory or order.
    """
    g_index, b_index = weight_index(global_dir), weight_index(base_dir)
    layers = n_layers(g_index)
    if layers != n_layers(b_index):
        raise SystemExit("the two checkpoints have different numbers of layers")
    sketches, full, norms = {}, ({} if keep_full else None), {}
    for layer in range(layers):
        for module in TARGET_MODULES:
            name = module_path(layer, module) + ".weight"
            delta = read_tensor(g_index, name) - read_tensor(b_index, name)
            sketches[(layer, module)] = grad.sketch({(layer, module): delta}, k=k, seed=seed)[(layer, module)]
            norms[f"L{layer}.{module}"] = float(delta.norm())
            if full is not None:
                full[(layer, module)] = delta
    return sketches, full, norms


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--global-model", type=Path, required=True, help="local Tiny Aya Global checkpoint")
    ap.add_argument("--base-model", type=Path, required=True, help="local Tiny Aya Base checkpoint")
    ap.add_argument("--out", type=Path, required=True, help="output folder")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full", action="store_true", help="also save the full float32 deltas (about 11 GB)")
    ap.add_argument("--no-verify", action="store_true", help="skip the shard checksums (tests only)")
    args = ap.parse_args(argv)

    if not args.no_verify:
        verify(args.global_model, GLOBAL_SHARDS)
        verify(args.base_model, BASE_SHARDS)
    t0 = time.perf_counter()
    sketches, full, norms = danger_deltas(args.global_model, args.base_model, args.k, args.seed, args.full)
    meta = {"source": "tiny-aya-global minus tiny-aya-base", "global_revision": BASE_REVISION,
            "base_revision": PRETRAINED_REVISION, "k": args.k, "seed": args.seed, "verified": not args.no_verify}
    args.out.mkdir(parents=True, exist_ok=True)
    grad.save(sketches, args.out / "danger_sketch.safetensors", meta)
    if full is not None:
        grad.save(full, args.out / "danger_full.safetensors", meta)
    (args.out / "danger_norms.json").write_text(json.dumps({**meta, "frobenius": norms}, indent=1))
    log.info("danger deltas for %d modules in %.0fs (total Frobenius norm %.2f)", len(sketches),
             time.perf_counter() - t0, sum(v * v for v in norms.values()) ** 0.5)


if __name__ == "__main__":
    main()
