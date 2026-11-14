"""Shared constants and small helpers. Every module imports from here; do not duplicate these values."""

from __future__ import annotations

import os
from pathlib import Path

# ---- Pinned models (verified on the Hub, 2026-11-14) -------------------------------------------
BASE_REPO = "CohereLabs/tiny-aya-global"
BASE_REVISION = "af89d219b53ed9b13b8a4645f9c8028973510324"
PRETRAINED_REPO = "CohereLabs/tiny-aya-base"
PRETRAINED_REVISION = "d62ac17e241c871fc52e9a95d7b0e2615189d7d5"
GGUF_REPO = "CohereLabs/tiny-aya-global-GGUF"
GGUF_REVISION = "a602ea7eeec3a4ad6f77a1b8cf6a53512824922b"

# ---- Tiny Aya (Cohere2) architecture, verified from public configs -----------------------------
N_LAYERS = 36
HIDDEN = 2048
INTERMEDIATE = 11008
N_HEADS = 16
N_KV_HEADS = 4
HEAD_DIM = 128
VOCAB = 262144

ATTN_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_MODULES = ("gate_proj", "up_proj", "down_proj")
TARGET_MODULES = ATTN_MODULES + MLP_MODULES
# Modules whose output is added straight into the residual stream (Cohere uses a parallel block:
# h = x + attn(ln(x)) + mlp(ln(x))).
RESIDUAL_WRITERS = ("o_proj", "down_proj")

# Generation stops on token 3, but chat turns end with token 6: always pass both.
STOP_TOKEN_IDS = (3, 6)

LANGS = ("en", "yo", "pcm")
# "shipped": the model's own chat template, which injects the ~366-token "contextual safety mode"
# system preamble. "minimal": a single user turn with no system preamble.
CONDITIONS = ("shipped", "minimal")

# ---- Paths -------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
# Private material (population manifest, downloaded third-party adapters, raw generations of
# third-party adapters) lives OUTSIDE the repository so it can never be committed by accident.
PRIVATE_DIR = Path(os.environ.get("WEIGHBRIDGE_PRIVATE_DIR", REPO_ROOT.parent / "private"))
CACHE_DIR = Path(os.environ.get("WEIGHBRIDGE_CACHE_DIR", PRIVATE_DIR / "cache"))
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = Path(os.environ.get("WEIGHBRIDGE_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))


def module_path(layer: int, module: str) -> str:
    """Fully qualified module name inside Cohere2ForCausalLM, e.g. model.layers.3.mlp.down_proj."""
    if module not in TARGET_MODULES:
        raise ValueError(f"unknown target module: {module}")
    parent = "self_attn" if module in ATTN_MODULES else "mlp"
    return f"model.layers.{layer}.{parent}.{module}"


def module_shape(module: str, hidden: int = HIDDEN, intermediate: int = INTERMEDIATE,
                 n_heads: int = N_HEADS, n_kv_heads: int = N_KV_HEADS, head_dim: int = HEAD_DIM) -> tuple[int, int]:
    """(out_features, in_features) of a target module's weight matrix."""
    return {
        "q_proj": (n_heads * head_dim, hidden),
        "k_proj": (n_kv_heads * head_dim, hidden),
        "v_proj": (n_kv_heads * head_dim, hidden),
        "o_proj": (hidden, n_heads * head_dim),
        "gate_proj": (intermediate, hidden),
        "up_proj": (intermediate, hidden),
        "down_proj": (hidden, intermediate),
    }[module]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
