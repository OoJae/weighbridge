"""First-order gradient of the refusal margin with respect to every target weight, its low-rank sketch,
and safetensors persistence with JSON metadata.

G[(layer, module)] = d mean_margin / d W has the weight's shape (out_features, in_features), so the
first-order change in margin under a LoRA update is <G, scaling * B @ A>.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .common import TARGET_MODULES
from .probe import encode_batches, final_logits, margin_from_logits, opener_token_ids, target_module

Key = tuple[int, str]
Factors = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
META_KEY = "weighbridge"
FACTOR_NAMES = ("U", "S", "V")
# Static loss scale for float16 backward passes (the T4 has no bfloat16): small weight gradients underflow in
# float16 without one. Gradients are unscaled into the float32 accumulators.
FP16_LOSS_SCALE = 2.0**10


# ---- Gradients ---------------------------------------------------------------------------------

def _target_weights(model: torch.nn.Module) -> dict[Key, torch.nn.Parameter]:
    """Weight parameter of every target module in every decoder layer."""
    n_layers = len(model.get_decoder().layers)
    return {(layer, name): target_module(model, layer, name).weight
            for layer in range(n_layers) for name in TARGET_MODULES}


@contextmanager
def _only_grad_on(model: torch.nn.Module, params: Sequence[torch.nn.Parameter]) -> Iterator[None]:
    """Freeze every parameter except `params`, restoring the original flags and clearing grads afterwards."""
    flags = {p: p.requires_grad for p in model.parameters()}
    for p in flags:
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
        p.grad = None
    try:
        with torch.enable_grad():
            yield
    finally:
        for p, flag in flags.items():
            p.requires_grad_(flag)
        for p in params:
            p.grad = None


def metric_gradient(model: torch.nn.Module, tokenizer: Any, prompts: Sequence[str], condition: str,
                    set_a: Sequence[int], set_b: Sequence[int], batch_size: int = 4) -> dict[Key, torch.Tensor]:
    """Gradient of the mean over prompts of logsumexp(logits[set_a]) - logsumexp(logits[set_b]) at the first
    response position, for every target weight, as float32 CPU tensors keyed by (layer, module).

    Gradients are accumulated batch by batch (each batch's loss is pre-divided by the number of prompts) and
    each weight's grad is cleared between batches, so memory stays at the weights plus one batch of
    activations plus G. Works in whatever dtype the model is in; a float16 model's backward pass uses the
    static loss scale `FP16_LOSS_SCALE`, and a non-finite result raises FloatingPointError.
    """
    weights = _target_weights(model)
    scale = FP16_LOSS_SCALE if next(model.parameters()).dtype == torch.float16 else 1.0
    G = {k: torch.zeros(w.shape, dtype=torch.float32) for k, w in weights.items()}
    with _only_grad_on(model, list(weights.values())):
        for ids, mask in encode_batches(model, tokenizer, prompts, condition, batch_size):
            loss = margin_from_logits(final_logits(model, ids, mask), set_a, set_b).sum() / len(prompts)
            (loss * scale).backward()
            for k, w in weights.items():
                G[k] += w.grad.float().cpu() / scale
                w.grad = None
    bad = [k for k, g in G.items() if not torch.isfinite(g).all()]
    if bad:
        raise FloatingPointError(f"non-finite gradient for {sorted(bad)[:3]}; lower FP16_LOSS_SCALE or use float32")
    return G


def margin_gradient(model: torch.nn.Module, tokenizer: Any, prompts: Sequence[str], lang: str, condition: str,
                    batch_size: int = 4) -> dict[Key, torch.Tensor]:
    """G = d mean margin / d W for every target module in every layer (see probe.margin for the margin)."""
    refuse, comply = opener_token_ids(tokenizer, lang)
    return metric_gradient(model, tokenizer, prompts, condition, refuse, comply, batch_size)


def random_token_sets(tokenizer: Any, vocab_size: int, sizes: tuple[int, int], exclude: Sequence[int],
                      seed: int) -> tuple[list[int], list[int]]:
    """Two disjoint, seeded random token-id sets of the given sizes, avoiding special and excluded ids."""
    banned = set(tokenizer.all_special_ids) | set(exclude)
    pool = torch.tensor([i for i in range(min(vocab_size, len(tokenizer))) if i not in banned])
    if len(pool) < sum(sizes):
        raise ValueError("vocabulary too small for two disjoint random token sets")
    perm = pool[torch.randperm(len(pool), generator=torch.Generator().manual_seed(seed))]
    return sorted(perm[:sizes[0]].tolist()), sorted(perm[sizes[0]:sum(sizes)].tolist())


def irrelevant_token_sets(model: torch.nn.Module, tokenizer: Any, seed: int,
                          lang: str = "en") -> tuple[list[int], list[int]]:
    """The control metric's two random disjoint token sets, sized like `lang`'s refusal and compliance opener
    sets and excluding the opener ids themselves."""
    refuse, comply = opener_token_ids(tokenizer, lang)
    vocab_size = model.get_output_embeddings().weight.shape[0]
    return random_token_sets(tokenizer, vocab_size, (len(refuse), len(comply)), refuse + comply, seed)


def irrelevant_gradient(model: torch.nn.Module, tokenizer: Any, prompts: Sequence[str], condition: str, seed: int,
                        lang: str = "en", batch_size: int = 4) -> dict[Key, torch.Tensor]:
    """Control: the gradient of the log-sum-exp difference between the two sets of `irrelevant_token_sets`."""
    set_a, set_b = irrelevant_token_sets(model, tokenizer, seed, lang)
    return metric_gradient(model, tokenizer, prompts, condition, set_a, set_b, batch_size)


# ---- Sketches ----------------------------------------------------------------------------------

def sketch(G: Mapping[Key, torch.Tensor], k: int = 64, seed: int = 0) -> dict[Key, Factors]:
    """Rank-k randomized SVD of each module's gradient: {key: (U (out x k), S (k,), V (in x k))}, float32.

    k is capped at the smaller matrix dimension. The global RNG is forked, seeded once and restored, and
    modules are processed in sorted key order, so the result depends only on `seed`.
    """
    out: dict[Key, Factors] = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for key in sorted(G):
            g = G[key].float()
            U, S, V = torch.svd_lowrank(g, q=min(k, *g.shape), niter=4)
            out[key] = (U.contiguous(), S.contiguous(), V.contiguous())
    return out


# ---- Persistence -------------------------------------------------------------------------------

def _flatten(obj: Mapping[Key, torch.Tensor | Factors]) -> dict[str, torch.Tensor]:
    """Safetensors names: "L{layer}.{module}.G" for a full gradient, ".U" / ".S" / ".V" for a sketch."""
    flat: dict[str, torch.Tensor] = {}
    for (layer, module), value in obj.items():
        if isinstance(value, torch.Tensor):
            flat[f"L{layer}.{module}.G"] = value.float().contiguous()
        else:
            for name, t in zip(FACTOR_NAMES, value, strict=True):
                flat[f"L{layer}.{module}.{name}"] = t.float().contiguous()
    return flat


def _unflatten(flat: Mapping[str, torch.Tensor]) -> dict[Key, torch.Tensor | Factors]:
    """Inverse of _flatten."""
    parts: dict[Key, dict[str, torch.Tensor]] = {}
    for name, t in flat.items():
        layer, module, kind = name.split(".")
        parts.setdefault((int(layer[1:]), module), {})[kind] = t
    return {key: (p["G"] if "G" in p else tuple(p[n] for n in FACTOR_NAMES)) for key, p in sorted(parts.items())}


def save(obj: Mapping[Key, torch.Tensor | Factors], path: str | Path, meta: Mapping[str, Any] | None = None) -> Path:
    """Write a full gradient or a sketch to safetensors, with `meta` stored as JSON in the file header.

    Suggested metadata: base revision, prompt panel ids, condition, lang, opener sets, dtype and k.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(_flatten(obj), str(path), metadata={META_KEY: json.dumps(dict(meta or {}), sort_keys=True)})
    return path


def load(path: str | Path) -> dict[Key, torch.Tensor | Factors]:
    """Read a full gradient ({key: G}) or a sketch ({key: (U, S, V)}) written by `save`."""
    return _unflatten(load_file(str(path)))


def load_meta(path: str | Path) -> dict[str, Any]:
    """The JSON metadata stored by `save`."""
    with safe_open(str(path), framework="pt") as f:
        header = f.metadata() or {}
    return json.loads(header.get(META_KEY, "{}"))
