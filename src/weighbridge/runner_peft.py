"""Behaviour runner on Transformers + PEFT: greedy generations and forward-pass (T1) screens.

One base model is loaded once; adapters are hot-swapped onto it with ``load_adapter`` / ``set_adapter`` /
``delete_adapter``. The base model itself is run under the alias ``__base__``. Every output file is JSONL,
appended once per batch, and resumable: rows whose key is already present are never recomputed.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch

from .common import BASE_REPO, BASE_REVISION, STOP_TOKEN_IDS, ensure_dir

BASE_ALIAS = "__base__"
GEN_KEY = ("adapter_alias", "condition", "prompt_id")
T1_KEY = ("adapter_alias", "condition", "lang", "split", "scale")

AdapterSpec = tuple[str, str | Path]
Directions = Mapping[int, torch.Tensor] | Mapping[str, Mapping[int, torch.Tensor]]


# ---- JSONL helpers (shared with runner_vllm and the jobs) ----------------------------------------

def read_jsonl(path: str | Path) -> list[dict]:
    """All complete rows of a JSONL file; a missing file gives []. A truncated last line is ignored."""
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def append_jsonl(path: str | Path, rows: Sequence[dict]) -> None:
    """Append rows with a single O_APPEND write followed by fsync, so a batch lands whole or not at all."""
    if not rows:
        return
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    ensure_dir(Path(path).parent)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)


def _drop_partial_tail(path: Path) -> None:
    """Truncate a file to its last newline, removing a half-written final row left by a killed process."""
    if not path.exists() or path.stat().st_size == 0:
        return
    data = path.read_bytes()
    if not data.endswith(b"\n"):
        with path.open("r+b") as f:
            f.truncate(data.rfind(b"\n") + 1)


def done_keys(path: str | Path, key: Sequence[str]) -> set[tuple]:
    """Keys of the rows already written to ``path``, after repairing a partial last line."""
    p = Path(path)
    _drop_partial_tail(p)
    return {tuple(r[k] for k in key) for r in read_jsonl(p)}


def timings_path(out_path: str | Path) -> Path:
    """Sidecar file with wall-clock seconds per adapter, e.g. ``gen.jsonl`` -> ``gen.timings.jsonl``."""
    return Path(out_path).with_suffix(".timings.jsonl")


def record_timing(out_path: str | Path, alias: str, stage: str, runner: str, seconds: float, n_rows: int) -> None:
    """Append one timing row for the cost ladder."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    append_jsonl(timings_path(out_path), [{"adapter_alias": alias, "stage": stage, "runner": runner,
                                           "seconds": round(seconds, 3), "n_rows": n_rows, "device": device}])


# ---- Model loading and adapter hot-swapping -------------------------------------------------------

def model_dtype() -> torch.dtype:
    """bfloat16 on a CUDA device with native bf16 (compute capability 8.0 or newer), float16 on an older one such as
    a T4 (7.5), float32 on the CPU.

    torch.cuda.is_bf16_supported() is not used: it also returns True on a T4, where bf16 only runs emulated.
    """
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def load_base(base_model_id: str = BASE_REPO, revision: str | None = None) -> tuple[Any, Any]:
    """Load the base model (in :func:`model_dtype`) and its tokenizer, pinned to a revision.

    ``base_model_id`` may be a Hub id or a local folder. ``revision`` defaults to ``common.BASE_REVISION`` when
    ``base_model_id`` is the Tiny Aya base repo.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rev = revision if revision is not None else (BASE_REVISION if base_model_id == BASE_REPO else None)
    model = AutoModelForCausalLM.from_pretrained(base_model_id, revision=rev, dtype=model_dtype())
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, revision=rev)
    return model, tokenizer


def peft_adapter_name(alias: str) -> str:
    """The PEFT adapter name for an alias: PEFT module names may not contain dots (``S1a__wise-0.25``)."""
    return alias.replace(".", "_")


class AdapterHost:
    """Keeps one base model and swaps PEFT adapters onto it, holding at most one adapter at a time."""

    def __init__(self, model: Any) -> None:
        self.base = model
        self.peft: Any = None
        self.loaded: str | None = None

    @contextmanager
    def use(self, alias: str, path: str | Path | None = None) -> Iterator[Any]:
        """Yield a model whose forward pass is the base (alias ``__base__``) or the base plus that adapter."""
        if alias == BASE_ALIAS:
            if self.peft is None:
                yield self.base
            else:
                with self.peft.disable_adapter():
                    yield self.peft
            return
        if path is None:
            raise ValueError(f"adapter {alias} needs a path")
        self._swap_in(alias, path)
        yield self.peft

    def _swap_in(self, alias: str, path: str | Path) -> None:
        """Load ``alias`` and make it the only adapter: load the new one, activate it, delete the old one."""
        if self.loaded == alias:
            return
        name = peft_adapter_name(alias)
        if self.peft is None:
            from peft import PeftModel

            self.peft = PeftModel.from_pretrained(self.base, str(path), adapter_name=name)
        else:
            self.peft.load_adapter(str(path), adapter_name=name)
            self.peft.set_adapter(name)
            self.peft.delete_adapter(peft_adapter_name(self.loaded))
        self.peft.eval()
        self.loaded = alias

    def close(self) -> None:
        """Remove every injected LoRA layer, leaving the caller's base model exactly as it was given."""
        if self.peft is not None:
            self.peft.unload()
            self.peft, self.loaded = None, None


@contextmanager
def _hosted(base_model_id: str, revision: str | None, model: Any, tokenizer: Any) -> Iterator[tuple[AdapterHost, Any]]:
    """Load the base unless a model and tokenizer were passed in, and always strip adapters on exit."""
    if (model is None) != (tokenizer is None):
        raise ValueError("pass both model and tokenizer, or neither")
    if model is None:
        model, tokenizer = load_base(base_model_id, revision)
    host = AdapterHost(model)
    try:
        yield host, tokenizer
    finally:
        host.close()


@contextmanager
def scaled_adapter(model: Any, adapter_name: str, lam: float) -> Iterator[Any]:
    """Temporarily multiply the LoRA scaling of ``adapter_name`` in every LoraLayer by ``lam``.

    Scaling ΔW by lam is exact for LoRA (ΔW = scaling·B@A), so lam=0 gives the base model and lam=1 the
    adapter. The original scalings are restored on exit, even after an exception.
    """
    from peft.tuners.lora import LoraLayer

    layers = [m for m in model.modules() if isinstance(m, LoraLayer) and adapter_name in m.scaling]
    if not layers:
        raise ValueError(f"no LoRA layer holds adapter {adapter_name!r}")
    saved = [m.scaling[adapter_name] for m in layers]
    for m, s in zip(layers, saved):
        m.scaling[adapter_name] = s * lam
    try:
        yield model
    finally:
        for m, s in zip(layers, saved):
            m.scaling[adapter_name] = s


def _all_specs(adapters: Sequence[AdapterSpec], include_base: bool) -> list[tuple[str, Path | None]]:
    """The adapter list with the base model first when requested."""
    specs: list[tuple[str, Path | None]] = [(BASE_ALIAS, None)] if include_base else []
    return specs + [(alias, Path(path)) for alias, path in adapters]


# ---- Generation ---------------------------------------------------------------------------------

def _device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _pad_left(batch: Sequence[Sequence[int]], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad token id lists into (input_ids, attention_mask)."""
    width = max(len(ids) for ids in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, ids in enumerate(batch):
        input_ids[i, width - len(ids):] = torch.tensor(ids, dtype=torch.long)
        mask[i, width - len(ids):] = 1
    return input_ids.to(device), mask.to(device)


def split_at_stop(token_ids: Sequence[int]) -> tuple[list[int], str]:
    """Content tokens before the first stop token, and the finish reason ("stop" or "length")."""
    for i, tok in enumerate(token_ids):
        if tok in STOP_TOKEN_IDS:
            return list(token_ids[:i]), "stop"
    return list(token_ids), "length"


@torch.inference_mode()
def _greedy(model: Any, batch: Sequence[Sequence[int]], pad_id: int, max_new_tokens: int) -> list[list[int]]:
    """Greedy-decode a batch of prompts and return only the newly generated token ids per prompt."""
    input_ids, mask = _pad_left(batch, pad_id, _device(model))
    out = model.generate(input_ids=input_ids, attention_mask=mask, do_sample=False, max_new_tokens=max_new_tokens,
                         eos_token_id=list(STOP_TOKEN_IDS), pad_token_id=pad_id)
    return out[:, input_ids.shape[1]:].tolist()


def _length_batches(items: Sequence[tuple[dict, list[int]]], batch_size: int) -> list[list[tuple[dict, list[int]]]]:
    """Sort (row, prompt ids) pairs by prompt length and cut them into batches, keeping padding small."""
    ordered = sorted(items, key=lambda it: (len(it[1]), it[0]["id"]))
    return [ordered[i:i + batch_size] for i in range(0, len(ordered), batch_size)]


def generation_row(alias: str, condition: str, prompt: dict, text: str, n_tokens: int, finish_reason: str,
                   runner: str, decode: dict) -> dict:
    """One Generations row in the contract's format."""
    return {"adapter_alias": alias, "condition": condition, "prompt_id": prompt["id"], "lang": prompt["lang"],
            "kind": prompt["kind"], "text": text, "n_tokens": n_tokens, "finish_reason": finish_reason,
            "runner": runner, "decode": decode}


def pending_prompts(panel_rows: Sequence[dict], alias: str, condition: str, done: set[tuple]) -> list[dict]:
    """Panel rows not yet generated for this adapter and condition."""
    return [r for r in panel_rows if (alias, condition, r["id"]) not in done]


def _pad_id(tokenizer: Any) -> int:
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else STOP_TOKEN_IDS[0]


def _generate_condition(model: Any, tokenizer: Any, alias: str, condition: str, rows: Sequence[dict],
                        max_new_tokens: int, batch_size: int, decode: dict, out_path: Path) -> int:
    """Generate and append rows for one adapter and condition; returns the number of rows written."""
    from .probe import format_prompt

    items = [(r, format_prompt(tokenizer, r["text"], condition)) for r in rows]
    written = 0
    for batch in _length_batches(items, batch_size):
        outputs = _greedy(model, [ids for _, ids in batch], _pad_id(tokenizer), max_new_tokens)
        out_rows = []
        for (prompt, _), gen in zip(batch, outputs):
            content, finish = split_at_stop(gen)
            text = tokenizer.decode(content, skip_special_tokens=True)
            out_rows.append(generation_row(alias, condition, prompt, text, len(content), finish, "peft", decode))
        append_jsonl(out_path, out_rows)
        written += len(out_rows)
    return written


def generate_peft(base_model_id: str, adapters: Sequence[AdapterSpec], panel_rows: Sequence[dict],
                  conditions: Sequence[str], max_new_tokens: int = 160, batch_size: int = 16, *,
                  out_path: str | Path, include_base: bool = True, model: Any = None, tokenizer: Any = None,
                  revision: str | None = None) -> Path:
    """Greedy generations for the base model and each adapter, written as Generations rows to ``out_path``.

    ``adapters`` is a list of (alias, local adapter folder). Pass ``model`` and ``tokenizer`` to reuse an
    already-loaded base (it is returned unmodified). Rows already in ``out_path`` are skipped, and the
    wall-clock seconds per adapter go to ``timings_path(out_path)``.
    """
    out = Path(out_path)
    done = done_keys(out, GEN_KEY)
    with _hosted(base_model_id, revision, model, tokenizer) as (host, tok):
        decode = {"greedy": True, "max_new_tokens": max_new_tokens, "stop_token_ids": list(STOP_TOKEN_IDS),
                  "batch_size": batch_size, "dtype": str(next(host.base.parameters()).dtype).removeprefix("torch.")}
        for alias, path in _all_specs(adapters, include_base):
            todo = {c: pending_prompts(panel_rows, alias, c, done) for c in conditions}
            if not any(todo.values()):
                continue
            start = time.perf_counter()
            with host.use(alias, path) as m:
                n = sum(_generate_condition(m, tok, alias, c, rows, max_new_tokens, batch_size, decode, out)
                        for c, rows in todo.items() if rows)
            record_timing(out, alias, "generate", "peft", time.perf_counter() - start, n)
    return out


# ---- Forward-pass screens (T1) -------------------------------------------------------------------

def _directions_for(directions: Directions, condition: str) -> Mapping[int, torch.Tensor]:
    """Directions may be one {layer: vector} map or one map per condition ({condition: {layer: vector}})."""
    return directions[condition] if condition in directions else directions  # type: ignore[return-value]


def default_layer(directions: Mapping[int, torch.Tensor]) -> int:
    """The middle layer among those with a direction, used when no projection layer is given."""
    layers = sorted(directions)
    return layers[len(layers) // 2]


def _panel_groups(panel_rows: Sequence[dict]) -> dict[tuple[str, str], list[dict]]:
    """Panel rows grouped by (lang, split), in panel order."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in panel_rows:
        groups.setdefault((r["lang"], r["split"]), []).append(r)
    return groups


def _t1_row(model: Any, tokenizer: Any, alias: str, condition: str, group: tuple[str, str], rows: Sequence[dict],
            directions: Mapping[int, torch.Tensor], layer: int, scale: float, batch_size: int) -> dict:
    """Margin and refusal-direction projection for one prompt group, as one T1 row (float32 math)."""
    from .probe import margin, projection

    texts = [r["text"] for r in rows]
    lang, split = group
    m = margin(model, tokenizer, texts, lang, condition, batch_size=batch_size).float().cpu()
    p = projection(model, tokenizer, texts, directions, condition, layer, batch_size=batch_size).float().cpu()
    return {"adapter_alias": alias, "condition": condition, "lang": lang, "split": split, "scale": scale,
            "layer": layer, "prompt_ids": [r["id"] for r in rows],
            "margin_mean": m.mean().item(), "margin_per_prompt": m.tolist(),
            "proj_mean": p.mean().item(), "proj_per_prompt": p.tolist(), "runner": "peft"}


def _t1_adapter(model: Any, tokenizer: Any, alias: str, conditions: Sequence[str], scales: Sequence[float],
                groups: dict[tuple[str, str], list[dict]], directions: Directions, layer: int | None,
                batch_size: int, done: set[tuple], out: Path) -> int:
    """All pending T1 rows of one adapter (every scale, condition and group); returns rows written."""
    written = 0
    for scale in scales:
        for condition in conditions:
            dirs = _directions_for(directions, condition)
            lay = layer if layer is not None else default_layer(dirs)
            todo = [g for g in groups if (alias, condition, g[0], g[1], scale) not in done]
            if not todo:
                continue
            ctx = nullcontext(model) if alias == BASE_ALIAS else scaled_adapter(model, peft_adapter_name(alias), scale)
            with ctx:
                rows = [_t1_row(model, tokenizer, alias, condition, g, groups[g], dirs, lay, scale, batch_size)
                        for g in todo]
            append_jsonl(out, rows)
            written += len(rows)
    return written


def t1_peft(base_model_id: str, adapters: Sequence[AdapterSpec], panel_rows: Sequence[dict],
            conditions: Sequence[str], directions: Directions, out_path: str | Path, *,
            scales: Sequence[float] = (1.0,), layer: int | None = None, batch_size: int = 8,
            include_base: bool = True, model: Any = None, tokenizer: Any = None,
            revision: str | None = None) -> Path:
    """Forward-pass screens: one T1 row per adapter x condition x lang x split (x scale), no generation.

    ``directions`` is {layer: unit vector} or {condition: {layer: unit vector}}; ``layer`` picks the
    projection layer (default: ``default_layer``). ``scales`` multiplies each adapter's ΔW, e.g.
    (0.25, 0.5, 1.0) for the linearity probe; the base model is screened once, at scale 1.0. Resumable,
    and timings go to ``timings_path(out_path)``.
    """
    out = Path(out_path)
    done = done_keys(out, T1_KEY)
    groups = _panel_groups(panel_rows)
    with _hosted(base_model_id, revision, model, tokenizer) as (host, tok):
        for alias, path in _all_specs(adapters, include_base):
            alias_scales = (1.0,) if alias == BASE_ALIAS else tuple(scales)
            if all((alias, c, g[0], g[1], s) in done for c in conditions for g in groups for s in alias_scales):
                continue
            start = time.perf_counter()
            with host.use(alias, path) as m:
                n = _t1_adapter(m, tok, alias, conditions, alias_scales, groups, directions, layer,
                                batch_size, done, out)
            record_timing(out, alias, "t1", "peft", time.perf_counter() - start, n)
    return out
