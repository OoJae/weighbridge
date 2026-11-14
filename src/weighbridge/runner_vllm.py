"""Behaviour runner on vLLM: the same Generations contract as ``runner_peft.generate_peft``, for CUDA jobs.

vLLM is imported lazily (it never runs on the Mac). Prompts are tokenised with the same Transformers
tokenizer and ``probe.format_prompt`` as the PEFT runner and passed to vLLM as token ids, so both runners
see identical inputs. Each adapter is one ``LoRARequest``; requests are grouped by adapter.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .common import BASE_REPO, BASE_REVISION, STOP_TOKEN_IDS
from .runner_peft import (
    BASE_ALIAS,
    GEN_KEY,
    AdapterSpec,
    append_jsonl,
    done_keys,
    generation_row,
    pending_prompts,
    record_timing,
)

MAX_MODEL_LEN = 2048
MAX_LORA_RANK = 64
MAX_LORAS = 8


def _revision(base_model_id: str, revision: str | None) -> str | None:
    """The pinned base revision by default for the Tiny Aya base repo."""
    return revision if revision is not None else (BASE_REVISION if base_model_id == BASE_REPO else None)


def build_llm(base_model_id: str = BASE_REPO, revision: str | None = None,
              gpu_memory_utilization: float = 0.9, seed: int = 0) -> Any:
    """A vLLM engine with LoRA enabled and prefix caching on (the shipped preamble is a shared prefix)."""
    from vllm import LLM

    return LLM(model=base_model_id, revision=_revision(base_model_id, revision), dtype="bfloat16",
               enable_lora=True, max_lora_rank=MAX_LORA_RANK, max_loras=MAX_LORAS, enable_prefix_caching=True,
               max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=gpu_memory_utilization, seed=seed)


def load_tokenizer(base_model_id: str = BASE_REPO, revision: str | None = None) -> Any:
    """The Transformers tokenizer used to build prompt token ids (identical to the PEFT runner's)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(base_model_id, revision=_revision(base_model_id, revision))


def _lora_request(alias: str, path: Path | None, lora_id: int) -> Any:
    """A LoRARequest for an adapter, or None for the base model."""
    if path is None:
        return None
    from vllm.lora.request import LoRARequest

    return LoRARequest(lora_name=alias, lora_int_id=lora_id, lora_path=str(path))


def _output_row(alias: str, condition: str, prompt: dict, completion: Any, tokenizer: Any, decode: dict) -> dict:
    """Convert one vLLM CompletionOutput into a Generations row (stop token excluded from n_tokens)."""
    ids = list(completion.token_ids)
    if completion.finish_reason == "stop" and ids and ids[-1] in STOP_TOKEN_IDS:
        ids = ids[:-1]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    finish = "stop" if completion.finish_reason == "stop" else "length"
    return generation_row(alias, condition, prompt, text, len(ids), finish, "vllm", decode)


def _generate_condition(llm: Any, tokenizer: Any, alias: str, lora: Any, condition: str, rows: Sequence[dict],
                        sampling: Any, decode: dict, out_path: Path) -> int:
    """All pending prompts of one adapter and condition in one vLLM call, appended at once."""
    from .probe import format_prompt

    prompts = [{"prompt_token_ids": format_prompt(tokenizer, r["text"], condition)} for r in rows]
    results = llm.generate(prompts, sampling_params=sampling, lora_request=lora, use_tqdm=False)
    out_rows = [_output_row(alias, condition, r, res.outputs[0], tokenizer, decode) for r, res in zip(rows, results)]
    append_jsonl(out_path, out_rows)
    return len(out_rows)


def generate_vllm(base_model_id: str, adapters: Sequence[AdapterSpec], panel_rows: Sequence[dict],
                  conditions: Sequence[str], max_new_tokens: int = 160, *, out_path: str | Path,
                  include_base: bool = True, llm: Any = None, tokenizer: Any = None, revision: str | None = None,
                  first_lora_id: int = 1) -> Path:
    """Greedy generations with vLLM, written as Generations rows (``runner="vllm"``) to ``out_path``.

    Pass ``llm`` (from ``build_llm``) and ``tokenizer`` to reuse an engine across calls; then give each
    call a ``first_lora_id`` past the ids already used, because vLLM caches adapters by integer id.
    Resumable, with wall-clock seconds per adapter in ``timings_path(out_path)``.
    """
    from vllm import SamplingParams

    out = Path(out_path)
    done = done_keys(out, GEN_KEY)
    llm = llm if llm is not None else build_llm(base_model_id, revision)
    tokenizer = tokenizer if tokenizer is not None else load_tokenizer(base_model_id, revision)
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, stop_token_ids=list(STOP_TOKEN_IDS))
    decode = {"greedy": True, "max_new_tokens": max_new_tokens, "stop_token_ids": list(STOP_TOKEN_IDS),
              "dtype": "bfloat16"}
    specs: list[tuple[str, Path | None, int]] = [
        (alias, Path(path), first_lora_id + i) for i, (alias, path) in enumerate(adapters)]
    if include_base:
        specs.insert(0, (BASE_ALIAS, None, 0))
    for alias, path, lora_id in specs:
        todo = {c: pending_prompts(panel_rows, alias, c, done) for c in conditions}
        if not any(todo.values()):
            continue
        start = time.perf_counter()
        lora = _lora_request(alias, path, lora_id)
        n = sum(_generate_condition(llm, tokenizer, alias, lora, c, rows, sampling, decode, out)
                for c, rows in todo.items() if rows)
        record_timing(out, alias, "generate", "vllm", time.perf_counter() - start, n)
    return out
