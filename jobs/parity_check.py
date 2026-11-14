"""Parity check between the vLLM and PEFT runners on a CUDA machine.

Runs both runners on a seeded pick of adapters (default 3) and prompts (default 10) under one condition,
then reports how often the two greedy generations agree: exact text match, same first token, and the
per-token agreement rate. Token ids are recovered by re-encoding each text with the base tokenizer, which
is exact whenever the two texts are equal and a close proxy otherwise.

Outputs (both generation files and ``parity.json``) go to ``--out-dir``, which defaults to a folder under
``common.PRIVATE_DIR`` because the generations come from third-party adapters.

Example::

    python jobs/parity_check.py --shard <private>/shards/shard_00.jsonl --panel data/panels/harmful_en.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from weighbridge.common import BASE_REPO, BASE_REVISION, CACHE_DIR, CONDITIONS, PRIVATE_DIR, ensure_dir
from weighbridge.runner_peft import GEN_KEY, read_jsonl


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line options of the parity check."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shard", type=Path, required=True, help="local JSONL of manifest rows")
    ap.add_argument("--panel", type=Path, required=True, help="panel JSONL to draw prompts from")
    ap.add_argument("--condition", default="shipped", choices=CONDITIONS)
    ap.add_argument("--n-adapters", type=int, default=3)
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=PRIVATE_DIR / "parity")
    ap.add_argument("--base-model", default=BASE_REPO)
    ap.add_argument("--revision", default=BASE_REVISION)
    return ap.parse_args(argv)


def seeded_pick(rows: Sequence[dict], n: int, seed: int) -> list[dict]:
    """A seeded random sample of at most n rows, in their original order."""
    idx = sorted(random.Random(seed).sample(range(len(rows)), min(n, len(rows))))
    return [rows[i] for i in idx]


def token_agreement(a: Sequence[int], b: Sequence[int]) -> float:
    """Share of positions, up to the longer sequence, where both have the same token (1.0 if both are empty)."""
    width = max(len(a), len(b))
    if width == 0:
        return 1.0
    return sum(x == y for x, y in zip(a, b)) / width


def compare(peft_rows: Sequence[dict], vllm_rows: Sequence[dict], tokenizer: Any) -> dict[str, float | int]:
    """Exact-match rate, first-token agreement and mean per-token agreement over rows present in both runs."""
    vllm_by_key = {tuple(r[k] for k in GEN_KEY): r for r in vllm_rows}
    pairs = [(r, vllm_by_key[key]) for r in peft_rows if (key := tuple(r[k] for k in GEN_KEY)) in vllm_by_key]
    if not pairs:
        raise ValueError("the two runs share no rows")
    exact, first, per_token = 0, 0, 0.0
    for p, v in pairs:
        a = tokenizer.encode(p["text"], add_special_tokens=False)
        b = tokenizer.encode(v["text"], add_special_tokens=False)
        exact += p["text"] == v["text"]
        first += a[:1] == b[:1]
        per_token += token_agreement(a, b)
    n = len(pairs)
    return {"n_pairs": n, "exact_match_rate": exact / n, "first_token_agreement": first / n,
            "per_token_agreement": per_token / n}


def main(argv: Sequence[str] | None = None) -> None:
    """Run both runners on the same adapters and prompts and write the agreement report."""
    from weighbridge.hub import download_adapter
    from weighbridge.runner_peft import generate_peft, load_base
    from weighbridge.runner_vllm import build_llm, generate_vllm

    args = parse_args(argv)
    shard = [r for r in read_jsonl(args.shard) if r.get("status", "included") == "included"]
    rows = seeded_pick(shard, args.n_adapters, args.seed)
    prompts = seeded_pick(read_jsonl(args.panel), args.n_prompts, args.seed)
    adapters = [(r["alias"], Path(download_adapter(r, ensure_dir(CACHE_DIR / "adapters") / r["alias"])))
                for r in rows]
    out_dir = ensure_dir(args.out_dir)
    for name in ("vllm.jsonl", "peft.jsonl", "vllm.timings.jsonl", "peft.timings.jsonl"):
        (out_dir / name).unlink(missing_ok=True)  # a parity run always starts from scratch

    # vLLM first, with half the GPU, so the Transformers model still fits afterwards.
    llm = build_llm(args.base_model, args.revision, gpu_memory_utilization=0.5)
    vllm_out = generate_vllm(args.base_model, adapters, prompts, [args.condition], args.max_new_tokens,
                             out_path=out_dir / "vllm.jsonl", include_base=False, llm=llm, revision=args.revision)
    model, tok = load_base(args.base_model, args.revision)
    peft_out = generate_peft(args.base_model, adapters, prompts, [args.condition], args.max_new_tokens,
                             out_path=out_dir / "peft.jsonl", include_base=False, model=model, tokenizer=tok)

    report = {"condition": args.condition, "adapters": [a for a, _ in adapters], "prompts": len(prompts),
              **compare(read_jsonl(peft_out), read_jsonl(vllm_out), tok)}
    (out_dir / "parity.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
