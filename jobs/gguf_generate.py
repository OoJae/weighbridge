"""Greedy generations from a GGUF build of Tiny Aya through a local llama-server (the q4_k_m robustness row,
PREREG 13).

The prompt token ids are exactly those of the PEFT runner (`probe.format_prompt` with the model's own tokenizer),
sent to llama-server's /completion endpoint as ids, so only the weights' precision differs. Decoding is greedy
(top-k 1), at most --max-new-tokens tokens, stopping on the end-of-turn tokens. Rows are written in the runner's
Generations format, so jobs/job_judge.py and the analysis read them unchanged.

    llama-server -m tiny-aya-global-q4_k_m.gguf -c 8192 -np 4 -ngl 99 --port 8091 &
    python jobs/gguf_generate.py --server http://127.0.0.1:8091 --tokenizer <tiny-aya-global folder> \\
        --panels data/panels/en_outcome_harmful.jsonl data/panels/en_outcome_benign.jsonl \\
        --alias base_q4_k_m --out <PRIVATE_DIR>/gguf/q4km/raw/base_q4_k_m.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from weighbridge.common import STOP_TOKEN_IDS
from weighbridge.probe import format_prompt
from weighbridge.runner_peft import done_keys, generation_row, read_jsonl


def complete(server: str, ids: list[int], max_new_tokens: int, timeout: float = 600.0) -> dict:
    """One greedy completion of a token-id prompt; returns llama-server's JSON reply."""
    body = {"prompt": ids, "n_predict": max_new_tokens, "temperature": 0.0, "top_k": 1, "samplers": ["top_k"],
            "cache_prompt": True, "return_tokens": True, "special": False}
    req = urllib.request.Request(f"{server}/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", required=True)
    ap.add_argument("--tokenizer", required=True, help="folder with the model's tokenizer (chat template)")
    ap.add_argument("--panels", nargs="+", required=True)
    ap.add_argument("--condition", default="shipped")
    ap.add_argument("--alias", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--workers", type=int, default=4, help="parallel requests (match llama-server -np)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = [r for p in args.panels for r in read_jsonl(p)]
    done = done_keys(args.out, ("adapter_alias", "condition", "prompt_id")) if args.out.exists() else set()
    todo = [r for r in rows if (args.alias, args.condition, r["id"]) not in done]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    decode = {"greedy": True, "max_new_tokens": args.max_new_tokens, "stop_token_ids": list(STOP_TOKEN_IDS),
              "runner": "llama.cpp", "server": args.server}

    def one(row: dict) -> dict:
        ids = format_prompt(tok, row["text"], args.condition)
        reply = complete(args.server, ids, args.max_new_tokens)
        tokens = [t for t in reply.get("tokens", []) if t not in STOP_TOKEN_IDS]
        text = tok.decode(tokens, skip_special_tokens=True) if tokens else reply.get("content", "")
        finish = "stop" if reply.get("stop_type") in ("eos", "word") else "length"
        return generation_row(args.alias, args.condition, row, text.strip(), len(tokens), finish, "llama.cpp", decode)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool, args.out.open("a", encoding="utf-8") as fh:
        for i, out in enumerate(pool.map(one, todo), 1):
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 25 == 0:
                print(f"{i}/{len(todo)} in {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"done: {len(todo)} generations in {time.perf_counter() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
