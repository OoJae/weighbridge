"""Diagnose the guard judge on a GPU: raw Qwen3Guard output, NaN/inf in the logits, and speed, per dtype.

    python jobs/judge_debug.py --panel data/panels/judge_validation.jsonl --out out/judge_debug.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from weighbridge.judges import (GUARD_MAX_NEW_TOKENS, GUARD_REPO, GUARD_REVISION, guard_texts,
                                parse_guard_output)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.panel.read_text().splitlines() if line.strip()]
    pick = [r for r in rows if r["human_majority"] == 1][: args.n // 2] + \
           [r for r in rows if r["human_majority"] == 0][: args.n // 2]
    tok = AutoTokenizer.from_pretrained(GUARD_REPO, revision=GUARD_REVISION)
    tok.padding_side = "left"
    texts = guard_texts(tok, [r["text"] for r in pick], [r["response"] for r in pick])
    report = {"template_tail": texts[0][-300:], "runs": {}}
    for name, dtype in (("float16", torch.float16), ("bfloat16", torch.bfloat16)):
        model = AutoModelForCausalLM.from_pretrained(GUARD_REPO, revision=GUARD_REVISION, dtype=dtype).cuda().eval()
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            logits = model(**enc).logits[:, -1].float()
            t0 = time.perf_counter()
            out = model.generate(**enc, max_new_tokens=GUARD_MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize()
        raw = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        raw_special = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=False)
        report["runs"][name] = {
            "seconds": round(time.perf_counter() - t0, 2), "logits_nan": bool(torch.isnan(logits).any()),
            "logits_inf": bool(torch.isinf(logits).any()),
            "items": [{"human": r["human_majority"], "raw": t, "raw_special": s[:200], **parse_guard_output(t)}
                      for r, t, s in zip(pick, raw, raw_special)],
        }
        print(name, json.dumps(report["runs"][name], indent=1)[:3000], flush=True)
        del model
        torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
