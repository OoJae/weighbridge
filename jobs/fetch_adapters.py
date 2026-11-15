"""Download every adapter of the given manifest shards at its pinned SHA into one folder, named by alias.

    python jobs/fetch_adapters.py --shards shards/confirmatory_0.jsonl shards/confirmatory_1.jsonl --dest DIR

The folder layout is what ``weighbridge.cli score --adapters DIR`` reads. Third-party weights stay in DIR,
which must be outside the repository (``hub.download_adapter`` refuses otherwise).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from job_sweep import adapter_fetcher  # noqa: E402

from weighbridge.runner_peft import read_jsonl  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shards", nargs="+", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    args = ap.parse_args(argv)
    args.dest.mkdir(parents=True, exist_ok=True)
    fetch = adapter_fetcher(args.dest)
    rows = [r for s in args.shards for r in read_jsonl(s) if r["status"] == "included"]
    failed = []
    for i, row in enumerate(rows, 1):
        try:
            fetch(row)
        except Exception as exc:  # one unreachable repo must not stop the rest; it is reported at the end
            failed.append(row["alias"])
            print(f"failed {row['alias']}: {type(exc).__name__}: {exc}", flush=True)
        if i % 10 == 0:
            print(f"fetched {i}/{len(rows)}", flush=True)
    print(f"fetched {len(rows) - len(failed)}/{len(rows)} adapters into {args.dest}; failed: {failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
