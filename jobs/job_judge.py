# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "vllm",
#   "fasttext>=0.9.3",
#   "numpy>=1.26",
#   "huggingface_hub>=1.5",
#   "pyyaml>=6",
# ]
# ///
"""HF Jobs entry point (CUDA container): judge labels for one generation run.

Reads ``<run-id>/raw/<alias>.jsonl`` generation files from the private dataset repo --gen-repo, looks up
each prompt's text in --panels, and labels every generation with Qwen3Guard-Gen-4B (vLLM), the string
refusal matcher and GlotLID. Label rows (generation keys plus guard, string-refusal, LID and label3
fields; never the generated text) are appended to ``<work-dir>/labels/<run-id>.jsonl``, one adapter at a
time so a restarted job skips adapters already labelled, then uploaded to ``labels/<run-id>.jsonl`` in
the private dataset repo --out-repo.

The weighbridge package must be importable. If it is not installed in the container, pass --code-repo: a
dataset repo holding this repository's tree (src/ and configs/), which is downloaded and put on sys.path.

Example:
  hf jobs uv run --flavor a10g-large --secrets HF_TOKEN jobs/job_judge.py \\
      --gen-repo <owner>/<runs-repo> --run-id sweep1 \\
      --panels hf://datasets/<owner>/<panels-repo>/en.jsonl hf://datasets/<owner>/<panels-repo>/yo.jsonl \\
      --code-repo <owner>/<code-repo> --out-repo <owner>/<runs-repo>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("job_judge")

TIMINGS_SUFFIX = ".timings.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--gen-repo", help="private dataset repo holding <run-id>/raw/*.jsonl")
    src.add_argument("--gen-dir", type=Path, help="local folder of per-adapter generation files (*.jsonl)")
    p.add_argument("--backend", choices=("hf", "vllm"), default="hf",
                   help="Qwen3Guard backend; 'hf' on a T4 (no bf16, vLLM untested there)")
    p.add_argument("--run-id", required=True)
    p.add_argument("--panels", nargs="+", required=True,
                   help="panel jsonl files: local paths or hf://datasets/<owner>/<repo>/<path>")
    p.add_argument("--out-repo", help="private dataset repo for the label file; omit to keep it local")
    p.add_argument("--work-dir", type=Path, default=Path("judge_out"))
    p.add_argument("--code-repo", help="dataset repo with this repository's tree, if weighbridge is not installed")
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args(argv)


def ensure_weighbridge(code_repo: str | None, token: str | None) -> None:
    """Make `import weighbridge` work, downloading the repository tree from `code_repo` when needed."""
    try:
        import weighbridge  # noqa: F401
    except ImportError:
        if not code_repo:
            raise SystemExit("weighbridge is not installed; pass --code-repo") from None
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(code_repo, repo_type="dataset", token=token))
        sys.path.insert(0, str(root / "src"))


# ---- Inputs ------------------------------------------------------------------------------------

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Every non-blank row of a JSONL file."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_panel(source: str, token: str | None) -> list[dict[str, Any]]:
    """Rows of a panel jsonl given as a local path or as hf://datasets/<owner>/<repo>/<path>."""
    prefix = "hf://datasets/"
    if source.startswith(prefix):
        from huggingface_hub import hf_hub_download

        owner, repo, filename = source[len(prefix):].split("/", 2)
        source = hf_hub_download(f"{owner}/{repo}", filename, repo_type="dataset", token=token)
    return read_jsonl(source)


def prompt_texts(panels: list[str], token: str | None) -> dict[str, str]:
    """prompt_id -> prompt text over all panels."""
    return {row["id"]: row["text"] for source in panels for row in read_panel(source, token)}


def generation_files(gen_repo: str, run_id: str, token: str | None) -> list[Path]:
    """Download the run's per-adapter generation files (timing sidecars excluded), sorted by name."""
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(gen_repo, repo_type="dataset", token=token,
                                  allow_patterns=[f"{run_id}/raw/*.jsonl"]))
    return local_generation_files(root / run_id / "raw")


def local_generation_files(gen_dir: Path) -> list[Path]:
    """Per-adapter generation files in a local folder (timing sidecars excluded), sorted by name."""
    return sorted(p for p in gen_dir.glob("*.jsonl") if not p.name.endswith(TIMINGS_SUFFIX))


# ---- Labelling ---------------------------------------------------------------------------------

def label_file(rows: list[dict[str, Any]], prompts: dict[str, str], batch_size: int,
               backend: str = "hf") -> list[dict[str, Any]]:
    """Label rows for the generations of one adapter."""
    from weighbridge.judges import label_row, lang_id, lang_id_backend, qwen3guard_labels

    responses = [r["text"] for r in rows]
    guards = qwen3guard_labels([prompts[r["prompt_id"]] for r in rows], responses, backend=backend,
                               batch_size=batch_size)
    lids = lang_id(responses)
    backend = lang_id_backend()
    return [label_row(row, guard, lid) | {"lid_backend": backend} for row, guard, lid in zip(rows, guards, lids)]


def labelled_aliases(out_path: Path) -> set[str]:
    """Adapters already present in the label file (for resuming)."""
    return {row["adapter_alias"] for row in read_jsonl(out_path)} if out_path.exists() else set()


def append_rows(out_path: Path, rows: list[dict[str, Any]]) -> None:
    """Append label rows to the JSONL output."""
    with out_path.open("a", encoding="utf-8") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def upload(out_path: Path, out_repo: str, run_id: str, token: str | None) -> None:
    """Upload the label file to labels/<run-id>.jsonl in a private dataset repo."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(out_repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=str(out_path), path_in_repo=f"labels/{run_id}.jsonl", repo_id=out_repo,
                    repo_type="dataset", commit_message=f"{run_id}: judge labels")


def main(argv: list[str] | None = None) -> None:
    """Label every generation file of the run, then upload the label file."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    token = os.environ.get("HF_TOKEN")
    ensure_weighbridge(args.code_repo, token)

    prompts = prompt_texts(args.panels, token)
    out_path = args.work_dir / "labels" / f"{args.run_id}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = labelled_aliases(out_path)
    files = (local_generation_files(args.gen_dir) if args.gen_dir
             else generation_files(args.gen_repo, args.run_id, token))
    for path in files:
        rows = [r for r in read_jsonl(path) if r["adapter_alias"] not in done]
        if rows:
            append_rows(out_path, label_file(rows, prompts, args.batch_size, args.backend))
            log.info("labelled %d generations from %s", len(rows), path.name)
    if args.out_repo:
        upload(out_path, args.out_repo, args.run_id, token)


if __name__ == "__main__":
    main()
