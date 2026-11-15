"""HF Jobs entry point: behaviour sweep of one shard of adapters on a CUDA container.

For every adapter in the shard (manifest rows with ``alias``, ``repo_id``, ``sha``, ...): download it at its
pinned SHA, generate on the panels with the chosen runner, then run the forward-pass screen (T1) on the
D1 and Y2 panels, optionally at the linearity scales. Outputs are keyed by alias only and pushed to a
private dataset repo after every adapter, so a killed job loses at most one adapter; on restart the job
pulls what is already there and resumes.

Layout under ``<work-dir>/<run-id>/`` and ``<out-repo>/<run-id>/``::

    raw/<alias>.jsonl            Generations rows (+ raw/<alias>.timings.jsonl)
    t1/<alias>.jsonl             Forward-pass screen rows (+ t1/<alias>.timings.jsonl)
    directions.safetensors       refusal directions per condition, fitted on the base model

Example::

    python jobs/job_sweep.py --shard shards/shard_03.jsonl --shard-repo <private-dataset> \\
        --panels data/panels/en_outcome_harmful.jsonl data/panels/en_outcome_benign.jsonl \\
        --t1-panels data/panels/d1_probe.jsonl data/panels/y2_heldout.jsonl --directions <probe output> \\
        --runner peft --run-id sweep1 --out-repo <private-dataset>
"""

from __future__ import annotations

import argparse
import itertools
import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from weighbridge.common import BASE_REPO, BASE_REVISION, CONDITIONS, PRIVATE_DIR, ensure_dir
from weighbridge.runner_peft import BASE_ALIAS, read_jsonl, timings_path

log = logging.getLogger("job_sweep")

LINEARITY_SCALES = (0.25, 0.5, 1.0)
DIRECTIONS_FILE = "directions.safetensors"

Fetch = Callable[[dict], Path]
Directions = dict[str, dict[int, torch.Tensor]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line options of the sweep job."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shard", required=True, help="JSONL of manifest rows: a local path, or a file in --shard-repo")
    ap.add_argument("--shard-repo", help="private dataset repo holding the shard file")
    ap.add_argument("--panels", nargs="*", default=[], help="panel JSONL files to generate on")
    ap.add_argument("--t1-panels", nargs="*", default=[], help="panel JSONL files for the forward-pass screen")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    ap.add_argument("--runner", choices=("peft", "vllm"), default="vllm")
    ap.add_argument("--linearity", action="store_true", help=f"screen adapters at scales {LINEARITY_SCALES}")
    ap.add_argument("--skip-base", action="store_true", help="do not run the base model (another shard does)")
    ap.add_argument("--directions", help="shared directions: this job's directions.safetensors or a job_probe output "
                                         "folder, as a local path or hf://datasets/<owner>/<repo>/<path>")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out-repo", help="private dataset repo for outputs; omit to keep outputs local")
    ap.add_argument("--work-dir", type=Path, default=PRIVATE_DIR / "jobs")
    ap.add_argument("--base-model", default=BASE_REPO)
    ap.add_argument("--revision", default=BASE_REVISION)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=16)
    return ap.parse_args(argv)


# ---- Inputs ------------------------------------------------------------------------------------

def read_shard(shard: str, shard_repo: str | None) -> list[dict]:
    """Included manifest rows of the shard, from a local file or a private dataset repo."""
    path = Path(shard)
    if shard_repo is not None:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(repo_id=shard_repo, filename=shard, repo_type="dataset"))
    return [r for r in read_jsonl(path) if r.get("status", "included") == "included"]


def load_panels(paths: Sequence[str]) -> list[dict]:
    """Rows of every panel file, in order."""
    return [row for p in paths for row in read_jsonl(p)]


def adapter_fetcher(dest_root: Path) -> Fetch:
    """A memoised downloader: manifest row -> local adapter folder at the row's pinned SHA.

    A row with ``local_path`` (our own sentinels and their repaired variants) is used where it is, never downloaded.
    """
    from weighbridge.hub import download_adapter

    cache: dict[str, Path] = {}

    def fetch(row: dict) -> Path:
        if row["alias"] not in cache:
            if row.get("local_path"):
                cache[row["alias"]] = Path(row["local_path"])
            else:
                cache[row["alias"]] = Path(download_adapter(row, dest_root / row["alias"]))
        return cache[row["alias"]]

    return fetch


def try_fetch(fetch: Fetch, row: dict) -> Path | None:
    """The adapter folder, or None (logged) when the pinned revision can no longer be downloaded.

    An author can delete or rewrite a repository after the snapshot; that adapter is reported as missing for the
    stages it did not reach, and the rest of the shard still runs.
    """
    try:
        return fetch(row)
    except Exception as exc:  # noqa: BLE001 - any download failure means the adapter is unavailable
        log.warning("adapter %s unavailable, skipped: %s", row["alias"], str(exc).splitlines()[0])
        return None


# ---- Output sync -------------------------------------------------------------------------------

class RunSync:
    """Mirrors ``<work-dir>/<run-id>`` to ``<out-repo>/<run-id>`` in a private dataset repo."""

    def __init__(self, out_repo: str, work_dir: Path, run_id: str) -> None:
        from huggingface_hub import HfApi

        self.api, self.repo, self.work_dir, self.run_id = HfApi(), out_repo, work_dir, run_id
        self.api.create_repo(out_repo, repo_type="dataset", private=True, exist_ok=True)

    def pull(self) -> None:
        """Download outputs from an earlier attempt of this run, so the runners resume from them."""
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=self.repo, repo_type="dataset", allow_patterns=[f"{self.run_id}/*"],
                          local_dir=self.work_dir)

    def push(self, paths: Sequence[Path], message: str) -> None:
        """Upload files in one commit. A failure is logged, not raised: the final ``push_all`` retries it."""
        from huggingface_hub import CommitOperationAdd

        ops = [CommitOperationAdd(path_in_repo=p.relative_to(self.work_dir).as_posix(), path_or_fileobj=str(p))
               for p in paths if p.exists()]
        if not ops:
            return
        try:
            self.api.create_commit(self.repo, operations=ops, commit_message=message, repo_type="dataset")
        except Exception:  # noqa: BLE001 - a network blip must not kill a paid GPU job mid-shard
            log.exception("upload failed (%s); will retry at the end", message)

    def push_all(self) -> None:
        """Upload the whole run folder."""
        self.api.upload_folder(repo_id=self.repo, repo_type="dataset", folder_path=self.work_dir / self.run_id,
                               path_in_repo=self.run_id, commit_message=f"{self.run_id}: final sync")


def _push(sync: RunSync | None, out: Path, message: str) -> None:
    if sync is not None:
        sync.push([out, timings_path(out)], message)


# ---- Generation stage --------------------------------------------------------------------------

Generate = Callable[[list[tuple[str, Path]], bool, Path], Path]


def make_generator(args: argparse.Namespace, panel_rows: Sequence[dict], hf: tuple[Any, Any] | None) -> Generate:
    """generate(adapters, include_base, out_path) for the chosen runner, reusing one loaded model or engine."""
    if args.runner == "peft":
        from weighbridge.runner_peft import generate_peft

        model, tok = hf

        def generate_with_peft(adapters: list[tuple[str, Path]], include_base: bool, out_path: Path) -> Path:
            return generate_peft(args.base_model, adapters, panel_rows, args.conditions, args.max_new_tokens,
                                 args.batch_size, out_path=out_path, include_base=include_base, model=model,
                                 tokenizer=tok, revision=args.revision)

        return generate_with_peft

    from weighbridge.runner_vllm import build_llm, generate_vllm, load_tokenizer

    # vLLM does not reliably release GPU memory in-process, so leave room for the T1 stage's model up front.
    llm = build_llm(args.base_model, args.revision, gpu_memory_utilization=0.6 if args.t1_panels else 0.9)
    tok = load_tokenizer(args.base_model, args.revision)
    lora_ids = itertools.count(1)  # vLLM caches adapters by integer id, so an id is never reused

    def generate_with_vllm(adapters: list[tuple[str, Path]], include_base: bool, out_path: Path) -> Path:
        return generate_vllm(args.base_model, adapters, panel_rows, args.conditions, args.max_new_tokens,
                             out_path=out_path, include_base=include_base, llm=llm, tokenizer=tok,
                             revision=args.revision, first_lora_id=next(lora_ids))

    return generate_with_vllm


def run_generation(args: argparse.Namespace, rows: Sequence[dict], panel_rows: Sequence[dict], fetch: Fetch,
                   run_dir: Path, sync: RunSync | None, hf: tuple[Any, Any] | None) -> None:
    """Generate for the base model (unless skipped) and every adapter, pushing after each one."""
    raw = ensure_dir(run_dir / "raw")
    generate = make_generator(args, panel_rows, hf)
    if not args.skip_base:
        _push(sync, generate([], True, raw / f"{BASE_ALIAS}.jsonl"), f"{args.run_id}: generations {BASE_ALIAS}")
    for i, row in enumerate(rows):
        start = time.perf_counter()
        folder = try_fetch(fetch, row)
        if folder is None:
            continue
        out = generate([(row["alias"], folder)], False, raw / f"{row['alias']}.jsonl")
        _push(sync, out, f"{args.run_id}: generations {row['alias']}")
        log.info("generation %d/%d %s done in %.0fs", i + 1, len(rows), row["alias"], time.perf_counter() - start)


# ---- Forward-pass screen stage ------------------------------------------------------------------

def save_directions(directions: Directions, path: Path) -> None:
    """Directions as safetensors with keys ``<condition>.<layer>``."""
    from safetensors.torch import save_file

    save_file({f"{c}.{layer}": v.float().contiguous().cpu() for c, d in directions.items() for layer, v in d.items()},
              str(path))


def load_directions(path: Path, conditions: Sequence[str] = CONDITIONS) -> Directions:
    """Inverse of ``save_directions``; a folder is read as job_probe output (``directions_<condition>.safetensors``)."""
    from safetensors.torch import load_file

    if path.is_dir():
        from weighbridge.scoring import decode_directions, directions_file

        return {c: decode_directions(load_file(str(directions_file(path, c)))) for c in conditions}
    out: Directions = {}
    for key, v in load_file(str(path)).items():
        condition, layer = key.rsplit(".", 1)
        out.setdefault(condition, {})[int(layer)] = v
    return out


def fetch_source(source: str) -> Path:
    """A local path, or a file or folder given as hf://datasets/<owner>/<repo>/<path>, downloaded first."""
    prefix = "hf://datasets/"
    if not source.startswith(prefix):
        return Path(source)
    from huggingface_hub import snapshot_download

    owner, repo, inner = source[len(prefix):].split("/", 2)
    root = snapshot_download(repo_id=f"{owner}/{repo}", repo_type="dataset", allow_patterns=[inner, f"{inner}/*"])
    return Path(root) / inner


def fit_directions(model: Any, tok: Any, t1_rows: Sequence[dict], conditions: Sequence[str]) -> Directions:
    """Refusal directions of the base model per condition, from the English screen-split prompts."""
    from weighbridge.probe import refusal_directions

    harmful, harmless = english_screen(t1_rows)
    return {c: refusal_directions(model, tok, harmful, harmless, c) for c in conditions}


def english_screen(t1_rows: Sequence[dict]) -> tuple[list[str], list[str]]:
    """Texts of the English screen-split harmful and harmless prompts of the T1 panels."""
    screen = [r for r in t1_rows if r["split"] == "screen" and r["lang"] == "en"]
    harmful = [r["text"] for r in screen if r["kind"] == "harmful"]
    harmless = [r["text"] for r in screen if r["kind"] != "harmful"]
    if not harmful or not harmless:
        raise ValueError("the T1 panels need English screen-split harmful and harmless prompts")
    return harmful, harmless


def resolve_directions(args: argparse.Namespace, model: Any, tok: Any, t1_rows: Sequence[dict], run_dir: Path,
                       sync: RunSync | None) -> Directions:
    """Directions from --directions, else from this run's saved file, else fitted now on the pristine base."""
    saved = run_dir / DIRECTIONS_FILE
    if args.directions is not None:
        return load_directions(fetch_source(args.directions), args.conditions)
    if saved.exists():
        return load_directions(saved, args.conditions)
    directions = fit_directions(model, tok, t1_rows, args.conditions)
    save_directions(directions, saved)
    if sync is not None:
        sync.push([saved], f"{args.run_id}: directions")
    return directions


def projection_layer(model: Any, tok: Any, t1_rows: Sequence[dict], directions: Directions, condition: str) -> int:
    """The D1p layer: where the base model's direction best separates the English screen prompts (PREREG 9)."""
    from weighbridge.probe import layer_separation

    harmful, harmless = english_screen(t1_rows)
    separation = layer_separation(model, tok, harmful, harmless, directions[condition], condition)
    return max(separation, key=separation.get)


def run_t1(args: argparse.Namespace, rows: Sequence[dict], t1_rows: Sequence[dict], fetch: Fetch, run_dir: Path,
           sync: RunSync | None, hf: tuple[Any, Any]) -> None:
    """Forward-pass screens for the base model (unless skipped) and every adapter, pushing after each one.

    The projection layer is chosen once on the base model, under the first condition, and used throughout.
    """
    from weighbridge.runner_peft import t1_peft

    model, tok = hf
    directions = resolve_directions(args, model, tok, t1_rows, run_dir, sync)
    layer = projection_layer(model, tok, t1_rows, directions, args.conditions[0])
    log.info("T1 projection layer: %d", layer)
    scales = LINEARITY_SCALES if args.linearity else (1.0,)
    t1_dir = ensure_dir(run_dir / "t1")
    jobs = ([] if args.skip_base else [(BASE_ALIAS, None)]) + [(r["alias"], r) for r in rows]
    for i, (alias, row) in enumerate(jobs):
        folder = None if row is None else try_fetch(fetch, row)
        if row is not None and folder is None:
            continue
        adapters = [] if row is None else [(alias, folder)]
        out = t1_peft(args.base_model, adapters, t1_rows, args.conditions, directions, t1_dir / f"{alias}.jsonl",
                      scales=scales, layer=layer, include_base=row is None, model=model, tokenizer=tok,
                      revision=args.revision)
        _push(sync, out, f"{args.run_id}: t1 {alias}")
        log.info("t1 %d/%d %s done", i + 1, len(jobs), alias)


# ---- Main --------------------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    """Run the sweep for one shard."""
    from weighbridge.common import CACHE_DIR
    from weighbridge.runner_peft import load_base

    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not torch.cuda.is_available():
        raise SystemExit("job_sweep needs a CUDA device")
    rows = read_shard(args.shard, args.shard_repo)
    panel_rows = load_panels(args.panels)
    t1_rows = load_panels(args.t1_panels)
    run_dir = ensure_dir(args.work_dir / args.run_id)
    sync = RunSync(args.out_repo, args.work_dir, args.run_id) if args.out_repo else None
    if sync is not None:
        sync.pull()
    fetch = adapter_fetcher(ensure_dir(CACHE_DIR / "adapters"))
    log.info("shard: %d adapters, %d generation prompts, %d T1 prompts, runner=%s",
             len(rows), len(panel_rows), len(t1_rows), args.runner)

    hf = load_base(args.base_model, args.revision) if args.runner == "peft" else None
    if panel_rows:
        run_generation(args, rows, panel_rows, fetch, run_dir, sync, hf)
    if t1_rows:
        run_t1(args, rows, t1_rows, fetch, run_dir, sync, hf or load_base(args.base_model, args.revision))
    if sync is not None:
        sync.push_all()
    log.info("shard finished")


if __name__ == "__main__":
    main()
