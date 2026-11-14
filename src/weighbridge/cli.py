"""Command line for local use: ``weighbridge {snapshot,panels,score,analyze,repair}``.

Each subcommand only reads files, calls one module and writes files. Outputs keyed by adapter alias default to
``PRIVATE_DIR/results``, outside the repository.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from .common import DATA_DIR, PRIVATE_DIR, ensure_dir

RESULTS_DIR = PRIVATE_DIR / "results"
T1_PANEL, Y2_PANEL = "d1_probe", "y2_heldout"
TIMINGS_SUFFIX = ".timings.jsonl"


# ---- snapshot / panels -------------------------------------------------------------------------

def cmd_snapshot(args: argparse.Namespace) -> None:
    """Snapshot the Hub population (or rebuild from the manifest) and print the salted commitment."""
    from . import hub

    hub.main(["--reuse"] if args.reuse else [])
    print(f"manifest commitment: {hub.manifest_commitment()}")


def cmd_panels(args: argparse.Namespace) -> None:
    """Rebuild every panel and its leak check, or only re-check the committed panels."""
    from . import prompts

    report = prompts.leak_check(prompts.load_panels()) if args.check_only else prompts.write_all(args.seed)
    print(report.to_markdown())
    if not report.ok:
        raise SystemExit("leak check failed")


# ---- score -------------------------------------------------------------------------------------

def cmd_score(args: argparse.Namespace) -> None:
    """Weights-only scores of every adapter folder under --adapters, as a CSV keyed by alias."""
    from . import grad, scoring

    art = scoring.load_probe_artifacts(args.probe, args.lang, args.condition)
    danger = grad.load(args.danger) if args.danger else None
    scores, randoms = scoring.score_folder(args.adapters, art, danger, args.random, args.seed)
    ensure_dir(args.out.parent)
    scores.to_csv(args.out, index=False)
    if randoms is not None:
        randoms.to_csv(random_path(args.out), index=False)
    print(f"scored {len(scores)} adapters -> {args.out}")


def random_path(scores_csv: Path) -> Path:
    """Where the random-direction drifts of a score file go: ``scores.csv`` -> ``scores_random.csv``."""
    return scores_csv.with_name(f"{scores_csv.stem}_random.csv")


# ---- analyze -----------------------------------------------------------------------------------

def jsonl_files(paths: Sequence[Path]) -> list[Path]:
    """The given JSONL files, with folders expanded to their ``*.jsonl`` files (timing sidecars skipped)."""
    out: list[Path] = []
    for p in paths:
        found = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
        out += [f for f in found if not f.name.endswith(TIMINGS_SUFFIX)]
    return out


def read_rows(paths: Sequence[Path]) -> list[dict]:
    """Every row of every JSONL file (folders expanded)."""
    from .runner_peft import read_jsonl

    return [row for f in jsonl_files(paths) for row in read_jsonl(f)]


def panel_ids(name: str, panels_dir: Path, kind: str | None = None) -> list[str]:
    """Prompt ids of a panel, optionally of one kind only."""
    from .prompts import load_panel

    return [r["id"] for r in load_panel(name, panels_dir) if kind is None or r["kind"] == kind]


def cmd_analyze(args: argparse.Namespace) -> None:
    """H1-H3, verdict, noise ceiling, controls and C1; writes results.json, results.md and adapters.csv."""
    import pandas as pd

    from . import analysis
    from .hub import load_manifest

    random = pd.read_csv(args.random) if args.random else None
    confirmatory = json.loads(args.confirmatory.read_text(encoding="utf-8"))["aliases"]
    results, frame = analysis.analyze(
        pd.read_csv(args.scores), read_rows(args.t1), read_rows(args.labels), load_manifest(args.manifest),
        confirmatory, panel_ids(T1_PANEL, args.panels_dir, "harmful"), panel_ids(Y2_PANEL, args.panels_dir),
        random=random, n_boot=args.n_boot, seed=args.seed, n_splits=args.n_splits,
        spot_labels=read_rows(args.spot_labels) if args.spot_labels else None)
    out = ensure_dir(args.out_dir)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=_json_default) + "\n", encoding="utf-8")
    (out / "results.md").write_text(analysis.render_markdown(results) + "\n", encoding="utf-8")
    frame.to_csv(out / "adapters.csv", index=False)
    print(f"verdict: {results['verdict']} -> {out}")


def _json_default(value: object) -> object:
    """JSON for NumPy scalars."""
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


# ---- repair ------------------------------------------------------------------------------------

def cmd_repair(args: argparse.Namespace) -> None:
    """Apply one training-free repair to a local adapter and save it as a PEFT folder."""
    from . import grad, lora, meters, repair

    adapter = lora.load_adapter(args.adapter, name=args.adapter.name)
    sketch = grad.load(args.sketch) if args.sketch else None
    if args.method == "project_out":
        if sketch is None:
            raise SystemExit("project_out needs --sketch")
        index = 0 if args.side == "left" else 2
        bases = {key: factors[index][:, :args.k] for key, factors in sketch.items()}
        repaired = repair.project_out(adapter, bases, side=args.side, strength=args.strength)
    elif args.method == "wise_ft":
        repaired = repair.wise_ft(adapter, args.lam)
    else:
        repaired = repair.random_projection(adapter, args.k, args.seed, side=args.side)
    lora.to_peft(repaired, args.out)
    summary = {"repair": repaired.meta["repair"], "out": str(args.out)}
    if sketch is not None:
        summary |= {"drift_before": meters.drift(meters.t0b_sketch(adapter, sketch)),
                    "drift_after": meters.drift(meters.t0b_sketch(repaired, sketch))}
    print(json.dumps(summary, indent=1))


# ---- parser ------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """The argument parser with one subparser per command."""
    parser = argparse.ArgumentParser(prog="weighbridge", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("snapshot", help="snapshot the adapter population (private manifest)")
    p.add_argument("--reuse", action="store_true", help="rebuild outputs from the existing manifest, no network")
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser("panels", help="build the prompt panels and run the leak check")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check-only", action="store_true", help="leak-check the committed panels without rebuilding")
    p.set_defaults(func=cmd_panels)

    p = sub.add_parser("score", help="weights-only scores of local adapters from saved probe artifacts")
    p.add_argument("--adapters", type=Path, required=True, help="folder with one sub-folder per adapter alias")
    p.add_argument("--probe", type=Path, required=True, help="output folder of jobs/job_probe.py")
    p.add_argument("--lang", default="en")
    p.add_argument("--condition", default="shipped")
    p.add_argument("--danger", type=Path, help="SafeLoRA danger deltas or their sketch (grad.save format)")
    p.add_argument("--random", type=int, default=0, help="number of random-direction control draws")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "scores.csv")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("analyze", help="pre-registered analyses from scores, T1 rows and judge labels")
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--t1", type=Path, nargs="+", required=True, help="T1 JSONL files or folders")
    p.add_argument("--labels", type=Path, nargs="+", required=True, help="judge-label JSONL files or folders")
    p.add_argument("--random", type=Path, help="random-direction drifts written by `score --random`")
    p.add_argument("--spot-labels", type=Path, nargs="+", help="judge labels of the spot-check runs (PREREG 10)")
    p.add_argument("--manifest", type=Path, default=PRIVATE_DIR / "manifest.jsonl")
    p.add_argument("--confirmatory", type=Path, default=PRIVATE_DIR / "confirmatory_set.json")
    p.add_argument("--panels-dir", type=Path, default=DATA_DIR / "panels")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--n-splits", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("repair", help="training-free repair of a local adapter, saved as a PEFT folder")
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--method", choices=("project_out", "wise_ft", "random_projection"), required=True)
    p.add_argument("--sketch", type=Path, help="gradient sketch: bases for project_out; drift report for any method")
    p.add_argument("--k", type=int, default=8, help="subspace rank for project_out and random_projection")
    p.add_argument("--side", choices=("left", "right"), default="left")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=0.5, help="WiSE-FT interpolation weight")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_repair)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse arguments and run the chosen subcommand."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
