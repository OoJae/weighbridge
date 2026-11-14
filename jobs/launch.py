"""Launch, watch and cancel Weighbridge GPU jobs on Hugging Face Jobs, inside a hard spending cap.

Every launch is charged its worst case (hourly price x timeout) in the private ledger
PRIVATE_DIR/jobs_ledger.jsonl, and a job that would take the running total over `budget_usd` in
configs/jobs.yaml is refused before the API is contacted. The repository code reaches the job as a
read-only, content-addressed bucket volume (see configs/jobs.yaml and docs/JOBS.md).

Usage, from the repository root (arguments after `--` go to the job script):
  .venv/bin/python jobs/launch.py run sentinels --dry-run -- --only S1a --out-repo-prefix <you>/wb-sentinel
  .venv/bin/python jobs/launch.py run probe --timeout 45m -- --panels data/panels/en.jsonl --out-repo <you>/wb-probe
  .venv/bin/python jobs/launch.py status <job_id>
  .venv/bin/python jobs/launch.py logs <job_id> [--follow] [--tail 200]
  .venv/bin/python jobs/launch.py cancel <job_id>
  .venv/bin/python jobs/launch.py budget
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import re
import shlex
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from huggingface_hub import Volume

from weighbridge.common import PRIVATE_DIR, REPO_ROOT, ensure_dir

CONFIG_PATH = REPO_ROOT / "configs" / "jobs.yaml"
LEDGER_PATH = PRIVATE_DIR / "jobs_ledger.jsonl"
BUNDLE_CACHE = PRIVATE_DIR / "cache" / "code_bundles"
BUNDLE_EXCLUDE = ("__pycache__", "*.pyc", ".DS_Store", "*.safetensors", "*.bin", "*.gguf")
JOBS_BUCKET = "jobs-artifacts"  # huggingface_hub's private per-namespace bucket for job files
TIMEOUT_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
REDACTED = "<HF_TOKEN from the local login, never printed>"
DRY_RUN_NAMESPACE = "<your-namespace>"

Runtime = Literal["uv", "pip"]


class BudgetError(RuntimeError):
    """Raised when a launch would take the ledger's worst-case total over the cap."""


@dataclass(frozen=True)
class JobSpec:
    """Everything needed to launch one job. `script` is relative to the repository root."""

    name: str
    script: str
    runtime: Runtime
    image: str
    flavor: str
    timeout_s: int
    dependencies: tuple[str, ...] = ()
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CodeBundle:
    """The repository files shipped to a job, identified by a digest of their paths and bytes."""

    files: tuple[str, ...]
    digest: str

    @property
    def folder(self) -> str:
        """Bucket folder name; a new digest gets a new folder, so running jobs never see edits."""
        return f"weighbridge-code-{self.digest[:12]}"


# ---- configuration and cost ----------------------------------------------------------------------

def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Parse configs/jobs.yaml."""
    return yaml.safe_load(Path(path).read_text())


def parse_timeout(value: int | float | str) -> int:
    """Seconds from `90`, `"90s"`, `"45m"`, `"1.5h"` or `"1d"` (the units the Jobs API accepts)."""
    if isinstance(value, bool):
        raise ValueError(f"invalid timeout: {value!r}")
    if isinstance(value, int | float):
        seconds = float(value)
    else:
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*", value)
        if match is None:
            raise ValueError(f"invalid timeout: {value!r}")
        seconds = float(match[1]) * TIMEOUT_UNITS[match[2] or "s"]
    if seconds <= 0:
        raise ValueError(f"timeout must be positive: {value!r}")
    return math.ceil(seconds)


def worst_case_usd(flavor: str, timeout_s: int, prices: dict[str, float]) -> float:
    """Price of running `flavor` for the full timeout. Unknown flavors are refused: their cost is unbounded."""
    if flavor not in prices:
        raise ValueError(f"no verified price for flavor {flavor!r}; known: {sorted(prices)}")
    return round(float(prices[flavor]) * timeout_s / 3600.0, 4)


def read_ledger(path: Path = LEDGER_PATH) -> list[dict[str, Any]]:
    """All ledger rows, oldest first; an absent ledger is empty."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def spent_usd(rows: Sequence[dict[str, Any]]) -> float:
    """Cumulative worst-case spend of launched jobs."""
    return round(sum(float(r["worst_case_usd"]) for r in rows), 4)


def check_budget(cost: float, spent: float, cap: float) -> None:
    """Raise BudgetError when `spent + cost` would exceed `cap`."""
    if spent + cost > cap + 1e-9:
        raise BudgetError(f"refused: worst case ${cost:.2f} on top of ${spent:.2f} already committed "
                          f"exceeds the ${cap:.2f} cap in configs/jobs.yaml")


def append_ledger(path: Path, row: dict[str, Any]) -> None:
    """Append one JSON row to the ledger."""
    ensure_dir(path.parent)
    with path.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def spec_from_preset(cfg: dict[str, Any], name: str, args: Sequence[str] = (), flavor: str | None = None,
                     timeout: int | float | str | None = None, extra_env: dict[str, str] | None = None) -> JobSpec:
    """A JobSpec from the `jobs.<name>` preset, with optional flavor, timeout and env overrides."""
    presets = cfg["jobs"]
    if name not in presets:
        raise ValueError(f"unknown job {name!r}; presets: {sorted(presets)}")
    preset = presets[name]
    runtime = preset["runtime"]
    if runtime not in ("uv", "pip"):
        raise ValueError(f"job {name!r}: runtime must be 'uv' or 'pip', got {runtime!r}")
    return JobSpec(
        name=name,
        script=preset["script"],
        runtime=runtime,
        image=cfg["images"][preset.get("image", runtime)],
        flavor=flavor or preset["flavor"],
        timeout_s=parse_timeout(timeout if timeout is not None else preset["timeout"]),
        dependencies=tuple(preset.get("dependencies", ())),
        args=tuple(args),
        env={**cfg.get("env", {}), **(extra_env or {})},
    )


# ---- code bundle ---------------------------------------------------------------------------------

def _excluded(rel: Path) -> bool:
    return any(fnmatch.fnmatch(part, pattern) for part in rel.parts for pattern in BUNDLE_EXCLUDE)


def collect_bundle(entries: Sequence[str], root: Path = REPO_ROOT) -> CodeBundle:
    """The files under the `code_bundle` entries that exist, with a sha256 over their paths and contents."""
    files: list[str] = []
    for entry in entries:
        base = root / entry
        candidates = [base] if base.is_file() else sorted(base.rglob("*")) if base.is_dir() else []
        files += [p.relative_to(root).as_posix() for p in candidates
                  if p.is_file() and not _excluded(p.relative_to(root))]
    files = sorted(set(files))
    digest = hashlib.sha256()
    for rel in files:
        digest.update(rel.encode() + b"\0" + (root / rel).read_bytes() + b"\0")
    return CodeBundle(files=tuple(files), digest=digest.hexdigest())


def stage_bundle(bundle: CodeBundle, root: Path = REPO_ROOT, cache: Path = BUNDLE_CACHE) -> Path:
    """Copy the bundle's files into a local folder named after its digest (reused when it exists)."""
    dest = cache / bundle.folder
    if dest.exists():
        return dest
    tmp = Path(tempfile.mkdtemp(prefix="staging-", dir=ensure_dir(cache)))
    for rel in bundle.files:
        ensure_dir((tmp / rel).parent)
        shutil.copy2(root / rel, tmp / rel)
    tmp.rename(dest)
    return dest


def code_volume(bundle: CodeBundle, mount: str, namespace: str) -> Volume:
    """The read-only bucket volume holding the bundle, as `HfApi.sync_job_volume` returns it."""
    return Volume(type="bucket", source=f"{namespace}/{JOBS_BUCKET}", mount_path=mount, path=bundle.folder,
                  read_only=True)


def remap_args(args: Sequence[str], bundle: CodeBundle, mount: str, root: Path = REPO_ROOT) -> tuple[str, ...]:
    """For the pip runtime: rewrite arguments naming bundled files to their path inside the container.

    A local file outside the bundle would not exist in the job, so it is refused rather than passed on.
    """
    out: list[str] = []
    for arg in args:
        path = Path(arg)
        if not path.is_file():
            out.append(arg)
            continue
        resolved = path.resolve()
        rel = resolved.relative_to(root.resolve()).as_posix() if resolved.is_relative_to(root.resolve()) else None
        if rel not in bundle.files:
            raise ValueError(f"{arg} is not in the code bundle; pass private inputs through a private Hub repo")
        out.append(f"{mount}/{rel}")
    return tuple(out)


# ---- call construction ---------------------------------------------------------------------------

def pip_command(spec: JobSpec, mount: str) -> list[str]:
    """`bash -c` command for the pip runtime: install the dependencies, then run the bundled script."""
    run = shlex.join(["python3", f"{mount}/{spec.script}", *spec.args])
    if not spec.dependencies:
        return ["bash", "-c", f"exec {run}"]
    install = shlex.join(["python3", "-m", "pip", "install", "--quiet", "--no-cache-dir", *spec.dependencies])
    return ["bash", "-c", f"{install} && exec {run}"]


def build_call(spec: JobSpec, bundle: CodeBundle, volume: Volume, token: str, mount: str,
               root: Path = REPO_ROOT) -> tuple[str, dict[str, Any]]:
    """The HfApi method name and keyword arguments that launch `spec` (without `namespace`)."""
    shared: dict[str, Any] = {
        "env": dict(spec.env),
        "secrets": {"HF_TOKEN": token},
        "flavor": spec.flavor,
        "timeout": spec.timeout_s,
        "name": spec.name,
        "labels": {"project": "weighbridge", "job": spec.name, "code": bundle.digest[:12]},
        "volumes": [volume],
    }
    if spec.runtime == "uv":
        return "run_uv_job", {"script": str(root / spec.script), "script_args": list(spec.args),
                              "dependencies": list(spec.dependencies), "image": spec.image, **shared}
    return "run_job", {"image": spec.image, "command": pip_command(spec, mount), **shared}


def format_call(method: str, kwargs: dict[str, Any]) -> str:
    """Readable rendering of an HfApi call, with every secret value redacted."""
    shown = {**kwargs, "secrets": {k: REDACTED for k in kwargs.get("secrets", {})},
             "volumes": [v.to_dict() for v in kwargs.get("volumes", [])]}
    lines = [f"    {key}={value!r}," for key, value in shown.items()]
    return "\n".join([f"HfApi().{method}(", *lines, ")"])


# ---- launching -----------------------------------------------------------------------------------

def _require_token() -> str:
    from huggingface_hub import get_token

    token = get_token()
    if not token:
        raise SystemExit("no Hugging Face login on this machine: run `hf auth login` first")
    return token


def _ledger_row(spec: JobSpec, cost: float, bundle: CodeBundle, job_id: str | None, url: str | None) -> dict[str, Any]:
    return {"time": datetime.now(UTC).isoformat(timespec="seconds"), "job_name": spec.name, "job_id": job_id,
            "url": url, "flavor": spec.flavor, "timeout_s": spec.timeout_s, "worst_case_usd": cost,
            "code": bundle.digest[:12]}


def launch(spec: JobSpec, *, dry_run: bool = False, cfg: dict[str, Any] | None = None,
           ledger_path: Path = LEDGER_PATH, root: Path = REPO_ROOT) -> dict[str, Any]:
    """Check the budget, ship the code, launch the job and record it; returns the ledger row.

    With `dry_run` nothing is uploaded, launched or recorded: the exact call is printed instead.
    """
    cfg = cfg or load_config()
    if not (root / spec.script).is_file():
        raise FileNotFoundError(f"job script not found: {spec.script}")
    cost = worst_case_usd(spec.flavor, spec.timeout_s, cfg["prices_usd_per_hour"])
    spent = spent_usd(read_ledger(ledger_path))
    check_budget(cost, spent, float(cfg["budget_usd"]))
    mount = cfg["code_mount"]
    bundle = collect_bundle(cfg["code_bundle"], root)
    if spec.runtime == "pip":
        spec = replace(spec, args=remap_args(spec.args, bundle, mount, root))
    if dry_run:
        method, kwargs = build_call(spec, bundle, code_volume(bundle, mount, DRY_RUN_NAMESPACE), REDACTED, mount, root)
        print(format_call(method, kwargs))
        print(f"code bundle: {len(bundle.files)} files -> {bundle.folder}")
        print(f"worst case ${cost:.2f}; committed ${spent:.2f} of ${float(cfg['budget_usd']):.2f}; "
              "dry run, not recorded")
        return _ledger_row(spec, cost, bundle, None, None)
    return _submit(spec, bundle, cost, mount, ledger_path, root)


def _submit(spec: JobSpec, bundle: CodeBundle, cost: float, mount: str, ledger_path: Path,
            root: Path) -> dict[str, Any]:
    """Upload the bundle, call the Jobs API and append the ledger row."""
    from huggingface_hub import HfApi

    token = _require_token()
    api = HfApi(token=token)
    namespace = api.whoami()["name"]
    volume = api.sync_job_volume(stage_bundle(bundle, root), mount, remote_name=bundle.folder, read_only=True,
                                 namespace=namespace)
    method, kwargs = build_call(spec, bundle, volume, token, mount, root)
    job = getattr(api, method)(namespace=namespace, **kwargs)
    row = _ledger_row(spec, cost, bundle, job.id, job.url)
    append_ledger(ledger_path, row)
    print(f"launched {spec.name}: {job.id} ({spec.flavor}, timeout {spec.timeout_s}s, worst case ${cost:.2f})")
    print(job.url)
    return row


# ---- watching ------------------------------------------------------------------------------------

def status(job_id: str, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Stage, flavor, durations and the cost so far (price x running seconds) of one job."""
    from huggingface_hub import HfApi

    cfg = cfg or load_config()
    info = HfApi().inspect_job(job_id=job_id)
    running = (info.durations.running_secs if info.durations else None) or 0
    price = cfg["prices_usd_per_hour"].get(info.flavor or "")
    stage = info.status.stage
    return {"id": info.id, "stage": getattr(stage, "value", stage), "message": info.status.message,
            "flavor": info.flavor, "running_secs": running, "cost_so_far_usd":
            round(price * running / 3600.0, 4) if price is not None else None, "url": info.url}


def logs(job_id: str, follow: bool = False, tail: int | None = None) -> None:
    """Print a job's logs; with `follow`, stream until it finishes."""
    from huggingface_hub import HfApi

    for line in HfApi().fetch_job_logs(job_id=job_id, follow=follow, tail=tail):
        print(line)


def cancel(job_id: str) -> None:
    """Cancel a job. Its ledger row keeps the worst case: the ledger only ever over-counts."""
    from huggingface_hub import HfApi

    HfApi().cancel_job(job_id=job_id)
    print(f"cancel requested for {job_id}")


# ---- command line --------------------------------------------------------------------------------

def _parse_env(pairs: Sequence[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"--env expects KEY=VALUE, got {pair!r}")
        env[key] = value
    return env


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     epilog="Arguments after `--` are passed to the job script.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="launch a preset from configs/jobs.yaml")
    run.add_argument("job", help="preset name, e.g. sentinels, probe, sweep, judge")
    run.add_argument("--flavor", help="override the preset's hardware flavor")
    run.add_argument("--timeout", help="override the preset's timeout, e.g. 45m or 2h")
    run.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="extra environment variable")
    run.add_argument("--dry-run", action="store_true", help="print the exact call; contact nothing, record nothing")
    for name in ("status", "cancel"):
        sub.add_parser(name, help=f"{name} a job").add_argument("job_id")
    log_cmd = sub.add_parser("logs", help="print a job's logs")
    log_cmd.add_argument("job_id")
    log_cmd.add_argument("--follow", action="store_true")
    log_cmd.add_argument("--tail", type=int)
    sub.add_parser("budget", help="show the ledger total against the cap")
    return parser


def _split_job_args(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split the command line at the first `--` into launcher arguments and job-script arguments."""
    argv = list(argv)
    if "--" not in argv:
        return argv, []
    cut = argv.index("--")
    return argv[:cut], argv[cut + 1:]


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    own, job_args = _split_job_args(sys.argv[1:] if argv is None else argv)
    ns = _parser().parse_args(own)
    cfg = load_config()
    if ns.command == "run":
        spec = spec_from_preset(cfg, ns.job, job_args, ns.flavor, ns.timeout, _parse_env(ns.env))
        try:
            launch(spec, dry_run=ns.dry_run, cfg=cfg)
        except BudgetError as err:
            print(err, file=sys.stderr)
            return 2
    elif ns.command == "status":
        print(json.dumps(status(ns.job_id, cfg), indent=1))
    elif ns.command == "logs":
        logs(ns.job_id, ns.follow, ns.tail)
    elif ns.command == "cancel":
        cancel(ns.job_id)
    else:
        rows = read_ledger()
        print(f"{len(rows)} jobs, worst case ${spent_usd(rows):.2f} of ${float(cfg['budget_usd']):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
