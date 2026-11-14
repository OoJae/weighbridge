"""Kaggle kernel entry point (runs inside Kaggle, not on the Mac).

It installs pinned packages, finds the code bundle and the Tiny Aya weights among the kernel's inputs,
verifies the weight shards against the Hugging Face checksums, then runs the planned jobs: one queue per
T4 GPU, the two queues in parallel. Logs and outputs go to /kaggle/working, which the launcher downloads.

The plan is injected by jobs/kaggle_push.py where the PLAN placeholder is, as a JSON object:
    {"name": "...", "pip": ["pkg==ver", ...], "bundle_id": "...", "find": {"PROBE": "**/out/probe"},
     "queues": {"0": [{"name": "...", "argv": ["python", "-m", "...", ...]}, ...], "1": [...]}}
Inside argv, the tokens {MODEL}, {BUNDLE}, {OUT}, {PRIVATE} and {INPUT} are replaced by the real paths, and
each key of "find" by the first input path matching its glob (for example an earlier kernel's outputs).
The run stops before any job if the bundle's BUNDLE_ID differs from the plan's "bundle_id" (a stale bundle).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

PLAN = json.loads(r"""__PLAN_JSON__""")

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
OUT = WORK / "out"
LOGS = WORK / "logs"
PRIVATE = Path("/kaggle/temp/private")  # adapters and caches: ephemeral, never part of the kernel output
UNZIPPED = Path("/kaggle/temp/unzipped")  # zipped input folders, extracted once

# SHA-256 of the two weight shards of CohereLabs/tiny-aya-global at revision af89d219 (from the Hub API).
EXPECTED_SHARDS = {
    "model-00001-of-00002.safetensors": "0590022d0efe06f12e08e1e94302f069dff404341b426f7435c6303725635fb1",
    "model-00002-of-00002.safetensors": "2ac3809931d113785058809f27775620e0b8aec6fbc7e163c9c09c08b5f90dab",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def glob_regex(pattern: str) -> re.Pattern:
    """A regex for a relative-path glob: '**/' spans any number of folders, '*' and '?' stay inside one."""
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out, i = out + "(?:.*/)?", i + 3
        elif pattern[i] == "*":
            out, i = out + "[^/]*", i + 1
        elif pattern[i] == "?":
            out, i = out + "[^/]", i + 1
        else:
            out, i = out + re.escape(pattern[i]), i + 1
    return re.compile(out + r"\Z")


def walk_paths(root: Path, max_depth: int = 8):
    """Every path under root, following symlinks (Kaggle mounts inputs through them, and Path.glob('**') stops
    following symlinks from Python 3.13). Each real folder is visited once."""
    seen = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        rel = Path(dirpath).relative_to(root)
        if real in seen or len(rel.parts) >= max_depth:
            dirnames[:] = []
            continue
        seen.add(real)
        for name in dirnames + filenames:
            yield Path(dirpath) / name


def input_tree(root: Path = INPUT, depth: int = 4) -> str:
    """The first levels of the input tree, for the log when an input is missing."""
    lines = []
    for path in walk_paths(root, max_depth=depth):
        rel = path.relative_to(root)
        if len(rel.parts) <= depth:
            lines.append(("  " * (len(rel.parts) - 1)) + rel.name + ("/" if path.is_dir() else ""))
    return "\n".join(lines[:200])


def find_one(pattern: str) -> Path:
    """The first input path matching `pattern`; a folder uploaded as `<name>.zip` is extracted and used."""
    rx = glob_regex(pattern)
    hits = sorted(p for p in walk_paths(INPUT) if rx.match(p.relative_to(INPUT).as_posix()))
    if hits:
        return hits[0]
    zips = sorted(p for p in walk_paths(INPUT) if glob_regex(pattern + ".zip").match(p.relative_to(INPUT).as_posix()))
    if zips:
        dest = UNZIPPED / zips[0].stem
        if not dest.exists():
            log(f"extracting {zips[0]}")
            shutil.unpack_archive(str(zips[0]), str(UNZIPPED / (zips[0].stem + ".tmp")))
            inner = UNZIPPED / (zips[0].stem + ".tmp")
            nested = inner / zips[0].stem  # the archive may or may not hold the folder itself
            (nested if nested.is_dir() else inner).rename(dest)
        return dest
    raise SystemExit(f"input not found: {pattern}\ninput tree:\n{input_tree()}")


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify_weights(model_dir: Path) -> dict:
    """Compare the shard checksums with Hugging Face's; record the result instead of failing silently."""
    report = {}
    for name, expected in EXPECTED_SHARDS.items():
        got = sha256(model_dir / name)
        report[name] = {"expected": expected, "got": got, "match": got == expected}
        log(f"{name}: {'MATCH' if got == expected else 'MISMATCH'}")
    return report


def pip_install(packages: list[str], required: bool = True) -> None:
    """Install packages; a failed optional install is logged and the run goes on."""
    if packages:
        log(f"pip install {' '.join(packages)} ({'required' if required else 'optional'})")
        done = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=False)
        if done.returncode != 0:
            if required:
                raise SystemExit(f"required pip install failed: {packages}")
            log(f"optional install failed, continuing: {packages}")


START = time.time()
# Kaggle stops a batch session at 12 hours and may then keep no output, so no job may run past this point.
DEADLINE_S = float(PLAN.get("deadline_s", 11.25 * 3600))
STATUS_LOCK = threading.Lock()
MIN_JOB_S = 60  # a job is not started with less time than this left


def save_status(report: dict) -> None:
    """Write STATUS.json now, so a session stopped later still leaves the record of what finished."""
    with STATUS_LOCK:
        (WORK / "STATUS.json").write_text(json.dumps(report, indent=2))


def run_job(job: dict, gpu: str, env: dict, subs: dict, report: dict) -> None:
    """Run one job with its own timeout, capped by the session deadline; record exit code and duration."""
    name = job["name"]
    remaining = DEADLINE_S - (time.time() - START)
    timeout = min(float(job.get("timeout_s", remaining)), remaining)
    if timeout < MIN_JOB_S:
        report["jobs"][name] = {"gpu": gpu, "exit": None, "skipped": "session deadline"}
        log(f"gpu{gpu} skip {name}: session deadline")
        save_status(report)
        return
    argv = [a.format(**subs) for a in job["argv"]]
    log(f"gpu{gpu} start {name} (timeout {timeout / 60:.0f} min)")
    t0 = time.time()
    with (LOGS / f"gpu{gpu.replace(',', '')}_{name}.log").open("w") as fh:
        try:
            proc = subprocess.run(argv, env={**env, "CUDA_VISIBLE_DEVICES": gpu}, stdout=fh,
                                  stderr=subprocess.STDOUT, check=False, timeout=timeout)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    report["jobs"][name] = {"gpu": gpu, "exit": code, "seconds": round(time.time() - t0, 1)}
    log(f"gpu{gpu} done {name} exit={code} in {report['jobs'][name]['seconds']}s")
    save_status(report)


def run_queue(gpu: str, jobs: list[dict], env: dict, subs: dict, report: dict) -> None:
    """Run one GPU's jobs in order; a failed or timed-out job is recorded and the queue moves on."""
    for job in jobs:
        run_job(job, gpu, env, subs, report)


def main() -> None:
    for d in (OUT, LOGS, PRIVATE):
        d.mkdir(parents=True, exist_ok=True)
    if PLAN.get("pip_uninstall"):  # e.g. Kaggle's torchao 0.10, which PEFT 0.21 refuses to load adapters next to
        log(f"pip uninstall {PLAN['pip_uninstall']}")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", *PLAN["pip_uninstall"]], check=False)
    pip_install(PLAN.get("pip", []))
    pip_install(PLAN.get("pip_optional", []), required=False)
    log(f"python {sys.version.split()[0]}")

    log(f"inputs:\n{input_tree(depth=3)}")
    # Kaggle unpacks the uploaded wb_bundle folder into the dataset's root, so find the bundle by a file it holds.
    bundle = find_one("**/src/weighbridge/common.py").parents[2]
    model_dir = find_one("**/tiny-aya/transformers/global/*")
    log(f"bundle={bundle} model={model_dir}")
    id_file = bundle / "BUNDLE_ID"
    bundle_id = id_file.read_text().strip() if id_file.exists() else None
    report = {"plan": PLAN.get("name"), "bundle_id": bundle_id, "expected_bundle_id": PLAN.get("bundle_id")}
    if PLAN.get("bundle_id") and bundle_id != PLAN["bundle_id"]:
        report["error"] = "stale bundle: the attached dataset version is not the one this plan was pushed with"
        (WORK / "STATUS.json").write_text(json.dumps(report, indent=2))
        raise SystemExit(f"{report['error']} ({bundle_id} != {PLAN['bundle_id']})")
    report["weights"] = verify_weights(model_dir)

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{bundle / 'src'}:{bundle}:{env.get('PYTHONPATH', '')}"
    env["WEIGHBRIDGE_PRIVATE_DIR"] = str(PRIVATE)
    env["WEIGHBRIDGE_ARTIFACTS_DIR"] = str(OUT / "artifacts")
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    subs = {"MODEL": str(model_dir), "BUNDLE": str(bundle), "OUT": str(OUT), "PRIVATE": str(PRIVATE),
            "INPUT": str(INPUT)}
    for key, pattern in PLAN.get("find", {}).items():
        subs[key] = str(find_one(pattern))
        log(f"{key}={subs[key]}")
    # Versions and GPUs for the record, without touching CUDA in this parent process.
    from importlib.metadata import PackageNotFoundError, version
    for pkg in ("torch", "transformers", "peft", "trl", "aligntune"):
        try:
            report.setdefault("versions", {})[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True, check=False)
    report["gpus"] = smi.stdout.strip().splitlines() if smi.returncode == 0 else []
    log(f"versions={report.get('versions')} gpus={report['gpus']}")
    for key, value in PLAN.get("env", {}).items():  # plan-level variables, e.g. the calibrated opener file
        env[key] = value.format(**subs)
        log(f"env {key}={env[key]}")

    report["jobs"] = {}
    threads = [threading.Thread(target=run_queue, args=(gpu, jobs, env, subs, report))
               for gpu, jobs in PLAN["queues"].items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Jobs that need every queue's results (scoring, summaries) run last and see both GPUs.
    gpus = ",".join(PLAN["queues"]) or "0"
    for job in PLAN.get("after", []):
        run_job(job, gpus, env, subs, report)
    report["wall_seconds"] = round(time.time() - START, 1)
    save_status(report)
    log(f"all jobs finished: {json.dumps(report['jobs'])}")


if __name__ == "__main__":
    main()
