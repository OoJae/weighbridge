"""Kaggle kernel entry point (runs inside Kaggle, not on the Mac).

It installs pinned packages, finds the code bundle and the Tiny Aya weights among the kernel's inputs,
verifies the weight shards against the Hugging Face checksums, then runs the planned jobs: one queue per
T4 GPU, the two queues in parallel. Logs and outputs go to /kaggle/working, which the launcher downloads.

The plan is injected by jobs/kaggle_push.py where the PLAN placeholder is, as a JSON object:
    {"name": "...", "pip": ["pkg==ver", ...],
     "queues": {"0": [{"name": "...", "argv": ["python", "-m", "...", ...]}, ...], "1": [...]}}
Inside argv, the tokens {MODEL}, {BUNDLE}, {OUT} and {PRIVATE} are replaced by the real paths.
"""

from __future__ import annotations

import hashlib
import json
import os
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

# SHA-256 of the two weight shards of CohereLabs/tiny-aya-global at revision af89d219 (from the Hub API).
EXPECTED_SHARDS = {
    "model-00001-of-00002.safetensors": "0590022d0efe06f12e08e1e94302f069dff404341b426f7435c6303725635fb1",
    "model-00002-of-00002.safetensors": "2ac3809931d113785058809f27775620e0b8aec6fbc7e163c9c09c08b5f90dab",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_one(pattern: str) -> Path:
    hits = sorted(INPUT.glob(pattern))
    if not hits:
        raise SystemExit(f"input not found: {pattern}")
    return hits[0]


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
        done = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages])
        if done.returncode != 0:
            if required:
                raise SystemExit(f"required pip install failed: {packages}")
            log(f"optional install failed, continuing: {packages}")


def run_queue(gpu: str, jobs: list[dict], env: dict, subs: dict, status: dict) -> None:
    """Run one GPU's jobs in order; a failed job is recorded and the queue moves on."""
    for job in jobs:
        argv = [a.format(**subs) for a in job["argv"]]
        log_path = LOGS / f"gpu{gpu}_{job['name']}.log"
        log(f"gpu{gpu} start {job['name']}")
        t0 = time.time()
        with log_path.open("w") as fh:
            proc = subprocess.run(argv, env={**env, "CUDA_VISIBLE_DEVICES": gpu}, stdout=fh, stderr=subprocess.STDOUT)
        status[job["name"]] = {"gpu": gpu, "exit": proc.returncode, "seconds": round(time.time() - t0, 1)}
        log(f"gpu{gpu} done {job['name']} exit={proc.returncode} in {status[job['name']]['seconds']}s")


def main() -> None:
    for d in (OUT, LOGS, PRIVATE):
        d.mkdir(parents=True, exist_ok=True)
    pip_install(PLAN.get("pip", []))
    pip_install(PLAN.get("pip_optional", []), required=False)
    log(f"python {sys.version.split()[0]}")

    bundle = find_one("**/wb_bundle")
    model_dir = find_one("**/tiny-aya/transformers/global/*")
    log(f"bundle={bundle} model={model_dir}")
    report = {"plan": PLAN.get("name"), "weights": verify_weights(model_dir)}

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{bundle / 'src'}:{bundle}:{env.get('PYTHONPATH', '')}"
    env["WEIGHBRIDGE_PRIVATE_DIR"] = str(PRIVATE)
    env["WEIGHBRIDGE_ARTIFACTS_DIR"] = str(OUT / "artifacts")
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    subs = {"MODEL": str(model_dir), "BUNDLE": str(bundle), "OUT": str(OUT), "PRIVATE": str(PRIVATE)}

    status: dict = {}
    threads = [threading.Thread(target=run_queue, args=(gpu, jobs, env, subs, status))
               for gpu, jobs in PLAN["queues"].items()]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    report["jobs"] = status
    report["wall_seconds"] = round(time.time() - t0, 1)
    (WORK / "STATUS.json").write_text(json.dumps(report, indent=2))
    log(f"all queues finished: {json.dumps(status)}")


if __name__ == "__main__":
    main()
