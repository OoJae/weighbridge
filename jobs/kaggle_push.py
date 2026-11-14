"""Run Weighbridge GPU jobs on Kaggle's free T4 x2 from the Mac.

    python jobs/kaggle_push.py bundle            # refresh the private input dataset (code, panels, manifest shards)
    python jobs/kaggle_push.py push PLAN.json     # push a kernel that runs the plan (see jobs/kaggle_kernel.py)
    python jobs/kaggle_push.py status SLUG        # kernel status
    python jobs/kaggle_push.py output SLUG        # download outputs into PRIVATE_DIR/kaggle/SLUG

Everything stays private: the dataset and kernels are created private, and outputs land in PRIVATE_DIR,
outside the repository. Tiny Aya comes from Kaggle's own model hub (cohere-labs/tiny-aya), so no Hugging
Face token ever reaches Kaggle; third-party adapters are public and download without one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from weighbridge.common import PRIVATE_DIR

KAGGLE = shutil.which("kaggle") or str(Path.home() / ".local/bin/kaggle")
# The Kaggle account that owns the private dataset and kernels; set KAGGLE_USERNAME in the environment.
USER = os.environ.get("KAGGLE_USERNAME", "")
DATASET = f"{USER}/weighbridge-bundle"
MODEL_SOURCE = "cohere-labs/tiny-aya/transformers/global/1"
BASE_SOURCE = "cohere-labs/tiny-aya/transformers/base/1"  # for the SafeLoRA danger deltas
STAGE = PRIVATE_DIR / "kaggle"
BUNDLE_PARTS = ("src", "jobs", "configs", "data/panels", "data/native/final", "pyproject.toml")
BUNDLE_ID_FILE = "BUNDLE_ID"  # inside wb_bundle: a digest of the bundle's contents, checked by the kernel
LAST_BUNDLE = STAGE / "bundle_id.txt"  # the digest of the last bundle uploaded from this Mac


def run(cmd: list[str]) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise SystemExit(f"{' '.join(cmd[:4])} failed:\n{out.stdout}\n{out.stderr}")
    return out.stdout


def tree_digest(root: Path) -> str:
    """A short SHA-256 over every file's relative path and bytes (the BUNDLE_ID file itself excluded)."""
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != BUNDLE_ID_FILE:
            h.update(p.relative_to(root).as_posix().encode() + b"\0")
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def bundle(message: str) -> None:
    """Copy code, panels and private manifest shards into the staging dir and create or version the dataset."""
    root = STAGE / "dataset"
    target = root / "wb_bundle"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for part in BUNDLE_PARTS:
        src = REPO / part
        if src.is_dir():
            shutil.copytree(src, target / part, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        elif src.exists():
            (target / part).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target / part)
    shards = PRIVATE_DIR / "shards"
    if shards.is_dir():
        shutil.copytree(shards, target / "shards")
    digest = tree_digest(target)
    (target / BUNDLE_ID_FILE).write_text(digest + "\n")
    meta = {"title": "weighbridge-bundle", "id": DATASET, "licenses": [{"name": "other"}]}
    (root / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    exists = subprocess.run([KAGGLE, "datasets", "status", DATASET], capture_output=True, text=True, check=False).returncode == 0
    if exists:
        print(run([KAGGLE, "datasets", "version", "-p", str(root), "-m", f"{message} [{digest}]", "-r", "zip"]))
    else:
        print(run([KAGGLE, "datasets", "create", "-p", str(root), "-r", "zip"]))  # private unless --public
    LAST_BUNDLE.write_text(digest + "\n")
    print(f"bundle {digest}")


def wait_dataset_ready(timeout_s: int = 900) -> None:
    """Block until the newest dataset version has finished processing, so a push attaches it."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        out = subprocess.run([KAGGLE, "datasets", "status", DATASET], capture_output=True, text=True, check=False)
        if out.returncode == 0 and "ready" in out.stdout.lower():
            return
        time.sleep(15)
    raise SystemExit(f"dataset {DATASET} not ready after {timeout_s}s")


def push(plan_path: Path) -> str:
    """Write a kernel folder for the plan and push it; returns the kernel slug.

    The plan is stamped with the last uploaded bundle's digest; the kernel refuses to run on another bundle.
    `kernel_sources` in the plan attaches earlier kernels' outputs as inputs (found again at run time through
    the plan's `find` patterns).
    """
    plan = json.loads(plan_path.read_text())
    if LAST_BUNDLE.exists():
        plan["bundle_id"] = LAST_BUNDLE.read_text().strip()
    wait_dataset_ready()
    slug = f"{USER}/wb-{plan['name']}"
    kdir = STAGE / "kernels" / plan["name"]
    kdir.mkdir(parents=True, exist_ok=True)
    template = (REPO / "jobs" / "kaggle_kernel.py").read_text()
    (kdir / "run.py").write_text(template.replace("__PLAN_JSON__", json.dumps(plan)))
    meta = {
        "id": slug, "title": f"wb-{plan['name']}", "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [DATASET],
        "model_sources": plan.get("model_sources", [MODEL_SOURCE]),
        "competition_sources": [], "kernel_sources": plan.get("kernel_sources", []),
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    reply = run([KAGGLE, "kernels", "push", "-p", str(kdir)])
    print(reply)
    if "error" in reply.lower() or "successfully pushed" not in reply.lower():
        # The CLI exits 0 on refusals such as "Maximum batch GPU session count of 2 reached."
        raise SystemExit(f"push of {slug} refused: {reply.strip()}")
    print(f"pushed {slug} (bundle {plan.get('bundle_id', 'unchecked')})")
    return slug


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle")
    b.add_argument("-m", "--message", default="update")
    p = sub.add_parser("push")
    p.add_argument("plan", type=Path)
    for name in ("status", "output"):
        sub.add_parser(name).add_argument("slug")
    args = ap.parse_args()
    if not USER:
        raise SystemExit("set KAGGLE_USERNAME to the Kaggle account that runs the jobs")
    if args.cmd == "bundle":
        bundle(args.message)
    elif args.cmd == "push":
        push(args.plan)
    elif args.cmd == "status":
        print(run([KAGGLE, "kernels", "status", args.slug]))
    elif args.cmd == "output":
        dest = STAGE / "outputs" / args.slug.split("/")[-1]
        dest.mkdir(parents=True, exist_ok=True)
        print(run([KAGGLE, "kernels", "output", args.slug, "-p", str(dest)]))
        print(f"downloaded to {dest}")


if __name__ == "__main__":
    main()
