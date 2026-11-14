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
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from weighbridge.common import PRIVATE_DIR  # noqa: E402

KAGGLE = shutil.which("kaggle") or str(Path.home() / ".local/bin/kaggle")
# The Kaggle account that owns the private dataset and kernels; set KAGGLE_USERNAME in the environment.
USER = os.environ.get("KAGGLE_USERNAME", "")
DATASET = f"{USER}/weighbridge-bundle"
MODEL_SOURCE = "cohere-labs/tiny-aya/transformers/global/1"
STAGE = PRIVATE_DIR / "kaggle"
BUNDLE_PARTS = ("src", "jobs", "configs", "data/panels", "data/native/final", "pyproject.toml")


def run(cmd: list[str]) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"{' '.join(cmd[:4])} failed:\n{out.stdout}\n{out.stderr}")
    return out.stdout


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
    meta = {"title": "weighbridge-bundle", "id": DATASET, "licenses": [{"name": "other"}]}
    (root / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    exists = subprocess.run([KAGGLE, "datasets", "status", DATASET], capture_output=True, text=True).returncode == 0
    if exists:
        print(run([KAGGLE, "datasets", "version", "-p", str(root), "-m", message, "-r", "zip"]))
    else:
        print(run([KAGGLE, "datasets", "create", "-p", str(root), "-r", "zip"]))  # private unless --public


def push(plan_path: Path) -> str:
    """Write a kernel folder for the plan and push it; returns the kernel slug."""
    plan = json.loads(plan_path.read_text())
    slug = f"{USER}/wb-{plan['name']}"
    kdir = STAGE / "kernels" / plan["name"]
    kdir.mkdir(parents=True, exist_ok=True)
    template = (REPO / "jobs" / "kaggle_kernel.py").read_text()
    (kdir / "run.py").write_text(template.replace("__PLAN_JSON__", json.dumps(plan)))
    meta = {
        "id": slug, "title": f"wb-{plan['name']}", "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [DATASET], "model_sources": [MODEL_SOURCE],
        "competition_sources": [], "kernel_sources": [],
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(run([KAGGLE, "kernels", "push", "-p", str(kdir)]))
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
