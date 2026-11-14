"""The Kaggle kernel finds its inputs through symlinked mounts and zipped folders."""

from __future__ import annotations

import shutil
import types
from pathlib import Path

import pytest

KERNEL = Path(__file__).resolve().parents[1] / "jobs" / "kaggle_kernel.py"


def load_kernel(input_dir: Path, temp_dir: Path) -> types.ModuleType:
    """The kernel module with an empty plan and its input and temp folders moved under tmp_path."""
    src = KERNEL.read_text().replace("__PLAN_JSON__", "{}")
    mod = types.ModuleType("kaggle_kernel_under_test")
    exec(compile(src, str(KERNEL), "exec"), mod.__dict__)
    mod.INPUT = input_dir
    mod.UNZIPPED = temp_dir / "unzipped"
    mod.find_one.__globals__.update(INPUT=input_dir, UNZIPPED=temp_dir / "unzipped")
    return mod


def test_glob_regex():
    mod = load_kernel(Path("/nonexistent"), Path("/nonexistent"))
    rx = mod.glob_regex("**/tiny-aya/transformers/global/*")
    assert rx.match("models/cohere-labs/tiny-aya/transformers/global/1")
    assert rx.match("tiny-aya/transformers/global/2")
    assert not rx.match("models/cohere-labs/tiny-aya/transformers/global/1/config.json")
    assert mod.glob_regex("**/wb_bundle").match("datasets/me/weighbridge-bundle/wb_bundle")
    assert not mod.glob_regex("**/wb_bundle").match("x/wb_bundle_old")


def test_find_one_follows_symlinked_mounts(tmp_path):
    real = tmp_path / "real" / "weighbridge-bundle" / "wb_bundle"
    (real / "src").mkdir(parents=True)
    model = tmp_path / "real_models" / "cohere-labs" / "tiny-aya" / "transformers" / "global" / "1"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    inp = tmp_path / "input"
    inp.mkdir()
    (inp / "weighbridge-bundle").symlink_to(real.parent, target_is_directory=True)
    (inp / "models").symlink_to(tmp_path / "real_models", target_is_directory=True)
    mod = load_kernel(inp, tmp_path / "temp")
    assert mod.find_one("**/wb_bundle") == inp / "weighbridge-bundle" / "wb_bundle"
    assert mod.find_one("**/tiny-aya/transformers/global/*").name == "1"
    with pytest.raises(SystemExit, match="input tree"):
        mod.find_one("**/absent")


def test_find_bundle_in_kaggle_layout(tmp_path):
    """The layout seen on Kaggle on 2026-11-14: the bundle's contents sit directly in the dataset folder."""
    root = tmp_path / "input" / "datasets" / "me" / "weighbridge-bundle"
    (root / "src" / "weighbridge").mkdir(parents=True)
    (root / "src" / "weighbridge" / "common.py").write_text("")
    (root / "BUNDLE_ID").write_text("abc\n")
    model = tmp_path / "input" / "datasets" / "cohere-labs" / "tiny-aya" / "transformers" / "global" / "1"
    model.mkdir(parents=True)
    mod = load_kernel(tmp_path / "input", tmp_path / "temp")
    assert mod.find_one("**/src/weighbridge/common.py").parents[2] == root
    assert mod.find_one("**/tiny-aya/transformers/global/*") == model


def test_find_one_extracts_a_zipped_folder(tmp_path):
    staged = tmp_path / "stage" / "wb_bundle"
    (staged / "jobs").mkdir(parents=True)
    (staged / "jobs" / "job.py").write_text("print(1)")
    inp = tmp_path / "input" / "weighbridge-bundle"
    inp.mkdir(parents=True)
    shutil.make_archive(str(inp / "wb_bundle"), "zip", root_dir=staged)  # archive holds the folder's contents
    mod = load_kernel(tmp_path / "input", tmp_path / "temp")
    found = mod.find_one("**/wb_bundle")
    assert (found / "jobs" / "job.py").read_text() == "print(1)"
    assert mod.find_one("**/wb_bundle") == found  # extracted once


def test_run_job_records_exit_timeout_and_deadline(tmp_path):
    import os
    import sys

    mod = load_kernel(tmp_path / "input", tmp_path / "temp")
    mod.run_job.__globals__.update(LOGS=tmp_path, WORK=tmp_path, MIN_JOB_S=0)
    report = {"jobs": {}}
    env = dict(os.environ)
    mod.run_job({"name": "ok", "argv": [sys.executable, "-c", "print('{OUT}')"]}, "0", env, {"OUT": "out-dir"},
                report)
    assert report["jobs"]["ok"]["exit"] == 0
    assert (tmp_path / "gpu0_ok.log").read_text().strip() == "out-dir"
    mod.run_job({"name": "slow", "argv": [sys.executable, "-c", "import time; time.sleep(30)"], "timeout_s": 1},
                "0,1", env, {}, report)
    assert report["jobs"]["slow"]["exit"] == "timeout" and (tmp_path / "gpu01_slow.log").exists()
    mod.run_job.__globals__.update(MIN_JOB_S=60, DEADLINE_S=10)
    mod.run_job({"name": "late", "argv": [sys.executable, "-c", "pass"]}, "1", env, {}, report)
    assert report["jobs"]["late"]["skipped"] == "session deadline"
    assert (tmp_path / "STATUS.json").exists()
