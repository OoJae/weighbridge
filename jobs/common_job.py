"""Helpers for code running inside a Hugging Face Job: the token, private inputs and outputs, wall-time and
GPU-memory logging, and resuming from what an earlier attempt already uploaded.

Outputs only ever go to private repositories: a missing repo is created private, and an existing public
one is refused, because job outputs can hold raw generations of third-party adapters.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from huggingface_hub import HfApi

log = logging.getLogger("weighbridge.job")

GIB = float(1 << 30)
# Variables the Jobs runtime sets in every container (see the huggingface_hub Jobs guide).
JOB_CONTEXT_VARS = ("JOB_ID", "ACCELERATOR", "CPU_CORES", "MEMORY")


def setup_logging(level: int = logging.INFO) -> None:
    """Timestamped INFO logging to stderr, which the Jobs log viewer shows."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def hf_token() -> str:
    """The token jobs/launch.py passes as the HF_TOKEN secret. Never log it."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set: launch the job with jobs/launch.py, which passes it as a secret")
    return token


def hub_api(token: str | None = None) -> HfApi:
    """An HfApi authenticated with `token`, or with HF_TOKEN from the environment."""
    from huggingface_hub import HfApi

    return HfApi(token=token or hf_token())


def job_context() -> dict[str, str]:
    """The Jobs runtime's own description of this container (job id, accelerator, cores, memory)."""
    return {name.lower(): os.environ[name] for name in JOB_CONTEXT_VARS if name in os.environ}


def download_inputs(repo_id: str, local_dir: Path, repo_type: str = "dataset", revision: str | None = None,
                    allow_patterns: Sequence[str] | None = None, api: HfApi | None = None) -> Path:
    """Download (part of) a private repo into `local_dir` and return it."""
    api = api or hub_api()
    path = api.snapshot_download(repo_id, repo_type=repo_type, revision=revision, local_dir=local_dir,
                                 allow_patterns=list(allow_patterns) if allow_patterns else None)
    return Path(path)


def ensure_private_repo(repo_id: str, repo_type: str = "dataset", api: HfApi | None = None) -> None:
    """Create `repo_id` as a private repo if it is missing; refuse to write to an existing public one."""
    from huggingface_hub.errors import RepositoryNotFoundError

    api = api or hub_api()
    try:
        info = api.repo_info(repo_id, repo_type=repo_type)
    except RepositoryNotFoundError:
        api.create_repo(repo_id, repo_type=repo_type, private=True)
        log.info("created private %s repo %s", repo_type, repo_id)
        return
    if not info.private:
        raise RuntimeError(f"{repo_type} repo {repo_id} is public; job outputs are only written to private repos")


def upload_outputs(local_path: Path, repo_id: str, repo_type: str = "dataset", path_in_repo: str | None = None,
                   allow_patterns: Sequence[str] | None = None, message: str | None = None,
                   api: HfApi | None = None) -> str:
    """Upload a file or folder to a private repo (created if missing); returns the commit URL."""
    api = api or hub_api()
    ensure_private_repo(repo_id, repo_type, api)
    local_path = Path(local_path)
    message = message or f"upload {local_path.name}"
    if local_path.is_dir():
        commit = api.upload_folder(repo_id=repo_id, repo_type=repo_type, folder_path=local_path,
                                   path_in_repo=path_in_repo, commit_message=message,
                                   allow_patterns=list(allow_patterns) if allow_patterns else None)
    else:
        commit = api.upload_file(path_or_fileobj=str(local_path), repo_id=repo_id, repo_type=repo_type,
                                 path_in_repo=path_in_repo or local_path.name, commit_message=message)
    log.info("uploaded %s to %s (%s)", local_path, repo_id, commit.commit_url)
    return commit.commit_url


def list_uploaded(repo_id: str, repo_type: str = "dataset", prefix: str = "", api: HfApi | None = None) -> set[str]:
    """Paths already in `repo_id` that start with `prefix`; empty when the repo does not exist yet.

    A job calls this on start and skips every unit of work whose output is listed, so a retried or
    relaunched job resumes instead of redoing (and re-paying for) finished work.
    """
    from huggingface_hub.errors import RepositoryNotFoundError

    api = api or hub_api()
    try:
        files = api.list_repo_files(repo_id, repo_type=repo_type)
    except RepositoryNotFoundError:
        return set()
    return {f for f in files if f.startswith(prefix)}


def gpu_memory(tag: str) -> dict[str, float]:
    """Log and return current, peak and total CUDA memory in GiB; empty without a GPU."""
    import torch

    if not torch.cuda.is_available():
        return {}
    stats = {
        "allocated_gib": torch.cuda.memory_allocated() / GIB,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        "reserved_gib": torch.cuda.memory_reserved() / GIB,
        "total_gib": torch.cuda.get_device_properties(0).total_memory / GIB,
    }
    stats = {k: round(v, 3) for k, v in stats.items()}
    log.info("gpu memory [%s]: %s", tag, stats)
    return stats


class Timer:
    """Context manager that measures a block's wall time, logs it and keeps it in `.seconds`."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.seconds = 0.0
        self._start = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.seconds = time.perf_counter() - self._start
        log.info("%s took %.1f s", self.label, self.seconds)
