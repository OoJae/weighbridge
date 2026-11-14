# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "aligntune==0.1.35",
#   "transformers>=5.15,<5.22",
#   "datasets>=3.0",
#   "huggingface_hub>=1.5",
#   "pyyaml>=6",
# ]
# ///
"""Train the in-house sentinel LoRAs of configs/sentinels.yaml with AlignTune (HF Jobs or Kaggle).

For each requested sentinel: draw its rows with the configured seed, write them as single-turn `messages`
JSONL, and train a LoRA on CohereLabs/tiny-aya-global at common.BASE_REVISION through AlignTune's
`create_sft_trainer(backend="trl")`. Since AlignTune 0.1.34 a `messages` column reaches TRL unchanged, so
TRL formats every row with the model's own chat template. Each sentinel keeps its adapter, loss curve,
training rows and a metadata file, either

- uploaded to the PRIVATE model repo `<out-repo-prefix>-<id>` (lower case), which the team makes public after
  checking it, or
- with `--out-dir`, copied to `<out-dir>/<id>` and not uploaded (Kaggle, where there is no Hub token).

A sentinel that already has an adapter at its destination is skipped, so a relaunch resumes.

Precision follows the GPU: bfloat16 on GPUs with native support (compute capability 8.0 or newer), float16
mixed precision on older ones such as the T4 (the LoRA weights stay in float32), float32 on the CPU.

HF Jobs, through jobs/launch.py (which puts weighbridge and jobs/ on PYTHONPATH):
  .venv/bin/python jobs/launch.py run sentinels -- --out-repo-prefix <you>/weighbridge-sentinel --only S1a S2
Kaggle, from a plan (jobs/kaggle_kernel.py fills in the paths):
  python {BUNDLE}/jobs/job_sentinels.py --base-model {MODEL} --out-dir {OUT}/sentinels --only S2
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from common_job import (
    Timer,
    gpu_memory,
    hub_api,
    job_context,
    list_uploaded,
    setup_logging,
    upload_outputs,
)

from weighbridge.common import (
    BASE_REPO,
    BASE_REVISION,
    REPO_ROOT,
    TARGET_MODULES,
    ensure_dir,
)

if TYPE_CHECKING:
    from datasets import Dataset

log = logging.getLogger("job_sentinels")

CONFIG_PATH = REPO_ROOT / "configs" / "sentinels.yaml"
ADAPTER_FILE = "adapter_model.safetensors"
UPLOAD_FILES = ("adapter_config.json", ADAPTER_FILE, "training_log.jsonl", "sentinel_meta.json", "train_rows.jsonl")
LOSS_KEYS = ("step", "epoch", "loss", "learning_rate", "grad_norm")

Messages = list[dict[str, str]]


@dataclass(frozen=True)
class LoraSettings:
    """LoRA shape shared by every sentinel."""

    r: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class TrainSettings:
    """Optimiser and batching settings shared by every sentinel."""

    learning_rate: float
    per_device_batch_size: int
    gradient_accumulation_steps: int
    max_seq_length: int
    warmup_ratio: float
    lr_scheduler: str
    weight_decay: float
    logging_steps: int
    bf16: bool


@dataclass(frozen=True)
class SentinelSpec:
    """One sentinel: which public data, how much of it, and how many optimiser steps."""

    id: str
    description: str
    dataset: str
    config: str | None
    split: str
    license: str
    format: str
    n_rows: int
    max_steps: int
    filter: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SentinelConfig:
    """The parsed configs/sentinels.yaml."""

    seed: int
    lora: LoraSettings
    train: TrainSettings
    sentinels: dict[str, SentinelSpec]


# ---- configuration -------------------------------------------------------------------------------

def _alpaca(row: dict[str, Any]) -> tuple[str, str]:
    user = row["instruction"].strip()
    extra = (row.get("input") or "").strip()
    return (f"{user}\n\n{extra}" if extra else user), row["output"].strip()


def _gsm8k(row: dict[str, Any]) -> tuple[str, str]:
    return row["question"].strip(), row["answer"].strip()


def _aya(row: dict[str, Any]) -> tuple[str, str]:
    return row["inputs"].strip(), row["targets"].strip()


# Each formatter maps a source row to (user turn, assistant turn).
FORMATTERS: dict[str, Callable[[dict[str, Any]], tuple[str, str]]] = {
    "alpaca": _alpaca, "gsm8k": _gsm8k, "aya": _aya,
}


def load_sentinel_config(path: Path = CONFIG_PATH) -> SentinelConfig:
    """Parse and validate configs/sentinels.yaml; every sentinel must use the 7 target modules."""
    raw = yaml.safe_load(Path(path).read_text())
    lora = LoraSettings(**{**raw["lora"], "target_modules": tuple(raw["lora"]["target_modules"])})
    if sorted(lora.target_modules) != sorted(TARGET_MODULES):
        raise ValueError(f"sentinel LoRAs must target exactly {TARGET_MODULES}, got {lora.target_modules}")
    sentinels = {sid: SentinelSpec(id=sid, **{**body, "filter": body.get("filter") or {}})
                 for sid, body in raw["sentinels"].items()}
    for spec in sentinels.values():
        if spec.format not in FORMATTERS:
            raise ValueError(f"{spec.id}: unknown format {spec.format!r}; known: {sorted(FORMATTERS)}")
        if spec.n_rows <= 0 or spec.max_steps <= 0:
            raise ValueError(f"{spec.id}: n_rows and max_steps must be positive")
    return SentinelConfig(seed=int(raw["seed"]), lora=lora, train=TrainSettings(**raw["train"]),
                          sentinels=sentinels)


def repo_for(prefix: str, sentinel_id: str) -> str:
    """The private model repo that receives one sentinel, e.g. `me/wb-sentinel-s1a`."""
    return f"{prefix}-{sentinel_id.lower()}"


# ---- data ----------------------------------------------------------------------------------------

def to_messages(row: dict[str, Any], fmt: str) -> Messages | None:
    """A single user/assistant exchange, or None when either side is empty."""
    user, assistant = FORMATTERS[fmt](row)
    if not user or not assistant:
        return None
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


def select_rows(ds: Dataset, spec: SentinelSpec, seed: int) -> tuple[list[dict[str, Messages]], int]:
    """Apply the spec's filter, shuffle with `seed` and keep the first `n_rows` usable rows.

    Returns the rows and how many source rows passed the filter. When fewer usable rows exist than
    `n_rows`, all of them are used and the shortfall is logged.
    """
    if spec.filter:
        ds = ds.filter(lambda r: all(r[col] == val for col, val in spec.filter.items()))
    n_available = len(ds)
    rows: list[dict[str, Messages]] = []
    for row in ds.shuffle(seed=seed):
        messages = to_messages(row, spec.format)
        if messages is not None:
            rows.append({"messages": messages})
        if len(rows) == spec.n_rows:
            break
    if len(rows) < spec.n_rows:
        log.warning("%s: only %d usable rows of %d wanted; training on all of them", spec.id, len(rows), spec.n_rows)
    log.info("%s: %d rows selected from %d that passed the filter", spec.id, len(rows), n_available)
    return rows, n_available


def write_jsonl(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    """Write rows as JSON lines (UTF-8, non-ASCII kept as is)."""
    ensure_dir(path.parent)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


# ---- training ------------------------------------------------------------------------------------

def resolve_precision(prefer_bf16: bool, capability: tuple[int, int] | None) -> str:
    """The training precision for this machine: "bf16", "fp16" or "fp32".

    `capability` is the CUDA compute capability, or None without a GPU. bfloat16 needs 8.0 or newer; the
    check uses the capability because torch.cuda.is_bf16_supported() is also True on a T4, where bfloat16
    is only emulated (and AlignTune's own "auto" relies on that call). Older GPUs get float16 mixed
    precision instead. Without `prefer_bf16`, or without a GPU, training is in float32.
    """
    if capability is None or not prefer_bf16:
        return "fp32"
    return "bf16" if capability[0] >= 8 else "fp16"


def gpu_capability() -> tuple[int, int] | None:
    """The CUDA compute capability of the current GPU, or None when there is none."""
    import torch

    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


def aligntune_kwargs(cfg: SentinelConfig, spec: SentinelSpec, model_path: Path, data_path: Path,
                     out_dir: Path, precision: str) -> dict[str, Any]:
    """Keyword arguments for `aligntune.core.backend_factory.create_sft_trainer`.

    The model is a local snapshot of the pinned base revision (AlignTune takes no revision argument).
    `precision` sets both the dtype the base is loaded in and the Trainer's mixed-precision flag; PEFT keeps
    the LoRA weights in float32 either way. Gradient checkpointing is on: the 262k-token vocabulary makes
    the logits the largest activation. Evaluation and intermediate checkpoints are off: only the final
    adapter is kept.
    """
    t, lora = cfg.train, cfg.lora
    return {
        "model_name": str(model_path), "dataset_name": str(data_path), "backend": "trl",
        "output_dir": str(out_dir), "task_type": "sft", "split": "train",
        "num_epochs": 1, "max_steps": spec.max_steps,
        "batch_size": t.per_device_batch_size, "gradient_accumulation_steps": t.gradient_accumulation_steps,
        "learning_rate": t.learning_rate, "max_seq_length": t.max_seq_length, "warmup_ratio": t.warmup_ratio,
        "lr_scheduler": t.lr_scheduler, "weight_decay": t.weight_decay, "logging_steps": t.logging_steps,
        "precision": precision, "bf16": precision == "bf16", "fp16": precision == "fp16",
        "gradient_checkpointing": True, "use_gradient_checkpointing": True,
        "use_peft": True, "lora_r": lora.r, "lora_alpha": lora.alpha, "lora_dropout": lora.dropout,
        "lora_target_modules": list(lora.target_modules),
        "seed": cfg.seed, "data_seed": cfg.seed, "split_seed": cfg.seed,
        "eval_strategy": "no", "save_strategy": "no", "load_best_model_at_end": False,
        "group_by_length": False, "loggers": [], "report_to": "none", "trust_remote_code": False,
    }


def loss_curve(log_history: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The per-logging-step training rows of a Trainer's `state.log_history`."""
    return [{k: entry[k] for k in LOSS_KEYS if k in entry} for entry in log_history if "loss" in entry]


def pin_adapter_config(adapter_cfg: dict[str, Any], lora: LoraSettings) -> dict[str, Any]:
    """Point a saved adapter_config at the Hub base and its pinned revision, and check its LoRA shape."""
    if adapter_cfg["r"] != lora.r or adapter_cfg["lora_alpha"] != lora.alpha:
        raise ValueError(f"saved adapter has r={adapter_cfg['r']}, alpha={adapter_cfg['lora_alpha']}; "
                         f"expected r={lora.r}, alpha={lora.alpha}")
    if sorted(adapter_cfg["target_modules"]) != sorted(lora.target_modules):
        raise ValueError(f"saved adapter targets {adapter_cfg['target_modules']}")
    return {**adapter_cfg, "base_model_name_or_path": BASE_REPO, "revision": BASE_REVISION}


def _versions() -> dict[str, str]:
    return {pkg: version(pkg) for pkg in ("aligntune", "trl", "transformers", "peft", "torch", "datasets")}


def train_sentinel(cfg: SentinelConfig, spec: SentinelSpec, model_path: Path, work_dir: Path) -> Path:
    """Train one sentinel and return its output folder (adapter, loss curve, rows and metadata)."""
    from aligntune.core.backend_factory import create_sft_trainer
    from datasets import load_dataset

    out_dir = ensure_dir(work_dir / spec.id)
    rows, n_available = select_rows(load_dataset(spec.dataset, spec.config, split=spec.split), spec, cfg.seed)
    data_path = write_jsonl(rows, out_dir / "train_rows.jsonl")
    capability = gpu_capability()
    precision = resolve_precision(cfg.train.bf16, capability)
    log.info("%s: training in %s (GPU compute capability %s)", spec.id, precision, capability)
    trainer = create_sft_trainer(**aligntune_kwargs(cfg, spec, model_path, data_path, out_dir, precision))
    with Timer(f"{spec.id} training") as timer:
        result = trainer.train()
    curve = loss_curve(trainer.trainer.state.log_history)
    write_jsonl(curve, out_dir / "training_log.jsonl")
    config_path = out_dir / "adapter_config.json"
    config_path.write_text(json.dumps(pin_adapter_config(json.loads(config_path.read_text()), cfg.lora), indent=2))
    meta = {
        **asdict(spec), "seed": cfg.seed, "lora": asdict(cfg.lora), "train": asdict(cfg.train),
        "base_repo": BASE_REPO, "base_revision": BASE_REVISION, "base_path": str(model_path),
        "precision": precision, "gpu_capability": capability, "gradient_checkpointing": True,
        "rows_passing_filter": n_available, "rows_selected": len(rows), "rows_after_aligntune": len(trainer.dataset),
        "global_step": result["steps"], "final_loss": curve[-1]["loss"] if curve else None,
        "mean_train_loss": result["final_loss"], "train_wall_s": round(timer.seconds, 1),
        "gpu_memory": gpu_memory(spec.id), "versions": _versions(), "job": job_context(),
    }
    (out_dir / "sentinel_meta.json").write_text(json.dumps(meta, indent=1))
    log.info("%s: %d steps, final logged loss %s, %.0f s", spec.id, meta["global_step"], meta["final_loss"],
             timer.seconds)
    del trainer
    _free_gpu()
    return out_dir


def _free_gpu() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ---- entry point ---------------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    dest = p.add_mutually_exclusive_group(required=True)
    dest.add_argument("--out-repo-prefix", help="upload to private model repos <prefix>-<id>, e.g. me/wb-sentinel")
    dest.add_argument("--out-dir", type=Path, help="keep each sentinel in <out-dir>/<id> instead of uploading it")
    p.add_argument("--base-model", type=Path,
                   help="local copy of the pinned base (default: download it from the Hub, which needs a token)")
    p.add_argument("--only", nargs="+", help="sentinel ids to train (default: all in the config)")
    p.add_argument("--config", type=Path, default=CONFIG_PATH)
    p.add_argument("--work-dir", type=Path, default=Path("/tmp/weighbridge/sentinels"))
    p.add_argument("--force", action="store_true", help="retrain even when the destination already has an adapter")
    return p.parse_args(argv)


def keep_local(work_dir: Path, dest: Path) -> Path:
    """Copy a trained sentinel's published files from its work folder to `dest`; returns `dest`."""
    ensure_dir(dest)
    for name in UPLOAD_FILES:
        shutil.copy2(work_dir / name, dest / name)
    return dest


def main(argv: Sequence[str] | None = None) -> None:
    """Train every requested sentinel that has no adapter at its destination, saving each as soon as it is done."""
    setup_logging()
    args = parse_args(argv)
    cfg = load_sentinel_config(args.config)
    ids = args.only or list(cfg.sentinels)
    unknown = sorted(set(ids) - set(cfg.sentinels))
    if unknown:
        raise SystemExit(f"unknown sentinels {unknown}; known: {sorted(cfg.sentinels)}")
    api = None if args.out_dir and args.base_model else hub_api()
    if args.out_dir:
        done = {sid for sid in ids if (args.out_dir / sid / ADAPTER_FILE).exists()}
    else:
        done = {sid for sid in ids
                if ADAPTER_FILE in list_uploaded(repo_for(args.out_repo_prefix, sid), "model", api=api)}
    todo = [sid for sid in ids if args.force or sid not in done]
    log.info("to train: %s (already done: %s)", todo, sorted(set(ids) - set(todo)))
    if not todo:
        return
    if args.base_model:
        model_path = args.base_model
    else:
        with Timer("base download"):
            model_path = Path(api.snapshot_download(BASE_REPO, revision=BASE_REVISION))
    gpu_memory("start")
    for sid in todo:
        out_dir = train_sentinel(cfg, cfg.sentinels[sid], model_path, args.work_dir)
        if args.out_dir:
            log.info("%s saved to %s", sid, keep_local(out_dir, args.out_dir / sid))
        else:
            upload_outputs(out_dir, repo_for(args.out_repo_prefix, sid), repo_type="model",
                           allow_patterns=UPLOAD_FILES,
                           message=f"sentinel {sid} ({cfg.sentinels[sid].max_steps} steps, seed {cfg.seed})",
                           api=api)


if __name__ == "__main__":
    main()
