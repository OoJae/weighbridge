# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch>=2.4",
#   "transformers>=5.15,<5.22",
#   "safetensors>=0.4",
#   "huggingface_hub>=1.5",
#   "pyyaml>=6",
# ]
# ///
"""HF Jobs entry point (CUDA container): every base-model probe the weights-only screens need.

Loads CohereLabs/tiny-aya-global at common.BASE_REVISION (or a local copy given with --base-model) in
bfloat16, or float16 on a GPU without bfloat16 such as a T4, and writes to --out-dir:
  first_tokens.json                     top first response tokens, harmful vs harmless, both conditions
  directions_{condition}.safetensors    refusal directions per layer ("L{layer}")
  input_means_{lang}.safetensors        o_proj / down_proj input means ("L{layer}.{module}"), shipped condition
  margins.jsonl                         base-model margin of every panel row under both conditions
  sketch_{name}.safetensors             rank-k sketches of each margin gradient and of the irrelevant control
  grad_{name}.safetensors               full gradients for the names listed in --full-grads
  parity.json                           margins on 20 prompts: the run dtype vs float16 (bfloat16 runs) or
                                        vs float32 on the CPU (float16 runs)
  run.json                              arguments, opener sets, prompt ids, timings and peak GPU memory
then uploads the folder to the private dataset repo --out-repo.

The weighbridge package must be importable. If it is not installed in the container, pass --code-repo: a
dataset repo holding this repository's tree (src/ and configs/), which is downloaded and put on sys.path.

Example:
  hf jobs uv run --flavor a10g-large --secrets HF_TOKEN jobs/job_probe.py \\
      --panels hf://datasets/<owner>/<panels-repo>/en.jsonl hf://datasets/<owner>/<panels-repo>/yo.jsonl \\
      --code-repo <owner>/<code-repo> --out-repo <owner>/<probe-results-repo>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger("job_probe")

GRADIENT_SPECS = (("en", "shipped"), ("en", "minimal"), ("yo", "shipped"), ("pcm", "shipped"))
HARMLESS_KINDS = ("benign", "xstest_safe")
PARITY_PROMPTS = 20
LANGS = ("en", "yo", "pcm")
REQUIRED_SETS = ("en_harmful", "en_harmless")

Model = torch.nn.Module
Sets = dict[str, list[dict[str, Any]]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--panels", nargs="+", required=True,
                   help="panel jsonl files: local paths or hf://datasets/<owner>/<repo>/<path>")
    p.add_argument("--out-dir", type=Path, default=Path("probe_out"))
    p.add_argument("--base-model", default=None, help="local folder of the base model (default: the pinned Hub repo)")
    p.add_argument("--out-repo", help="private dataset repo to upload --out-dir to; omit to skip the upload")
    p.add_argument("--code-repo", help="dataset repo with this repository's tree, if weighbridge is not installed")
    p.add_argument("--max-prompts", type=int, default=None, help="cap on prompts per screen set")
    p.add_argument("--k", type=int, default=64, help="sketch rank")
    p.add_argument("--full-grads", nargs="*", default=["en_shipped"], help="gradient names to also save in full")
    p.add_argument("--batch-size", type=int, default=8, help="batch size for forward-only probes")
    p.add_argument("--grad-batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0, help="seed of the irrelevant-metric control and the sketches")
    return p.parse_args(argv)


def ensure_weighbridge(code_repo: str | None, token: str | None) -> None:
    """Make `import weighbridge` work, downloading the repository tree from `code_repo` when needed."""
    try:
        import weighbridge  # noqa: F401
    except ImportError:
        if not code_repo:
            raise SystemExit("weighbridge is not installed; pass --code-repo") from None
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(code_repo, repo_type="dataset", token=token))
        sys.path.insert(0, str(root / "src"))


# ---- Inputs ------------------------------------------------------------------------------------

def read_panel(source: str, token: str | None) -> list[dict[str, Any]]:
    """Rows of a panel jsonl given as a local path or as hf://datasets/<owner>/<repo>/<path>."""
    prefix = "hf://datasets/"
    if source.startswith(prefix):
        from huggingface_hub import hf_hub_download

        owner, repo, filename = source[len(prefix):].split("/", 2)
        source = hf_hub_download(f"{owner}/{repo}", filename, repo_type="dataset", token=token)
    with open(source, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def screen_rows(rows: list[dict[str, Any]], lang: str, kinds: tuple[str, ...],
                cap: int | None) -> list[dict[str, Any]]:
    """Screen-split rows of one language and kind set, in file order, capped at `cap`."""
    picked = [r for r in rows if r["split"] == "screen" and r["lang"] == lang and r["kind"] in kinds]
    return picked[:cap]


def screened_langs(sets: Sets) -> list[str]:
    """Languages with harmful screen prompts; English is always required, Yoruba and Pidgin are optional."""
    return [lang for lang in LANGS if sets.get(f"{lang}_harmful")]


def texts(rows: list[dict[str, Any]]) -> list[str]:
    """The prompt text of each row."""
    return [r["text"] for r in rows]


def load_base(base_model: str | None) -> tuple[torch.nn.Module, Any]:
    """The base model on CUDA in eval mode (pinned Hub revision, or a local folder) and its tokenizer."""
    from weighbridge.common import BASE_REPO
    from weighbridge.runner_peft import load_base as load

    if not torch.cuda.is_available():
        raise SystemExit("job_probe needs a CUDA device")
    return load(base_model or BASE_REPO)


# ---- Bookkeeping -------------------------------------------------------------------------------

@contextmanager
def timed(stage: str, record: dict[str, dict[str, float]]) -> Iterator[None]:
    """Log and record wall time and peak GPU memory of one stage."""
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    yield
    seconds = time.perf_counter() - start
    peak_gb = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0
    record[stage] = {"seconds": round(seconds, 2), "peak_gpu_gb": round(peak_gb, 3)}
    log.info("%s: %.1fs, peak GPU %.2f GB", stage, seconds, peak_gb)


def save_tensors(path: Path, tensors: dict[str, torch.Tensor], meta: dict[str, Any]) -> None:
    """Float32 tensors to safetensors with JSON metadata under the "weighbridge" header key."""
    from safetensors.torch import save_file

    save_file({k: v.float().contiguous().cpu() for k, v in tensors.items()}, str(path),
              metadata={"weighbridge": json.dumps(meta, sort_keys=True)})


def upload(out_dir: Path, repo_id: str, token: str | None) -> None:
    """Upload the output folder to a private dataset repo, creating it if needed."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=repo_id, repo_type="dataset",
                      commit_message="base-model probes")


# ---- Stages ------------------------------------------------------------------------------------

def run_first_tokens(model: Model, tokenizer: Any, sets: Sets, args: argparse.Namespace, out: Path) -> None:
    """Top first response tokens on English harmful and harmless prompts, under both conditions."""
    from weighbridge import probe

    result = {f"{kind}_{cond}": probe.first_token_distribution(model, tokenizer, texts(sets[f"en_{kind}"]), cond,
                                                               batch_size=args.batch_size)
              for kind in ("harmful", "harmless") for cond in ("shipped", "minimal")}
    (out / "first_tokens.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))


def run_directions(model: Model, tokenizer: Any, sets: Sets, args: argparse.Namespace, out: Path,
                   meta: dict[str, Any]) -> None:
    """English refusal directions per layer, under both conditions."""
    from weighbridge import probe, scoring

    for cond in ("shipped", "minimal"):
        dirs = probe.refusal_directions(model, tokenizer, texts(sets["en_harmful"]), texts(sets["en_harmless"]),
                                        cond, batch_size=args.batch_size)
        save_tensors(scoring.directions_file(out, cond), scoring.encode_directions(dirs),
                     {**meta, "condition": cond, "lang": "en"})


def run_input_means(model: Model, tokenizer: Any, sets: Sets, args: argparse.Namespace, out: Path,
                    meta: dict[str, Any]) -> None:
    """Residual-writer input means on each language's harmful screen prompts, shipped condition.

    Languages without screen prompts are skipped.
    """
    from weighbridge import probe, scoring

    for lang in screened_langs(sets):
        means = probe.module_input_means(model, tokenizer, texts(sets[f"{lang}_harmful"]), "shipped",
                                         batch_size=args.batch_size)
        save_tensors(scoring.input_means_file(out, lang), scoring.encode_input_means(means),
                     {**meta, "condition": "shipped", "lang": lang,
                      "prompt_ids": [r["id"] for r in sets[f"{lang}_harmful"]]})


def run_margins(model: Model, tokenizer: Any, panels: Sets, args: argparse.Namespace, out: Path) -> None:
    """Base-model margin of every panel row, in that row's language, under both conditions."""
    from weighbridge import probe

    with open(out / "margins.jsonl", "w", encoding="utf-8") as f:
        for panel, rows in panels.items():
            for lang in sorted({r["lang"] for r in rows}):
                group = [r for r in rows if r["lang"] == lang]
                for cond in ("shipped", "minimal"):
                    values = probe.margin(model, tokenizer, texts(group), lang, cond, batch_size=args.batch_size)
                    for r, v in zip(group, values.tolist(), strict=True):
                        f.write(json.dumps({"panel": panel, "id": r["id"], "lang": lang, "kind": r["kind"],
                                            "split": r["split"], "condition": cond, "margin": v}) + "\n")


def save_gradient(G: dict[tuple[int, str], torch.Tensor], name: str, args: argparse.Namespace, out: Path,
                  meta: dict[str, Any]) -> None:
    """Rank-k sketch of G, plus the full G when `name` is listed in --full-grads."""
    from weighbridge import grad

    grad.save(grad.sketch(G, k=args.k, seed=args.seed), out / f"sketch_{name}.safetensors", {**meta, "k": args.k})
    if name in args.full_grads:
        grad.save(G, out / f"grad_{name}.safetensors", meta)


def run_gradients(model: Model, tokenizer: Any, sets: Sets, args: argparse.Namespace, out: Path,
                  meta: dict[str, Any], timings: dict[str, dict[str, float]]) -> None:
    """Margin gradients for each (lang, condition) in GRADIENT_SPECS whose language has screen prompts, then the
    irrelevant-metric control."""
    from weighbridge import grad, probe

    for lang, cond in GRADIENT_SPECS:
        if lang not in screened_langs(sets):
            continue
        name, rows = f"{lang}_{cond}", sets[f"{lang}_harmful"]
        refuse, comply = probe.opener_token_ids(tokenizer, lang)
        with timed(f"grad_{name}", timings):
            G = grad.margin_gradient(model, tokenizer, texts(rows), lang, cond, batch_size=args.grad_batch_size)
        save_gradient(G, name, args, out, {**meta, "metric": "margin", "lang": lang, "condition": cond,
                                           "prompt_ids": [r["id"] for r in rows],
                                           "openers": {"refuse": refuse, "comply": comply}})
        del G
    rows = sets["en_harmful"]
    set_a, set_b = grad.irrelevant_token_sets(model, tokenizer, args.seed, "en")
    with timed("grad_irrelevant", timings):
        G = grad.irrelevant_gradient(model, tokenizer, texts(rows), "shipped", args.seed,
                                     batch_size=args.grad_batch_size)
    save_gradient(G, "irrelevant", args, out, {**meta, "metric": "irrelevant", "lang": "en", "condition": "shipped",
                                               "prompt_ids": [r["id"] for r in rows], "seed": args.seed,
                                               "openers": {"set_a": set_a, "set_b": set_b}})


def run_calibration(model: Model, tokenizer: Any, sets: Sets, args: argparse.Namespace, out: Path) -> None:
    """Calibrate the English opener sets on the base model (PREREG section 8) and make every later stage use them.

    Writes openers_calibrated.json; later margins and gradients read it through probe.OPENERS_ENV, and the sweep
    jobs are given the same file.
    """
    from weighbridge import probe

    cal = probe.calibrate_openers(model, tokenizer, texts(sets["en_harmful"]), texts(sets["en_harmless"]),
                                  batch_size=args.batch_size)
    path = out / "openers_calibrated.json"
    path.write_text(json.dumps(cal, indent=1, ensure_ascii=False))
    os.environ[probe.OPENERS_ENV] = str(path)
    en = cal["en"]
    log.info("calibrated openers: refuse %s | comply %s | fallback=%s", en["refuse_tokens"], en["comply_tokens"],
             en["fallback"])


def parity_reference(dtype: torch.dtype) -> tuple[torch.dtype, str]:
    """The (dtype, device) a run in `dtype` is checked against: float16 on the GPU for a bfloat16 run, else
    float32 on the CPU (a float16 run on a T4 has no bfloat16 to compare with)."""
    return (torch.float16, "cuda") if dtype == torch.bfloat16 else (torch.float32, "cpu")


def run_parity(model: Model, tokenizer: Any, sets: Sets, args: argparse.Namespace, out: Path) -> None:
    """Margins on 20 prompts in the run dtype, then with the model cast to the parity reference.

    Leaves the model in the reference dtype and device, so it must be the last stage.
    """
    from weighbridge import probe

    prompts = texts(sets["en_harmful"][:PARITY_PROMPTS])
    run_dtype = next(model.parameters()).dtype
    ref_dtype, ref_device = parity_reference(run_dtype)
    run = probe.margin(model, tokenizer, prompts, "en", "shipped", batch_size=args.batch_size)
    ref = probe.margin(model.to(ref_device, ref_dtype), tokenizer, prompts, "en", "shipped",
                       batch_size=args.batch_size)
    diff = (run - ref).abs()
    corr = torch.corrcoef(torch.stack([run, ref]))[0, 1].item()
    report = {"n": len(prompts), "run_dtype": str(run_dtype), "reference": f"{ref_dtype} on {ref_device}",
              "max_abs_diff": diff.max().item(), "mean_abs_diff": diff.mean().item(), "pearson": corr,
              "run": run.tolist(), "reference_margins": ref.tolist()}
    (out / "parity.json").write_text(json.dumps(report, indent=1))
    log.info("%s vs %s margins: max |diff| %.4f, pearson %.4f", run_dtype, report["reference"],
             report["max_abs_diff"], corr)


def main(argv: list[str] | None = None) -> None:
    """Run every stage, write the output folder and upload it."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    token = os.environ.get("HF_TOKEN")
    ensure_weighbridge(args.code_repo, token)
    from weighbridge.common import BASE_REVISION

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    panels = {Path(src).stem: read_panel(src, token) for src in args.panels}
    rows = [r for panel_rows in panels.values() for r in panel_rows]
    sets = {f"{lang}_harmful": screen_rows(rows, lang, ("harmful",), args.max_prompts) for lang in LANGS}
    sets["en_harmless"] = screen_rows(rows, "en", HARMLESS_KINDS, args.max_prompts)
    empty = [name for name in REQUIRED_SETS if not sets[name]]
    if empty:
        raise SystemExit(f"no screen-split prompts for: {empty}")
    log.info("screen prompts per set: %s; languages probed: %s",
             {name: len(s) for name, s in sets.items()}, screened_langs(sets))

    timings: dict[str, dict[str, float]] = {}
    with timed("load", timings):
        model, tokenizer = load_base(args.base_model)
    meta = {"base_revision": BASE_REVISION, "base_model": args.base_model or "hub",
            "dtype": str(next(model.parameters()).dtype).removeprefix("torch.")}
    with timed("first_tokens", timings):
        run_first_tokens(model, tokenizer, sets, args, out)
    with timed("calibrate_openers", timings):
        run_calibration(model, tokenizer, sets, args, out)
    with timed("directions", timings):
        run_directions(model, tokenizer, sets, args, out,
                       {**meta, "harmful_ids": [r["id"] for r in sets["en_harmful"]],
                        "harmless_ids": [r["id"] for r in sets["en_harmless"]]})
    with timed("input_means", timings):
        run_input_means(model, tokenizer, sets, args, out, meta)
    with timed("margins", timings):
        run_margins(model, tokenizer, panels, args, out)
    run_gradients(model, tokenizer, sets, args, out, meta, timings)
    with timed("parity", timings):
        run_parity(model, tokenizer, sets, args, out)

    run = {"args": {k: str(v) for k, v in vars(args).items()}, **meta, "timings": timings,
           "prompt_ids": {name: [r["id"] for r in s] for name, s in sets.items()}}
    (out / "run.json").write_text(json.dumps(run, indent=1))
    if args.out_repo:
        upload(out, args.out_repo, token)


if __name__ == "__main__":
    main()
