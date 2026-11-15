# GPU jobs

The job scripts in `jobs/` (`job_probe.py`, `job_sweep.py`, `job_judge.py`, `job_sentinels.py`, `job_c3.py`) are plain
command-line programs. Stage 1 ran them in two ways, both free:

- **Kaggle (what produced every Stage 1 number).** `jobs/kaggle_push.py bundle` uploads this repository as a private
  Kaggle dataset; `jobs/kaggle_push.py push <plan.json>` creates a private batch notebook from `jobs/kaggle_kernel.py`
  that checks the base-model weights against their published SHA-256, then runs the plan's jobs: one queue per T4 GPU,
  an optional `cpu` queue, and `after` jobs once the queues finish. A plan names its inputs (earlier notebooks'
  outputs, found again by glob), a per-job timeout and a session deadline, and records a `STATUS.json`. A plan with
  `"enable_gpu": false` runs on a CPU session, which uses no GPU quota (weights-only scoring ran this way). Every job
  resumes from the rows it already wrote, so a timed-out run is continued by pushing the same plan again.
- **Colab (judging only).** `job_judge.py --gen-dir ... --prior-labels ...` on a free T4 labels the generation files
  copied to the VM and skips adapters already labelled, so a reclaimed VM loses at most one adapter's work. The gated
  base model and the Hugging Face token never leave the laptop and Kaggle.

The rest of this page describes the original launcher for [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)
(`jobs/launch.py`, presets in `configs/jobs.yaml`), which takes the same job arguments.

## Before the first launch

1. Log in once with a token that can write to your namespace: `hf auth login`. The launcher reads that token
   with `huggingface_hub.get_token()` and hands it to the job as the encrypted secret `HF_TOKEN`. It never
   prints it, not even in a dry run.
2. The same account must have accepted the terms of the gated `CohereLabs/tiny-aya-global`. The jobs
   download it inside the container. Your laptop never needs it.
3. Always dry-run first. A dry run prints the exact `HfApi` call. It does not contact the Hub, upload
   anything or charge the ledger:

   ```bash
   .venv/bin/python jobs/launch.py run sentinels --dry-run -- --out-repo-prefix <you>/weighbridge-sentinel
   ```

## How a launch works

- **Spending cap.** A launch is charged its worst case: hourly price × timeout. The charge goes to the private
  ledger `PRIVATE_DIR/jobs_ledger.jsonl`. If the running total would go over `budget_usd` (USD 15), the
  launch is refused before anything is uploaded. Cancelling a job does not refund its row, so the ledger can
  only over-count. `launch.py budget` shows the total. Only these flavors have verified prices:
  t4-small $0.40/h, l4x1 $0.80/h, a10g-small $1.00/h, a10g-large $1.50/h, a100-large $2.50/h. Any other
  flavor is refused.
- **Timeout.** Every launch passes a timeout, either the preset's or `--timeout 45m`. Without one, the Jobs
  default of 30 minutes would kill long runs.
- **Code.** The launcher collects `src/`, `jobs/`, `configs/` and `data/panels/` and skips caches and weight
  files. It copies them into a content-addressed folder (`weighbridge-code-<sha12>`) of your private
  `<namespace>/jobs-artifacts` bucket with `HfApi.sync_job_volume`. That folder is mounted read-only at
  `/code`, and `PYTHONPATH=/code/src:/code/jobs`. A running job never sees later edits, and no git push or
  package build is needed. Only public code and panels are in the bundle.
- **Runtimes.**
  - `uv` (probe, sentinels, judge) uses `HfApi.run_uv_job` on `ghcr.io/astral-sh/uv:python3.12-bookworm`.
    The script's own PEP 723 header lists its dependencies. Local files named in the arguments are shipped
    to `/data` by `run_uv_job`.
  - `pip` (sweep) uses `HfApi.run_job` on `vllm/vllm-openai:v0.33.0`: `pip install <deps>`, then
    `python3 /code/<script>`. Arguments that name bundled files are rewritten to `/code/...`. Any other
    local file is refused, because it would not exist in the container.
- **Private outputs.** Jobs write only to private repos. `common_job.upload_outputs` creates a missing repo
  as private and refuses to write to a public one. Third-party adapter ids, shards and generations reach a
  job only through a private dataset repo, never through the bundle.
- **Resuming.** Each job lists what its output repo already holds (`common_job.list_uploaded`) and skips
  that work, so relaunching after a timeout continues where the last run stopped.

Watch and stop jobs with:

```bash
.venv/bin/python jobs/launch.py status <job_id>          # stage, running seconds, cost so far
.venv/bin/python jobs/launch.py logs <job_id> --follow   # stream logs (or --tail 200)
.venv/bin/python jobs/launch.py cancel <job_id>
.venv/bin/python jobs/launch.py budget
```

## The jobs

Times below are planning estimates from model size and typical throughput. They have not been measured
yet. Replace them with the measured wall times (logged by every job) after the first run.

| Job | Preset flavor | Timeout | Worst case | Expected |
| --- | --- | --- | --- | --- |
| sentinels | l4x1 | 3 h | $2.40 | 1.5–2 h, about $1.40 |
| probe | a100-large | 1 h | $2.50 | 20–40 min, about $1.20 |
| sweep (one shard) | a10g-large | 4 h | $6.00 | depends on shard size, see below |
| judge | l4x1 | 2 h | $1.60 | 40–70 min, about $0.80 |
| **all presets once** | | | **$12.50** | leaves $2.50 of the $15 cap for one rerun |

Override the flavor or timeout per launch with `--flavor` and `--timeout`. The worst case changes to match.

### probe: base-model directions, means, margins and gradients

```bash
.venv/bin/python jobs/launch.py run probe -- \
    --panels data/panels/fit_directions.jsonl data/panels/d1_probe.jsonl data/panels/y2_heldout.jsonl \
    --out-repo <you>/weighbridge-probe
```

- **Inputs:** the public panels (bundled) and the gated base at the pinned revision.
- **Outputs**, in the private dataset repo `--out-repo`:
  - refusal directions per condition, module input means and base margins;
  - rank-k gradient sketches, plus full gradients for `--full-grads`;
  - a float16 vs bfloat16 parity check;
  - `run.json` with timings and peak memory.

  See the header of `jobs/job_probe.py`.
- **Languages:** English screen prompts are required; Yoruba and Pidgin directions, means and gradients are computed
  only when screen-split prompts in that language are among the panels (none exist yet), and skipped otherwise.
- **Local weights:** `--base-model <folder>` loads a local copy of the base (for example Kaggle's) instead of the
  pinned Hub revision. On a GPU without bfloat16 (T4) the model runs in float16, gradients use a static loss scale
  (`grad.FP16_LOSS_SCALE`), and the parity check compares against float32 on the CPU.
- **Scoring afterwards:** `weighbridge score --adapters <folder of alias sub-folders> --probe <this output>`.
- **Flavor:** a100-large. The float32 margin gradient over all 252 target matrices is about 11 GB, on top of
  6.7 GB of bf16 weights, which is tight on a 24 GB A10G. Pass `--flavor a10g-large` only with
  `--full-grads` empty.

### sentinels: in-house LoRAs with known benign training

```bash
.venv/bin/python jobs/launch.py run sentinels -- --out-repo-prefix <you>/weighbridge-sentinel
.venv/bin/python jobs/launch.py run sentinels --timeout 45m -- --out-repo-prefix <you>/weighbridge-sentinel --only S2
```

| Id | Data (licence) | Rows | Steps |
| --- | --- | --- | --- |
| S1a | `tatsu-lab/alpaca` (CC-BY-NC-4.0) | 2,000 | 250 |
| S1b | the same 2,000 rows | 2,000 | 1,000 |
| S2 | `openai/gsm8k` main/train (MIT) | 2,000 | 250 |
| S3 | `CohereLabs/aya_dataset` rows with `language == "Yoruba"` (Apache-2.0; 11,758 available) | 2,000 | 250 |

- **Shared settings** (`configs/sentinels.yaml`): LoRA r=16, alpha=32 and dropout 0.05 on the 7 target
  modules; lr 2e-4 with a cosine schedule and 3% warmup; batch 4 × 2 accumulation; bf16; seed 0 for row
  sampling, data order and initialisation. One step is 8 rows, so 250 steps is one epoch.
- **Training stack:** AlignTune 0.1.35, `create_sft_trainer(backend="trl", task_type="sft")`.
  1. The job samples the rows and writes them as single-turn `messages` JSONL.
  2. AlignTune 0.1.34+ passes the `messages` column to TRL unchanged. TRL then applies the tokenizer's own
     chat template, so every row carries the shipped system preamble, exactly as the deployed model sees
     it. The loss covers the whole formatted conversation (TRL's language-modelling format), as in most
     TRL-trained third-party adapters.
  3. AlignTune takes no model revision, so the job snapshots the base at `common.BASE_REVISION` and passes
     the local path.
  4. After saving, `adapter_config.json` is pointed back at `CohereLabs/tiny-aya-global` with the pinned
     `revision`, and its r, alpha and targets are checked against the config.
- **Outputs:** one private model repo per sentinel, `<prefix>-s1a`, `-s1b`, `-s2` and `-s3`, holding:
  - `adapter_config.json` and `adapter_model.safetensors`;
  - `training_log.jsonl`, the loss curve: step, loss, learning rate and gradient norm every 5 steps;
  - `train_rows.jsonl`, the exact rows used;
  - `sentinel_meta.json`: row counts before and after AlignTune's data checks, final loss, training wall
    time, peak GPU memory, package versions and the job id.

  The team makes a repo public only after checking it. A sentinel whose repo already holds an adapter is
  skipped. `--force` retrains it.
- **Time:** about 3 s per step on an L4 (estimate). S1a, S2 and S3 take 12–15 min each, S1b 50–60 min, plus
  about 10 min of install and base download.

### sweep: behaviour of every adapter (one shard per job)

```bash
.venv/bin/python jobs/launch.py run sweep --timeout 90m -- \
    --shard shard_00.jsonl --shard-repo <you>/weighbridge-private \
    --panels data/panels/en_outcome_harmful.jsonl data/panels/en_outcome_benign.jsonl data/panels/yo_tukabench.jsonl \
    --t1-panels data/panels/d1_probe.jsonl data/panels/y2_heldout.jsonl \
    --run-id sweep1 --out-repo <you>/weighbridge-runs
```

- **Inputs:**
  - a shard of included manifest rows, held in a private dataset repo (`--shard-repo`, never bundled);
  - the bundled panels;
  - the adapters at their pinned SHAs, downloaded in the job.
- **Directions:** pass `--directions` with the probe job's output (a local folder, or
  `hf://datasets/<you>/weighbridge-probe` when launched on HF Jobs) so D0a and the T1 projection share one set of
  directions. Without it the sweep fits its own from the T1 panels' English screen prompts. The projection layer
  (D1p) is the one where the base model's direction best separates those prompts (`probe.layer_separation`).
- **Outputs**, in the private `--out-repo`, keyed by alias only: `<run-id>/raw/<alias>.jsonl` generations,
  `<run-id>/t1/<alias>.jsonl` forward-pass screens and the fitted directions. Each adapter is uploaded as
  soon as it finishes, so a timeout loses at most one adapter, and a relaunch pulls and resumes.
- **Time:** about 650 greedy generations of up to 160 tokens per adapter, so about 100k tokens. At an
  assumed 1.5–2.5k LoRA tokens/s with vLLM on an A10G, that is 1–1.5 min per adapter, plus the forward-pass
  screen. Size each shard to about 30 adapters with `--timeout 90m` ($2.25 worst case). Time the first
  shard, then size the rest from the measured rate.

### judge: Qwen3Guard labels for one generation run

```bash
.venv/bin/python jobs/launch.py run judge -- \
    --gen-repo <you>/weighbridge-runs --run-id sweep1 \
    --panels data/panels/en_outcome_harmful.jsonl data/panels/en_outcome_benign.jsonl data/panels/yo_tukabench.jsonl \
    --out-repo <you>/weighbridge-runs
```

- **Inputs:** `<run-id>/raw/*.jsonl` from the private `--gen-repo`, and the panels for the prompt text.
- **Outputs:** `labels/<run-id>.jsonl` in the private `--out-repo`. Each row holds:
  - the guard safety and refusal labels;
  - the string-refusal flag;
  - the language ID;
  - `label3`.

  Rows never contain the generated text. Adapters already labelled are skipped on restart.
- **Flavor:** l4x1, since Qwen3Guard-Gen-4B in bf16 needs about 8 GB. This is a prefill-heavy job.
