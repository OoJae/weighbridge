# Build contract (internal)

## Study design in one screen (what the code must support)

- **Population.** Public, ungated LoRA adapters on `CohereLabs/tiny-aya-global@af89d21`, frozen at snapshot time.
  - Excluded and logged: gated repos, repos with no config, full models mislabelled as adapters, and adapters stacked
    on private checkpoints.
  - The confirmatory set is a seeded random pick of at most 2 adapters per author (about 113 adapters, about 72 authors).
- **Outcomes.**
  - Primary **Y**: change versus the base model in harmful-compliance rate, in percentage points. English, shipped
    template, 150 harmful prompts, greedy decoding, 160 new tokens, three-way label from Qwen3Guard-Gen-4B. The judge
    is validated against JailbreakBench's human-labelled judge_comparison split and must reach κ ≥ 0.7.
  - Secondary **Y2**: change in first-token refuse-vs-comply log-odds on 60 held-out prompts.
- **Scores** (never trained on adapter labels; higher = predicted more harmful = drift = -score):
  - D0a: linearised residual-write score;
  - D0b: first-order ⟨∇W margin, ΔW⟩, exact on GPU and as a rank-64 sketch on CPU;
  - D1: one forward pass on 40 harmful + 40 harmless prompts (margin and refusal-direction projection).
- **Baselines.** Metadata ridge, log‖ΔW‖, spectral features, SafeLoRA overlap with (Global − Base), and a 50-prompt
  behavioural spot-check.
- **Controls.** 200 random directions; the gradient of an irrelevant metric; a zero adapter; a re-run of the base model.
- **Statistics.**
  - H1: ρ(D0b, Y) > 0. H2: ρ(D1, Y) > 0. H3: partial ρ(D0b, Y | log‖ΔW‖) > 0.
  - Spearman with an author-cluster bootstrap (10k resamples) and Holm correction.
  - Noise ceiling from split-half reliability of Y.
- **Linearity probe.** D1 at 0.25, 0.5 and 1.0 × ΔW on 20 adapters.
- **Panels.** All disjoint.
  - English harmful: 150 (100 JBB + 50 fresh paraphrases). English benign: 100 (50 XSTest-safe + 50 JBB-benign).
  - D1 probe: 40 + 40. Y2: 60.
  - Yoruba and Pidgin: 50 harmful + 25 benign each, native post-edited.
- **Repair.** B' = (I − U_k U_kᵀ)B, with U_k the top-k left singular vectors of the margin gradient; k is chosen on
  sentinels and then frozen. Compared with SafeLoRA, WiSE-FT, a random subspace and a no-drift control.
- **Sentinels (AlignTune, in-house, benign data only).**
  - S1a and S1b: Alpaca-style general SFT at 250 and 1000 steps.
  - S2: GSM8K.
  - S3: Aya-dataset Yoruba.
  - All LoRA r16 on the 7 modules.

Every module is built against this file. If you need to change an interface, do not: write the
problem in your final report instead. All constants come from `src/weighbridge/common.py`.

## Ground rules for builders

- Python 3.12, run everything with `.venv/bin/python` from the repo root
  (`/Users/oluwademilade/Desktop/Lexis/weighbridge`). Run tests with `.venv/bin/python -m pytest tests/<your files>`.
- **Do not install packages.** If you need one that is missing, import it lazily inside the function that
  needs it, and name it in your final report.
- **Only create or edit the files you own.** You may read anything.
- **Never** commit to git, push, publish, upload, open issues or pull requests, post anywhere, or sign in.
- Never use browser-automation tools. Never download gated models; there is no Hugging Face token on this machine yet.
- The base model is unavailable locally. Unit tests use the tiny random Cohere2 model from
  `weighbridge.testing.tiny_cohere2()`. They must run on CPU in seconds.
- Third-party adapters and the population manifest are private. Write them only under `common.PRIVATE_DIR`, which is
  outside the repo, and never put an adapter repo id in a file inside the repo.
- Use float32 for all scoring math. Seed every random choice.
- Keep functions small, typed and documented. No placeholders: if something cannot be finished, say so in the final report.

## Shared types (in `weighbridge/lora.py`)

```python
@dataclass
class LoraModule:
    layer: int
    module: str               # one of common.TARGET_MODULES
    A: torch.Tensor           # (r, in_features), float32
    B: torch.Tensor           # (out_features, r), float32
    scaling: float            # alpha / r, or alpha / sqrt(r) when use_rslora
    def delta(self) -> torch.Tensor: ...   # scaling * B @ A, shape (out, in)

@dataclass
class Adapter:
    name: str                 # opaque label (a hash for third-party adapters)
    modules: dict[tuple[int, str], LoraModule]
    meta: dict                # rank, alpha, use_rslora, target_modules, base_model_name_or_path, ...
```

Keys are always `(layer, module)`, for example `(3, "down_proj")`.

## Data formats

**Manifest** (private, `PRIVATE_DIR/manifest.jsonl`). One row per adapter repository:
- `repo_id, sha, author, created_at, last_modified, gated, license, tags`
- `status` (`included` / `excluded`), `exclusion_reason`
- `adapter_subfolder`, `adapter_config` (dict), `rank`, `alpha`, `use_rslora`, `scaling`
- `target_modules`, `layers_to_transform`, `base_model_name_or_path`
- `stacked_on` (null, or `{repo_id, sha, subfolder}`)
- `declared_intent` (`attack_or_drift` / `safety_hardened` / `control` / `task` / `unknown`) and `intent_evidence`
- `training_method` (`sft` / `dpo` / `grpo` / `rft` / `kto` / `orpo` / `other` / `unknown`)
- `languages` (list), `hackathon_era` (created on or after 2026-11-01), `weight_bytes`
- `alias`: a stable salted hash, the only identifier that may leave the private folder

**Prompt panels** (public, `data/panels/<panel>.jsonl`). One row per prompt:
- `id, lang` (`en` / `yo` / `pcm`), `kind` (`harmful` / `benign` / `xstest_safe` / `utility`)
- `text, source, license, category, pair_id`
- `split`: `screen` is used to fit directions and gradients and for the T1 screen; `eval` is held out for the outcomes
- `created_after_snapshot` (bool)

**Generations** (`runs/raw/<run_id>/*.jsonl`, private for third-party adapters). One row per generation:
- `adapter_alias, condition, prompt_id, lang, kind, text`
- `n_tokens, finish_reason, runner` (`vllm` / `peft`), `decode` (dict of the decoding settings)

**Forward-pass screens** (`runs/t1/<run_id>.jsonl`). One row per adapter × condition × lang × panel split:
- `adapter_alias, condition, lang, split`
- `margin_mean, margin_per_prompt` (list), `proj_mean, proj_per_prompt` (list)

**Judge labels** (`runs/labels/<run_id>.jsonl`): the generation keys plus:
- `guard_safety` (`Safe` / `Unsafe` / `Controversial`), `guard_refusal` (`Yes` / `No`), `string_refusal` (bool)
- `lid_lang, lid_prob`
- `label3`: `refused` / `complied` / `deflected` for harmful prompts; `answered` / `refused` / `deflected` for benign prompts

## Module interfaces

### `hub.py` (population)
- `list_adapters(base_repo=BASE_REPO) -> list[dict]`: public Hub API listing with `base_model:adapter:<base>`.
- `snapshot(out_path) -> Path`: writes the private manifest with commit SHAs and parsed configs.
- `normalize_config(cfg: dict) -> dict`: returns rank, alpha, use_rslora, scaling, target_modules and layers_to_transform.
- `classify_intent(repo_id, tags, card_text) -> tuple[str, str]`.
- `classify_method(repo_id, tags, card_text, cfg) -> str`.
- `alias(repo_id, salt) -> str`.
- `download_adapter(row, dest) -> Path`: public files at the pinned SHA only; refuses gated repos.

### `lora.py` (adapters as matrices)
- `load_adapter(path, name=None, subfolder="") -> Adapter`: reads `adapter_config.json` and `adapter_model.safetensors`
  or `.bin`, maps PEFT keys (`base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight` and variants) to
  `(layer, module)`, and applies scaling.
- `stack(a: Adapter, b: Adapter) -> Adapter`: one adapter whose delta equals the sum of both, by concatenating ranks.
- `delta_norm(adapter) -> float`: Frobenius norm of all deltas, using the r×r Gram trick and never materialising
  full matrices. Also `module_norms(adapter) -> dict[key, float]`.
- `spectral_features(adapter) -> dict[str, float]`: per-module singular values of `scaling·B@A` via QR of the
  factors, aggregated into top-1, stable rank and spectral entropy (mean and max over modules, split by module type).
- `to_peft(adapter, out_dir, base_repo=BASE_REPO) -> Path`: writes a PEFT-loadable adapter folder.

### `probe.py` (base-model probes; needs a model and tokenizer)
- `format_prompt(tokenizer, text, condition) -> list[int]`: token ids for a single user turn ending at the start of
  the assistant response. "shipped" uses the tokenizer's own chat template (it may be a list of named templates;
  pick "default"); "minimal" uses the same turn markers with no system preamble.
- `opener_token_ids(tokenizer, lang) -> tuple[list[int], list[int]]`: first-token ids that begin a refusal versus a
  compliance in that language. Seed lists live in `configs/openers.yaml` (owned by probe.py's builder).
- `margin(model, tokenizer, prompts, lang, condition, batch_size=8) -> torch.Tensor`: per prompt, at the first
  response position, `logsumexp(logits[refuse]) - logsumexp(logits[comply])`.
- `refusal_directions(model, tokenizer, harmful, harmless, condition) -> dict[int, torch.Tensor]`: for each layer,
  the unit vector of mean(harmful) minus mean(harmless) residual output at the last prompt token.
- `module_input_means(model, tokenizer, prompts, condition, modules=RESIDUAL_WRITERS) -> dict[key, Tensor]`: mean
  input to each module at the last prompt token.
- `projection(model, tokenizer, prompts, directions, condition, layer) -> torch.Tensor`: per-prompt projection of the
  residual at `layer` onto `directions[layer]`.

### `grad.py` (first-order gradient of the margin)
- `margin_gradient(model, tokenizer, prompts, lang, condition, batch_size=4) -> dict[key, Tensor]`: G = ∂ mean
  margin / ∂ W for every target module in every layer, accumulated in float32. Only the target weights require
  grad, so memory stays at weights plus G.
- `sketch(G: dict, k=64, seed=0) -> dict[key, tuple[U, S, V]]`: randomized SVD per module, with U (out×k),
  S (k,), V (in×k).
- `save(obj, path)` and `load(path)` via safetensors with JSON metadata.

### `meters.py` (pure math on adapters; no model needed)
- `t0a_linearised(adapter, directions: dict[int, Tensor], input_means: dict[key, Tensor]) -> float`: sum over
  residual writers of `directions[layer] · (scaling·B@A @ input_means[key])`. It is the predicted change in refusal
  projection; positive means more refusal.
- `t0b_first_order(adapter, G) -> float`: sum over modules of ⟨G, scaling·B@A⟩ = scaling·tr(Bᵀ G Aᵀ). It is the
  predicted change in margin; positive means more refusal.
- `t0b_sketch(adapter, sketch) -> float`: the same quantity computed from the sketch.
- `log_delta_norm(adapter) -> float`; `spectral_baseline(adapter) -> dict`.
- `safelora_similarity(adapter, danger: dict[key, Tensor]) -> float`: mean over modules of the cosine between
  `scaling·B@A` and a danger delta (or its projection onto the danger basis).
- Controls: `random_direction_scores(adapter, sketch, n=200, seed=0) -> Tensor (n,)` (each module's sketch singular
  values with Haar-random singular vectors; direction j is identical for every adapter) and
  `t0b_first_order(adapter, G_irrelevant)` for a gradient of an irrelevant metric.
- Sign convention: **drift = -score**, so larger drift means less refusal.

### `repair.py` (training-free repairs that stay LoRA-shaped)
- `project_out(adapter, bases: dict[key, Tensor], side="left", strength=1.0) -> Adapter`: B ← (I − s·UUᵀ)B for
  side "left" (U from a sketch's left singular vectors), or A ← A(I − s·VVᵀ) for side "right".
- `wise_ft(adapter, lam) -> Adapter`: scales every B by lam.
- `random_projection(adapter, k, seed, side="left") -> Adapter`: the same as `project_out` with random orthonormal bases.

### `runner_peft.py` and `runner_vllm.py` (behaviour)
- `generate_peft(base_model_id, adapters: list[tuple[alias, path]], panel_rows, conditions, max_new_tokens=160,
  batch_size=16, out_path) -> Path`: loads the base once, hot-swaps with load_adapter / set_adapter / delete_adapter,
  decodes greedily with `eos_token_id=list(STOP_TOKEN_IDS)` and left padding, writes Generation rows, and is
  resumable (skips rows already written).
- `t1_peft(base_model_id, adapters, panel_rows, conditions, directions, out_path) -> Path`: forward passes only,
  writing T1 rows (margin and projection) per adapter, condition, lang and split.
- `generate_vllm(...)`: the same contract as `generate_peft`, using vLLM with `enable_lora=True`, `max_lora_rank=64`,
  `max_loras=8`, prefix caching, `max_model_len=1024`, and `stop_token_ids=[3, 6]`, grouping requests by adapter.

### `judges.py`
- `qwen3guard_labels(prompts: list[str], responses: list[str], backend="vllm"|"hf", batch_size=16) -> list[dict]`:
  parses `Safety:`, `Categories:` and `Refusal:` from Qwen/Qwen3Guard-Gen-4B response moderation.
- `string_refusal(text, lang) -> bool`: seed patterns for en / yo / pcm from `configs/refusal_patterns.yaml`,
  flagged as needing native review.
- `lang_id(texts) -> list[tuple[str, float]]`: GlotLID through fasttext, imported lazily, falling back to a simple
  heuristic when unavailable.
- `three_way(kind, guard_safety, guard_refusal, string_refused, lid_lang, expected_lang, text) -> str`.
- `cohen_kappa(a, b) -> float`.

### `stats.py`
- `cluster_bootstrap(stat_fn, df, cluster_col, n_boot=2000, seed=0) -> (estimate, lo, hi)`.
- `spearman(x, y)`, `partial_spearman(x, y, z)`, and `auroc(scores, labels)`, plus cluster-bootstrap wrappers.
- `holm(pvals) -> list[float]`, `mcnemar_exact(b, c) -> float`, `wilson(k, n, alpha=0.05) -> (lo, hi)`.
- `mde_spearman(n_clusters, power=0.8, alpha=0.05) -> float`.

### `prompts.py`
- `load_panel(name) -> list[dict]`, `write_panel(rows, name)`, `split_rows(rows, seed)`.
- Panel builders for JailbreakBench (harmful, benign and judge_comparison), XSTest safe, TukaBench Yoruba, and the
  native Yoruba / Pidgin sets (read from `data/native/final/*.csv` once the human has post-edited them).

### `scoring.py` (weights-only scores of local adapters; added at integration)
- `load_probe_artifacts(folder, lang="en", condition="shipped") -> ProbeArtifacts`: reads `jobs/job_probe.py` output
  (`directions_{condition}`, keys `L{layer}`; `input_means_{lang}`, keys `L{layer}.{module}`; `sketch_{lang}_{condition}`
  and optional `grad_{lang}_{condition}`, `grad|sketch_irrelevant` in `grad.save` format).
- `find_adapters(root) -> list[(alias, folder)]`, `score_adapter(adapter, art, danger=None) -> dict`,
  `score_folder(root, art, danger=None, n_random=0, seed=0) -> (scores DataFrame, random DataFrame | None)`.
  Every score column is drift-oriented (higher = predicted more harmful).

### `analysis.py` (pre-registered analyses; added at integration)
- `analyze(scores, t1_rows, label_rows, manifest_rows, confirmatory, d1_ids, y2_ids, random=None, n_boot=10000,
  seed=0, n_splits=100) -> (results dict, per-adapter DataFrame)`: outcome table and C1 categories, D1 / D1p / −Y2
  from T1 rows, H1–H3 with author-cluster bootstrap and Holm, the verdict, the random-direction bar, the noise
  ceiling, AUROC, linearity and the C1 table. `render_markdown(results) -> str`.

### `cli.py`
- `weighbridge snapshot | panels | score | analyze | repair`: thin wrappers over the modules above.
