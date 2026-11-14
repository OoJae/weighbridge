# Weighbridge: can you screen a Tiny Aya fine-tune's safety without running it?

**Claim.** On [N] public LoRA adapters of Tiny Aya Global from [K] authors, a score read from the weights alone
([t] CPU-s per adapter) ranked measured harmful-compliance drift at Spearman ρ = [ρ_D0b] ([lo, hi], author-clustered),
against ρ = [ρ_D1] for one 40-prompt forward pass ([g] GPU-s) and ρ = [ρ_size] for weight-delta size. A training-free
projection cut a drifted adapter's harmful compliance from [x]% to [y]%, with over-refusal [a]% → [b]%.

> Content warning: this project studies requests for harmful content. Harmful completions are never shown; they
> appear only as labels and hashes.

## Tracks and problem statements

- **Track 2b**, problem statement: *repairing safety drift without retraining.*
- **Track 2b**, problem statement: *locating where safety lives and making a targeted fix.*

## Abstract

People publish LoRA adapters of Tiny Aya every day, and anyone deploying one needs to know whether it is still safe.
Running a full safety evaluation for each adapter is slow and needs a GPU. We ask whether cheaper screens work. We
froze a snapshot of every public, ungated LoRA adapter of Tiny Aya Global, measured each one's change in harmful
compliance against the base model, and tested three screens against it: a weights-only score that runs on a laptop
CPU, a single forward pass with no generation, and the size of the weight change. The analysis was pre-registered
before any behaviour was measured, including a rule that kills the weights-only screen if its interval touches zero.
We also report what fine-tuning did to safety in English, Yoruba and Nigerian Pidgin, with native-speaker labels, and
we test a training-free repair that projects the drifted direction out of an adapter while keeping it a LoRA.
[Result in one sentence per claim.]

## For judges: one minute

| Question | Answer |
|---|---|
| What did we test? | Whether an adapter's weights alone predict its measured safety drift, on [N] adapters other people released |
| Headline | Weights-only ρ = [..] ([lo, hi]); one forward pass ρ = [..] ([lo, hi]); noise ceiling [..] |
| Pre-registered verdict | [usable screen / triage only / kill] (`PREREG.md` §12) |
| Before → after (repair) | harmful compliance [x]% → [y]%; over-refusal [a]% → [b]%; task score [c] → [d] |
| Yoruba and Pidgin | [p]% / [q]% of task adapters more harmful; judge κ vs native labels [..] / [..] |
| Cost | weights only [t] CPU-s per adapter; forward pass [g] GPU-s; full evaluation [G] GPU-s |
| Kits | CuratorKIT, AuditKIT, SafeTune, AlignTune (CircuitKIT in Stage 2) |
| Honest negative | [the result that did not work] |

## Problem and approach

**Why it matters.** Anyone deploying a community adapter needs to know whether it is safe. Fine-tuning on harmless
data can erode a model's refusals (Qi et al. 2023), and a public Hub listing gives no warning. A full evaluation of
every adapter costs a GPU and a judge model per adapter. A screen that works from the weights alone could run at
upload time, on a CPU, for every adapter.

**Approach.**
1. Freeze every public, ungated LoRA adapter of `CohereLabs/tiny-aya-global` at a pinned revision.
2. Measure each adapter's behaviour: harmful compliance on 150 English prompts, over-refusal on 100 benign look-alikes,
   and 75 prompts each in Yoruba and Pidgin.
3. Score each adapter without generating: from its weights (D0a, D0b), and from one forward pass (D1).
4. Test, with author-clustered intervals and a pre-registered kill rule, whether the scores rank the measured drift.
5. Repair drifted adapters by projecting the top directions of the refusal-margin gradient out of each module's B
   factor, so the result is still a LoRA of the same rank.

What is and is not new is set out in `docs/RELATED_WORK.md`.

## Kits used

### CuratorKIT
- The adapter manifest as a dataset, with counts before and after each exclusion rule.
- Prompt panels: exact, MinHash and embedding deduplication, and leak checks between every panel and its sources.
- Author-grouped splits with a leakage report, for the leave-one-author-out baselines.
- Pinned commit: [sha].

### AuditKIT
- Rescoring of saved generations with its guard judge (Qwen3Guard) and multilingual refusal scorer.
- Judge calibration against human labels (κ).
- Paired bootstrap per adapter and per-language deltas.
- Logs pushed to the Hub with provenance, for the base model and our own adapters only.
- Pinned commit: [sha].

### SafeTune
- Refusal directions (the activation-space reference for D0a and D1p).
- SafeLoRA and WiSE-FT, as the repair comparators and the SafeLoRA screening baseline.
- Over-refusal evaluation.
- Pinned commit: [sha].

### AlignTune
- Trains the four sentinel adapters (known-answer controls) on benign data, and exports GGUF.
- Pinned commit: [sha].

### CircuitKIT (Stage 2)
- A version of D0b restricted to the units CircuitKIT attributes to refusal.

### How they connect

```
 Hub snapshot ──► CuratorKIT ──► manifest (private) + prompt panels (public) + author splits
                                     │                     │
                                     ▼                     ▼
              weights-only meters (D0a, D0b)      runners + AuditKIT judges ──► Y, Y2, C1
              forward-pass meter (D1)                       │
                                     │                     │
                                     └──────► statistics ◄─┘  (H1–H3, controls, noise ceiling)
                                                  │
 AlignTune sentinels ──► known answers ───────────┤
                                                  ▼
                         SafeTune (SafeLoRA, WiSE-FT) vs gradient projection ──► repair (C3)
```

## Models and licences

| Model | Revision | Licence | Role |
|---|---|---|---|
| `CohereLabs/tiny-aya-global` | `af89d21` | CC-BY-NC 4.0 + Cohere acceptable-use addendum | base model |
| `CohereLabs/tiny-aya-base` | `d62ac17` | CC-BY-NC 4.0 + Cohere acceptable-use addendum | Global − Base for the SafeLoRA baseline |
| `CohereLabs/tiny-aya-global-GGUF` | `a602ea7` | CC-BY-NC 4.0 + Cohere acceptable-use addendum | q4_k_m check |
| `Qwen/Qwen3Guard-Gen-4B` | [sha] | Apache-2.0 | judge |
| GlotLID | [sha] | [licence] | language identification |
| Third-party adapters | pinned per adapter (private) | various; aggregate counts in Results | measured, never redistributed |
| Our sentinels S1a, S1b, S2, S3 | [links] | [licence] | known-answer controls |

## Datasets and licences

| Dataset | Licence | Used | Size used | Leak check |
|---|---|---|---|---|
| JailbreakBench JBB-Behaviors (harmful, benign) | MIT | outcome and benign panels | 100 + 50 | disjointness report |
| JailbreakBench judge_comparison | MIT | judge validation | [n] | not used for any score |
| XSTest (safe prompts) | CC-BY 4.0 | benign panel | 50 | disjointness report |
| Fresh English paraphrases (ours) | [licence] | outcome panel | 50 | written after the snapshot |
| Yoruba and Pidgin panels (ours, native post-edited) | [licence] | Yoruba and Pidgin outcomes | 75 + 75 | written after the snapshot |
| TukaBench Yoruba | CC-BY-NC 4.0 | [role, if used] | [n] | disjointness report |
| D1, Y2 and fitting prompts | [sources and licences] | screens | 80, 60, [n] | disjoint from the outcome panels |
| Alpaca-style SFT data | [licence] | S1a, S1b | [n] | sentinels only |
| GSM8K | MIT | S2 | [n] | sentinels only |
| Aya dataset (Yoruba subset) | Apache-2.0 | S3 | [n] | sentinels only |

Leak checks: [summary of the CuratorKIT disjointness report]; public-versus-fresh compliance gap [..] points.

## Metrics and why

| Metric | Direction | Why |
|---|---|---|
| Harmful-compliance change Y (points) | ↓ | what a deployer cares about; separates compliance from format collapse |
| First-token margin change Y2 | ↑ | needs no judge |
| Over-refusal on benign look-alikes | ↓ | a fix that refuses everything is not a fix |
| Deflection | reported | format collapse is not safety |
| Spearman ρ, partial ρ | ↑ | works with few drifted adapters; partial ρ removes weight-delta size |
| AUROC (only with ≥ 8 positives from ≥ 5 authors) | ↑ | screening at a threshold |
| Cost per adapter (CPU-s, GPU-s) | ↓ | the point of a screen |
| Drift recovered, KL to the unrepaired adapter, task score | ↑, ↓, ↑ | repair quality |
| Judge κ per language | ↑ | trust in labels |

## Method

Full details, thresholds and the analysis plan are in `PREREG.md`, committed before any behaviour was measured.

- **Population.** [N_total] adapters listed; [N_incl] included after the exclusion rules; confirmatory set of at most 2
  per author ([N] adapters, [K] authors).
- **Behaviour.** Greedy decoding, 160 new tokens, the shipped chat template (primary) and no system preamble
  (secondary). Labels: refused, complied or deflected, by Qwen3Guard-Gen-4B validated at κ = [..] against JailbreakBench
  human labels; native-speaker labels for Yoruba and Pidgin.
- **Scores.** D0a: linearised change in the refusal-direction projection. D0b: first-order change in the refusal
  margin, ⟨∂margin/∂W, ΔW⟩, exact and as a rank-64 sketch. D1: margin change in one forward pass on 40 harmful
  prompts. None is fitted to adapter labels.
- **Statistics.** Spearman with an author-cluster bootstrap (10,000 resamples), Holm correction over H1–H3, 200
  random-direction controls, a noise ceiling from split-half reliability.
- **Repair.** B' = (I − U_k U_kᵀ)B per module, with U_k the top-k left singular vectors of the margin gradient; k
  chosen on one sentinel, then frozen.

## Results

### Screening (C2)

| Score | Cost per adapter | ρ with Y [95% CI] | Partial ρ given size | Beats random 95th pct | Verdict |
|---|---|---|---|---|---|
| D0b exact | [..] | [..] | [..] | [..] | [..] |
| D0b sketch (CPU) | [..] | [..] | [..] | [..] | |
| D0a | [..] | [..] | | [..] | |
| D1 (one forward pass) | [..] | [..] | | [..] | |
| log‖ΔW‖ | [..] | [..] | | | |
| Spectral (leave-author-out) | [..] | [..] | | | |
| Metadata (leave-author-out) | [..] | [..] | | | |
| SafeLoRA overlap | [..] | [..] | | | |
| 50-prompt spot-check | [..] | [..] | | | |
| Noise ceiling | | [..] | | | |

Controls: random directions 95th percentile ρ = [..]; irrelevant-metric gradient ρ = [..]; zero adapter [..]; base
re-run noise [..] points.

### What fine-tuning did (C1)

| Language and condition | More harmful | Format collapse | Safer or over-refusing | Unchanged | Adapters / authors |
|---|---|---|---|---|---|
| English, shipped | [..]% [lo, hi] | [..] | [..] | [..] | [..] |
| English, no preamble | [..] | [..] | [..] | [..] | [..] |
| Yoruba | [..] | [..] | [..] | [..] | [..] |
| Nigerian Pidgin | [..] | [..] | [..] | [..] | [..] |

Every cell covers at least 5 authors.

### Mechanism

- Linearity probe: ρ(D0b, D1) at 0.25 / 0.5 / 1.0 × ΔW = [..] / [..] / [..].
- Where the damage sits: [layers and module types].

### Repair (C3)

| Method | Drift recovered | Over-refusal change | KL to unrepaired | Task score |
|---|---|---|---|---|
| Gradient projection (k = [..]) | [..] | [..] | [..] | [..] |
| SafeLoRA | [..] | [..] | [..] | [..] |
| WiSE-FT λ = 0.5 | [..] | [..] | [..] | [..] |
| Random projection | [..] | [..] | [..] | [..] |
| No-drift control | n/a | [..] | [..] | [..] |

### Judges

| Language | Judge κ | Against | Native test-retest κ |
|---|---|---|---|
| English | [..] | JailbreakBench human labels | n/a |
| Yoruba | [..] | native labels | [..] |
| Nigerian Pidgin | [..] | native labels | [..] |

## Cost / ROI

| Tier | Per adapter | Whole snapshot | ρ |
|---|---|---|---|
| Weights only | [..] CPU-s | [..] | [..] |
| One forward pass | [..] GPU-s | [..] | [..] |
| 50-prompt spot-check | [..] GPU-s | [..] | [..] |
| Full evaluation | [..] GPU-s | [..] | 1 (by definition) |

Total spend: [$..] of $20 in credits. [One sentence on what a deployer saves.]

## Limitations

- One base model, one snapshot. Results may not transfer to other bases or later uploads.
- The confirmatory set has about [K] author clusters, so intervals are wide (minimum detectable ρ about 0.30 one-sided, 0.33 two-sided, at 80% power).
- The English judge is validated against human labels; Yoruba and Pidgin rely on one native labeller.
- Greedy decoding at 160 tokens; late compliance is checked on only 10 adapters.
- D0a is blind to adapters that touch neither `o_proj` nor `down_proj`.
- Third-party results are aggregate only until final results, so readers cannot check single adapters yet.
- [Deviations from the pre-registration, from `DECISIONS.md`.]

## With more credits

| Step | Expected before → after |
|---|---|
| Prospective test: commit hashed D0 and D1 predictions for adapters uploaded after the snapshot, then measure them | out-of-sample ρ_D1 0.5–0.7; ρ_D0b 0.1–0.4 |
| Wider population: all Tiny Aya bases (about 290 adapters) | about 72 → about 100 author clusters; interval half-width about ±0.20 → ±0.17 |
| Second-order variant (Gauss–Newton, Safe Delta cost) | tests whether curvature rescues the weights-only screen |
| CircuitKIT-restricted D0b | expected null |
| Repair at scale, with q4_k_m re-checks | 50–80% of drift recovered at ≤ 5 points of extra over-refusal; GGUF within ±3 points of bf16 |
| Author notice and opt-out, then per-adapter release | per-adapter rows for authors who do not opt out |

## Artifacts

- Code: this repository (MIT).
- Pre-registration: `PREREG.md`, commit [sha]; manifest commitment [SHA-256].
- Decisions log: `DECISIONS.md`. Ethics: `docs/ETHICS.md`. Related work: `docs/RELATED_WORK.md`.
- Hugging Face model collection: [link] (sentinels, repaired sentinels, base-model meter artifacts).
- Hugging Face dataset collection: [link] (prompt panels, aggregate tables, base-model and sentinel logs, native labels).
- Upstream pull requests: [links, if filed].

## Reproduction

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest            # unit tests, CPU, seconds
```

- Every table is rebuilt from the published logs on CPU by `notebooks/reproduce.ipynb`.
- GPU steps (base-model artifacts, generation, forward-pass screens, repair) run from `jobs/`: [commands].
- Third-party adapter results need the private manifest and cannot be rebuilt from public files; the aggregate tables
  and the manifest commitment are published instead.

## Declaration of outside work and LLM usage

- **Outside work.** All code, data and analysis were produced during the hackathon. Kits, models and datasets are
  credited above. We built on ideas from two other entries, credited without verdicts in `docs/RELATED_WORK.md` §5.
- **LLM usage.** Anthropic's Claude (through Claude Code) was used to survey prior work, write code and tests, draft
  the Yoruba and Pidgin prompts (post-edited by a native speaker) and draft this write-up. Every number comes from a
  logged run. A human chose the design, labelled the Yoruba and Pidgin outputs and reviewed the claims.

## Teammates

Solo entry. Name withheld for double-blind review.
