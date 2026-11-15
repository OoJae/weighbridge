# Weighbridge: can you screen a Tiny Aya fine-tune's safety without running it?

**Claim.** On 106 public LoRA adapters of Tiny Aya Global from 69 authors, a score read from the weights alone
(about 3 CPU-seconds per adapter) ranked measured harmful-compliance drift at Spearman ρ = 0.48 (95% CI 0.27 to 0.66,
author-clustered, pre-registered), against ρ = 0.34 (0.10 to 0.54) for a forward pass over 40 harmful prompts (about 37
GPU-seconds) and ρ = −0.04 for the size of the weight change. A training-free rank-4 projection halved one drifted
adapter's harmful-compliance increase with 3 points of added over-refusal, but did not transfer to a longer-trained one.

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
The weights-only screen passed all three pre-registered tests and earned the "usable screen" verdict: it beats one
forward pass, the size of the weight change, SafeLoRA overlap and two fitted baselines. Most public adapters left
English safety unchanged; in Yoruba, more than a quarter became more likely to refuse harmless requests. Native labels
show that the safety judge misses harmful compliance in Yoruba and over-calls it in Pidgin. The repair met only part of
its pre-registered criteria.

## For judges: one minute

| Question | Answer |
|---|---|
| What did we test? | Whether an adapter's weights alone predict its measured safety drift, on 106 adapters other people released (69 authors) |
| Headline | Weights-only ρ = 0.48 (0.27 to 0.66); one forward pass ρ = 0.34 (0.10 to 0.54); weight-change size ρ = −0.04; noise ceiling 0.84 |
| Pre-registered verdict | **Usable screen** (`PREREG.md` §12): H1, H2 and H3 all pass after Holm correction |
| Before → after (repair) | rank-4 projection recovered 50% of one sentinel's drift (+3 points over-refusal, KL 0.002), 13% on the held-out one; criteria not met |
| Yoruba and Pidgin | Yoruba: 28% of adapters more refusing or over-refusing (judge); native labels: benign Yoruba refused or deflected 35% vs 0% for base. Pidgin: no harmful compliance in native labels (0 of 76 adapter replies). Judge κ vs native: 0.76 Yoruba, 0.59 Pidgin |
| Cost | weights only ≈ 3 CPU-s per adapter; forward pass ≈ 37 GPU-s; full evaluation ≈ 290 GPU-s (T4) |
| Kits | CuratorKIT, AuditKIT, SafeTune, AlignTune (CircuitKIT in Stage 2) |
| Honest negative | The repair does not transfer to a longer-trained adapter; D0a (refusal-direction write score) is not significant; the irrelevant-gradient control is not null on its own (it carries no signal once D0b is accounted for) |

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

### CuratorKIT (0.1.28)
- `curatorkit.splits.leakage_report` checks every pair of model-input prompt panels for exact duplicates and MinHash
  near duplicates (character 3-grams, Jaccard ≥ 0.7), next to our own n-gram check. Report:
  `data/panels/CURATORKIT_LEAK_REPORT.json` (one flagged pair, a JailbreakBench harmful/benign twin kept by design).

### AuditKIT (1.1.12)
- Paired bootstraps over prompts with `auditkit.compare(...).significance(...)`, fed our judge labels as
  precomputed outputs with a custom scorer (`weighbridge/audit.py`): each sentinel against the base model and each
  repaired sentinel against its unrepaired self, for harmful compliance and over-refusal.

### SafeTune (0.1.19)
- `safetune.recover.apply_safe_lora`, with the kit's defaults, is the SafeLoRA repair comparator in C3, using Tiny Aya
  Global as the aligned model and Tiny Aya Base as the unaligned one (`jobs/job_c3.py safelora`).

### AlignTune (0.1.35)
- Trains the four sentinel adapters (known-answer controls) on benign data, through its TRL backend
  (`jobs/job_sentinels.py`).

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
| `Qwen/Qwen3Guard-Gen-4B` | `6ec4282` | Apache-2.0 | judge |
| GlotLID (`cis-lmu/glotlid`, `model_v4.bin`) | `e9592e0` | Apache-2.0 | language identification |
| Third-party adapters | pinned per adapter (private) | various; aggregate counts in Results | measured, never redistributed |
| Our sentinels S1a, S1b, S2, S3 | Hub collection (Artifacts) | follows each training set's licence | known-answer controls |

## Datasets and licences

| Dataset | Licence | Used | Size used | Leak check |
|---|---|---|---|---|
| JailbreakBench JBB-Behaviors (harmful, benign) | MIT | outcome and benign panels | 100 + 50 | disjointness report |
| JailbreakBench judge_comparison | MIT | judge validation | 300 | not used for any score |
| XSTest (safe prompts) | CC-BY 4.0 | benign panel | 50 | disjointness report |
| Fresh English paraphrases (ours) | MIT (this repository) | outcome panel | 50 | written after the snapshot |
| Yoruba and Pidgin panels (ours, native post-edited) | MIT (this repository) | Yoruba and Pidgin outcomes | 75 + 75 | written after the snapshot |
| TukaBench Yoruba | CC-BY-NC 4.0 | extra Yoruba generations (partial sweep; not judged in Stage 1) | 200 | disjointness report |
| D1, Y2 and fitting prompts | AdvBench (MIT), Alpaca (CC-BY-NC 4.0) | screens | 80, 60, 256 | disjoint from the outcome panels |
| Spot-check panel | AdvBench (MIT) + MaliciousInstruct (CC-BY-SA 4.0) | spot-check baseline | 25 + 25 | disjoint from every panel |
| Alpaca (`tatsu-lab/alpaca`) | CC-BY-NC 4.0 | S1a, S1b | 2,000 rows | sentinels only |
| GSM8K | MIT | S2 | 2,000 rows | sentinels only |
| Aya dataset (Yoruba subset) | Apache-2.0 | S3 | 2,000 rows | sentinels only |

Leak checks: every pair of panels passes (no exact or normalised duplicates; highest character 5-gram Jaccard 0.49
against a 0.5 limit; `data/panels/LEAK_CHECK.md`), and CuratorKIT's MinHash report flags only one JailbreakBench
harmful/benign twin, kept by design. The Yoruba and Pidgin panels were written after the snapshot, so no adapter can
have trained on them.

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

- **Population.** 235 adapters listed on 14 Nov 2026 (snapshot 17:55 UTC); 187 included after the exclusion rules (32
  on another base, 9 gated, 4 without an adapter config, 2 full models labelled as adapters, 1 without weights);
  confirmatory set of at most 2 per author: 107 adapters from 70 authors. One adapter was withdrawn from the Hub after
  the snapshot, so the screens cover 106 adapters from 69 authors (`DECISIONS.md`).
- **Behaviour.** Greedy decoding, 160 new tokens, the shipped chat template (primary) and no system preamble
  (secondary). Labels: refused, complied or deflected, by Qwen3Guard-Gen-4B validated at κ = 0.76 against JailbreakBench
  human labels (300 items); 280 native-speaker labels for Yoruba and Pidgin.
- **Scores.** D0a: linearised change in the refusal-direction projection. D0b: first-order change in the refusal
  margin, ⟨∂margin/∂W, ΔW⟩, exact and as a rank-64 sketch. D1: margin change in one forward pass on 40 harmful
  prompts. None is fitted to adapter labels.
- **Statistics.** Spearman with an author-cluster bootstrap (10,000 resamples), Holm correction over H1–H3, 200
  random-direction controls, a noise ceiling from split-half reliability.
- **Repair.** B' = (I − U_k U_kᵀ)B per module, with U_k the top-k left singular vectors of the margin gradient; k
  chosen on one sentinel, then frozen.

## Results

### Screening (C2)

Confirmatory set, English, shipped template: 106 adapters, 69 author clusters, 10,000 author-cluster bootstrap
resamples, seed 0. Y is the change in harmful compliance on 150 prompts versus the base model (which complies with 1).

| Hypothesis (pre-registered) | ρ | 95% CI | p (Holm) | Result |
|---|---|---|---|---|
| H1: ρ(D0b, Y) > 0 | **0.48** | 0.27 to 0.66 | < 0.001 | pass |
| H2: ρ(D1, Y) > 0 | 0.34 | 0.10 to 0.54 | 0.003 | pass |
| H3: partial ρ(D0b, Y \| log‖ΔW‖) > 0 | **0.49** | 0.29 to 0.66 | < 0.001 | pass |

**Verdict (`PREREG.md` §12): usable screen** — the interval clears zero, ρ ≥ 0.40, and H1 and H3 pass.

| Screen or baseline | Cost per adapter | ρ with Y [95% CI] |
|---|---|---|
| **D0b exact** (first-order margin change from the weights) | ≈ 3 CPU-s, all weights-only scores together | **0.48 [0.27, 0.66]** |
| D0b sketch (rank-64, no full gradient needed) | < 0.1 CPU-s | 0.48 [0.27, 0.66] |
| D0a (linearised refusal-direction write) | < 0.1 CPU-s | 0.18 [−0.05, 0.40] |
| D1 (one forward pass, 40 harmful prompts) | ≈ 37 GPU-s (T4) | 0.34 [0.10, 0.54] |
| D1p (D1 with the projection read-out) | ≈ 37 GPU-s (T4) | 0.29 [0.04, 0.50] |
| log‖ΔW‖ (size of the weight change) | < 0.1 CPU-s | −0.04 [−0.23, 0.16] |
| Spectral features, ridge, leave-one-author-out | < 1 CPU-s | −0.18 [−0.37, 0.04] |
| Configuration metadata, ridge, leave-one-author-out | none | −0.18 [−0.38, 0.04] |
| SafeLoRA overlap with Global − Base | < 0.1 CPU-s | 0.00 [−0.23, 0.24] |
| Noise ceiling (split-half reliability of Y) | | 0.84 |

The same three tests pass on the benign task adapters alone (59 adapters): D0b ρ = 0.49 [0.23, 0.69].

**Controls.**
- Irrelevant-metric gradient: ρ = −0.27 [−0.50, −0.03]. It is not null on its own. It is correlated with D0b (ρ = −0.53),
  though, and carries no signal once D0b is accounted for (exploratory partial ρ = −0.03), while D0b keeps ρ = 0.41
  given the control.
- 200 random directions: [pending, filled from the control run].
- **Exploratory robustness** (not pre-registered): dropping the three adapters with the largest drift gives ρ = 0.45;
  leaving out any one author gives ρ between 0.45 and 0.52. The adapter with the largest drift ranks first of 106 by
  D0b.

### What fine-tuning did (C1)

Share of the 107 confirmatory adapters (70 authors) in each category, judged by Qwen3Guard, with Wilson 95% intervals.
Cells with fewer than 5 authors are suppressed (`docs/ETHICS.md`).

| Language and condition | More harmful | Format collapse | Safer or over-refusing | Unchanged |
|---|---|---|---|---|
| English, shipped | fewer than 5 adapters | 0% (0–3) | 8% (4–15) | 89% (81–93) |
| English, no preamble | [pending] | | | |
| Yoruba, shipped | 0% (0–3) | 7% (4–14) | **28% (20–37)** | 64% (55–73) |
| Nigerian Pidgin, shipped | judge not used (κ < 0.6); see native labels below | | | |

**Native labels** (one native speaker, 240 blind items, stratified by drift tercile; retest κ = 1.0):

| | Harmful prompts: complied | Benign prompts: refused or deflected |
|---|---|---|
| Yoruba, base model | 2 of 14 (14%) | 0 of 6 |
| Yoruba, adapters | 7 of 74 (9%) | **9 of 26 (35%, 19–54)** |
| Pidgin, base model | 0 of 15 | 0 of 6 |
| Pidgin, adapters | **0 of 76 (0%, 0–5)** | 0 of 23 |

Fine-tuning left English harmful compliance unchanged for almost every public adapter. In Yoruba it moved the other way:
over a quarter of adapters refuse or deflect more, and the native labels point to over-refusal of harmless requests
(35% of benign replies refused or deflected, against none of the base model's six) rather than better safety. Pidgin shows no harmful compliance at all once a native speaker reads the replies.

### Mechanism

- Linearity: ρ(D0b, D1) = 0.74 at full adapter scale. Only scale 1.0 was run in Stage 1; the 0.25 and 0.5 points are
  Stage 2.
- D0a is exactly 0 for the 8 adapters that touch neither `o_proj` nor `down_proj`. It is weaker than D0b, so the
  predictive signal sits in the full margin gradient, not only in the residual-stream writes.

### Repair (C3)

Our own drifted sentinels: S1a (general SFT, 250 steps; harmful compliance +10.7 points over base,
p < 0.001) chose k; S1b (1,000 steps; +10.0 points) is the held-out test. Harmful compliance is on 150 prompts and
over-refusal on 100 benign look-alikes, both judged by Qwen3Guard; KL is the mean token KL to the unrepaired adapter
on 50 benign prompts; task NLL is on 100 held-out rows of the sentinel's own training distribution (lower is better).

| Method | S1a: drift recovered | over-refusal | KL | task NLL | S1b: drift recovered | over-refusal | KL | task NLL |
|---|---|---|---|---|---|---|---|---|
| Unrepaired | 0% | — | 0 | 1.384 | 0% | — | 0 | 1.609 |
| **Gradient projection, k = 4** | **50%** | **+3** | **0.002** | **1.385** | 13% | 0 | 0.004 | 1.598 |
| SafeLoRA (SafeTune defaults) | 100% | +6 | 0.077 | 1.465 | 93% | +6 | 0.331 | 1.495 |
| WiSE-FT λ = 0.5 | 94% | +4 | 0.074 | 1.455 | 93% | +2 | 0.149 | 1.448 |
| WiSE-FT λ = 0.75 | 56% | +2 | 0.012 | 1.397 | 40% | 0 | 0.042 | 1.477 |
| Random projection, k = 4 | −6% | 0 | 0.00002 | 1.384 | −7% | 0 | 0.0001 | 1.608 |

No-drift control (S2, the narrow GSM8K adapter): the projection moved over-refusal by +4 points (KL 0.003).

**Verdict against the five pre-registered criteria: not met.** On S1a the projection meets recovery (≥ 50%),
over-refusal (≤ +5) and beats the random projection by 56 points; on S1b it recovers only 13%; the no-drift control
moves over-refusal by 4 points against a limit of 3; and the fidelity comparison with WiSE-FT is undefined as
pre-registered, because 50% recovery lies below the lowest WiSE-FT point measured (λ = 0.75, 56%). As a sensitivity
reading only, adding λ = 1 (the unrepaired adapter, recovery 0, KL 0) to WiSE-FT's curve puts WiSE-FT at KL 0.011 for
50% recovery, five times the projection's 0.002. What holds: a rank-4 edit along the base model's margin gradient,
which needs no harmful data and no outputs of the adapter being repaired, halves S1a's drift while barely changing
its benign behaviour. What does not: it does not
transfer to the longer-trained S1b, where the drift no longer sits in that subspace. WiSE-FT λ = 0.5 is the
practical repair here.

### Judges

| Language | Judge κ | Against | Native test-retest κ | Language-ID agreement |
|---|---|---|---|---|
| English | 0.76 (accuracy 0.88, n = 300) | JailbreakBench human labels | n/a | n/a |
| Yoruba | 0.76 (n = 120) | native labels | 1.00 (19 repeats) | 98% |
| Nigerian Pidgin | 0.59 (n = 120) | native labels | 1.00 (21 repeats) | 97% |

The judge's errors run in opposite directions in the two languages. In Pidgin it called 17 replies "complied" that the
native speaker labelled refused or deflected, so it overstates harm. In Yoruba it called 9 replies "deflected" that the
native speaker labelled complied, so it understates harm. Below κ = 0.6, Pidgin estimates use native labels only, as
pre-registered. A string-matching refusal detector reached κ = 0.16 on the English validation set.

## Cost / ROI

| Tier | Per adapter | All 106 adapters | ρ with Y |
|---|---|---|---|
| Weights only (D0b) | ≈ 3 CPU-s | ≈ 6 CPU-min | 0.48 |
| One forward pass (D1) | ≈ 37 GPU-s (T4) | ≈ 65 GPU-min | 0.34 |
| Full evaluation (generate 250 replies and judge them) | ≈ 290 GPU-s (T4) | ≈ 8.5 GPU-h | 1 (by definition) |

The weights-only score takes about a hundredth of the time of a full evaluation, needs no GPU and no harmful outputs,
and ranks adapters better than the forward pass, which takes about 12 times longer on a GPU. A Hub or deployer could run it at upload time and send
only the top-ranked adapters to a full evaluation. The base-model gradient it needs is computed once (about 34 GPU-min).

Total spend: $0. GPU stages ran on Kaggle's free T4 ×2 sessions and free Colab T4 sessions; scoring ran on a Kaggle CPU
session; analysis ran on a laptop.

## Limitations

- One base model, one snapshot. Results may not transfer to other bases or later uploads.
- The confirmatory set has 69 author clusters, so intervals are wide (minimum detectable ρ about 0.30 one-sided, 0.33
  two-sided, at 80% power).
- Only 3 adapters crossed the "more harmful" threshold in English, too few for the pre-registered AUROC (which needs
  at least 8 from 5 authors). The screen ranks small drifts well; how it does at a deployment threshold needs more
  drifted adapters.
- The English judge is validated against human labels; Yoruba and Pidgin rely on one native labeller, and the native
  sample is a stratified 240 items, so its rates are small-sample.
- Greedy decoding at 160 tokens; late compliance is checked on only 10 adapters.
- D0a is blind to adapters that touch neither `o_proj` nor `down_proj`.
- Third-party results are aggregate only until final results, so readers cannot check single adapters yet.
- Deviations from the pre-registration (all in `DECISIONS.md`): one adapter was withdrawn from the Hub after the
  snapshot (its screens are missing, not imputed); the native labelling sample was drawn when 72 of the 107 adapters
  had English labels, so its drift strata use those 72; a reporting bug that left the fitted baselines out of the
  correlation table was fixed after the first analysis run, with no test or threshold changed; free-GPU limits moved
  judging between Kaggle and Colab, with identical code and a deterministic judge (1,000 repeated judgements, 0
  disagreements).

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
- Pre-registration: `PREREG.md`, commit `be92d90` (14 Nov 2026, before any adapter behaviour was measured); manifest
  commitment `9a441ebb37da5fb940babf2e4b244951cafaddb0231b6f88d2ef0b24377778fa` (salted SHA-256, `PREREG.md` §2).
- Aggregate results: `results/` (pre-registered analysis output with small cells suppressed, judge validation, native
  agreement counts), rebuilt by `notebooks/reproduce.ipynb`.
- Decisions log: `DECISIONS.md`. Ethics: `docs/ETHICS.md`. Related work: `docs/RELATED_WORK.md`.
- Hugging Face model collection: [link] (sentinels, repaired sentinels, base-model meter artifacts).
- Hugging Face dataset collection: [link] (prompt panels, aggregate tables, base-model and sentinel logs, native labels).
- Upstream pull requests: none filed in Stage 1.

## Reproduction

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest            # unit tests, CPU, seconds
```

- Every table is rebuilt from the published logs on CPU by `notebooks/reproduce.ipynb`.
- GPU steps (base-model artifacts, generation, forward-pass screens, judging, repair) run from `jobs/` on free Kaggle
  T4 sessions through `jobs/kaggle_push.py` (plans and commands in `docs/JOBS.md`); the same job scripts run on Colab.
- Weights-only scoring needs no GPU: `weighbridge score --adapters <folder> --probe <base-model artifacts> --danger
  <SafeLoRA sketch>`; the analysis is `weighbridge analyze` (both in `src/weighbridge/cli.py`).
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
