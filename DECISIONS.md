# Decisions log

Every design decision, with its date and reason. After `PREREG.md` is committed, every change to it is logged here
too, with what was changed, why, and whether any outcome data had been seen at the time (`PREREG.md` §19).

Dates are 2026, West Africa Time.

---

## 2026-11-14: champion choice

**Decision.** Build Weighbridge: test whether a Tiny Aya fine-tune's safety can be screened without running it, on
every public adapter of Tiny Aya Global, with Yoruba and Nigerian Pidgin ground truth and a training-free repair.

**How it was chosen.**
- Scouts read the six kits, the models, the rules, the public entries and the literature.
- Fourteen candidate ideas were scored by a simulated five-judge panel (LLM agents) modelled on the real judging pool:
  a Lexsi researcher, a Cohere Aya researcher, a sceptical academic, a start-up founder and a Hugging Face engineer.
  The top five:

| Idea | Panel score /100 |
|---|---|
| Weighbridge (screening public adapters from their weights) | 76 |
| Do Yoruba and Pidgin safety fixes work by replying in English? | 70 |
| How small can a phone-sized Tiny Aya get before Yoruba and Pidgin safety breaks? | 67 |
| Plant known safety faults; which evaluations catch them? | 67 |
| Does safety hold for Yoruba typed without tone marks? | 61 |

**Why.**
- The field was saturated. The Nigerian-language gap with repair, circuit-restricted repair, 4-bit survival,
  abstention probes and reply-language drift were each already taken by several entries.
- Weighbridge rests on two things other teams structurally could not do: a study of the whole population of public
  adapters, and native-speaker ground truth in Yoruba and Pidgin.
- Its question is useful whatever the answer: a deployer either gains a CPU screen or learns that weights alone are
  not enough.
- Fallback, if the infrastructure fails or neither score tracks the sentinels: switch to the runner-up, which reuses
  the prompts, refusal directions, runner and native labels.

## 2026-11-14: four adversarial checks and their effect

Four independent checks attacked the champion before any code was written. None found a fatal problem. Each found
something fixable, and together they changed the headline, the outcome, the scope and the publication rules.

| Check | Score /10 | Main finding | What changed |
|---|---|---|---|
| Prior art | 5 (about 7 within the hackathon) | Population studies of real derivatives exist (Khan et al., 31 derivatives; Yoshida et al., 213 Hub adapters). Weights-only screening exists for backdoors and for self-trained adapters; SQSD already does first-order LoRA projection. Still open: a label-free weights-only score tested on adapters other people released, and drift in Yoruba and Pidgin. | The headline became the screening test. We never claim a first population study, a first prevalence estimate or a first weights-only screen. Prevalence became the secondary result. |
| Signal | 5.5 (6.5 with fixes) | Few benign adapters will drift (about 10–15%). The weights-only score is likely to fail (about 55–60%); one forward pass is likely to work (about 70%). Weights-only predictors have failed across training methods before. | A continuous outcome with Spearman and an author-cluster bootstrap; a kill rule on the interval's lower bound; three-way labels with a format-collapse category; at most three confirmatory tests; a linearity probe to explain a failure; in-house sentinels as known positives. |
| Engineering | 5 (7 once trimmed) | Compute fits (about $10–13 of $20), but the full plan needed 37–40 work-hours with about 41 left. vLLM with LoRA on Tiny Aya is untested in public; fp16 numerics on a T4 are unknown; no credible Yoruba or Pidgin judge exists. | Hard scope cuts (below). All GPU work in bf16 on A10G. vLLM only after a parity test against PEFT. Native labels are the Yoruba and Pidgin ground truth. |
| Norms | 6 | About 75% of the adapters are other entrants' work during double-blind judging. Hashed ids do not anonymise them: the list comes back from one API call. Repaired copies cannot be republished without naming their authors. | Aggregate-only results with at least 5 authors per cell until final results; no gated repositories; no redistribution; a private manifest with a salted commitment; public repairs on our own adapters only; a private question to the organizers; author notice and opt-out afterwards (`docs/ETHICS.md`). |

## 2026-11-14: primary outcome

**Decision.** The primary outcome Y is the change in harmful-compliance rate (points), judged by Qwen3Guard-Gen-4B
after validation at κ ≥ 0.7 against JailbreakBench's human labels. The change in first-token margin is the secondary
outcome Y2.

**Why.** The approved draft made the margin primary because it needs no judge. The final design kept the outcome
continuous but defined it on harmful compliance, which is what a deployer cares about, and which keeps format
collapse from being counted as safety drift. Y2 stays as the judge-free check, and a string matcher with human
adjudication is the fallback if the judge fails validation.

## 2026-11-14: scope for Stage 1

**In:** about 113 confirmatory adapters (at most 2 per author) in English with and without the preamble, plus Yoruba
and Pidgin; D0a, D0b (exact and sketch), D1; every baseline and control; H1–H3; the cost ladder; the linearity probe
on 20 adapters; the C1 table; C3 on sentinels in public and on at most 5 third-party adapters in private; native
labels; one q4_k_m row for the base model.

**Cut to Stage 2 or dropped:**
- the CircuitKIT-restricted score and the Space;
- second-order scores (Safe Delta, Gauss–Newton) and full-rank delta repairs;
- base models other than Tiny Aya Global, and Aya Expanse;
- rebuilding adapters stacked on private checkpoints (excluded instead);
- gated adapters;
- the no-preamble condition in Yoruba and Pidgin;
- per-adapter publication;
- any sentinel trained on harmful data; the narrow customer-support sentinel of the draft plan;
- WildGuard as a second judge (gated; it needs an access request and a token we do not have yet);
- vLLM, unless it passes parity with PEFT within about an hour.

**Minimum acceptable cut**, if time runs short: at least 60 third-party adapters from at least 35 authors in English
with the shipped template; D0b, D1, log‖ΔW‖ and the random-direction control with H1–H3; C1 in English; Yoruba and
Pidgin on at least 30 adapters with at least 120 native labels; C3 on 2 sentinels against WiSE-FT and a random
projection.

## 2026-11-14: sentinels

**Decision.** Four in-house adapters, trained with AlignTune on benign data only, LoRA rank 16 on the seven target
modules: S1a and S1b (Alpaca-style SFT at 250 and 1000 steps), S2 (GSM8K) and S3 (Aya dataset, Yoruba).

**Why.** They give known answers for the meter sanity check, a tuning set for the repair rank k that never touches the
confirmatory adapters, and adapters we may repair and publish without involving anyone else's work.

## 2026-11-14: SafeLoRA baseline uses Global − Base

**Decision.** The SafeLoRA baseline measures overlap with the alignment update, Tiny Aya Global minus Tiny Aya Base,
as SafeLoRA is designed.

**Why.** The draft plan built the danger direction from an in-house adapter trained on harmful data. That sentinel was
cut, so the baseline uses the published models only.

**Known risk.** The draft plan noted that Global − Base had broken other teams' repairs. SafeLoRA is still run as
designed, both as a baseline and as a repair comparator, and reported as it falls.

## 2026-11-14: engineering defaults

- Revisions pinned in `src/weighbridge/common.py`: Tiny Aya Global `af89d21`, Base `d62ac17`, GGUF `a602ea7`.
- All GPU generation and forward passes in bf16; all scoring math in float32. A T4 (fp16 only) is used only after an
  fp16-versus-bf16 parity check.
- Generation always stops on tokens 3 and 6: chat turns end with 6, not with the end-of-sequence token 3.
- Private material lives in `PRIVATE_DIR`, outside the repository, so it cannot be committed by accident.
- Unit tests use a tiny random Cohere2 model, because the real model is gated and not on this machine.

## 2026-11-14: etiquette ticket to the organizers

**Decision.** Ask the organizers privately, through a support ticket in the hackathon's `#report_issues` channel,
before publishing any analysis of other entrants' adapters. The entrant files it; we do not post anywhere ourselves.

**Text:**

> Hi organizers, a quick rules question before we start. Our Track 2b entry would measure safety drift across public
> LoRA adapters of Tiny Aya Global on the Hub, which include some other participants' hackathon adapters. We would
> download only public, ungated repos and never redistribute anyone's weights or outputs. Until final results are
> announced we would publish only aggregate results, with no adapter names or per-adapter scores, and we would contact
> authors before naming any adapter afterwards. Is this allowed under the originality and conduct rules? If you
> prefer, we can leave out every adapter created since Nov 1, or share the full list privately with staff. Thank you!

**Status.** Drafted 2026-11-14; not yet filed at the time of this entry. [Date filed; reply and date.]

**If there is no reply** within 24 hours of filing: keep the aggregate-only default, which is already our rule, and
say in the write-up that a ruling was requested. **If the organizers object:** switch to the fallback population in
`PREREG.md` §4 and log it here.

## 2026-11-14: compute moves to free Kaggle GPUs (before the pre-registration commit)

**Decision.** Run every GPU stage on Kaggle's free T4 x2 instead of Hugging Face Jobs, in float16, with the PEFT
runner; drop vLLM for Stage 1.

**Why.** The Hugging Face account has no prepaid credit, so Jobs refuse to start (HTTP 402), and the entrant chose
free compute only. Kaggle hosts Tiny Aya itself (`cohere-labs/tiny-aya`, variant `global`); its two weight shards
have the same sizes as the pinned Hub revision, and every Kaggle run checks their SHA-256 against the Hub's values
before doing anything else. A T4 has no bfloat16, so the parity check compares float16 on the T4 with float32 on the
CPU. Base-model-only work and all analysis also run on the entrant's Mac.

## 2026-11-14: opener sets are calibrated on the base model (before the pre-registration commit)

**Decision.** The English refusal and compliance opener sets that define the margin (Y2, D1 and the gradient behind
D0b) are calibrated by a fixed rule on the base model's own greedy openings on screen prompts, as written in
`PREREG.md` section 8, instead of being hand-picked.

**Why.** On the real tokenizer the seed lists contain ambiguous first tokens (for example "I", "#", "**" and a bare
space), which would make the margin noisy. The rule uses only the base model and screen prompts, so no outcome data
can influence it.

## 2026-11-14: hub.py changed after the snapshot

`hub.manifest_commitment` (the salted manifest commitment) was added to `hub.py` after the snapshot was taken. The
snapshot logic did not change, but `code_version` in `manifest_meta.json` and in `docs/population_summary.md` refers to
the snapshot-time hash of `hub.py` (sha256 prefix 120125a2bb94), not to the committed file.

---

## Deviations after the pre-registration commit

| Date | Change | Reason | Outcome data seen? |
|---|---|---|---|
| 2026-11-14 | Sentinel training (`configs/sentinels.yaml`): micro-batch 1 × 8 accumulation instead of 4 × 2 (same effective batch of 8, same steps, rows and seed); float16 mixed precision with gradient checkpointing on the T4, bfloat16 where the GPU has it; `job_sentinels.py` can keep adapters locally and take a local base model. | Sentinels train on Kaggle's free T4s, which have 16 GB and no bfloat16, and no Hub token. With a 262k-token vocabulary the logits of four 1,024-token rows alone need several GB. | No. No sentinel had been trained and no adapter output had been measured. |
| 2026-11-14 | Added `data/panels/spot_check.jsonl` (PREREG §7 and §10; SHA-256 `68853b9fae0cb4350d51ee8929c740eac27252c513f92f6bd21e4d50c86630b9`), which the pre-registration commit had listed but not built: the 25 harmful prompts left in the seed-shuffled, deduplicated AdvBench pool after the frozen panels, plus 25 MaliciousInstruct questions (`walledai/MaliciousInstruct@4669349`, CC-BY-SA-4.0) that are not near duplicates of any panel or judge goal, drawn with seed 0. The seven frozen panels are byte-identical and the leak check passes. | The AdvBench pool ran out at 25, and PREREG §7 forbids a prompt in two panels, so the spot-check could not reuse fitting prompts. | No. No adapter had been run on any panel. |
| 2026-11-15 | Pilot gate (PREREG §17), instruments only: base harmful compliance 0.7% (shipped) and 4.0% (no preamble); over-refusal 13.0% / 15.0%; on the 15 pilot adapters Y ranged from −0.7 to +95.3 points (shipped), with 2 'more harmful' (shipped) and 1 (no preamble). The judge passed validation: κ = 0.757 against JailbreakBench human labels (accuracy 0.88). Parity passed (above). Decision: go, with the design unchanged. No score was compared with any outcome. | Pre-registered gate. | Pilot outcomes only, as §17 allows; no score–outcome relation was examined. |
| 2026-11-15 | Pilot instruments (engineering only): Kaggle's preinstalled torchao 0.10 stopped PEFT from loading adapters, so every plan uninstalls it; the first judge-validation run read the wrong label keys and scored every item "not harmful" (κ = 0), a script bug since fixed with a regression test; a raw-output check showed Qwen3Guard-Gen-4B gives correct verdicts in float16 on a T4 (no NaN or inf), about 6× faster than bfloat16, so the judge stays in float16 there. Parity: float16 on the T4 versus float32 on the CPU, mean absolute margin difference 0.005 (max 0.012, Pearson 0.99998), within PREREG's 0.1 tolerance. | Found while running the pilot. | No adapter outcome had been computed: no adapter generation had been judged. |
| 2026-11-14 | Operational detail of the fitted baselines (PREREG §10), fixed before any outcome exists: both ridges are standardised per training fold with the penalty chosen on the training fold by leave-one-out over {0.01, 0.1, 1, 10, 100, 1000} (`analysis.ridge_loao`). "Stratified by training method" for the spectral ridge is implemented as training-method indicators added to the spectral features, since several methods have too few adapters for separate fits. The spot-check runs as its own generation run, so its harmful prompts never enter Y. | PREREG named the baselines but not these settings. | No. |
| 2026-11-15 | Native panels built from the native speaker's post-edited files (`data/panels/yo_native.jsonl` SHA-256 `648e45088d0e41a1b680204478d57c6c118f87f11c1ff78d2eee43ab0c71e351`, `pcm_native.jsonl` `92c0899ac77994a2300f38e81d1dab2f3bba4744a6ae75bc1863e3e17a606f7e`; 75 rows each, all kept; 3 Pidgin rows rewritten, 2 Yoruba rows annotated). Both leak checks now recognise a translation and its English source row as designed twins: a native row names the English row's id as its pair_id, while the English row keeps its JailbreakBench pair_id, so the pair had been missed and two Pidgin prompts that keep English words ("insider trading", "endangered animals") were flagged against their own sources. Pairs without that link are still checked. The frozen panels are byte-identical. | PREREG §7 already exempts a prompt and its translation; the checker did not implement it for the native files. | No. No adapter had been run on a native panel. |
