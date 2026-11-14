# Related work

Weighbridge sits where three lines of work meet: safety drift after benign fine-tuning, studies of real-world
derivative models, and screening adapters from their weights. Each line is published. This page says what we take
from each, and states exactly what is and is not new here.

## 1. Benign fine-tuning erodes safety

- **Qi et al. 2023** (arXiv 2310.03693). Fine-tuning an aligned model on benign data raises harmfulness. On
  Llama-2-7b-Chat, one epoch of full fine-tuning on Alpaca took the harmfulness rate from 0.3% to 16.1%; Dolly 0.6% to
  12.1%. This is the effect our outcome Y measures.
- **He et al. 2024** (arXiv 2404.01099). Which benign examples do the damage can be predicted from gradients: the 100
  examples most gradient-similar to harmful anchors gave 66.5% attack success on Llama-2-7B-Chat, against 13.0% for
  100 random ones. Gradient similarity to a safety signal is the idea behind our first-order score D0b.
- **Lyu et al. 2024, PTST** (arXiv 2402.18540). A safety prompt protects most when it is absent during training and
  present at test time. Tiny Aya's chat template always injects a safety preamble, so we test with it (primary) and
  without it (secondary), and record each adapter's training format.
- **Peng et al. 2024, safety basin** (arXiv 2405.17374). Safety stays flat under small random weight perturbations,
  then collapses in a step. If wild adapters sit inside or across such a basin, a linear score will mis-rank them.
  Our linearity probe tests this directly.
- **Hawkins et al.** (arXiv 2606.28843, ICML 2026). LoRA on 1,000 benign pairs in 9 languages across 6 small models
  raised compliance up to 4×, and evaluating in the training language amplified the change in about two-thirds of
  cases. This is the closest evidence that drift can differ by language, and motivates our Yoruba and Pidgin panels.

## 2. Drift in real-world derivatives

- **Khan et al.** (AIES 2026, arXiv 2604.24902). 31 real-world medical and legal derivatives of several open models
  (plus controlled runs; about 100 models in all). Drift was common and mixed in sign. Normalised L2 distance showed no
  systematic relation to drift. English only, with no clustering or multiplicity correction.
- **Yoshida et al.** (arXiv 2608.04347). 213 public Hugging Face PEFT adapters on two base models, with train and test
  splits grouped by owner. Their detector reads activations, not weights, and they report no ranking metric.

These two works already measure drift across populations of real derivatives. We do not claim a first population
study or a first prevalence estimate. Our census is the ground truth for testing a screen, not the contribution.

## 3. Screening adapters from their weights

- **Puertolas Merenciano et al.** (arXiv 2602.15195, ICLR 2026). Backdoor detection from LoRA weights alone, using
  spectral statistics of the attention projections. Strong results on the authors' own adapters. The target is
  backdoors, not refusal drift.
- **Paul** (arXiv 2604.08844). Pre-registered study of 38 adapters the author trained on Llama-3.2-3B-Instruct.
  Spectral features separate drifted adapters perfectly within one training method (AUC 1.00) and not at all across
  methods (AUC 0.00). A public population mixes training methods, so we stratify the spectral baseline by method and
  expect it to struggle.
- **SQSD** (arXiv 2605.04572, ICML 2026). Projects LoRA updates onto weight-space danger directions (harmful SFT minus
  base) and safety directions, to first order. Validated by monotone attack success across ranked subsets of its own
  fine-tunes; no correlations reported. This is direct prior art for our D0b; D0b differs in using the gradient of a
  behavioural margin at the base model rather than a difference of trained weights, and in being tested on adapters
  other people released.
- **SafeLoRA** (arXiv 2405.16833). Projects a layer's LoRA update onto the alignment matrix (aligned minus unaligned
  weights) when their similarity falls below a threshold. Training-free and data-free, but never validated as a
  detector across many adapters. We use its similarity as a baseline and its projection as a repair comparator.
- **Safe Delta** (arXiv 2505.12038). A second-order, per-parameter safety cost that drops the first-order term by
  assuming the aligned weights sit at an optimum. Our margin's gradient at the base model is not zero, which is why a
  first-order score is not trivially zero here. A second-order variant is in our Stage 2 plan.

## 4. Arguments against first-order screening

- **Springer et al.** (arXiv 2602.15799). Alignment lives in sharply curved, low-dimensional subspaces that
  first-order methods cannot detect or defend. If they are right on wild adapters, our kill rule for D0b fires. The
  linearity probe distinguishes their explanation (non-linearity) from a wrong direction.
- **Ponkshe et al.** (arXiv 2505.14185). Safety subspaces are not linearly distinct from general learning, in weights
  or activations. This predicts that our projection repair will cost task fidelity, which is why C3 has fidelity and
  no-drift criteria and is compared with WiSE-FT and a random projection.

## 5. Hackathon precedents

Credited for ideas we built on; we attach no verdict to either.

- **safetypulse** monitors the team's own LoRA training runs on Tiny Aya Global with a refusal-circuit score and a
  50-prompt behavioural spot-check, and compares them with weight-delta norm. Our 50-prompt spot-check baseline and
  our weight-delta-size baseline follow this comparison.
- **aftermarket** tests public releases from seven other teams under post-release attacks. It is the precedent for
  evaluating other entrants' public work.

## 6. What is new here, and what is not

**Not new, and not claimed:**
- that benign fine-tuning can erode safety (Qi et al.; He et al.; Lyu et al.; Hawkins et al.);
- measuring drift across a population of real derivatives, or estimating its prevalence (Khan et al.; Yoshida et al.);
- screening adapters from weights (Puertolas Merenciano et al.; Paul; SafeLoRA);
- first-order projection of LoRA updates onto safety-relevant directions (SQSD);
- projection-based, training-free repair (SafeLoRA and related methods).

**What this work adds:**
1. **A label-free weights-only score tested on adapters other people released.** D0b is never fitted to adapter
   labels. It is tested against measured harmful-compliance change under a pre-registered kill rule, with
   author-clustered intervals, Holm correction, a noise ceiling and a 200-direction random control. Prior weights-only
   screens were tested on adapters their authors trained, or fitted on labelled adapters.
2. **Three cost tiers on one population.** Weights only (CPU), one forward pass, and a behavioural spot-check, compared
   on the same adapters and outcome, with their cost per adapter.
3. **A direct test of the first-order objection on wild adapters.** The linearity probe asks whether a failure comes
   from non-linearity or from the wrong direction.
4. **Yoruba and Nigerian Pidgin ground truth.** We found no published measurement of drift across a population of
   adapters in either language. Labels come from a native speaker, with κ reported per language.
5. **A repair with pre-registered success criteria,** including a no-drift control and a random-projection control,
   kept LoRA-shaped so it can be served and quantised.

Whatever the verdict on D0b, the result is informative: a pass gives deployers a CPU-only screen; a kill extends the
negative results of Paul and Springer et al. from controlled runs to a real population.
