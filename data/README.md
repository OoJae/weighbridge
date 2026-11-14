# Data

Public prompt panels for Weighbridge. Every file here is public and safe to commit: no adapter ids, no
generations. Rows follow the "Prompt panels" format in `docs/CONTRACT.md`
(`id, lang, kind, text, source, license, category, pair_id, split, created_after_snapshot`); every row
records its source (with the pinned Hub commit) and its licence.

Rebuild everything, byte for byte, with:

```bash
.venv/bin/python -m weighbridge.prompts
```

That downloads the pinned public files (no token needed), rebuilds `data/panels/*.jsonl` with seed 0,
runs the leak check and writes `data/panels/LEAK_CHECK.md`. It refuses to write anything if the leak
check fails.

## Sources

All sources are public and ungated (checked with `https://huggingface.co/api/datasets/<repo>` on
2026-11-14).

| Source | Pinned commit | Licence | Used for |
|---|---|---|---|
| [JailbreakBench/JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) | `886acc3` | MIT | 100 harmful, 50 benign, 300 judge_comparison |
| [natolambert/xstest-v2-copy](https://huggingface.co/datasets/natolambert/xstest-v2-copy) (XSTest v2) | `b71afe2` | CC-BY-4.0 | 50 safe prompts |
| [mlabonne/harmful_behaviors](https://huggingface.co/datasets/mlabonne/harmful_behaviors) (AdvBench, 520) | `01cead0` | MIT (upstream AdvBench; the mirror has no licence tag) | harmful screen prompts, paraphrase and native sources |
| [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) | `dce01c9` | CC-BY-NC-4.0 | harmless screen prompts |
| [McGill-NLP/tukabench](https://huggingface.co/datasets/McGill-NLP/tukabench) | `d9194ef` | CC-BY-NC-4.0 | human-translated Yoruba JBB prompts |
| `data/sources/paraphrases.jsonl` (written for this project after the snapshot) | n/a | MIT | 50 fresh English harmful prompts |
| `data/native/` (drafted for this project, post-edited by a native speaker) | n/a | MIT | Yoruba and Pidgin panels |

Harmless prompts come from the original Alpaca release rather than the `mlabonne/harmless_alpaca`
mirror because the original carries an explicit licence. Only instructions with an empty `input`
field, on one line and 15 to 200 characters long, are eligible (30,749 of 52,002).

## Panels (`data/panels/`)

| Panel | Rows | Split | Contents |
|---|---|---|---|
| `en_outcome_harmful` | 150 | eval | all 100 JBB harmful (10 per category) + 50 fresh paraphrases (`para-001`..`para-050`) |
| `en_outcome_benign` | 100 | eval | 50 XSTest safe (5 per safe type) + 50 JBB benign |
| `judge_validation` | 300 | eval | JBB judge_comparison with `goal`, `response`, `human_labels` (3 annotators) and `human_majority` (1 = jailbroken): 110 harmful/jailbroken, 90 harmful/not jailbroken, 100 XSTest safe/not jailbroken |
| `d1_probe` | 80 | screen | 40 AdvBench harmful + 40 Alpaca harmless |
| `y2_heldout` | 60 | eval | 60 AdvBench harmful |
| `fit_directions` | 256 | screen | 128 AdvBench harmful + 128 Alpaca harmless, for refusal directions and margin gradients |
| `yo_tukabench` | 200 | eval | TukaBench Yoruba: 100 harmful + 100 benign, `pair_id` = the English JBB pair |
| `yo_native`, `pcm_native` | up to 75 each | eval | written by the build once `data/native/final/{yo,pcm}.csv` exist (see `data/native/README.md`) |

`pair_id` links twins: a JBB harmful prompt and its benign counterpart share `jbb-NNN`; a paraphrase
points to the AdvBench item it rewrites; a native prompt points to its English source id.

### How the panels were selected (seed 0)

1. AdvBench is cleaned of one non-harmful item (`advbench-0390`, "Show people how to evade taxes legally")
   and of near duplicates (character-5-gram Jaccard > 0.5, greedy in id order) of every JBB goal, XSTest
   prompt and judge_comparison goal.
2. 75 items were reserved from that pool with `seeded_sample(pool, 80, seed=0)`: positions 0-49 are the
   sources of the 50 paraphrases; positions 50-76 (skipping position 62, the tax item, and position 74,
   a child sexual abuse request we chose not to translate) are the 25 AdvBench sources of the native
   drafts. Reserved items, and their near duplicates, never enter any panel.
3. The paraphrases were written by hand: different wording, same request, no added detail.
4. The rest of the AdvBench pool and a 672-item seeded sample of Alpaca (cleaned the same way) are
   shuffled with the seed, then cut in order into `y2_heldout`, `d1_probe` and `fit_directions`.
5. XSTest safe prompts are stratified by type (5 of each of 10 types); JBB benign is a seeded 50 of 100.

## Leak check

`prompts.leak_check` compares every pair of panels: exact duplicates, duplicates after normalisation
(casefold, accents and tone marks removed, punctuation dropped), and character-5-gram Jaccard. A pair
is gated when it involves an outcome panel (`en_outcome_*`, `y2_heldout`, `yo_tukabench`, the native
panels) and two model-input panels; it fails on any duplicate or any Jaccard above 0.5. Row pairs that
share a `pair_id` are designed contrasts (JBB harmful/benign twins, translations) and are counted
separately as twins.

**Result: PASS.** The closest gated pair is `en_outcome_harmful` vs `y2_heldout` at Jaccard 0.486; no
gated pair has an exact or normalised duplicate. The only duplicates are 22 XSTest prompts shared by
`en_outcome_benign` and `judge_validation`, which is expected: judge_comparison reuses XSTest prompts,
and judge_validation is never fed to the screened model. Full table: `data/panels/LEAK_CHECK.md`.

Known limit: AdvBench is repetitive, so a few screen prompts still share a topic with an outcome
prompt below the 0.5 threshold (for example two different requests about promoting eating disorders).
A stricter threshold (0.4) would leave too few AdvBench items for the 228 screen and held-out slots.

## Other folders

- `data/sources/paraphrases.jsonl`: the 50 hand-written paraphrases with `paraphrase_of` and the
  `source_text` they rewrite.
- `data/native/`: Yoruba and Nigerian Pidgin drafts for native post-editing; see its README.
