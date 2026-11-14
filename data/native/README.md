# Native Yoruba and Nigerian Pidgin prompts

Two files of machine-assisted drafts wait for a native speaker to post-edit them:

- `drafts/yo_drafts.csv`: Yoruba, written with tone marks.
- `drafts/pcm_drafts.csv`: Nigerian Pidgin.

Each holds 75 rows: 50 harmful requests and 25 benign look-alikes (prompts that sound alarming but are
harmless, in the XSTest style, `kind = xstest_safe`). The two files cover the same English sources in
the same order, so `yo-para-001` and `pcm-para-001` are translations of one prompt.

How to edit them is in `docs/NATIVE_PROMPT_GUIDE.md`.

## Columns

| Column | Who fills it | Meaning |
|---|---|---|
| `id` | fixed | `<lang>-<English source id>`; never change it |
| `lang`, `kind`, `category` | fixed | `yo` or `pcm`; `harmful` or `xstest_safe`; JBB category or XSTest type |
| `english_source` | fixed | the English prompt this row translates |
| `draft` | fixed | the first draft |
| `final` | you | your corrected text; leave empty if the draft is already right |
| `keep` | you | `y` to keep the row, `n` to drop it; every row needs one |
| `notes` | you, optional | anything worth knowing (why you dropped it, a dialect choice) |

English source ids say where a prompt comes from:

- `para-NNN`: one of the 50 fresh English paraphrases in the English outcome panel
  (`data/panels/en_outcome_harmful.jsonl`), so these rows can be compared across all three languages.
- `advbench-NNNN`: an AdvBench item reserved for the native sets; it appears in no English panel.
- `look-NN`: a benign look-alike written for this project.

## Reading the finished files back

1. Save the edited files as `final/yo.csv` and `final/pcm.csv`, UTF-8, with the same nine columns
   (in Excel choose "CSV UTF-8"; Google Sheets and LibreOffice export UTF-8 by default).
2. Run `.venv/bin/python -m weighbridge.prompts`.

`prompts.load_native(lang)` reads `final/<lang>.csv` and:

- raises an error naming any row whose `keep` is not `y` or `n` (so nothing unreviewed slips in);
- drops rows marked `n`;
- uses `final` as the prompt text, or `draft` when `final` is empty;
- returns panel rows with `source = "weighbridge-native"`, `license = "MIT"`, `split = "eval"`,
  `created_after_snapshot = true`, `pair_id` = the English source id and an `english_source` field.

The build then writes `data/panels/yo_native.jsonl` and `data/panels/pcm_native.jsonl` and includes
them in the leak check. A file that is missing from `final/` is simply skipped, so the two languages
can be finished separately.
