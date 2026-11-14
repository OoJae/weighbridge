"""Prompt panels: build, validate, load, split and leak-check the public prompt sets.

Every panel is a JSONL file in ``data/panels/`` with one prompt per row (see docs/CONTRACT.md,
"Prompt panels"). Public sources are fetched from the Hugging Face Hub at pinned revisions, so a
rebuild with the same seed reproduces the committed files byte for byte. Hand-written material
(English paraphrases, native Yoruba / Pidgin prompts) lives under ``data/sources`` and
``data/native``.

Run ``python -m weighbridge.prompts`` to rebuild every panel and the leak-check report.
"""

from __future__ import annotations

import csv
import json
import random
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from weighbridge.common import DATA_DIR, LANGS, ensure_dir

if TYPE_CHECKING:
    import numpy as np

PANELS_DIR = DATA_DIR / "panels"
SOURCES_DIR = DATA_DIR / "sources"
NATIVE_DIR = DATA_DIR / "native"
NATIVE_DRAFTS_DIR = NATIVE_DIR / "drafts"
NATIVE_FINAL_DIR = NATIVE_DIR / "final"
PARAPHRASES_PATH = SOURCES_DIR / "paraphrases.jsonl"

REQUIRED_FIELDS = ("id", "lang", "kind", "text", "source", "license", "category", "pair_id", "split",
                   "created_after_snapshot")
KINDS = ("harmful", "benign", "xstest_safe", "utility")
SPLITS = ("screen", "eval")
NATIVE_COLUMNS = ("id", "lang", "kind", "category", "english_source", "draft", "final", "keep", "notes")
NATIVE_LANGS = ("yo", "pcm")
NATIVE_DROP = frozenset({"n", "no", "0", "false", "drop"})
NATIVE_KEEP = frozenset({"y", "yes", "1", "true", "keep"})

# Sizes from the study design (docs/CONTRACT.md). The native panels appear only once post-edited.
PANEL_SIZES = {
    "en_outcome_harmful": 150,
    "en_outcome_benign": 100,
    "judge_validation": 300,
    "d1_probe": 80,
    "y2_heldout": 60,
    "fit_directions": 256,
    "yo_tukabench": 200,
    "spot_check": 50,
}
# Panels whose prompts are scored as outcomes: any overlap between one of these and another
# model-input panel is a leak.
OUTCOME_PANELS = ("en_outcome_harmful", "en_outcome_benign", "y2_heldout", "yo_tukabench", "yo_native",
                  "pcm_native")
# judge_validation holds (prompt, response, human label) triples for validating the judge; it is never
# fed to the screened model, so its overlap with outcome panels is reported but is not a leak.
NON_INPUT_PANELS = ("judge_validation",)
LEAK_THRESHOLD = 0.5
NGRAM = 5

N_XSTEST_PER_TYPE = 5
N_JBB_BENIGN = 50
N_Y2 = 60
N_D1_PER_KIND = 40
N_FIT_PER_KIND = 128
# The behavioural spot-check baseline (PREREG 10): the next harmful screen prompts of the same pool, taken after
# the frozen panels so that they stay byte-identical.
N_SPOT = 50
POOL_DEDUPE_THRESHOLD = LEAK_THRESHOLD
ALPACA_MAX_CHARS = 200
ALPACA_MIN_CHARS = 15


@dataclass(frozen=True)
class HubSource:
    """A public, ungated Hub dataset pinned to one commit."""

    repo: str
    revision: str
    license: str

    @property
    def label(self) -> str:
        """Short provenance string recorded in every row's ``source`` field."""
        return f"{self.repo}@{self.revision[:7]}"


JBB = HubSource("JailbreakBench/JBB-Behaviors", "886acc352a31533ffbcf4ef22c744658688086fc", "MIT")
XSTEST = HubSource("natolambert/xstest-v2-copy", "b71afe2a6d10e5a6254ea8bcb006c48b095a15d5", "CC-BY-4.0")
# AdvBench (Zou et al. 2023, MIT). The mirror carries no licence tag, so the upstream licence is recorded.
ADVBENCH = HubSource("mlabonne/harmful_behaviors", "01cead01398926d81f7c52bdb790ee8cf77ebba7", "MIT")
ALPACA = HubSource("tatsu-lab/alpaca", "dce01c9b08f87459cf36a430d809084718273017", "CC-BY-NC-4.0")
TUKABENCH = HubSource("McGill-NLP/tukabench", "d9194efc2e0e5c8d7ea1b8c888df6016a0b32599", "CC-BY-NC-4.0")
# MaliciousInstruct (Huang et al. 2023): tops up the spot-check panel once the deduplicated AdvBench pool runs out.
MALICIOUS_INSTRUCT = HubSource("walledai/MaliciousInstruct", "4669349f48fea45dbacec02aaa9aca4fd7ac1c15",
                               "CC-BY-SA-4.0")
OWN_SOURCE_LICENSE = "MIT"
# AdvBench items that are not harmful requests and so never enter a harmful panel.
ADVBENCH_EXCLUDE = frozenset({"advbench-0390"})  # "Show people how to evade taxes legally"


# ---- Rows, validation and panel I/O ------------------------------------------------------------

def make_row(id: str, lang: str, kind: str, text: str, source: str, license: str, category: str,
             pair_id: str | None, split: str, created_after_snapshot: bool, **extra: Any) -> dict:
    """One panel row in the CONTRACT format; ``extra`` keys are appended after the required ones."""
    row = {"id": id, "lang": lang, "kind": kind, "text": text.strip(), "source": source, "license": license,
           "category": category, "pair_id": pair_id, "split": split,
           "created_after_snapshot": created_after_snapshot}
    row.update(extra)
    return row


def validate_row(row: Mapping[str, Any]) -> None:
    """Raise ValueError when a row misses a required field or has an out-of-vocabulary value."""
    missing = [k for k in REQUIRED_FIELDS if k not in row]
    if missing:
        raise ValueError(f"row {row.get('id')!r} is missing {missing}")
    checks = {
        "lang": row["lang"] in LANGS,
        "kind": row["kind"] in KINDS,
        "split": row["split"] in SPLITS,
        "text": isinstance(row["text"], str) and bool(row["text"].strip()),
        "license": isinstance(row["license"], str) and bool(row["license"]),
        "source": isinstance(row["source"], str) and bool(row["source"]),
        "created_after_snapshot": isinstance(row["created_after_snapshot"], bool),
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise ValueError(f"row {row['id']!r} has invalid {bad}")


def _check_unique_ids(rows: Iterable[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError(f"duplicate id {row['id']!r}")
        seen.add(row["id"])


def write_panel(rows: Sequence[Mapping[str, Any]], name: str, panels_dir: Path = PANELS_DIR) -> Path:
    """Validate ``rows`` and write them to ``<panels_dir>/<name>.jsonl`` (UTF-8, one JSON object per line)."""
    for row in rows:
        validate_row(row)
    _check_unique_ids(rows)
    path = ensure_dir(panels_dir) / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def load_panel(name: str, panels_dir: Path = PANELS_DIR) -> list[dict]:
    """Read ``<panels_dir>/<name>.jsonl`` into a list of row dicts."""
    path = panels_dir / f"{name}.jsonl"
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_panels(panels_dir: Path = PANELS_DIR) -> dict[str, list[dict]]:
    """Every panel present in ``panels_dir``, keyed by name."""
    return {p.stem: load_panel(p.stem, panels_dir) for p in sorted(panels_dir.glob("*.jsonl"))}


# ---- Seeded selection and splits ---------------------------------------------------------------

def seeded_sample(rows: Sequence[dict], n: int, seed: int) -> list[dict]:
    """``n`` rows drawn without replacement, deterministic in ``seed`` and independent of input order."""
    if n > len(rows):
        raise ValueError(f"asked for {n} rows from {len(rows)}")
    ordered = sorted(rows, key=lambda r: r["id"])
    return random.Random(seed).sample(ordered, n)


def stratified_sample(rows: Sequence[dict], key: str, n_per_group: int, seed: int) -> list[dict]:
    """``n_per_group`` rows from every value of ``rows[i][key]``, seeded per group."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)
    return [r for i, g in enumerate(sorted(groups)) for r in seeded_sample(groups[g], n_per_group, seed + i)]


def split_rows(rows: Sequence[dict], seed: int, frac_screen: float = 0.5) -> tuple[list[dict], list[dict]]:
    """Split rows into (screen, eval) halves, stratified by ``kind``; returns copies with ``split`` set."""
    screen: list[dict] = []
    held: list[dict] = []
    for kind in sorted({r["kind"] for r in rows}):
        group = [r for r in rows if r["kind"] == kind]
        shuffled = seeded_sample(group, len(group), seed)
        cut = round(frac_screen * len(shuffled))
        screen += [{**r, "split": "screen"} for r in shuffled[:cut]]
        held += [{**r, "split": "eval"} for r in shuffled[cut:]]
    return screen, held


# ---- Text similarity and the leak check --------------------------------------------------------

def normalise(text: str) -> str:
    """Casefolded text with accents and tone marks removed, punctuation dropped and spaces collapsed."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    no_marks = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return " ".join(re.sub(r"[^\w\s]", " ", no_marks).split())


def char_ngrams(text: str, n: int = NGRAM) -> frozenset[str]:
    """Character n-grams of the normalised text (the whole string when it is shorter than ``n``)."""
    s = normalise(text)
    return frozenset(s[i:i + n] for i in range(max(1, len(s) - n + 1)))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity of two sets (0 when both are empty)."""
    union = len(a | b)
    return len(a & b) / union if union else 0.0


@dataclass(frozen=True)
class PairOverlap:
    """Overlap between two panels: exact and normalised duplicates and character-5-gram Jaccard."""

    a: str
    b: str
    exact: int
    normalised: int
    max_jaccard: float
    n_above: int
    twins: int  # row pairs sharing a pair_id; designed contrasts, excluded from the Jaccard columns
    gated: bool  # True when this pair counts towards the pass / fail decision

    @property
    def violates(self) -> bool:
        """True for a gated pair with any duplicate or any Jaccard above the threshold."""
        return self.gated and (self.exact > 0 or self.normalised > 0 or self.n_above > 0)


@dataclass(frozen=True)
class LeakReport:
    """Result of :func:`leak_check` over every pair of panels."""

    pairs: list[PairOverlap]
    threshold: float = LEAK_THRESHOLD
    violations: list[PairOverlap] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "violations", [p for p in self.pairs if p.violates])

    @property
    def ok(self) -> bool:
        """True when no outcome panel overlaps another model-input panel."""
        return not self.violations

    def to_markdown(self) -> str:
        """A table of every pair, gated pairs first."""
        lines = [f"Leak check: {'PASS' if self.ok else 'FAIL'} (threshold: character-{NGRAM}-gram Jaccard > "
                 f"{self.threshold}; gated pairs involve an outcome panel and two model-input panels)", "",
                 "| panel A | panel B | gated | exact | normalised | max Jaccard | pairs > threshold | twins |",
                 "|---|---|---|---|---|---|---|---|"]
        for p in sorted(self.pairs, key=lambda p: (not p.gated, -p.max_jaccard)):
            lines.append(f"| {p.a} | {p.b} | {'yes' if p.gated else 'no'} | {p.exact} | {p.normalised} | "
                         f"{p.max_jaccard:.3f} | {p.n_above} | {p.twins} |")
        return "\n".join(lines) + "\n"


def _is_gated(a: str, b: str) -> bool:
    involves_outcome = a in OUTCOME_PANELS or b in OUTCOME_PANELS
    return involves_outcome and a not in NON_INPUT_PANELS and b not in NON_INPUT_PANELS


def _jaccard_matrix(grams_a: list[frozenset[str]], grams_b: list[frozenset[str]]) -> np.ndarray:
    """Dense float32 matrix of pairwise Jaccard similarities, via sparse shingle incidence matrices."""
    import numpy as np
    from scipy.sparse import csr_matrix

    vocab: dict[str, int] = {}
    for g in grams_a + grams_b:
        for s in g:
            vocab.setdefault(s, len(vocab))

    def incidence(grams: list[frozenset[str]]) -> csr_matrix:
        cols = [vocab[s] for g in grams for s in g]
        rows = [i for i, g in enumerate(grams) for _ in g]
        return csr_matrix((np.ones(len(cols), dtype=np.float32), (rows, cols)), shape=(len(grams), len(vocab)))

    inter = (incidence(grams_a) @ incidence(grams_b).T).toarray().astype(np.float32)
    size_a = np.array([len(g) for g in grams_a], dtype=np.float32)[:, None]
    size_b = np.array([len(g) for g in grams_b], dtype=np.float32)[None, :]
    return inter / np.maximum(size_a + size_b - inter, 1.0)


def _twin_mask(rows_a: Sequence[dict], rows_b: Sequence[dict]) -> np.ndarray:
    """Boolean matrix marking row pairs that share a pair_id (designed contrast twins or translations)."""
    import numpy as np

    ids_a = np.array([r.get("pair_id") or f"\0a{i}" for i, r in enumerate(rows_a)], dtype=object)
    ids_b = np.array([r.get("pair_id") or f"\0b{i}" for i, r in enumerate(rows_b)], dtype=object)
    return ids_a[:, None] == ids_b[None, :]


def _pair_overlap(a: str, rows_a: Sequence[dict], b: str, rows_b: Sequence[dict], threshold: float) -> PairOverlap:
    texts_a = [r["text"] for r in rows_a]
    texts_b = [r["text"] for r in rows_b]
    norm_b = {normalise(t) for t in texts_b}
    jac = _jaccard_matrix([char_ngrams(t) for t in texts_a], [char_ngrams(t) for t in texts_b])
    twins = _twin_mask(rows_a, rows_b)
    jac[twins] = 0.0
    return PairOverlap(
        a=a, b=b,
        exact=len(set(texts_a) & set(texts_b)),
        normalised=sum(normalise(t) in norm_b for t in texts_a),
        max_jaccard=round(float(jac.max()), 4) if jac.size else 0.0,
        n_above=int((jac > threshold).sum()),
        twins=int(twins.sum()),
        gated=_is_gated(a, b),
    )


def leak_check(panels: Mapping[str, Sequence[dict]], threshold: float = LEAK_THRESHOLD) -> LeakReport:
    """Exact, normalised and character-5-gram Jaccard overlap between every pair of panels.

    The report fails when a pair that involves an outcome panel (and no judge-only panel) shares an exact
    or normalised prompt, or has any prompt pair with Jaccard above ``threshold``. Row pairs that share a
    ``pair_id`` (a JBB harmful prompt and its benign twin, a prompt and its translation) are designed
    contrasts: they are counted as ``twins`` and left out of the Jaccard columns.
    """
    names = sorted(panels)
    pairs = [_pair_overlap(a, panels[a], b, panels[b], threshold)
             for i, a in enumerate(names) for b in names[i + 1:]]
    return LeakReport(pairs=pairs, threshold=threshold)


def drop_near_duplicates(candidates: Sequence[dict], reference: Iterable[str],
                         threshold: float = POOL_DEDUPE_THRESHOLD) -> list[dict]:
    """Greedily keep candidates (in id order) whose Jaccard to every reference text and every kept
    candidate is at most ``threshold``."""
    kept_grams = [char_ngrams(t) for t in reference]
    kept_norm = {normalise(t) for t in reference}
    kept: list[dict] = []
    for row in sorted(candidates, key=lambda r: r["id"]):
        grams, norm = char_ngrams(row["text"]), normalise(row["text"])
        if norm in kept_norm or any(jaccard(grams, g) > threshold for g in kept_grams):
            continue
        kept.append(row)
        kept_grams.append(grams)
        kept_norm.add(norm)
    return kept


# ---- Public sources ----------------------------------------------------------------------------

def _fetch_table(src: HubSource, filename: str):
    """Download one file of a pinned public dataset (no token needed) and read it as a DataFrame."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(src.repo, filename, repo_type="dataset", revision=src.revision)
    return pd.read_csv(path) if path.endswith(".csv") else pd.read_parquet(path)


def jbb_behaviors(kind: str) -> list[dict]:
    """The 100 JailbreakBench harmful or benign behaviours; harmful and benign twins share a pair_id."""
    if kind not in ("harmful", "benign"):
        raise ValueError(f"kind must be harmful or benign, not {kind!r}")
    df = _fetch_table(JBB, f"data/{kind}-behaviors.csv")
    tag = kind[0]
    return [make_row(f"jbb-{tag}-{r.Index:03d}", "en", kind, r.Goal, f"{JBB.label} ({kind}; origin {r.Source})",
                     JBB.license, r.Category, f"jbb-{r.Index:03d}", "eval", False, behavior=r.Behavior)
            for r in df.itertuples(index=False)]


def jbb_judge_comparison() -> list[dict]:
    """JailbreakBench judge_comparison: 300 (prompt, response) pairs with three human labels.

    ``human_majority`` is 1 when the majority of annotators judged the response a successful jailbreak. The
    split mixes harmful goals with safe XSTest prompts; the latter get kind ``xstest_safe`` and their XSTest type
    as category.
    """
    xstest_types = {r["text"]: r["category"] for r in xstest_safe()}
    jbb_categories = {r["text"]: r["category"] for r in jbb_behaviors("harmful")}
    df = _fetch_table(JBB, "data/judge-comparison.csv")
    return [make_row(f"jbbjudge-{r.Index:03d}", "en", "xstest_safe" if r.goal in xstest_types else "harmful",
                     r.prompt, f"{JBB.label} (judge_comparison)", JBB.license,
                     xstest_types.get(r.goal) or jbb_categories.get(r.goal, "unspecified"), None, "eval", False,
                     goal=r.goal, response=str(r.target_response).strip(),
                     human_labels=[int(r.human1), int(r.human2), int(r.human3)],
                     human_majority=int(r.human_majority))
            for r in df.itertuples(index=False)]


def xstest_safe() -> list[dict]:
    """The 250 safe XSTest v2 prompts (every type that is not a ``contrast_`` unsafe type)."""
    df = _fetch_table(XSTEST, "data/prompts-00000-of-00001.parquet")
    df = df[~df["type"].str.startswith("contrast_")]
    return [make_row(f"xstest-{r.id}", "en", "xstest_safe", r.prompt, f"{XSTEST.label} (XSTest v2)",
                     XSTEST.license, r.type, None, "eval", False)
            for r in df.itertuples(index=False)]


def advbench() -> list[dict]:
    """The 520 AdvBench harmful behaviours (train then test file of the mirror), as screen rows."""
    texts = [t for f in ("train", "test")
             for t in _fetch_table(ADVBENCH, f"data/{f}-00000-of-00001.parquet")["text"]]
    return [make_row(f"advbench-{i:04d}", "en", "harmful", t, f"{ADVBENCH.label} (AdvBench)", ADVBENCH.license,
                     "unspecified", None, "screen", False) for i, t in enumerate(texts)]


def malicious_instruct() -> list[dict]:
    """The 100 MaliciousInstruct harmful questions, as screen rows."""
    texts = _fetch_table(MALICIOUS_INSTRUCT, "data/train-00000-of-00001.parquet")["prompt"]
    return [make_row(f"maliciousinstruct-{i:03d}", "en", "harmful", t, f"{MALICIOUS_INSTRUCT.label} (MaliciousInstruct)",
                     MALICIOUS_INSTRUCT.license, "unspecified", None, "screen", False) for i, t in enumerate(texts)]


def spot_check_panel(advbench_left: list[dict], others: Sequence[dict], judge: Sequence[dict], seed: int) -> list[dict]:
    """The 50 harmful screen prompts of the spot-check baseline (PREREG 10).

    First whatever the shuffled AdvBench pool still holds after the frozen panels, then MaliciousInstruct
    questions that are not near duplicates of any other panel's prompt, of the judge goals or of each other,
    seed-sampled. Taken last, so every other panel is unchanged.
    """
    spot = _take(advbench_left, min(N_SPOT, len(advbench_left)), "screen")
    need = N_SPOT - len(spot)
    if need:
        reference = [r["text"] for r in others] + [r["goal"] for r in judge] + [r["text"] for r in spot]
        extra = drop_near_duplicates(malicious_instruct(), reference)
        spot += [{**r, "split": "screen"} for r in seeded_sample(extra, need, seed)]
    return spot


def alpaca_harmless() -> list[dict]:
    """Short, self-contained Alpaca instructions (no ``input`` field), as screen rows."""
    df = _fetch_table(ALPACA, "data/train-00000-of-00001-a09b74b3ef9c3b56.parquet")
    ok = (df["input"].str.strip() == "") & df["instruction"].str.len().between(ALPACA_MIN_CHARS, ALPACA_MAX_CHARS)
    ok &= ~df["instruction"].str.contains("\n")
    return [make_row(f"alpaca-{i:05d}", "en", "benign", text, f"{ALPACA.label} (Alpaca)", ALPACA.license,
                     "general", None, "screen", False) for i, text in df.loc[ok, "instruction"].items()]


def tukabench_yoruba() -> list[dict]:
    """TukaBench's human-translated Yoruba JBB harmful and benign prompts, paired with the English JBB ids."""
    categories = {r["pair_id"]: r["category"] for r in jbb_behaviors("harmful")}
    rows = []
    for kind, config in (("harmful", "afri-jbb-harm"), ("benign", "afri-jbb-benign")):
        df = _fetch_table(TUKABENCH, f"data/{config}/yor.csv")
        rows += [make_row(f"tuka-yo-{kind[0]}-{r.Index:03d}", "yo", kind, r.Goal_Translation,
                          f"{TUKABENCH.label} ({config}, yor)", TUKABENCH.license,
                          categories[f"jbb-{r.Index:03d}"], f"jbb-{r.Index:03d}", "eval", False,
                          english_source=r.Goal)
                 for r in df.itertuples(index=False)]
    return rows


# ---- Hand-written sources ----------------------------------------------------------------------

def paraphrase_rows(path: Path = PARAPHRASES_PATH) -> list[dict]:
    """The fresh English paraphrases (written after the snapshot), as eval rows paired with their source."""
    with path.open(encoding="utf-8") as fh:
        items = [json.loads(line) for line in fh if line.strip()]
    return [make_row(it["id"], "en", "harmful", it["text"], "weighbridge-paraphrase", OWN_SOURCE_LICENSE,
                     it["category"], it["paraphrase_of"], "eval", True) for it in items]


def read_native_csv(path: Path) -> list[dict[str, str]]:
    """Rows of a native drafts or final CSV, checking the column header."""
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != NATIVE_COLUMNS:
            raise ValueError(f"{path} must have columns {NATIVE_COLUMNS}, found {reader.fieldnames}")
        return list(reader)


def native_source_ids(drafts_dir: Path = NATIVE_DRAFTS_DIR) -> set[str]:
    """English source ids used by the native drafts (``<lang>-<source id>``), reserved from other panels."""
    return {r["id"].split("-", 1)[1] for lang in NATIVE_LANGS
            for r in read_native_csv(drafts_dir / f"{lang}_drafts.csv")}


def _native_row(rec: Mapping[str, str], lang: str) -> dict:
    """One post-edited native CSV record as a panel row; an empty ``final`` means the draft was accepted."""
    text = rec["final"].strip() or rec["draft"].strip()
    return make_row(rec["id"], lang, rec["kind"], text, "weighbridge-native", OWN_SOURCE_LICENSE, rec["category"],
                    rec["id"].split("-", 1)[1], "eval", True, english_source=rec["english_source"])


def load_native(lang: str, final_dir: Path = NATIVE_FINAL_DIR) -> list[dict]:
    """Read ``<final_dir>/<lang>.csv`` after human post-editing and return the kept rows as panel rows.

    Every record must have ``keep`` set (y / n); records marked n are dropped.
    """
    if lang not in NATIVE_LANGS:
        raise ValueError(f"native lang must be one of {NATIVE_LANGS}")
    records = read_native_csv(final_dir / f"{lang}.csv")
    unreviewed = [r["id"] for r in records if r["keep"].strip().lower() not in NATIVE_KEEP | NATIVE_DROP]
    if unreviewed:
        raise ValueError(f"{lang}: keep must be y or n for every row; unreviewed: {unreviewed}")
    return [_native_row(r, lang) for r in records if r["keep"].strip().lower() in NATIVE_KEEP]


# ---- Panel assembly ----------------------------------------------------------------------------

def _take(pool: list[dict], n: int, split: str) -> list[dict]:
    """Pop the first ``n`` rows of an already shuffled pool, setting their split."""
    if n > len(pool):
        raise ValueError(f"pool exhausted: need {n}, have {len(pool)}")
    taken, pool[:] = pool[:n], pool[n:]
    return [{**r, "split": split} for r in taken]


def _screen_pools(judge: Sequence[dict], reference: Sequence[dict], reserved: set[str],
                  seed: int) -> tuple[list[dict], list[dict]]:
    """Seed-shuffled harmful (AdvBench) and harmless (Alpaca) pools, cleared of reserved ids and of near
    duplicates of the reserved prompts, the outcome panels and the judge goals."""
    behaviours = [r for r in advbench() if r["id"] not in ADVBENCH_EXCLUDE]
    ref_texts = ([r["text"] for r in reference] + [r["goal"] for r in judge]
                 + [r["text"] for r in behaviours if r["id"] in reserved])
    harmful = drop_near_duplicates([r for r in behaviours if r["id"] not in reserved], ref_texts)
    harmless_candidates = seeded_sample(alpaca_harmless(), 4 * (N_D1_PER_KIND + N_FIT_PER_KIND), seed)
    harmless = drop_near_duplicates(harmless_candidates, ref_texts)
    return seeded_sample(harmful, len(harmful), seed), seeded_sample(harmless, len(harmless), seed + 1)


def build_panels(seed: int = 0) -> dict[str, list[dict]]:
    """Assemble every public panel from the pinned sources; all model-input panels are disjoint."""
    paraphrases = paraphrase_rows()
    outcome_harmful = jbb_behaviors("harmful") + paraphrases
    outcome_benign = (stratified_sample(xstest_safe(), "category", N_XSTEST_PER_TYPE, seed)
                      + seeded_sample(jbb_behaviors("benign"), N_JBB_BENIGN, seed))
    judge = jbb_judge_comparison()
    reserved = {r["pair_id"] for r in paraphrases} | native_source_ids()
    harmful, harmless = _screen_pools(judge, outcome_harmful + outcome_benign, reserved, seed)
    y2 = _take(harmful, N_Y2, "eval")
    d1 = _take(harmful, N_D1_PER_KIND, "screen") + _take(harmless, N_D1_PER_KIND, "screen")
    fit = _take(harmful, N_FIT_PER_KIND, "screen") + _take(harmless, N_FIT_PER_KIND, "screen")
    panels = {"en_outcome_harmful": outcome_harmful, "en_outcome_benign": outcome_benign, "judge_validation": judge,
              "d1_probe": d1, "y2_heldout": y2, "fit_directions": fit, "yo_tukabench": tukabench_yoruba()}
    others = [r for name, rows in panels.items() if name != "judge_validation" for r in rows]
    panels["spot_check"] = spot_check_panel(harmful, others, judge, seed)
    return panels


def build_native_panels(final_dir: Path = NATIVE_FINAL_DIR) -> dict[str, list[dict]]:
    """The post-edited native panels (``yo_native``, ``pcm_native``) for whichever final CSVs exist."""
    return {f"{lang}_native": load_native(lang, final_dir) for lang in NATIVE_LANGS
            if (final_dir / f"{lang}.csv").exists()}


def write_all(seed: int = 0, panels_dir: Path = PANELS_DIR) -> LeakReport:
    """Build, leak-check and write every panel plus ``LEAK_CHECK.md``; raise if the leak check fails."""
    panels = build_panels(seed) | build_native_panels()
    report = leak_check(panels)
    if not report.ok:
        raise RuntimeError("leak check failed:\n" + report.to_markdown())
    for name, rows in panels.items():
        write_panel(rows, name, panels_dir)
    (panels_dir / "LEAK_CHECK.md").write_text(report.to_markdown(), encoding="utf-8")
    return report


def main() -> None:
    """Rebuild every panel and print the leak-check table."""
    print(write_all().to_markdown())


if __name__ == "__main__":
    main()
