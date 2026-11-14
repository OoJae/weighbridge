"""Native-speaker labelling (PREREG 8): a blind, stratified sample of Yoruba and Pidgin replies with a
test-retest block, and the agreement between the judge and the native labels.

    python -m weighbridge.labelset build --gen <raw dirs or files> --labels <judge label files> \\
        --outcomes <adapters.csv or outcome rows> --panels-dir data/panels --out-dir <PRIVATE_DIR>/labelling
    python -m weighbridge.labelset agree --native <exported CSV> --key <out-dir>/key.jsonl --labels <label files>

Items are blind: the labeller sees an item id, the language, whether the prompt was harmful or benign, the prompt
and the reply, never the adapter. Third-party replies are written under PRIVATE_DIR only.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

N_ITEMS, N_RETEST, SEED = 240, 40, 0
LANGS = ("yo", "pcm")
DISAGREE_SHARE = 1 / 3
STRATA = ("base", "low", "mid", "high")


def _rows(paths: Iterable[str | Path]) -> list[dict]:
    out = []
    for p in map(Path, paths):
        for f in (sorted(p.glob("*.jsonl")) if p.is_dir() else [p]):
            if f.name.endswith(".timings.jsonl"):
                continue
            out += [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    return out


def drift_strata(drift: Mapping[str, float]) -> dict[str, str]:
    """Adapter alias -> drift tercile (low / mid / high) of its English Y; the base model is its own stratum."""
    ranked = sorted((y, a) for a, y in drift.items() if y == y)
    n = len(ranked)
    out = {a: STRATA[1 + min(2, (3 * i) // max(n, 1))] for i, (_, a) in enumerate(ranked)}
    out["__base__"] = "base"
    return out


def disagrees(label: Mapping[str, Any]) -> bool:
    """The guard and the string matcher disagree on whether the reply refuses."""
    guard_refused = label.get("guard_refusal") == "Yes"
    return guard_refused != bool(label.get("string_refusal"))


def build_label_set(gen_rows: Sequence[dict], label_rows: Sequence[dict], drift: Mapping[str, float],
                    prompts: Mapping[str, str], n_items: int = N_ITEMS, n_retest: int = N_RETEST,
                    seed: int = SEED, langs: Sequence[str] = LANGS) -> tuple[list[dict], list[dict]]:
    """Blind items for the labelling page and the private key that maps each item back to its reply.

    Per language (equal shares): up to a third of the items are replies where the guard and the string matcher
    disagree; the rest are spread evenly over the drift strata (base, and terciles of English Y). Then `n_retest`
    items are repeated with new ids in the second half of the list, for test-retest agreement.
    """
    rng = random.Random(seed)
    labels = {(r["adapter_alias"], r["condition"], r["prompt_id"]): r for r in label_rows}
    strata = drift_strata(drift)
    per_lang = n_items // len(langs)
    chosen: list[dict] = []
    for lang in langs:
        pool = [g for g in gen_rows if g["lang"] == lang and g.get("text", "").strip()
                and g["adapter_alias"] in strata and g["prompt_id"] in prompts]
        pool.sort(key=lambda g: (g["adapter_alias"], g["condition"], g["prompt_id"]))
        rng.shuffle(pool)
        picked: list[dict] = []
        seen = set()

        def take(rows: Iterable[dict], n: int) -> None:
            it = iter(rows)
            while len(picked) < n:  # check before pulling, so no candidate is consumed and dropped
                g = next(it, None)
                if g is None:
                    return
                key = (g["adapter_alias"], g["condition"], g["prompt_id"])
                if key not in seen:
                    seen.add(key)
                    picked.append(g)

        dis = [g for g in pool if disagrees(labels.get((g["adapter_alias"], g["condition"], g["prompt_id"]), {}))]
        take(dis, int(per_lang * DISAGREE_SHARE))
        by_stratum = defaultdict(list)
        for g in pool:
            by_stratum[strata[g["adapter_alias"]]].append(g)
        queues = [iter(by_stratum[s]) for s in STRATA if by_stratum[s]]
        while len(picked) < per_lang and queues:  # round-robin over strata
            for q in list(queues):
                before = len(picked)
                take(q, before + 1)
                if len(picked) == before:
                    queues.remove(q)
                if len(picked) >= per_lang:
                    break
        chosen += picked
    rng.shuffle(chosen)
    items, key = [], []
    for i, g in enumerate(chosen):
        item_id = f"L{i:03d}"
        items.append({"id": item_id, "lang": g["lang"], "kind": g["kind"], "prompt": prompts[g["prompt_id"]],
                      "reply": g["text"]})
        key.append({"id": item_id, "adapter_alias": g["adapter_alias"], "condition": g["condition"],
                    "prompt_id": g["prompt_id"], "lang": g["lang"], "stratum": strata[g["adapter_alias"]],
                    "retest_of": None})
    repeats = rng.sample(range(len(items)), min(n_retest, len(items)))
    half = len(items) // 2
    for j, src in enumerate(repeats):
        item_id = f"R{j:03d}"
        pos = rng.randint(half, len(items))
        items.insert(pos, {**items[src], "id": item_id})
        key.insert(pos, {**key[src], "id": item_id, "retest_of": key[src]["id"]})
    return items, key


# ---- agreement ---------------------------------------------------------------------------------

def read_native_csv(path: str | Path) -> list[dict]:
    """Rows exported by tools/label_ui.html (UTF-8 with BOM)."""
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def native_agreement(native: Sequence[dict], key: Sequence[dict], label_rows: Sequence[dict]) -> dict[str, Any]:
    """Per language: Cohen's κ between the judge's three-way label and the native label, the native test-retest
    κ, and the share of replies whose language the native speaker and the language identifier agree on."""
    from .judges import cohen_kappa

    lab = {(r["adapter_alias"], r["condition"], r["prompt_id"]): r for r in label_rows}
    by_id = {k["id"]: k for k in key}
    native_by_id = {n["id"]: n for n in native if n.get("label3")}
    out: dict[str, Any] = {}
    for lang in sorted({k["lang"] for k in key}):
        pairs, retest, lid = [], [], []
        for item_id, n in native_by_id.items():
            k = by_id.get(item_id)
            if k is None or k["lang"] != lang:
                continue
            if k["retest_of"]:
                first = native_by_id.get(k["retest_of"])
                if first:
                    retest.append((first["label3"], n["label3"]))
                continue
            judge = lab.get((k["adapter_alias"], k["condition"], k["prompt_id"]))
            if judge:
                pairs.append((judge["label3"], n["label3"]))
                if n.get("reply_lang"):
                    lid.append(judge.get("lid_lang") == n["reply_lang"])
        out[lang] = {
            "n": len(pairs),
            "kappa_judge_native": float(cohen_kappa([a for a, _ in pairs], [b for _, b in pairs])) if pairs else None,
            "n_retest": len(retest),
            "kappa_retest": float(cohen_kappa([a for a, _ in retest], [b for _, b in retest])) if retest else None,
            "lid_agreement": (sum(lid) / len(lid)) if lid else None,
            "native_only": bool(pairs) and float(cohen_kappa([a for a, _ in pairs], [b for _, b in pairs])) < 0.6,
        }
    return out


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--gen", nargs="+", required=True, help="raw generation folders or files (yo / pcm runs)")
    b.add_argument("--labels", nargs="+", required=True, help="judge label files for those runs")
    b.add_argument("--outcomes", required=True, help="CSV with columns alias and y (English drift), e.g. adapters.csv")
    b.add_argument("--panels-dir", type=Path, required=True)
    b.add_argument("--out-dir", type=Path, required=True)
    b.add_argument("--n-items", type=int, default=N_ITEMS)
    b.add_argument("--n-retest", type=int, default=N_RETEST)
    a = sub.add_parser("agree")
    a.add_argument("--native", required=True)
    a.add_argument("--key", required=True)
    a.add_argument("--labels", nargs="+", required=True)
    a.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        prompts = {r["id"]: r["text"] for r in _rows(sorted(args.panels_dir.glob("*.jsonl")))}
        with open(args.outcomes, encoding="utf-8") as fh:
            drift = {r["alias"]: float(r["y"]) for r in csv.DictReader(fh) if r.get("y") not in (None, "")}
        items, key = build_label_set(_rows(args.gen), _rows(args.labels), drift, prompts, args.n_items,
                                     args.n_retest)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "items.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items),
                                                  encoding="utf-8")
        (args.out_dir / "key.jsonl").write_text("".join(json.dumps(k) + "\n" for k in key), encoding="utf-8")
        print(f"{len(items)} items ({sum(k['retest_of'] is not None for k in key)} repeats) -> {args.out_dir}")
    else:
        result = native_agreement(read_native_csv(args.native), _rows([args.key]), _rows(args.labels))
        text = json.dumps(result, indent=1)
        if args.out:
            args.out.write_text(text)
        print(text)


if __name__ == "__main__":
    main()
