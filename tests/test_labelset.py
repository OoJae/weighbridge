"""Native labelling sample and judge-native agreement (PREREG 8)."""

from __future__ import annotations

import csv
import json

from weighbridge import labelset as L


def _data():
    aliases = ["__base__"] + [f"a{i}" for i in range(9)]
    drift = {f"a{i}": float(i) for i in range(9)}
    prompts = {f"{lang}-p{j}": f"prompt {lang} {j}" for lang in ("yo", "pcm") for j in range(30)}
    gen, labels = [], []
    for a in aliases:
        for lang in ("yo", "pcm"):
            for j in range(30):
                pid = f"{lang}-p{j}"
                gen.append({"adapter_alias": a, "condition": "shipped", "prompt_id": pid, "lang": lang,
                            "kind": "harmful" if j < 20 else "benign", "text": f"reply {a} {pid}"})
                labels.append({"adapter_alias": a, "condition": "shipped", "prompt_id": pid, "lang": lang,
                               "label3": "refused" if j % 2 else "complied", "guard_refusal": "Yes" if j % 2 else "No",
                               "string_refusal": bool(j % 2) if j % 7 else not bool(j % 2), "lid_lang": lang})
    return gen, labels, drift, prompts


def test_build_label_set_is_blind_balanced_and_has_a_retest_block():
    gen, labels, drift, prompts = _data()
    items, key = L.build_label_set(gen, labels, drift, prompts, n_items=60, n_retest=10, seed=0)
    assert len(items) == len(key) == 70
    assert all(set(i) == {"id", "lang", "kind", "prompt", "reply"} for i in items)  # no adapter alias shown
    main = [k for k in key if k["retest_of"] is None]
    assert sum(k["lang"] == "yo" for k in main) == sum(k["lang"] == "pcm" for k in main) == 30
    assert {k["stratum"] for k in main} == {"base", "low", "mid", "high"}
    assert len({(k["adapter_alias"], k["prompt_id"]) for k in main}) == 60
    lab = {(r["adapter_alias"], r["condition"], r["prompt_id"]): r for r in labels}
    n_dis = sum(L.disagrees(lab[(k["adapter_alias"], k["condition"], k["prompt_id"])]) for k in main)
    assert n_dis >= 2 * (30 // 3)  # the disagreement share is filled first in each language
    positions = [i for i, k in enumerate(key) if k["retest_of"]]
    assert len(positions) == 10 and min(positions) >= 30
    by_id = {i["id"]: i for i in items}
    for k in key:
        if k["retest_of"]:
            assert by_id[k["id"]]["reply"] == by_id[k["retest_of"]]["reply"]
    again, _ = L.build_label_set(gen, labels, drift, prompts, n_items=60, n_retest=10, seed=0)
    assert again == items


def test_native_agreement(tmp_path):
    gen, labels, drift, prompts = _data()
    items, key = L.build_label_set(gen, labels, drift, prompts, n_items=40, n_retest=6, seed=1)
    lab = {(r["adapter_alias"], r["condition"], r["prompt_id"]): r for r in labels}
    path = tmp_path / "native.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "lang", "kind", "label3", "reply_lang", "intelligibility", "notes"])
        w.writeheader()
        for k in key:
            src = k if not k["retest_of"] else next(x for x in key if x["id"] == k["retest_of"])
            judge = lab[(src["adapter_alias"], src["condition"], src["prompt_id"])]
            w.writerow({"id": k["id"], "lang": k["lang"], "kind": "harmful", "label3": judge["label3"],
                        "reply_lang": k["lang"], "intelligibility": 3, "notes": ""})
    result = L.native_agreement(L.read_native_csv(path), key, labels)
    for lang in ("yo", "pcm"):
        assert result[lang]["kappa_judge_native"] == 1.0 and result[lang]["lid_agreement"] == 1.0
        assert result[lang]["native_only"] is False
    assert sum(result[lang]["n_retest"] for lang in ("yo", "pcm")) == 6
    json.dumps(result)
