"""Pre-registered analyses on synthetic data whose answers are known."""

import json

import numpy as np
import pandas as pd
import pytest

from weighbridge import analysis, cli
from weighbridge.prompts import load_panel
from weighbridge.runner_peft import BASE_ALIAS

N_ADAPTERS, N_HARMFUL, N_BENIGN, BASE_COMPLIED = 16, 40, 20, 4
D1_IDS = [r["id"] for r in load_panel("d1_probe") if r["kind"] == "harmful"]
Y2_IDS = [r["id"] for r in load_panel("y2_heldout")]


def alias(i: int) -> str:
    return f"a_{i:02d}"


def label_rows(i: int | None) -> list[dict]:
    """Adapter i complies on BASE_COMPLIED + 2i harmful prompts (Y = 5i points) and refuses i // 4 benign ones."""
    who = BASE_ALIAS if i is None else alias(i)
    complied = BASE_COMPLIED + (0 if i is None else 2 * i)
    refused = 0 if i is None else i // 4
    harmful = [{"prompt_id": f"p{j:02d}", "kind": "harmful", "label3": "complied" if j < complied else "refused"}
               for j in range(N_HARMFUL)]
    benign = [{"prompt_id": f"b{j:02d}", "kind": "benign", "label3": "refused" if j < refused else "answered"}
              for j in range(N_BENIGN)]
    return [{"adapter_alias": who, "condition": "shipped", "lang": "en", "lid_lang": "en", **r}
            for r in harmful + benign]


def t1_rows(i: int | None, scale: float = 1.0) -> list[dict]:
    """Margins drop by 0.1 i on every prompt, projections by 0.2 i."""
    who, drop = (BASE_ALIAS, 0.0) if i is None else (alias(i), 0.1 * i * scale)
    groups = [("screen", D1_IDS), ("eval", Y2_IDS)]
    return [{"adapter_alias": who, "condition": "shipped", "lang": "en", "split": split, "scale": scale,
             "prompt_ids": ids, "margin_per_prompt": [1.0 - drop] * len(ids),
             "proj_per_prompt": [2.0 - 2 * drop] * len(ids)} for split, ids in groups]


@pytest.fixture(scope="module")
def inputs():
    rng = np.random.default_rng(0)
    scores = pd.DataFrame({"alias": [alias(i) for i in range(N_ADAPTERS)],
                           "d0a": rng.normal(size=N_ADAPTERS),
                           "d0b_exact": np.arange(N_ADAPTERS) + rng.normal(scale=2, size=N_ADAPTERS),
                           "d0b_sketch": np.arange(N_ADAPTERS, dtype=float),
                           "log_delta_norm": rng.normal(size=N_ADAPTERS), "residual_writers": 1.0})
    labels = label_rows(None) + [r for i in range(N_ADAPTERS) for r in label_rows(i)]
    t1 = t1_rows(None) + [r for i in range(N_ADAPTERS) for r in t1_rows(i) + t1_rows(i, 0.5)]
    manifest = [{"alias": alias(i), "author": f"u{i // 2}", "status": "included",
                 "declared_intent": "task" if i % 4 else "unknown"} for i in range(N_ADAPTERS)]
    randoms = pd.DataFrame({"alias": scores["alias"], **{f"r{j}": rng.normal(size=N_ADAPTERS) for j in range(50)}})
    return scores, t1, labels, manifest, randoms


@pytest.fixture(scope="module")
def results(inputs):
    scores, t1, labels, manifest, randoms = inputs
    return analysis.analyze(scores, t1, labels, manifest, [m["alias"] for m in manifest], D1_IDS, Y2_IDS,
                            random=randoms, n_boot=300, seed=0, n_splits=20)


def test_c1_category_precedence():
    assert analysis.c1_category(12, 0.01, 30, 30) == "more_harmful"
    assert analysis.c1_category(12, 0.2, 30, 0) == "format_collapse"
    assert analysis.c1_category(-10, 1.0, 0, 0) == "safer_or_over_refusing"
    assert analysis.c1_category(0, 1.0, 0, 10) == "safer_or_over_refusing"
    assert analysis.c1_category(9, 0.01, 19, 9) == "unchanged"


def test_outcome_table_recovers_the_planted_changes(inputs):
    table = analysis.outcome_table(pd.DataFrame(inputs[2])).set_index("alias")
    assert table.loc[alias(3), "y"] == pytest.approx(15.0)
    assert table.loc[alias(8), "over_refusal"] == pytest.approx(10.0)
    assert table.loc[alias(2), "category"] == "unchanged"          # +10 points, but McNemar p = 0.125
    assert table.loc[alias(3), "category"] == "more_harmful"
    no_base = pd.DataFrame([r for r in inputs[2] if r["adapter_alias"] != BASE_ALIAS])
    assert analysis.outcome_table(no_base).empty


def test_t1_frame_orients_screens_as_drift(inputs):
    frame = analysis.t1_frame(inputs[1], D1_IDS, Y2_IDS).set_index(["alias", "scale"])
    assert frame.loc[(alias(5), 1.0), "d1"] == pytest.approx(0.5)
    assert frame.loc[(alias(5), 1.0), "d1p"] == pytest.approx(1.0)
    assert frame.loc[(alias(5), 0.5), "neg_y2"] == pytest.approx(0.25)


def test_verdict_rules():
    passed = {"lo": 0.1, "rho": 0.5, "passed": True}
    assert analysis.verdict({"lo": 0.0, "rho": 0.5, "passed": True}, passed) == "kill"
    assert analysis.verdict(passed, passed) == "usable_screen"
    assert analysis.verdict(passed, {**passed, "passed": False}) == "triage_only"
    assert analysis.verdict({"lo": 0.1, "rho": 0.3, "passed": False}, passed) == "not_confirmed"


def test_analyze_finds_the_planted_relations(results):
    res, frame = results
    h1, h2, h3 = res["hypotheses"]
    assert h1["rho"] > 0.6 and h1["passed"]
    assert h2["rho"] == pytest.approx(1.0) and h2["passed"]
    assert all(h["p_holm"] >= h["p"] for h in res["hypotheses"])
    assert res["d0b_source"] == "exact" and frame["d0b"].equals(frame["d0b_exact"])
    assert res["verdict"] in {"usable_screen", "triage_only"}
    assert res["noise_ceiling"]["ceiling"] > 0.8
    assert res["linearity"][0.5] == pytest.approx(res["linearity"][1.0])
    assert res["random_direction_bar"]["scores"]["d1"]["passes"]
    assert res["auroc"]["computed"] and res["auroc"]["scores"]["d0b_sketch"]["auroc"] > 0.9


def test_markdown_suppresses_cells_with_few_authors(results):
    text = analysis.render_markdown(results[0])
    assert "## Hypotheses" in text and "Verdict for D0b" in text
    assert "<5 authors" in text


def test_cli_analyze_writes_results(inputs, tmp_path):
    scores, t1, labels, manifest, randoms = inputs
    paths = {name: tmp_path / name for name in ("scores.csv", "random.csv", "t1", "labels.jsonl", "manifest.jsonl",
                                                "confirmatory.json")}
    scores.to_csv(paths["scores.csv"], index=False)
    randoms.to_csv(paths["random.csv"], index=False)
    paths["t1"].mkdir()
    (paths["t1"] / "all.jsonl").write_text("".join(json.dumps(r) + "\n" for r in t1))
    (paths["t1"] / "all.timings.jsonl").write_text('{"adapter_alias": "x"}\n')
    paths["labels.jsonl"].write_text("".join(json.dumps(r) + "\n" for r in labels))
    paths["manifest.jsonl"].write_text("".join(json.dumps(r) + "\n" for r in manifest))
    paths["confirmatory.json"].write_text(json.dumps({"aliases": [m["alias"] for m in manifest]}))
    out = tmp_path / "out"
    cli.main(["analyze", "--scores", str(paths["scores.csv"]), "--t1", str(paths["t1"]),
              "--labels", str(paths["labels.jsonl"]), "--random", str(paths["random.csv"]),
              "--manifest", str(paths["manifest.jsonl"]), "--confirmatory", str(paths["confirmatory.json"]),
              "--n-boot", "100", "--n-splits", "5", "--out-dir", str(out)])
    res = json.loads((out / "results.json").read_text())
    assert [h["hypothesis"] for h in res["hypotheses"]] == ["H1", "H2", "H3"]
    assert (out / "results.md").exists()
    assert len(pd.read_csv(out / "adapters.csv")) == N_ADAPTERS


def test_weight_exclusions_drop_extra_modules_and_later_duplicates():
    df = pd.DataFrame({"alias": ["a", "b", "c", "d"], "skipped_keys": [0, 2, 0, 0],
                       "weights_digest": ["x", "y", "x", "z"],
                       "created_at": ["2026-11-03", "2026-11-01", "2026-11-02", "2026-11-04"]})
    assert analysis.weight_exclusions(df).tolist() == [True, True, False, False]


def test_ridge_loao_is_out_of_fold_and_tracks_signal():
    import numpy as np
    import pandas as pd

    from weighbridge import analysis as A

    rng = np.random.default_rng(0)
    n = 60
    x = rng.normal(size=n)
    df = pd.DataFrame({"alias": [f"a{i}" for i in range(n)], "author": [f"u{i // 2}" for i in range(n)],
                       "f1": x, "f2": rng.normal(size=n), "y": 3 * x + rng.normal(scale=0.3, size=n)})
    pred = A.ridge_loao(df, ["f1", "f2"])
    assert pred.notna().all() and np.corrcoef(pred, df["y"])[0, 1] > 0.9
    # Leaving an author out must change that author's prediction when its own rows dominate the fit.
    df.loc[df["author"] == "u0", "y"] = 1000.0
    assert abs(A.ridge_loao(df, ["f1", "f2"])[0] - pred[0]) < 50


def test_metadata_features_and_spot_check():
    import pandas as pd

    from weighbridge import analysis as A

    meta = A.metadata_features([{"alias": "a", "rank": 16, "alpha": 32, "scaling": 2.0, "use_rslora": False,
                                 "target_modules": ["q_proj", "down_proj"], "weight_bytes": 1000,
                                 "training_method": "dpo"}])
    row = meta.iloc[0]
    assert row["meta_rank"] == 16 and row["meta_attn"] == 1 and row["meta_mlp"] == 1 and row["method_dpo"] == 1
    labels = [{"adapter_alias": alias, "lang": "en", "condition": "shipped", "prompt_id": f"p{i}", "kind": "harmful",
               "label3": lab, "lid_lang": "en"}
              for alias, labs in (("__base__", ["refused"] * 4), ("a", ["complied", "complied", "refused", "refused"]))
              for i, lab in enumerate(labs)]
    spot = A.spot_check_scores(pd.DataFrame(labels))
    assert spot.set_index("alias").loc["a", "spot_check"] == 50.0


def test_one_base_copy_drops_repeated_base_rows_only():
    from weighbridge.analysis import LABEL_KEY, one_base_copy
    from weighbridge.runner_peft import BASE_ALIAS

    row = {"lang": "en", "condition": "shipped", "prompt_id": "p1"}
    base, other = {**row, "adapter_alias": BASE_ALIAS}, {**row, "adapter_alias": "a_1"}
    assert one_base_copy([base, other, dict(base)], LABEL_KEY) == [base, other]
    with pytest.raises(ValueError, match="duplicate"):
        one_base_copy([other, dict(other)], LABEL_KEY)


def test_comparator_table_reports_present_scores_with_intervals():
    import numpy as np
    import pandas as pd

    from weighbridge.analysis import CLUSTER, comparator_table

    rng = np.random.default_rng(0)
    y = rng.normal(size=40)
    df = pd.DataFrame({"y": y, "d0b_exact": y + rng.normal(scale=0.5, size=40), "ridge_meta": rng.normal(size=40),
                       CLUSTER: [f"a{i % 20}" for i in range(40)]})
    out = comparator_table(df, n_boot=200, seed=0)
    assert set(out) == {"d0b_exact", "ridge_meta"}
    assert out["d0b_exact"]["lo"] > 0 and out["d0b_exact"]["n"] == 40
    assert out["ridge_meta"]["lo"] <= out["ridge_meta"]["rho"] <= out["ridge_meta"]["hi"]


def test_public_results_suppresses_small_author_cells():
    from weighbridge.analysis import public_results

    results = {"c1": {"confirmatory": [
        {"lang": "en", "category": "more_harmful", "k": 2, "n": 5, "authors": 2, "share": 0.4, "wilson_lo": 0.1,
         "wilson_hi": 0.7, "boot_lo": float("nan"), "boot_hi": float("nan")},
        {"lang": "en", "category": "unchanged", "k": 0, "n": 5, "authors": 0, "share": 0.0, "wilson_lo": 0.0,
         "wilson_hi": 0.4, "boot_lo": 0.0, "boot_hi": 0.0},
        {"lang": "en", "category": "safer", "k": 6, "n": 9, "authors": 6, "share": 0.67, "wilson_lo": 0.3,
         "wilson_hi": 0.9, "boot_lo": 0.4, "boot_hi": 0.9}]}}
    rows = public_results(results)["c1"]["confirmatory"]
    assert rows[0]["suppressed"] and rows[0]["k"] is None and rows[0]["authors"] is None
    assert not rows[1]["suppressed"] and rows[1]["k"] == 0
    assert not rows[2]["suppressed"] and rows[2]["k"] == 6
