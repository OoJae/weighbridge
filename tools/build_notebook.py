"""Write notebooks/reproduce.ipynb: every published table rebuilt on CPU from the aggregate files in results/."""
import json
from pathlib import Path

CELLS = [
("md", """# Weighbridge: reproduce the published tables

Runs on a CPU in seconds. Inputs are the aggregate files in `results/` (no adapter is identified in them):

- `results_public.json`: the pre-registered analysis output (`weighbridge analyze`), with small-author cells suppressed;
- `judge_validation_en.json`: Qwen3Guard against JailbreakBench human labels, per item;
- `native_agreement.json`: judge-versus-native label counts for Yoruba and Pidgin.

Per-adapter rows of third-party adapters are private until final results (see `docs/ETHICS.md`), so the hypothesis
tests cannot be re-run from public files; this notebook re-derives everything that can be checked from the aggregates:
the verdict rule, the Holm correction, every κ, and the C1 table with its suppression."""),
("code", """import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path.cwd() if (Path.cwd() / "results").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
from weighbridge.analysis import verdict
from weighbridge.judges import cohen_kappa
from weighbridge.stats import holm
R = json.loads((ROOT / "results/results_public.json").read_text())
print(R["n_adapters"], "adapters,", R["n_authors"], "authors; D0b from the", R["d0b_source"], "gradient")"""),
("md", "## H1–H3 and the pre-registered verdict"),
("code", """h = pd.DataFrame(R["hypotheses"])
assert [round(x, 12) for x in holm(h["p"].tolist())] == [round(x, 12) for x in h["p_holm"]], "Holm mismatch"
print("verdict rule ->", verdict(R["hypotheses"][0], R["hypotheses"][2]), "| reported:", R["verdict"])
h[["hypothesis", "score", "given", "rho", "lo", "hi", "p_holm", "passed"]].round(3)"""),
("md", "## Screens and baselines (ρ with Y, author-cluster 95% CI)"),
("code", """pd.DataFrame(R["comparators"]).T[["n", "rho", "lo", "hi"]].astype(float).round(3)"""),
("code", """if "random_direction_bar" in R:
    bar = R["random_direction_bar"]
    print("95th percentile of", bar["n_directions"], "random directions:", round(bar["null_p95"], 3))
    display(pd.DataFrame(bar["scores"]).T)
print("noise ceiling:", {k: round(v, 3) for k, v in R["noise_ceiling"].items()})"""),
("md", "## C1: what fine-tuning did (cells with a non-zero count from fewer than 5 authors are suppressed)\n\n"
       "Pidgin rows are judge-based and shown for completeness only: the judge's κ against native labels is below 0.6, "
       "so the write-up uses native labels for Pidgin, as pre-registered."),
("code", """c1 = pd.DataFrame(R["c1"]["confirmatory"])
def cell(r):
    if r["suppressed"]:
        return "<5 authors"
    return f"{int(r['k'])}/{int(r['n'])} ({100 * r['share']:.0f}%, {100 * r['wilson_lo']:.0f}-{100 * r['wilson_hi']:.0f})"
c1["cell"] = c1.apply(cell, axis=1)
c1.pivot_table(index=["lang", "condition", "intent"], columns="category", values="cell", aggfunc="first")"""),
("md", "## Judges: English against human labels"),
("code", """J = json.loads((ROOT / "results/judge_validation_en.json").read_text())
human = [it["human"] for it in J["per_item"]]
judge = [int(it["judge_safety"] in ("Unsafe", "Controversial")) for it in J["per_item"]]
print("kappa recomputed:", round(cohen_kappa(human, judge), 3), "| reported:", round(J["judge"]["kappa"], 3))"""),
("md", "## Judges: Yoruba and Pidgin against native labels"),
("code", """N = json.loads((ROOT / "results/native_agreement.json").read_text())
rows = []
for lang, d in N["languages"].items():
    a = [p["judge"] for p in d["judge_vs_native"] for _ in range(p["count"])]
    b = [p["native"] for p in d["judge_vs_native"] for _ in range(p["count"])]
    r1 = [p["first"] for p in d["retest"] for _ in range(p["count"])]
    r2 = [p["second"] for p in d["retest"] for _ in range(p["count"])]
    rows.append({"lang": lang, "n": len(a), "kappa_judge_native": cohen_kappa(a, b), "n_retest": len(r1),
                 "kappa_retest": cohen_kappa(r1, r2), "lid_agreement": d["lid_agree"] / d["lid_n"]})
pd.DataFrame(rows).round(3)"""),
("code", """g = pd.concat([pd.DataFrame(d["native_by_group"]).assign(lang=l) for l, d in N["languages"].items()])
g.pivot_table(index=["lang", "group", "prompt"], columns="native", values="count", aggfunc="sum", fill_value=0)"""),
]


def main() -> None:
    cells = []
    for kind, src in CELLS:
        lines = src.splitlines(keepends=True)
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines})
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    Path("notebooks/reproduce.ipynb").write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
