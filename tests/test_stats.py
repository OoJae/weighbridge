import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from weighbridge import stats


def _tied_data(seed: int, n: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x, y, z with deliberate ties (rounded values)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = np.round(z + rng.normal(size=n), 1)
    y = np.round(0.5 * z + 0.5 * x + rng.normal(size=n), 1)
    return x, y, z


# ---- point statistics ----------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_spearman_matches_scipy(seed):
    x, y, _ = _tied_data(seed)
    assert stats.spearman(x, y) == pytest.approx(sps.spearmanr(x, y).statistic, abs=1e-12)


def test_spearman_constant_is_nan():
    assert math.isnan(stats.spearman([1, 1, 1, 1], [1, 2, 3, 4]))


def test_spearman_length_mismatch():
    with pytest.raises(ValueError):
        stats.spearman([1, 2, 3], [1, 2])


@pytest.mark.parametrize("seed", [0, 1])
def test_partial_spearman_matches_explicit(seed):
    x, y, z = _tied_data(seed)
    rx, ry, rz = sps.rankdata(x), sps.rankdata(y), sps.rankdata(z)
    design = np.column_stack([np.ones_like(rz), rz])
    ex = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    ey = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    assert stats.partial_spearman(x, y, z) == pytest.approx(np.corrcoef(ex, ey)[0, 1], abs=1e-12)


def test_partial_spearman_matches_first_order_formula():
    x, y, z = _tied_data(3)
    rxy, rxz, ryz = stats.spearman(x, y), stats.spearman(x, z), stats.spearman(y, z)
    formula = (rxy - rxz * ryz) / math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    assert stats.partial_spearman(x, y, z) == pytest.approx(formula, abs=1e-12)


def test_partial_spearman_removes_confounder():
    rng = np.random.default_rng(4)
    z = rng.normal(size=2000)
    x = z + 0.3 * rng.normal(size=2000)
    y = z + 0.3 * rng.normal(size=2000)
    assert stats.spearman(x, y) > 0.8
    assert abs(stats.partial_spearman(x, y, z)) < 0.1


def _auroc_pairs(scores: np.ndarray, labels: np.ndarray) -> float:
    """O(n^2) pair count: wins plus half ties over all positive-negative pairs."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    total = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return total / (len(pos) * len(neg))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_auroc_matches_pair_count_with_ties(seed):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=80)
    scores = np.round(labels * 0.5 + rng.normal(size=80), 0)   # heavy ties
    assert stats.auroc(scores, labels) == pytest.approx(_auroc_pairs(scores, labels), abs=1e-12)


def test_auroc_matches_sklearn():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(5)
    labels = rng.integers(0, 2, size=100)
    scores = np.round(rng.normal(size=100) + labels, 1)
    assert stats.auroc(scores, labels) == pytest.approx(metrics.roc_auc_score(labels, scores), abs=1e-12)


def test_auroc_single_class_is_nan():
    assert math.isnan(stats.auroc([0.1, 0.2], [1, 1]))


# ---- cluster bootstrap -----------------------------------------------------------------------------

def _clustered(seed: int, n_clusters: int = 40, per_cluster: int = 5, icc_sd: float = 1.0) -> pd.DataFrame:
    """Rows with a shared cluster effect, so rows within a cluster are strongly correlated."""
    rng = np.random.default_rng(seed)
    cluster = np.repeat(np.arange(n_clusters), per_cluster)
    effect = rng.normal(scale=icc_sd, size=n_clusters)[cluster]
    y = effect + 0.3 * rng.normal(size=cluster.size)
    x = y + rng.normal(size=cluster.size)
    return pd.DataFrame({"author": cluster, "row": np.arange(cluster.size), "x": x, "y": y})


def _mean_y(d: pd.DataFrame) -> float:
    return float(d["y"].mean())


def test_cluster_bootstrap_is_deterministic():
    df = _clustered(0)
    first = stats.cluster_bootstrap(_mean_y, df, "author", n_boot=200, seed=7)
    second = stats.cluster_bootstrap(_mean_y, df, "author", n_boot=200, seed=7)
    other = stats.cluster_bootstrap(_mean_y, df, "author", n_boot=200, seed=8)
    assert first == second
    assert first != other
    estimate, lo, hi = first
    assert estimate == pytest.approx(df["y"].mean())
    assert lo < estimate < hi


def test_cluster_bootstrap_keeps_clusters_whole():
    df = _clustered(1, n_clusters=10, per_cluster=3)
    sizes = stats.cluster_bootstrap_samples(lambda d: float(len(d)), df, "author", n_boot=50, seed=0)
    assert np.all(sizes == 30)
    multiples = stats.cluster_bootstrap_samples(
        lambda d: float(d.groupby("author").size().mod(3).sum()), df, "author", n_boot=50, seed=0)
    assert np.all(multiples == 0)


def test_cluster_bootstrap_coverage_beats_row_bootstrap():
    """With a strong cluster effect, the cluster bootstrap covers the true mean (0) near 95% of the
    time, while a naive row-level bootstrap is far too narrow."""
    cluster_hits, row_hits, n_sims = 0, 0, 60
    for sim in range(n_sims):
        df = _clustered(100 + sim)
        _, lo, hi = stats.cluster_bootstrap(_mean_y, df, "author", n_boot=300, seed=sim)
        cluster_hits += lo <= 0.0 <= hi
        _, lo, hi = stats.cluster_bootstrap(_mean_y, df, "row", n_boot=300, seed=sim)
        row_hits += lo <= 0.0 <= hi
    assert cluster_hits / n_sims >= 0.85
    assert row_hits / n_sims < 0.75


def test_ci_wrappers_match_point_statistics():
    df = _clustered(2)
    df["label"] = (df["y"] > 0).astype(int)
    df["z"] = df["y"] + np.random.default_rng(0).normal(size=len(df))
    rho, lo, hi = stats.spearman_ci(df, "x", "y", "author", n_boot=200)
    assert rho == pytest.approx(stats.spearman(df["x"], df["y"]))
    assert lo <= rho <= hi
    prho, plo, phi = stats.partial_spearman_ci(df, "x", "y", "z", "author", n_boot=200)
    assert prho == pytest.approx(stats.partial_spearman(df["x"], df["y"], df["z"]))
    assert plo <= prho <= phi
    auc, alo, ahi = stats.auroc_ci(df, "x", "label", "author", n_boot=200)
    assert auc == pytest.approx(stats.auroc(df["x"], df["label"]))
    assert 0.0 <= alo <= auc <= ahi <= 1.0


def test_bootstrap_p_greater():
    assert stats.bootstrap_p_greater(np.array([0.1, 0.2, 0.3, -0.1])) == pytest.approx(2 / 5)
    assert stats.bootstrap_p_greater(np.array([0.5, np.nan, 0.6])) == pytest.approx(1 / 3)


# ---- tests, intervals, power ---------------------------------------------------------------------

def test_holm_known_vector():
    assert stats.holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_caps_at_one_and_matches_brute_force():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=7)
    m = p.size
    order = np.argsort(p)
    brute = np.empty(m)
    for rank, idx in enumerate(order):
        brute[idx] = min(1.0, max((m - j) * p[order[j]] for j in range(rank + 1)))
    assert stats.holm(p) == pytest.approx(brute.tolist())
    assert max(stats.holm([0.9, 0.8])) == 1.0


@pytest.mark.parametrize("b, c, expected", [
    (1, 9, 2 * 11 / 1024),
    (5, 5, 1.0),
    (0, 0, 1.0),
    (0, 6, 2 / 64),
])
def test_mcnemar_exact_known_counts(b, c, expected):
    assert stats.mcnemar_exact(b, c) == pytest.approx(expected)


@pytest.mark.parametrize("b, c", [(3, 12), (20, 31), (7, 2)])
def test_mcnemar_matches_scipy_binomtest(b, c):
    assert stats.mcnemar_exact(b, c) == pytest.approx(sps.binomtest(b, b + c, 0.5).pvalue)


@pytest.mark.parametrize("k, n", [(5, 10), (0, 20), (20, 20), (37, 150)])
def test_wilson_matches_scipy(k, n):
    ci = sps.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    assert stats.wilson(k, n) == pytest.approx((ci.low, ci.high), abs=1e-12)


def test_wilson_empty():
    assert stats.wilson(0, 0) == (0.0, 1.0)


def test_mde_spearman():
    expected = math.tanh((sps.norm.ppf(0.95) + sps.norm.ppf(0.8)) * math.sqrt(1.06 / 110))
    assert stats.mde_spearman(113) == pytest.approx(expected)
    assert stats.mde_spearman(72, two_sided=True) > stats.mde_spearman(72)
    assert stats.mde_spearman(200) < stats.mde_spearman(72)
    with pytest.raises(ValueError):
        stats.mde_spearman(3)


def test_noise_ceiling():
    perfect = stats.noise_ceiling([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
    assert perfect == pytest.approx((1.0, 1.0, 1.0))
    rng = np.random.default_rng(0)
    truth = rng.normal(size=500)
    h1, h2 = truth + rng.normal(size=500), truth + rng.normal(size=500)
    r = stats.spearman(h1, h2)
    nc = stats.noise_ceiling(h1, h2)
    assert nc.split_half == pytest.approx(r)
    assert nc.reliability == pytest.approx(2 * r / (1 + r))
    assert nc.ceiling == pytest.approx(math.sqrt(2 * r / (1 + r)))
    assert math.isnan(stats.noise_ceiling([1, 2, 3], [3, 2, 1]).reliability)


def test_percentile_of():
    null = np.arange(100, dtype=float)
    assert stats.percentile_of(1000.0, null) == 100.0
    assert stats.percentile_of(-1.0, null) == 0.0
    assert stats.percentile_of(50.0, null) == pytest.approx(50.5)


def test_paired_delta():
    ids = [f"p{i}" for i in range(10)]
    base = dict.fromkeys(ids, "refused") | {"p0": "complied"}
    adapter = dict.fromkeys(ids, "complied") | {"p0": "refused", "p1": "deflected"}
    result = stats.paired_delta(base, adapter)
    # base complies on 1/10, adapter on 8/10: +70 points; b = 8, c = 1.
    assert result.delta_points == pytest.approx(70.0)
    assert (result.b, result.c, result.n) == (8, 1, 10)
    assert result.p_value == pytest.approx(stats.mcnemar_exact(8, 1))


def test_paired_delta_requires_same_prompts():
    with pytest.raises(ValueError):
        stats.paired_delta({"a": "complied"}, {"b": "complied"})
