"""Statistics for the screening study: rank correlations, AUROC, author-cluster bootstrap, multiplicity
correction, exact paired tests, interval estimates, power, and the noise ceiling of the outcome.

All arithmetic is float64 NumPy (at least float32 precision, as the project requires). Every random
choice goes through `numpy.random.default_rng(seed)`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats as sps

ArrayLike = Sequence[float] | np.ndarray | pd.Series
COMPLIED = "complied"


class NoiseCeiling(NamedTuple):
    """Split-half reliability of an outcome and the highest correlation any predictor can reach with it."""
    split_half: float       # Spearman correlation between the two halves
    reliability: float      # Spearman-Brown corrected full-length reliability
    ceiling: float          # sqrt(reliability): the attainable correlation with a noise-free predictor


class PairedDelta(NamedTuple):
    """Change in harmful-compliance rate (adapter minus base) with its exact McNemar test."""
    delta_points: float     # percentage points
    p_value: float
    b: int                  # base did not comply, adapter complied
    c: int                  # base complied, adapter did not
    n: int                  # paired prompts


# =================================================================================================
# Point statistics
# =================================================================================================

def _as_float(x: ArrayLike) -> np.ndarray:
    """A 1-D float64 copy of `x`."""
    return np.asarray(x, dtype=np.float64).ravel()


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; NaN when either input is constant or shorter than 3."""
    if a.size < 3:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / denom if denom > 0 else float("nan")


def spearman(x: ArrayLike, y: ArrayLike) -> float:
    """Spearman's rho: Pearson correlation of average-tie ranks."""
    x, y = _as_float(x), _as_float(y)
    _check_same_size(x, y)
    return _pearson(sps.rankdata(x), sps.rankdata(y))


def _residualise(v: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    """Residuals of an ordinary least-squares fit of `v` on an intercept and `covariate`."""
    design = np.column_stack([np.ones_like(covariate), covariate])
    coef, *_ = np.linalg.lstsq(design, v, rcond=None)
    return v - design @ coef


def partial_spearman(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> float:
    """Partial Spearman rho of x and y given z: rank all three, residualise the x and y ranks on the z
    ranks, and correlate the residuals."""
    x, y, z = _as_float(x), _as_float(y), _as_float(z)
    _check_same_size(x, y, z)
    rz = sps.rankdata(z)
    return _pearson(_residualise(sps.rankdata(x), rz), _residualise(sps.rankdata(y), rz))


def auroc(scores: ArrayLike, labels: ArrayLike) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (ties count one half).

    `labels` are 1 / True for positives. Equals P(score_pos > score_neg) + 0.5 P(tie); NaN if either
    class is empty.
    """
    s = _as_float(scores)
    pos = np.asarray(labels).astype(bool).ravel()
    _check_same_size(s, pos)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    u = float(sps.rankdata(s)[pos].sum()) - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def _check_same_size(*arrays: np.ndarray) -> None:
    """Raise if the arrays differ in length."""
    sizes = {a.size for a in arrays}
    if len(sizes) > 1:
        raise ValueError(f"length mismatch: {sorted(sizes)}")


# =================================================================================================
# Cluster bootstrap
# =================================================================================================

def cluster_bootstrap_samples(stat_fn: Callable[[pd.DataFrame], float], df: pd.DataFrame, cluster_col: str,
                              n_boot: int = 2000, seed: int = 0) -> np.ndarray:
    """`stat_fn` on `n_boot` resamples that draw whole clusters with replacement (rows keep their cluster)."""
    groups = list(df.groupby(cluster_col, sort=True).indices.values())
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        picks = rng.integers(0, len(groups), size=len(groups))
        samples[i] = stat_fn(df.take(np.concatenate([groups[j] for j in picks])))
    return samples


def cluster_bootstrap(stat_fn: Callable[[pd.DataFrame], float], df: pd.DataFrame, cluster_col: str,
                      n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """(estimate on the full data, lower, upper) percentile interval from a cluster bootstrap.

    Resamples where the statistic is undefined (NaN) are ignored when taking the percentiles.
    """
    samples = cluster_bootstrap_samples(stat_fn, df, cluster_col, n_boot, seed)
    lo, hi = np.nanpercentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat_fn(df)), float(lo), float(hi)


def bootstrap_p_greater(samples: np.ndarray, null: float = 0.0) -> float:
    """One-sided bootstrap p-value for "statistic > null": (1 + #{samples <= null}) / (1 + #samples)."""
    finite = samples[np.isfinite(samples)]
    return float((1 + np.sum(finite <= null)) / (1 + finite.size))


def spearman_ci(df: pd.DataFrame, x: str, y: str, cluster_col: str, n_boot: int = 2000,
                seed: int = 0) -> tuple[float, float, float]:
    """Spearman rho of columns x and y with a cluster-bootstrap 95% interval."""
    return cluster_bootstrap(lambda d: spearman(d[x], d[y]), df, cluster_col, n_boot, seed)


def partial_spearman_ci(df: pd.DataFrame, x: str, y: str, z: str, cluster_col: str, n_boot: int = 2000,
                        seed: int = 0) -> tuple[float, float, float]:
    """Partial Spearman rho of x and y given z with a cluster-bootstrap 95% interval."""
    return cluster_bootstrap(lambda d: partial_spearman(d[x], d[y], d[z]), df, cluster_col, n_boot, seed)


def auroc_ci(df: pd.DataFrame, score: str, label: str, cluster_col: str, n_boot: int = 2000,
             seed: int = 0) -> tuple[float, float, float]:
    """AUROC of column `score` for binary column `label` with a cluster-bootstrap 95% interval."""
    return cluster_bootstrap(lambda d: auroc(d[score], d[label]), df, cluster_col, n_boot, seed)


# =================================================================================================
# Tests, intervals, power
# =================================================================================================

def holm(pvals: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in the input order."""
    p = _as_float(pvals)
    order = np.argsort(p, kind="stable")
    m = p.size
    stepped = np.minimum(1.0, (m - np.arange(m)) * p[order])
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = np.maximum.accumulate(stepped)
    return adjusted.tolist()


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts b and c (binomial, p = 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2.0 * sps.binom.cdf(min(b, c), n, 0.5)))


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k / n; (0, 1) when n is 0."""
    if n == 0:
        return 0.0, 1.0
    z = float(sps.norm.ppf(1 - alpha / 2))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def mde_spearman(n_clusters: int, power: float = 0.8, alpha: float = 0.05, two_sided: bool = False) -> float:
    """Smallest Spearman rho detectable with `n_clusters` independent units, via Fisher's z.

    Uses the Fieller-Hartley-Pearson variance 1.06 / (n - 3) for the z-transformed Spearman rho. The
    default test is one-sided, matching the directional hypotheses H1 to H3.
    """
    if n_clusters <= 3:
        raise ValueError("n_clusters must exceed 3")
    z_alpha = sps.norm.ppf(1 - (alpha / 2 if two_sided else alpha))
    z_power = sps.norm.ppf(power)
    return float(math.tanh((z_alpha + z_power) * math.sqrt(1.06 / (n_clusters - 3))))


def noise_ceiling(y_half1: ArrayLike, y_half2: ArrayLike) -> NoiseCeiling:
    """Noise ceiling of an outcome measured on two disjoint prompt halves (Spearman-Brown corrected).

    A split-half correlation of -1 or NaN gives NaN reliability and ceiling.
    """
    return spearman_brown(spearman(y_half1, y_half2))


def spearman_brown(r: float) -> NoiseCeiling:
    """Noise ceiling from a split-half correlation `r` (for example the mean over many random splits).

    Reliability is the Spearman-Brown correction ``2r / (1 + r)``; the ceiling is its square root, floored at 0.
    An `r` of -1 or NaN gives NaN reliability and ceiling.
    """
    if not r > -1.0:
        return NoiseCeiling(r, float("nan"), float("nan"))
    reliability = 2 * r / (1 + r)
    return NoiseCeiling(r, reliability, math.sqrt(max(reliability, 0.0)))


def percentile_of(value: float, null_scores: ArrayLike) -> float:
    """Percentile (0 to 100) of `value` within a null distribution, counting ties as one half."""
    null = _as_float(null_scores)
    return float(100.0 * (np.sum(null < value) + 0.5 * np.sum(null == value)) / null.size)


def paired_delta(base_labels: Mapping[str, str], adapter_labels: Mapping[str, str]) -> PairedDelta:
    """Harmful-compliance change in points (adapter minus base) with an exact McNemar p-value.

    Both mappings go from prompt_id to label3 over the same harmful prompts; a label counts as compliance
    when it is "complied".
    """
    if base_labels.keys() != adapter_labels.keys():
        raise ValueError("base and adapter labels must cover the same prompt ids")
    ids = sorted(base_labels)
    base = np.array([base_labels[i] == COMPLIED for i in ids])
    adapter = np.array([adapter_labels[i] == COMPLIED for i in ids])
    b = int(np.sum(~base & adapter))
    c = int(np.sum(base & ~adapter))
    delta = 100.0 * (b - c) / len(ids) if ids else float("nan")
    return PairedDelta(delta, mcnemar_exact(b, c), b, c, len(ids))
