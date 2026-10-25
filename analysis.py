"""Statistics: tests and summaries of stored metrics, never a metric.

The metrics themselves (S_T, dAcc, S_E, hit rate, ...) are defined once in measurement.py and come
from results/store; this module only tests and summarises them.

Conventions
  wilcoxon_paired(a, b)   differences d = b - a (e.g. a = baseline / GE, b = FEC / GP); zeros dropped
                          (scipy's default); p only with >= MIN_PAIRS pairs (results.py's rule)
  rank-biserial           r = (W+ - W-) / (W+ + W-), ranks of |d| over non-zero d (Kerby 2014);
                          the reference implementation used scipy's two-sided statistic min(W+, W-), always <= 0 (fixed)
  holm, bonferroni, bh    adjusted p-values over the non-NaN entries, NaN kept, input order kept
  sd                      ddof = 1
  bayes_correlated_t      Bayesian correlated t-test (Corani & Benavoli 2015), one dataset: the paired
                          differences of repeated random train/test splits; Nadeau-Bengio variance for
                          random splits, 1/n + n_test/n_train (baycomp's own correction assumes k-fold
                          CV); posterior = baycomp's Posterior (Student t)
  bayes_signed_rank       Bayesian signed-rank test (Benavoli et al. 2017, baycomp), across datasets:
                          one value per dataset; fixed random_state, so the probabilities are reproducible
  Bayesian outputs        (P(left), P(rope), P(right)) for d = b - a: left = b worse by more than the rope
"""
from __future__ import annotations

import itertools

import baycomp
import numpy as np
import pandas as pd
from baycomp.single import Posterior
from scipy.stats import friedmanchisquare, pearsonr, rankdata, spearmanr, wilcoxon

MIN_PAIRS = 6            # = results.MIN_PAIRS_FOR_TEST
NAN = float("nan")


def _signed_rank(d, alternative: str = "two-sided") -> dict:
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    nz = d[d != 0]
    out = {"n": len(d), "n_nonzero": len(nz), "median_diff": float(np.median(d)) if len(d) else NAN,
           "w_plus": NAN, "w_minus": NAN, "r_rb": NAN, "p": NAN}
    if len(nz):
        ranks = rankdata(np.abs(nz))
        wp, wm = float(ranks[nz > 0].sum()), float(ranks[nz < 0].sum())
        out.update(w_plus=wp, w_minus=wm, r_rb=(wp - wm) / (wp + wm))
        if len(d) >= MIN_PAIRS:
            out["p"] = float(wilcoxon(d, alternative=alternative).pvalue)
    return out


def wilcoxon_paired(a, b) -> dict:
    """Paired Wilcoxon signed-rank on d = b - a (pairs with a NaN dropped)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired samples must have the same length")
    return _signed_rank(b - a)


def wilcoxon_one_sample(x, mu: float, alternative: str = "two-sided") -> dict:
    """One-sample Wilcoxon of x against mu (the reference implementation: speedups > 1 one-sided "greater", dAcc two-sided)."""
    return _signed_rank(np.asarray(x, dtype=float) - mu, alternative)


def friedman_kendall(table: pd.DataFrame) -> dict:
    """Friedman test over the columns (e.g. fractions) with rows = blocks (datasets); complete rows only; needs
    k >= 3 columns (scipy refuses two: use the paired Wilcoxon then).
    Kendall's W = chi2 / (n (k - 1))."""
    t = table.dropna()
    n, k = t.shape
    out = {"n": n, "k": k, "chi2": NAN, "p": NAN, "W": NAN}
    if n < 3 or k < 3 or np.allclose(t.to_numpy(), t.to_numpy()[:, :1]):
        return out
    chi2, p = friedmanchisquare(*[t[c].to_numpy() for c in t.columns])
    out.update(chi2=float(chi2), p=float(p), W=float(chi2 / (n * (k - 1))))
    return out


def pairwise_wilcoxon(table: pd.DataFrame) -> pd.DataFrame:
    """Every pair of columns, paired over rows; Bonferroni and Holm over the pairs."""
    rows = []
    for c1, c2 in itertools.combinations(table.columns, 2):
        pair = table[[c1, c2]].dropna()
        rows.append({"a": c1, "b": c2, **wilcoxon_paired(pair[c1], pair[c2])})
    df = pd.DataFrame(rows, columns=["a", "b", "n", "n_nonzero", "median_diff", "w_plus", "w_minus", "r_rb", "p"])
    df["p_bonferroni"] = bonferroni(df["p"])
    df["p_holm"] = holm(df["p"])
    return df


def _valid(p) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=float)
    return p, np.flatnonzero(~np.isnan(p))


def bonferroni(p) -> np.ndarray:
    p, idx = _valid(p)
    out = np.full(len(p), NAN)
    out[idx] = np.minimum(1.0, p[idx] * len(idx))
    return out


def holm(p) -> np.ndarray:
    p, idx = _valid(p)
    out = np.full(len(p), NAN)
    m = len(idx)
    order = idx[np.argsort(p[idx], kind="stable")]
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[j]))
        out[j] = running
    return out


def bayes_correlated_t(a, b, rope: float, n_test: float, n_train: float) -> tuple[float, float, float]:
    """P(b - a < -rope), P(|b - a| <= rope), P(b - a > rope) for paired scores of random train/test splits."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    if len(d) < 2:
        return NAN, NAN, NAN
    var = np.var(d, ddof=1) * (1 / len(d) + n_test / n_train)
    return tuple(float(p) for p in Posterior(float(d.mean()), var, len(d) - 1, rope).probs())


def bayes_signed_rank(a, b, rope: float, nsamples: int = 50000) -> tuple[float, float, float]:
    """P(left), P(rope), P(right) for b - a, one pair per dataset (a = baseline, b = FEC)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2:
        return NAN, NAN, NAN
    return tuple(float(p) for p in baycomp.SignedRankTest.probs(a, b, rope, nsamples=nsamples, random_state=0))


def bh_fdr(p) -> np.ndarray:
    p, idx = _valid(p)
    out = np.full(len(p), NAN)
    m = len(idx)
    order = idx[np.argsort(p[idx], kind="stable")]
    running = 1.0
    for i in range(m - 1, -1, -1):
        j = order[i]
        running = min(running, p[j] * m / (i + 1))
        out[j] = running
    return out


def correlations(df: pd.DataFrame, outcomes: list[str], characteristics: list[str],
                 method: str = "spearman") -> pd.DataFrame:
    """One row per (outcome, characteristic): n, rho, p; q = BH-FDR over the whole grid.
    One row of df = one dataset (the unit of analysis)."""
    fn = {"spearman": spearmanr, "pearson": pearsonr}[method]
    rows = []
    for o, c in itertools.product(outcomes, characteristics):
        pair = df[[o, c]].apply(pd.to_numeric, errors="coerce").dropna()
        rho = p = NAN
        if len(pair) >= 3 and pair[o].nunique() > 1 and pair[c].nunique() > 1:
            res = fn(pair[o], pair[c])
            rho, p = float(res[0]), float(res[1])
        rows.append({"outcome": o, "characteristic": c, "n": len(pair), "rho": rho, "p": p})
    out = pd.DataFrame(rows, columns=["outcome", "characteristic", "n", "rho", "p"])
    out["q_bh"] = bh_fdr(out["p"])
    return out


def iqr_flags(x) -> np.ndarray:
    """True outside [Q1 - 1.5 IQR, Q3 + 1.5 IQR]; all False with fewer than 4 values."""
    x = np.asarray(x, dtype=float)
    ok = ~np.isnan(x)
    if ok.sum() < 4:
        return np.zeros(len(x), dtype=bool)
    q1, q3 = np.percentile(x[ok], [25, 75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    return ok & ((x < lo) | (x > hi))


def mean_sd(df: pd.DataFrame, by: list[str], cols: list[str]) -> pd.DataFrame:
    """<col>_mean, <col>_sd (ddof = 1) and n per group."""
    g = df.groupby(by, dropna=False)[cols]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd"))
    out["n"] = g.size()
    return out.reset_index()
