"""014 statistics: textbook values (hand-computed) and the NaN rules."""
import math

import numpy as np
import pandas as pd
import pytest

import analysis as an
import results


def test_min_pairs_same_as_store():
    assert an.MIN_PAIRS == results.MIN_PAIRS_FOR_TEST


def test_rank_biserial_matched_pairs():
    # d = 1, 2, 3, -4: ranks 1..4, W+ = 6, W- = 4, r = (6 - 4) / 10 = 0.2 (sign: b larger)
    r = an.wilcoxon_paired([0, 0, 0, 0], [1, 2, 3, -4])
    assert (r["w_plus"], r["w_minus"]) == (6, 4) and r["r_rb"] == pytest.approx(0.2)
    assert math.isnan(r["p"])                                    # < 6 pairs: no test
    assert an.wilcoxon_paired([1, 2, 3], [0, 1, 2])["r_rb"] == -1.0   # all b smaller


def test_wilcoxon_p_and_zero_rules():
    a = np.arange(8.0)
    r = an.wilcoxon_paired(a, a + np.array([1, 2, 3, 4, 5, 6, 7, -0.5]))
    assert 0 < r["p"] < 0.05 and r["r_rb"] > 0.9
    zero = an.wilcoxon_paired(a, a)
    assert math.isnan(zero["p"]) and math.isnan(zero["r_rb"])    # nothing to test
    one = an.wilcoxon_one_sample([1.2, 1.3, 1.1, 1.4, 1.5, 1.25], 1.0)
    assert one["r_rb"] == 1.0 and one["p"] < 0.05
    greater = an.wilcoxon_one_sample([1.2, 1.3, 1.1, 1.4, 1.5, 1.25], 1.0, "greater")
    assert greater["p"] == pytest.approx(one["p"] / 2)


def test_holm_bonferroni_bh():
    p = [0.01, 0.04, 0.03, 0.005]
    assert an.holm(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert an.bonferroni(p) == pytest.approx([0.04, 0.16, 0.12, 0.02])
    assert an.bh_fdr(p) == pytest.approx([0.02, 0.04, 0.04, 0.02])
    with_nan = an.holm([0.01, float("nan"), 0.04])
    assert math.isnan(with_nan[1]) and with_nan[0] == pytest.approx(0.02) and with_nan[2] == pytest.approx(0.04)


def test_friedman_kendall_perfect_agreement():
    t = pd.DataFrame({"0.01": [1, 1, 1], "0.05": [2, 2, 2], "0.10": [3, 3, 3]}, dtype=float)
    r = an.friedman_kendall(t)
    assert r["chi2"] == pytest.approx(6.0) and r["W"] == pytest.approx(1.0)
    assert math.isnan(an.friedman_kendall(t.iloc[:2])["p"])    # < 3 datasets


def test_pairwise_and_correlations():
    rng = np.random.default_rng(0)
    t = pd.DataFrame({"a": rng.normal(size=8), "b": rng.normal(size=8) + 3, "c": rng.normal(size=8)})
    pw = an.pairwise_wilcoxon(t)
    assert len(pw) == 3 and (pw["p_holm"] >= pw["p"]).all() and (pw["p_bonferroni"] >= pw["p_holm"] - 1e-12).all()
    df = pd.DataFrame({"S_T": [1, 2, 3, 4, 5], "rows": [10, 20, 30, 40, 50], "flat": [1, 1, 1, 1, 1]})
    c = an.correlations(df, ["S_T"], ["rows", "flat"])
    assert c.loc[0, "rho"] == pytest.approx(1.0) and math.isnan(c.loc[1, "rho"])


def test_iqr_and_mean_sd():
    assert an.iqr_flags([1, 2, 3, 4, 100]).tolist() == [False, False, False, False, True]
    assert not an.iqr_flags([1, 100, 2]).any()                   # < 4 values
    m = an.mean_sd(pd.DataFrame({"g": [0, 0, 1], "v": [1.0, 3.0, 5.0]}), ["g"], ["v"])
    assert m.loc[0, "v_sd"] == pytest.approx(math.sqrt(2)) and m.loc[0, "n"] == 2   # ddof = 1


def test_bayes_correlated_t_matches_baycomp():
    """With n_test/n_train = 1/(k-1) the random-split correction equals baycomp's k-fold one (k = 10, 3 runs)."""
    import baycomp
    rng = np.random.default_rng(1)
    a = rng.uniform(0.7, 0.8, 30)
    b = a + rng.normal(0, 0.01, 30)
    assert np.allclose(an.bayes_correlated_t(a, b, 0.01, 1, 9), baycomp.CorrelatedTTest.probs(a, b, rope=0.01, runs=3))
    assert an.bayes_correlated_t(a, a + 0.03, 0.01, 0.2, 0.8)[2] > 0.99      # clearly better, beyond the rope
    assert sum(an.bayes_correlated_t(a, b, 0.01, 0.2, 0.8)) == pytest.approx(1)


def test_bayes_signed_rank_matches_baycomp():
    import baycomp
    rng = np.random.default_rng(2)
    a = rng.uniform(0.6, 0.9, 13)
    b = a + rng.normal(0, 0.004, 13)
    ours = an.bayes_signed_rank(a, b, 0.01)
    assert np.allclose(ours, baycomp.two_on_multiple(a, b, rope=0.01), atol=0.01)       # Monte Carlo
    assert ours == an.bayes_signed_rank(a, b, 0.01)                                       # fixed random_state
    assert ours[1] > 0.95
