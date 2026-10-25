"""Tests for results.py (runs the laptop's fetched pilot if present, else a tiny local batch)."""
import json

import pandas as pd
import pytest

import measurement
import results


@pytest.fixture(scope="module")
def runs():
    runs = [r for r in results.load_runs("default", "prnn_crabs") if r.get("algorithm", "GE") == "GE"]
    if len(runs) < 38:
        pytest.skip("needs the pilot_small runs of prnn_crabs under results/ge/default (fetch or pilot first)")
    return runs


def test_metrics_match_measurement(runs):
    table = results.metrics_table()
    row = table[(table.algorithm == "GE") & (table.dataset == "prnn_crabs") & (table.sampler == "random")
                & (table.fraction == 0.05)].iloc[0]
    pick = lambda mode: [r for r in runs if r["mode"] == mode and (mode == "baseline" or
                         (r["fec"]["sampler"] == "random" and r["fec"]["fraction"] == 0.05))]
    m = measurement.metrics(pick("baseline"), pick("fec"), pick("fec_diagnostic"))
    assert row["S_T"] == pytest.approx(m["S_T"]) and row["dAcc"] == pytest.approx(m["dAcc"])
    assert row["n_runs"] == m["n_runs"] and pd.isna(row["p_eval_time"])          # 2 pairs: no test


def test_integrate_refuses_two_tables(runs, tmp_path, monkeypatch):
    fake = json.loads(json.dumps(runs[0]))
    fake["table_checksum"] = "0" * 64
    monkeypatch.setattr(results, "load_runs", lambda *a: runs + [fake])
    monkeypatch.setattr(results, "STORE", tmp_path)
    with pytest.raises(results.datasets.DatasetError, match="two tables"):
        results.integrate()


def test_difficulty_class_cuts():
    """Fixed cut-offs on gradient-boosting balanced accuracy (2026-10-03)."""
    assert [results.difficulty_class(v) for v in (0.61, 0.7999, 0.80, 0.8799, 0.88, 0.99)] == \
        ["hard", "hard", "medium", "medium", "simple", "simple"]
    assert results.difficulty_class(float("nan")) == ""


def test_tost_equivalence():
    """TOST: small p when the paired differences sit well inside the margin, large when they sit outside."""
    import numpy as np
    rng = np.random.default_rng(0)
    a = rng.uniform(0.7, 0.8, 30)
    assert results._tost_p(a, a + rng.normal(0, 0.002, 30)) < 0.01          # within +-1 pp
    assert results._tost_p(a, a - 0.03 + rng.normal(0, 0.002, 30)) > 0.5    # 3 pp worse
    assert np.isnan(results._tost_p(a[:3], a[:3]))                          # too few pairs
