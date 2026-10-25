import json
import math
import subprocess

import numpy as np
import pytest

import datasets as ds
import fec
import ge
import measurement
from grape import grape


@pytest.mark.parametrize("name, n", [("appendicitis", 20), ("prnn_crabs", 40), ("MagicTelescope", 300)])
def test_fps_bit_identical(name, n):
    if not (ds.DATA_DIR / f"{name}.csv").is_file():
        pytest.skip(f"needs data/{name}.csv: python datasets.py recreate {name}")
    X, _, _ = ds.load_table(name)
    X = X[:3000].T                                          # features × rows
    assert np.array_equal(fec.farthest_point(X, n, 30), fec.farthest_point_reference(X, n, 30))


def test_fps_boolean_ties():
    X = np.random.default_rng(0).integers(0, 2, size=(12, 500)).astype(float)
    assert np.array_equal(fec.farthest_point(X, 60, 7), fec.farthest_point_reference(X, 60, 7))


class Ind:
    def __init__(self, phenotype, invalid=False):
        self.phenotype, self.invalid = phenotype, invalid


def test_one_row_fingerprint_constant_is_invalid():
    x = np.array([[0.5, -1.0, 2.0]])
    y = np.array([1, 0, 1])
    ev = fec.FECEvaluator(x[:, :1])
    assert ev(Ind("greater_than_or_equal(x[0],+0.00)"), [x, y])[0] == 0.0       # valid: stored
    assert math.isnan(ev(Ind("less_than_or_equal(+1.00,+2.00)"), [x, y])[0])    # constant: invalid
    assert math.isnan(ge.fitness(Ind("less_than_or_equal(+1.00,+2.00)"), [x, y])[0])
    assert ev.summary()["hits"] == 0


def test_hit_miss_and_fake_hit():
    x = np.array([[0.5, -1.0, 2.0, -3.0]])
    y = np.array([1, 0, 1, 1])
    ev = fec.FECEvaluator(x[:, [0, 2]], diagnostic=True)                        # fingerprint rows 0.5, 2.0
    a = ev(Ind("greater_than_or_equal(x[0],+0.00)"), [x, y])[0]                 # miss: full error 0.25
    b = ev(Ind("greater_than_or_equal(x[0],-2.00)"), [x, y])[0]                 # same on 0.5, 2.0: hit
    assert a == b == 0.25 and ev.total["hits"] == 1
    assert ev.total["fake_hits"] == 1                                           # its full error is 0.5


def run_pair(name, seed, diagnostic, zscore=True):
    X, y, info = ds.load_table(name)
    train, test = ge.split(X, y, seed, 0.2, zscore, info)
    p = {**ge.PARAMS, "population": 100, "generations": 5, "min_init_depth": ge.recorded_depth(name)}
    idx = fec.farthest_point(train[0], fec.sample_size(0.1, len(train[1])), seed)
    ev = fec.FECEvaluator(train[0][:, idx], diagnostic=diagnostic)
    rows, hof, _ = ge.evolve(grape.Grammar(str(ds.ROOT / "grammars" / f"{name}.bnf")), train, test, p, seed, ev)
    for r in rows:                                     # what evolution does not decide (measurement._NOT_EVOLUTION)
        for k in measurement._NOT_EVOLUTION:
            r.pop(k, None)
    return rows, hof.items[0].phenotype, ev.summary()


@pytest.mark.parametrize("name", ["appendicitis", "prnn_crabs"])
def test_diagnostic_evolves_like_timing(name):
    t_rows, t_best, t_sum = run_pair(name, 30, False)
    d_rows, d_best, d_sum = run_pair(name, 30, True)
    assert t_rows == d_rows and t_best == d_best
    assert (t_sum["hits"], t_sum["misses"]) == (d_sum["hits"], d_sum["misses"]) and "fake_hits" in d_sum


def test_smaller_samples_nest_in_larger_ones():
    name = "MagicTelescope"
    if not (ds.DATA_DIR / f"{name}.csv").is_file():
        pytest.skip(f"needs data/{name}.csv: python datasets.py recreate {name}")
    X, y, info = ds.load_table("MagicTelescope")
    (Xt, yt), _ = ge.split(X, y, 30, 0.2, True, info)
    for small, large in [(107, 535), (535, 1070)]:
        assert np.array_equal(fec.farthest_point(Xt, small, 30), fec.farthest_point(Xt, large, 30)[:small])
        assert np.array_equal(fec.random_rows(Xt, small, 30), fec.random_rows(Xt, large, 30)[:small])


def test_fps_cache_prefix_and_time():
    X = np.random.default_rng(1).normal(size=(4, 300))
    idx, cum = fec.farthest_point(X, 30, 5, times=True)
    rows, secs = fec.sample_rows("farthest_point", X, None, 10, 5, {"indices": idx, "cum_times": cum})
    assert np.array_equal(rows, fec.farthest_point(X, 10, 5)) and secs == cum[9]
    assert np.all(np.diff(cum) >= 0)


def test_prepared_cache_used_by_run():
    small = {"population": 50, "generations": 2}
    path = fec.prepare_fps("prnn_crabs", 30, 0.10, small)          # same setup as the runs below
    cached = ge.run("prnn_crabs", 30, small, {"fraction": 0.05, "fps_cache": True})
    direct = ge.run("prnn_crabs", 30, small, {"fraction": 0.05})
    assert cached["fec"]["fps_from_cache"] and not direct["fec"]["fps_from_cache"]
    assert cached["best_phenotype"] == direct["best_phenotype"] and cached["generations"][-1]["min"] == direct["generations"][-1]["min"]
    path.unlink()


def test_stratified_equal_per_class():
    y = np.array([0] * 80 + [1] * 20)
    rows = fec.stratified_rows(y, 21, 3)
    assert len(rows) == 21 and (y[rows] == 0).sum() == 11 and (y[rows] == 1).sum() == 10
