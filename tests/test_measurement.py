"""Tests for measurement.py and the evaluators' metering (spec 007 SC-001..004)."""
import numpy as np
import pytest

import fec
import ge
import measurement as ms


class Ind:
    def __init__(self, phenotype, invalid=False):
        self.phenotype, self.invalid = phenotype, invalid


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_operator_calls():
    assert ms.operator_calls("and_(less_than_or_equal(x[0],+1.00),not_(x[1]))") == 3
    assert ms.operator_calls("x[0]") == 0


def test_diagnostic_check_not_timed(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(ms.time, "perf_counter", clock)
    x = np.array([[0.5, -1.0, 2.0, -3.0]])
    y = np.array([1, 0, 1, 1])
    ev = fec.FECEvaluator(x[:, [0, 2]], diagnostic=True)
    real = ev.full_fitness

    def slow_fitness(ind, points):                  # every full evaluation "takes" 100 s
        clock.now += 100.0
        return real(ind, points)
    ev.full_fitness = slow_fitness
    ev(Ind("greater_than_or_equal(x[0],+0.00)"), [x, y])     # miss: full evaluation is timed
    ev(Ind("greater_than_or_equal(x[0],-2.00)"), [x, y])     # hit: fake-hit check is not
    assert ev.meter.eval_time_sec == 100.0 and ev.meter.n_evals == 2
    assert ev.total["fake_hits"] == 1
    assert ev.meter.n_full == 1 and ev.meter.effort_full == 4 and ev.meter.effort_sample == 2 * 2


def test_test_evaluations_not_metered():
    result = ge.run("appendicitis", 30, {"population": 50, "generations": 3})
    assert result["measures"]["n_evals"] == result["n_train_evals"]


def test_same_evolution_and_metrics():
    small = {"population": 50, "generations": 3}
    base = [ge.run("prnn_crabs", s, small) for s in (30, 31)]
    timing = [ge.run("prnn_crabs", s, small, {"fraction": 0.1}) for s in (30, 31)]
    diag = [ge.run("prnn_crabs", s, small, {"fraction": 0.1, "diagnostic": True}) for s in (30, 31)]
    ms.same_evolution(timing[0], diag[0])
    with pytest.raises(ValueError, match="did not evolve identically"):
        ms.same_evolution(timing[0], diag[1])
    m = ms.metrics(base, timing, diag)
    bt = np.mean([r["measures"]["eval_time_sec"] for r in base])
    ft = np.mean([r["measures"]["eval_time_sec"] for r in timing])
    assert m["S_T"] == pytest.approx(bt / ft)
    assert m["dAcc_pairs"] == [(1 - timing[i]["best_test_error"]) - (1 - base[i]["best_test_error"]) for i in (0, 1)]
    assert 0 <= m["fake_hit_ratio"] <= 1 and 0 <= m["hit_rate"] <= 1
    with pytest.raises(ValueError, match="seeds differ"):
        ms.metrics(base[:1], timing[1:])


def test_metrics_hand_computed():
    def run(seed, t, err, effort, hits=0, misses=0):
        return {"seed": seed, "best_test_error": err, "generation_time_sec": t * 2, "wall_time_sec": t * 3,
                "measures": {"eval_time_sec": t, "effort": effort},
                "fec": {"hits": hits, "misses": misses, "sampling_time_sec": 0.5}}
    base = [run(1, 10.0, 0.20, 1000.0), run(2, 30.0, 0.30, 3000.0)]
    fecs = [run(1, 5.0, 0.25, 500.0, 8, 2), run(2, 5.0, 0.30, 500.0, 6, 4)]
    m = ms.metrics(base, fecs)
    assert m["S_T"] == 4.0 and m["S_E"] == 4.0 and m["S_gen"] == 4.0 and m["S_wall"] == 4.0
    assert m["dAcc"] == pytest.approx(-0.025) and m["dAcc_pairs"] == pytest.approx([-0.05, 0.0])
    assert m["hit_rate"] == 0.7


def test_wall_time_includes_cached_farthest_point_sampling():
    """S_wall counts the fingerprint sampling also when the farthest-point picks came from the cache."""
    def run(seed, wall, fec=None):
        return {"seed": seed, "best_test_error": 0.2, "generation_time_sec": wall, "wall_time_sec": wall,
                "measures": {"eval_time_sec": wall, "effort": 1.0}, "fec": fec or {}}
    base = [run(1, 100.0), run(2, 100.0)]
    cached = [run(s, 50.0, {"hits": 1, "misses": 1, "sampling_time_sec": 150.0, "fps_from_cache": True}) for s in (1, 2)]
    inline = [run(s, 50.0, {"hits": 1, "misses": 1, "sampling_time_sec": 0.01, "fps_from_cache": False}) for s in (1, 2)]
    assert ms.metrics(base, cached)["S_wall"] == pytest.approx(0.5)      # 100 / (50 + 150)
    assert ms.metrics(base, inline)["S_wall"] == pytest.approx(2.0)      # sampling already inside the run
    assert ms.metrics(base, cached)["S_T"] == pytest.approx(2.0)         # S_T: evaluation time only


def test_effort_by_operator_sums_to_effort():
    m = ms.Meter()
    m.add_full("add(x[0],pdiv(x[1],add(x[2],+0.50)))", 10)
    m.add_sample("not_(less_than_or_equal(x[0],x[1]))", 3)
    d = m.as_dict()
    assert d["effort_by_operator"] == {"full": {"add": 20, "pdiv": 10}, "sample": {"less_than_or_equal": 3, "not_": 3}}
    assert ms.effort_with_weights(d, {}) == d["effort"] == 36
    assert ms.effort_with_weights(d, {"pdiv": 5.0}) == 36 + 4 * 10
    assert ms.operator_counts("argmax3(where(x[0],0.1,0.2),0.3,0.4)") == {"argmax3": 1, "where": 1}


@pytest.mark.parametrize("engine", ["ge", "gp"])
@pytest.mark.parametrize("fec_opts", [None, {"fraction": 0.1, "sampler": "random", "diagnostic": False, "fps_cache": False}])
def test_runs_record_effort_by_operator(engine, fec_opts):
    import typed_gp
    run = (ge if engine == "ge" else typed_gp).run("prnn_crabs", 0, {"population": 20, "generations": 2}, fec_opts)
    m = run["measures"]
    assert set(m["effort_by_operator"]) == {"full", "sample"}
    assert sum(sum(p.values()) for p in m["effort_by_operator"].values()) == m["effort"]
    assert sum(m["effort_by_operator"]["full"].values()) == m["effort_full"]
    assert ms.effort_with_weights(m, {}) == m["effort"]


@pytest.mark.parametrize("engine", ["ge", "gp"])
def test_runs_record_best_per_generation_and_fingerprint_rows(engine):
    import typed_gp
    mod = ge if engine == "ge" else typed_gp
    p = {"population": 20, "generations": 3}
    base = mod.run("prnn_crabs", 0, dict(p))
    assert len(base["best_phenotype_per_generation"]) == 4
    assert base["best_phenotype_per_generation"][-1] == base["best_phenotype"]
    assert all("best_phenotype" not in g for g in base["generations"])
    f = mod.run("prnn_crabs", 0, dict(p), {"fraction": 0.1, "sampler": "stratified", "diagnostic": False, "fps_cache": False})
    rows = f["fec"]["sample_rows"]
    assert len(rows) == f["fec"]["sample_size"] == len(set(rows)) and all(0 <= i < f["n_train"] for i in rows)
