"""instrumented diagnostic runs."""
import json
import random
from pathlib import Path

import numpy as np
import pytest
from deap import base, creator, tools
from scipy.stats import rankdata, spearmanr

import batch
import experiment
import ge
import instrument
import results
import typed_gp
from datasets import DatasetError

SMALL = {"population": 60, "generations": 4}
FEC = {"sampler": "random", "fps_cache": False}

if not hasattr(creator, "FitnessMin019"):
    creator.create("FitnessMin019", base.Fitness, weights=(-1.0,))
    creator.create("Ind019", list, fitness=creator.FitnessMin019)


def _ind(used, true):
    i = creator.Ind019([0])
    i.fitness.values = (used,)
    i.fec_true = true
    return i


def _runs(mod, name, frac):
    fec = {**FEC, "fraction": frac}
    return (mod.run(name, 30, SMALL, {**fec, "diagnostic": False}), mod.run(name, 30, SMALL, {**fec, "diagnostic": True}),
            mod.run(name, 30, SMALL, {**fec, "diagnostic": True}, instrument=True))


# ----------------------------------------------------------------

def test_select_tournament_is_deaps_tournament():
    for rep in range(50):
        rng = random.Random(rep)
        pool = [_ind(rng.choice([0.1, 0.2, 0.2, 0.3, 0.5]), rng.random()) for _ in range(rng.randint(5, 40))]
        random.seed(rep)
        a = tools.selTournament(pool, 30, 4)
        state_a = random.getstate()
        random.seed(rep)
        b = instrument.select_tournament(pool, 30, 4, instrument.Recorder(4, [0, 1]))
        assert [id(x) for x in a] == [id(x) for x in b]
        assert random.getstate() == state_a                               # no extra draws


def test_changed_tournaments_hand_count():
    rec = instrument.Recorder(4, [0, 1])
    a, b, c, d = _ind(0.1, 0.4), _ind(0.2, 0.2), _ind(0.3, 0.3), _ind(0.2, 0.2)
    rec.tournament([a, b, c], a)          # real winner a (true 0.4), correct b (0.2): changed, gap 0.2
    rec.tournament([b, d, c], b)          # tie in true between b and d: not changed
    rec.tournament([c, b], b)             # real winner b is also correct: not changed
    assert (rec.n_tourn, rec.n_changed) == (3, 1)
    assert rec.gaps == [pytest.approx(0.2)]


def test_population_measures_by_hand():
    used = [0.1, 0.2, 0.3, 0.4, 0.4]
    true = [0.1, 0.25, 0.3, 0.35, 0.4]
    m = instrument.population_measures([_ind(u, t) for u, t in zip(used, true)])
    assert m["ins_n_wrong"] == 2 and m["ins_n_valid"] == 5
    assert m["ins_spearman"] == pytest.approx(spearmanr(used, true).statistic)
    shift = np.abs(rankdata(used) - rankdata(true))[[1, 3]]
    assert m["ins_rank_shift_mean"] == pytest.approx(shift.mean()) and m["ins_rank_shift_max"] == pytest.approx(shift.max())
    const = instrument.population_measures([_ind(0.2, 0.2), _ind(0.2, 0.3)])
    assert np.isnan(const["ins_spearman"]) and const["ins_spearman_undefined"] == 1
    none = instrument.population_measures([_ind(0.1, 0.1), _ind(0.2, 0.2)])
    assert none["ins_n_wrong"] == 0 and np.isnan(none["ins_rank_shift_mean"])


@pytest.mark.parametrize("mod,name,frac", [(ge, "appendicitis", 0.05), (ge, "prnn_crabs", 0.01),
                                           (typed_gp, "prnn_crabs", 0.05), (typed_gp, "appendicitis", 0.01)])
def test_instrumented_run_reproduces_both_twins(mod, name, frac):
    timing, diag, inst = _runs(mod, name, frac)
    assert instrument.compare(inst, timing, False) == []
    assert instrument.compare(inst, diag, True) == []
    assert inst["setup"] == ge.instrument_setup(SMALL) and diag["setup"] == ge.setup_id(SMALL)
    assert not any(k.startswith("ins_") for g in diag["generations"] for k in g)      # plain runs unchanged
    cols = {"ins_spearman", "ins_n_wrong", "ins_rank_shift_mean", "ins_tourn_changed", "ins_tourn_gap_mean",
            "ins_elite_used", "ins_elite_true", "ins_fake_err_p90", "ins_hits_same_pred", "ins_exact_hits",
            "ins_cache_bytes", "ins_distinct_parents", "ins_unchanged_offspring"}
    assert len(inst["generations"]) == SMALL["generations"] + 1
    assert all(cols <= set(g) for g in inst["generations"])
    g0, g1 = inst["generations"][0], inst["generations"][1]
    assert np.isnan(g0["ins_tournaments"]) and g1["ins_tournaments"] == SMALL["population"] - 1
    for g in inst["generations"]:                                        # data-model validation rules
        assert g["ins_n_wrong"] <= g["ins_n_valid"]
        assert g["ins_hits_same_pred"] + g["ins_hits_equal_fit_only"] + g["fec_fake_hits"] == g["fec_hits"]
        if g["gen"] > 0:
            assert g["ins_tourn_changed"] <= g["ins_tournaments"]


def test_instrument_only_for_diagnostic_runs():
    with pytest.raises(DatasetError):
        ge.run("appendicitis", 30, SMALL, {**FEC, "fraction": 0.05, "diagnostic": False}, instrument=True)
    with pytest.raises(DatasetError):
        ge.run("appendicitis", 30, SMALL, None, instrument=True)


# ----------------------------------------------------------------

@pytest.mark.parametrize("mod", [ge, typed_gp])
def test_full_fingerprint_has_no_fake_and_no_equal_fitness_only_hits(mod):
    r = mod.run("appendicitis", 30, SMALL, {**FEC, "fraction": 1.0, "diagnostic": True}, instrument=True)
    assert r["fec"]["fake_hits"] == 0
    assert all(g["ins_hits_equal_fit_only"] == 0 and g["ins_n_wrong"] == 0 for g in r["generations"])
    ins = r["instrument"]
    assert ins["peak_rss_bytes"] > 0 and ins["cache_bytes_final"] > 0
    assert sum(ins["fingerprint_class_counts"].values()) == r["fec"]["sample_size"]


# ----------------------------------------------------------------

@pytest.mark.parametrize("mod,name", [(ge, "appendicitis"), (typed_gp, "prnn_crabs")])
def test_logs_and_snapshot_recomputation(mod, name):
    r = mod.run(name, 30, SMALL, {**FEC, "fraction": 0.01, "diagnostic": True}, instrument=True)
    logs = r["instrument_logs"]
    assert len(logs["hits"]) == r["fec"]["hits"]
    assert logs["entries"]["reuses"].sum() == r["fec"]["hits"]
    assert len(logs["fakes"]) == r["fec"]["fake_hits"] == int(logs["hits"]["fake"].sum())
    snap = logs["snapshots"]
    assert set(snap["gen"]) == {0, SMALL["generations"]}
    for gen in (0, SMALL["generations"]):
        s = snap[(snap["gen"] == gen) & (snap["valid"] == 1)]
        row = r["generations"][gen]
        assert int((s["used"] != s["true"]).sum()) == row["ins_n_wrong"]
        if row["ins_spearman_undefined"] == 0:
            assert row["ins_spearman"] == pytest.approx(spearmanr(s["used"], s["true"]).statistic)


def test_save_writes_logs_and_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(ge.datasets, "ROOT", tmp_path)
    r = ge.run("appendicitis", 30, SMALL, {**FEC, "fraction": 0.05, "diagnostic": True}, instrument=True)
    path = ge.save_result(dict(r))
    rec = json.loads(path.read_text())
    assert set(rec["instrument"]["files"]) == set(instrument.LOG_KINDS)
    assert all((path.parent / f).is_file() for f in rec["instrument"]["files"].values())
    with pytest.raises(DatasetError):
        ge.save_result(dict(r))


def test_check_twin_detects_a_difference(tmp_path):
    timing, diag, inst = _runs(ge, "appendicitis", 0.05)
    inst.pop("instrument_logs")
    base_dir = tmp_path / "results" / "ge"
    sub = "appendicitis"
    paths = {"inst": base_dir / inst["setup"] / sub / "fec_diagnostic_random_frac0.05" / "seed30.json",
             "diag": base_dir / diag["setup"] / sub / "fec_diagnostic_random_frac0.05" / "seed30.json",
             "timing": base_dir / diag["setup"] / sub / "fec_random_frac0.05" / "seed30.json"}
    for k, rec in (("inst", inst), ("diag", diag), ("timing", timing)):
        paths[k].parent.mkdir(parents=True, exist_ok=True)
        paths[k].write_text(json.dumps(rec))
    # setups with overrides: twin setup = setup without the 'instrumented_' prefix
    assert instrument.check_twin(paths["inst"])["ok"]
    diag["fec"]["fake_hits"] += 1
    paths["diag"].write_text(json.dumps(diag))
    out = instrument.check_twin(paths["inst"])
    assert not out["ok"] and any("fake_hits" in m for m in out["mismatches"])
    assert paths["inst"].with_suffix(".mismatch").is_file()


# ----------------------------------------------------------------

def test_experiment_key_and_plan(tmp_path):
    e = experiment.load(Path("experiments/instrumented_ge.yaml"))
    jobs = batch.plan(e)
    assert e["setup"] == "instrumented" and len(jobs) == 9 * 3 * 30
    assert all(j.phase == "diagnostic" and "--instrument" in j.command() and "/instrumented/" in str(j.out) for j in jobs)
    bad = tmp_path / "bad.yaml"
    bad.write_text("algorithm: GE\ndatasets: [appendicitis]\nfec: {samplers: [random], fractions: [0.05]}\n"
                   "modes: [fec, fec_diagnostic]\ninstrument: true\n")
    with pytest.raises(DatasetError):
        experiment.load(bad)


def test_store_keeps_instrumented_runs_apart():
    assert "instrumented" in results.SEPARATE_SETUPS and "phase_timing" in results.SEPARATE_SETUPS
    assert all("instrumented" not in str(p) for _, p in results._run_paths())
