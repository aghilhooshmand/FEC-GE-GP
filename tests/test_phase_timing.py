"""per-phase timing of FEC (spec to , data-model identities)."""
import json
from pathlib import Path

import numpy as np
import pytest

import batch
import experiment
import ge
import phase_timing
import typed_gp
from datasets import DatasetError

SMALL = {"population": 50, "generations": 4}
FEC = {"sampler": "random", "diagnostic": False, "fps_cache": False}


def _pair(mod, name, frac):
    fec = {**FEC, "fraction": frac}
    return mod.run(name, 30, SMALL, fec), mod.run(name, 30, SMALL, fec, phase_timing=True)


@pytest.mark.parametrize("mod,name,frac", [(ge, "appendicitis", 0.05), (ge, "prnn_crabs", 0.10),
                                           (typed_gp, "prnn_crabs", 0.10), (typed_gp, "appendicitis", 0.05)])
def test_phase_timed_run_is_its_twin_and_adds_up(mod, name, frac):
    plain, timed = _pair(mod, name, frac)
    assert phase_timing.compare(timed, plain) == []                       # , SC-001
    assert "phases" not in plain and plain["setup"] == ge.setup_id(SMALL)  # default path unchanged
    assert timed["setup"] == ge.phase_setup(SMALL)
    ph, m = timed["phases"], timed["measures"]
    assert ph.get("n_hits", 0) + ph.get("n_misses", 0) + ph.get("n_invalid_fp", 0) == ph["n_reached_fp"]
    assert ph["n_reached_fp"] + ph.get("n_invalid_pre", 0) == m["n_evals"]
    assert ph.get("n_misses", 0) == m["n_full_evals"]
    times = {k: v for k, v in ph.items() if k.startswith("t_") and k != "t_unaccounted"}
    assert all(v >= 0 for v in times.values()) and ph["t_unaccounted"] >= 0
    assert ph["clock_ns_per_lap"] > 0
    # the uncovered part is the clock's own reads outside the laps (begin + the Meter): small against the total
    assert ph["t_unaccounted"] < 0.25 * m["eval_time_sec"]
    gen_hits = sum(g.get("ph_n_hits", 0) for g in timed["generations"])
    assert gen_hits == ph.get("n_hits", 0)                                 # per generation sums to the run


def test_phase_timing_only_for_fec_timing_runs():
    with pytest.raises(DatasetError):
        ge.run("appendicitis", 30, SMALL, None, phase_timing=True)
    with pytest.raises(DatasetError):
        ge.run("appendicitis", 30, SMALL, {**FEC, "fraction": 0.05, "diagnostic": True}, phase_timing=True)


def _record(ph, setup="phase_timing"):
    return {"algorithm": "GE", "dataset": "x", "seed": 30, "setup": setup, "fec": {"sampler": "random",
            "fraction": 0.05}, "measures": {"eval_time_sec": 1.0, "n_evals": 10, "n_full_evals": 2,
            "n_sample_evals": 8}, "phases": ph, "generations": [], "best_phenotype": "p", "n_train_evals": 10}


def test_c_meas_hand_calculation_and_no_miss_flag():
    ph = {"t_fp_eval": 0.004, "t_fp_key": 0.001, "t_lookup": 0.001, "n_reached_fp": 8,
          "t_full_eval": 0.020, "n_misses": 2}
    row = phase_timing.run_row(_record(ph))
    assert row["ms_fp_per_candidate"] == pytest.approx(1000 * 0.006 / 8)
    assert row["ms_full_per_miss"] == pytest.approx(1000 * 0.020 / 2)
    assert row["c_meas"] == pytest.approx((0.006 / 8) / (0.020 / 2))
    none = phase_timing.run_row(_record({**ph, "n_misses": 0, "t_full_eval": 0.0}))
    assert np.isnan(none["c_meas"]) and none["c_meas_flag"] == "no_misses"


def test_twin_check_detects_a_difference(tmp_path):
    twin_dir = tmp_path / "results" / "ge" / "default" / "x" / "fec_random_frac0.05"
    run_dir = tmp_path / "results" / "ge" / "phase_timing" / "x" / "fec_random_frac0.05"
    twin_dir.mkdir(parents=True), run_dir.mkdir(parents=True)
    rec = _record({})
    (twin_dir / "seed30.json").write_text(json.dumps({**rec, "setup": "default"}))
    (run_dir / "seed30.json").write_text(json.dumps(rec))
    assert phase_timing.check_twin(run_dir / "seed30.json")["ok"]
    (run_dir / "seed30.json").write_text(json.dumps({**rec, "best_phenotype": "q"}))
    out = phase_timing.check_twin(run_dir / "seed30.json")
    assert not out["ok"] and "best_phenotype" in out["mismatches"]
    assert (run_dir / "seed30.mismatch").is_file() and not (run_dir / "seed30.json").exists()


@pytest.mark.parametrize("algo", ["ge", "gp"])
def test_phase_timing_experiment_plans_only_fec_timing_jobs(algo):
    exp = experiment.load(Path(f"experiments/phase_timing_{algo}.yaml"))
    jobs = batch.plan(exp)
    assert len(jobs) == 17 * 30 * 3                                         # 1,530 per algorithm
    assert {j.tag.split("_frac")[0] for j in jobs} == {"fec_random"}
    assert {j.setup for j in jobs} == {"phase_timing"} and all("--phase-timing" in j.command() for j in jobs)


def test_default_experiment_plans_unchanged():
    exp = experiment.load(Path("experiments/line1_hard_ge.yaml"))
    tags = {j.tag.split("_")[0] for j in batch.plan(exp, seeds=[30]) if j.phase != "fps"}
    assert tags == {"baseline", "fec"} and all(not j.phase_timing for j in batch.plan(exp, seeds=[30]))


def test_phase_timing_needs_fec_only_mode(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("algorithm: GE\ndatasets: [appendicitis]\nphase_timing: true\n")
    with pytest.raises(DatasetError):
        experiment.load(bad)


TWIN = Path("results/ge/default/phoneme_balanced/fec_random_frac0.01/seed30.json")


@pytest.mark.skipif(not TWIN.is_file(), reason="needs the stored line-1 run")
def test_uncapped_twin_is_reproduced_exactly():
    """A stored run made before the 015 retry cap (no cap_attempts) is reproduced with GRAPE's own operators."""
    twin = json.loads(TWIN.read_text())
    assert "cap_attempts" not in twin
    run = ge.run("phoneme_balanced", 30, {}, {**FEC, "fraction": 0.01}, phase_timing=True)
    assert run["cap_attempts"] is None and phase_timing.compare(run, twin) == []
