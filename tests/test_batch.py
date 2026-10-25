"""Tests for batch.py (planning, order, skipping, pilot gate, cores)."""
import os
import sys
from pathlib import Path

import batch
import experiment

EXP = {"name": "t", "datasets": ["a", "b"], "seeds": [30, 31], "samplers": ["farthest_point", "random", "stratified"],
       "fractions": [0.01, 0.05, 0.10], "params": {}, "hive": {"workers": None, "max_busy": 0.1}}


def exp_for(names, seeds):
    return {**EXP, "datasets": names, "seeds": list(seeds)}


def test_plan_counts_and_phases():
    jobs = batch.plan(exp_for(["a", "b"], (30, 31)))
    count = {p: sum(j.phase == p for j in jobs) for p in ("fps", "timing", "diagnostic")}
    assert count == {"fps": 4, "timing": 2 * 2 * 10, "diagnostic": 2 * 2 * 9}
    phases = [j.phase for j in jobs]
    assert phases == sorted(phases, key=["fps", "timing", "diagnostic"].index)


def test_timing_interleaves_baseline_and_fec():
    timing = [j for j in batch.plan(exp_for(["a"], (30, 31))) if j.phase == "timing"]
    first_seed = timing[:10]
    assert {j.seed for j in first_seed} == {30} and sum(j.config["mode"] == "baseline" for j in first_seed) == 1
    assert first_seed[0].config != timing[10].config or first_seed[1].config != timing[11].config  # shuffled per seed
    assert [j.config for j in timing[:10]] == [j.config for j in batch.plan(exp_for(["a"], (30, 31))) if j.phase == "timing"][:10]


def test_commands_and_paths():
    jobs = batch.plan(exp_for(["prnn_crabs"], (30,)))
    fps = next(j for j in jobs if j.phase == "fps")
    assert fps.command()[1:] == ["fec.py", "prepare", "prnn_crabs", "30", "0.1"]
    d = next(j for j in jobs if j.phase == "diagnostic" and j.config["sampler"] == "farthest_point")
    assert "--diagnostic" in d.command() and "--fps-cache" in d.command()
    assert d.out.parts[-4:-1] == ("default", "prnn_crabs", f"fec_diagnostic_farthest_point_frac{d.config['fraction']:g}")


class FakeJob(batch.Job):
    """A job that writes its output file with a tiny process."""
    def __init__(self, path: Path, fail: bool = False):
        super().__init__("timing", "x", 0, {"mode": "baseline"})
        self._path, self._fail = path, fail

    @property
    def out(self):
        return self._path

    @property
    def log(self):
        return self._path.with_suffix(".log")

    def command(self):
        code = "import sys; sys.exit(3)" if self._fail else f"open({str(self._path)!r}, 'w').write('{{}}')"
        return [sys.executable, "-c", code]


def test_execute_skips_done_and_reports_failures(tmp_path):
    done_already = tmp_path / "a.json"
    done_already.write_text("{}")
    jobs = [FakeJob(done_already), FakeJob(tmp_path / "b.json"), FakeJob(tmp_path / "c.json", fail=True)]
    cpu = sorted(os.sched_getaffinity(0))[0]
    done, failed = batch.execute(jobs, [cpu])
    assert [j.out.name for j in done] == ["b.json"] and [j.out.name for j in failed] == ["c.json"]


def test_pilot_gate_lists_missing():
    assert len(batch.pilot_missing(exp_for(["no_such_dataset"], (30, 31)))) == 19   # 1 + 9 FEC + 9 diagnostic


def test_overrides_change_setup_and_commands():
    job = batch.plan({**exp_for(["prnn_crabs"], (30,)), "params": {"population": 200}})[1]
    assert job.setup.startswith("p") and job.out.parts[-4] == job.setup
    assert "--param" in job.command() and "population=200" in job.command()


def test_physical_cores_cover_cpus():
    groups = batch.physical_cores()
    cpus = sorted(c for g in groups for c in g)
    assert groups and len(cpus) == len(set(cpus))


class DependentJob(FakeJob):
    def __init__(self, path, need, order):
        super().__init__(path)
        self._need, self._order = need, order

    def needs(self):
        return self._need

    def command(self):
        return [sys.executable, "-c", f"open({str(self._path)!r}, 'w').write('{{}}'); "
                                      f"open({str(self._order)!r}, 'a').write({self._path.name!r} + ' ')"]


class SlowMaker(FakeJob):
    def __init__(self, path, order):
        super().__init__(path)
        self.phase, self._order = "fps", order

    def command(self):
        return [sys.executable, "-c", f"import time; time.sleep(1.5); open({str(self._path)!r}, 'w').write('x'); "
                                      f"open({str(self._order)!r}, 'a').write('picks ')"]


def test_dependent_job_waits_for_its_picks_others_do_not(tmp_path):
    order = tmp_path / "order.txt"
    picks = tmp_path / "picks.npz"
    jobs = [SlowMaker(picks, order), DependentJob(tmp_path / "fps_run.json", picks, order),
            DependentJob(tmp_path / "baseline.json", None, order)]
    cpus = sorted(os.sched_getaffinity(0))[:3]
    done, failed = batch.execute(jobs, cpus)
    assert not failed and len(done) == 3
    assert order.read_text().split() == ["baseline.json", "picks", "fps_run.json"]


def test_failed_picks_do_not_block(tmp_path):
    picks = tmp_path / "picks.npz"
    maker = FakeJob(picks, fail=True)
    maker.phase = "fps"
    dependent = DependentJob(tmp_path / "fps_run.json", picks, tmp_path / "order.txt")
    done, failed = batch.execute([maker, dependent], sorted(os.sched_getaffinity(0))[:1])
    assert [j.out.name for j in done] == ["fps_run.json"] and [j.out.name for j in failed] == ["picks.npz"]


def test_gp_jobs_run_typed_gp():
    job = batch.plan({**exp_for(["prnn_crabs"], (30,)), "algorithm": "GP"})[1]
    assert job.command()[1] == "typed_gp.py" and job.out.parts[-5] == "gp"
