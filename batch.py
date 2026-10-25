"""Batch engine: plan and run one experiment's GE runs on the free physical cores.

The experiment file (experiments/<name>.yaml, see experiment.py) gives the datasets, seeds,
FEC samplers and fractions and the GE parameter overrides (the "setup"). Per dataset and seed:
  phase fps         farthest-point picks at the largest fraction, computed once (fec.py prepare)
  phase timing      baseline + FEC (samplers x fractions), interleaved: seed-major, each seed's
                    configs in a fixed shuffled order, so running jobs always mix configs; shares
                    one queue with fps: a farthest-point run starts as soon as its own picks exist
  phase diagnostic  the FEC configs again with the fake-hit check (separate phase: it changes
                    cache behaviour and so the load; pairs are checked by measurement.same_evolution)
One job = one `python ge.py run ...` process, pinned to one hardware thread of an idle physical
core, BLAS/OpenMP threads = 1, its own log; finished results are skipped on restart.

Commands (python batch.py <command> experiments/<name>.yaml [--workers N] [--max-busy F]):
  plan     jobs per phase, done and remaining
  pilot    the first two seeds of the exact configuration (required before `run`)
  run      every seed, after a complete pilot
  status   progress and an estimate of the remaining time
  check    timing/diagnostic pairs evolved identically
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import datasets
import experiment
import fec
import ge
import measurement

ROOT = datasets.ROOT
THREAD_ENV = {v: "1" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
BUSY = 0.10                     # a core used more than this at start is skipped
MEM_MARGIN_GB = 4.0             # don't start a job below this much available memory


@dataclass
class Job:
    phase: str
    name: str
    seed: int
    config: dict = field(default_factory=dict)
    overrides: dict = field(default_factory=dict)
    max_fraction: float = 0.10
    algorithm: str = "GE"
    save_individuals: bool = False
    phase_timing: bool = False          # FEC timing runs with the phase clock, setup phase_timing

    @property
    def setup(self) -> str:
        return ge.phase_setup(self.overrides) if self.phase_timing else ge.setup_id(self.overrides)

    @property
    def tag(self) -> str:
        c = self.config
        if self.phase == "fps":
            return "fps"
        return "baseline" if c["mode"] == "baseline" else f"{c['mode']}_{c['sampler']}_frac{c['fraction']:g}"

    @property
    def out(self) -> Path:
        if self.phase == "fps":
            return fec.fps_cache_path(self.name, self.seed, self.setup)
        return ROOT / "results" / self.algorithm.lower() / self.setup / self.name / self.tag / f"seed{self.seed}.json"

    @property
    def log(self) -> Path:
        return ROOT / "logs" / "batch" / self.algorithm.lower() / self.setup / self.name / self.tag / f"seed{self.seed}.log"

    def needs(self) -> Path | None:
        """The farthest-point picks this job reads, if any (it waits for them, see execute)."""
        if self.phase != "fps" and self.config.get("sampler") == "farthest_point":
            return fec.fps_cache_path(self.name, self.seed, self.setup)
        return None

    def command(self) -> list[str]:
        params = [f"{k}={json.dumps(v)}" for k, v in sorted(self.overrides.items())]
        if self.phase == "fps":
            return [sys.executable, "fec.py", "prepare", self.name, str(self.seed), str(self.max_fraction), *params]
        c = self.config
        cmd = [sys.executable, "typed_gp.py" if self.algorithm == "GP" else "ge.py", "run", self.name, "--seed", str(self.seed)]
        cmd += [a for p in params for a in ("--param", p)]
        if c["mode"] != "baseline":
            cmd += ["--fraction", str(c["fraction"]), "--sampler", c["sampler"]]
            cmd += ["--diagnostic"] if c["mode"] == "fec_diagnostic" else []
            cmd += ["--fps-cache"] if c["sampler"] == "farthest_point" else []
        cmd += ["--save-individuals"] if self.save_individuals else []
        cmd += ["--phase-timing"] if self.phase_timing else []
        return cmd


def configs(exp: dict, mode: str = "fec") -> list[dict]:
    """The FEC configs of `mode` (fec or fec_diagnostic): samplers x fractions."""
    return [{"mode": mode, "sampler": s, "fraction": f} for s in exp["samplers"] for f in exp["fractions"]]


def plan(exp: dict, seeds=None) -> list[Job]:
    seeds = exp["seeds"] if seeds is None else seeds
    kw = {"overrides": exp["params"], "max_fraction": max(exp["fractions"]), "algorithm": exp.get("algorithm", "GE"),
          "save_individuals": exp.get("save_individuals", False), "phase_timing": exp.get("phase_timing", False)}
    names = exp["datasets"]
    modes = exp.get("modes", ["baseline", "fec", "fec_diagnostic"])
    jobs = [Job("fps", n, s, **kw) for s in seeds for n in names] if "farthest_point" in exp["samplers"] else []
    for phase, mode, extra in (("timing", "fec", [{"mode": "baseline"}]), ("diagnostic", "fec_diagnostic", [])):
        if mode not in modes:
            continue
        extra = extra if "baseline" in modes else []
        for s in seeds:
            for n in names:
                cfgs = extra + configs(exp, mode)
                random.Random(f"{n}/{s}/{phase}").shuffle(cfgs)       # fixed, mixed order per seed
                jobs += [Job(phase, n, s, c, **kw) for c in cfgs]
    return jobs


def pilot_seeds(exp: dict) -> list[int]:
    return list(exp["seeds"][:2])


# ---------------------------------------------------------------- cores, memory

def physical_cores() -> list[list[int]]:
    """Hardware threads grouped by physical core (sysfs topology)."""
    groups = set()
    for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/topology/core_cpus_list"):
        cpus = []
        for part in path.read_text().strip().split(","):
            lo, _, hi = part.partition("-")
            cpus += list(range(int(lo), int(hi or lo) + 1))
        groups.add(tuple(sorted(cpus)))
    return [list(g) for g in sorted(groups)]


def _cpu_times() -> dict[int, tuple[int, int]]:
    out = {}
    for line in Path("/proc/stat").read_text().splitlines():
        parts = line.split()
        if parts[0].startswith("cpu") and parts[0] != "cpu":
            vals = list(map(int, parts[1:]))
            out[int(parts[0][3:])] = (sum(vals), vals[3] + vals[4])          # total, idle + iowait
    return out


def free_cores(max_workers: int | None = None, interval: float = 1.0, max_busy: float = BUSY) -> list[int]:
    """One hardware thread of each physical core whose threads are all idle (< BUSY use)."""
    a = _cpu_times()
    time.sleep(interval)
    b = _cpu_times()
    use = {c: 1 - (b[c][1] - a[c][1]) / max(1, b[c][0] - a[c][0]) for c in b}
    cores = [g[0] for g in physical_cores() if all(use.get(c, 1.0) < max_busy for c in g)]
    return cores[:max_workers] if max_workers else cores


def available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 ** 2
    return float("inf")


# ---------------------------------------------------------------- execution

def _shown(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def execute(jobs: list[Job], cores: list[int]) -> tuple[list[Job], list[Job]]:
    """Run the jobs not done yet, one per core, in list order; a job whose `needs()` file is not there
    yet waits (unless the job meant to make it failed: then it runs and computes the picks itself).
    Returns (done, failed)."""
    pending = [j for j in jobs if not j.out.is_file()]
    making = {j.out for j in pending if j.phase == "fps"}
    failed_outs: set[Path] = set()

    def ready(job: Job) -> bool:
        need = job.needs()
        return need is None or need.is_file() or need not in making or need in failed_outs

    running: dict[int, tuple[subprocess.Popen, Job, object]] = {}
    done, failed, total = [], [], len(pending)
    env = {**os.environ, **THREAD_ENV}
    while pending or running:
        for cpu in [c for c in cores if c not in running]:
            job = next((j for j in pending if ready(j)), None)
            if job is None or available_gb() < MEM_MARGIN_GB:
                break
            pending = [j for j in pending if j is not job]       # by identity, not dataclass equality
            job.log.parent.mkdir(parents=True, exist_ok=True)
            log = open(job.log, "w")
            proc = subprocess.Popen(job.command(), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    preexec_fn=lambda c=cpu: os.sched_setaffinity(0, {c}))
            running[cpu] = (proc, job, log)
        time.sleep(0.5)
        for cpu, (proc, job, log) in list(running.items()):
            if proc.poll() is None:
                continue
            log.close()
            del running[cpu]
            ok = proc.returncode == 0 and job.out.is_file()
            (done if ok else failed).append(job)
            if not ok:
                failed_outs.add(job.out)
            print(f"[{len(done) + len(failed)}/{total}] {'ok  ' if ok else 'FAIL'} {job.name} {job.tag} seed {job.seed}"
                  + ("" if ok else f"  (log: {_shown(job.log)})"), flush=True)
    return done, failed


def run_phases(jobs: list[Job], cores: list[int]) -> int:
    print(f"{len(cores)} workers on cores {cores}")
    failed_all = []
    # fps and timing share one queue (a dataset's FPS runs start once its own picks exist);
    # diagnostic runs are a separate phase after every timing run (they change the load)
    for label, phases in (("fps + timing", ("fps", "timing")), ("diagnostic", ("diagnostic",))):
        phase_jobs = [j for j in jobs if j.phase in phases]
        todo = sum(not j.out.is_file() for j in phase_jobs)
        print(f"--- phase {label}: {todo} of {len(phase_jobs)} jobs to run", flush=True)
        _, failed = execute(phase_jobs, cores)
        failed_all += failed
    for j in failed_all:
        print(f"FAILED {j.name} {j.tag} seed {j.seed}: see {_shown(j.log)}")
    return 1 if failed_all else 0


def pilot_missing(exp: dict) -> list[str]:
    """(dataset, config) pairs whose pilot runs are missing."""
    return sorted({f"{j.name} {j.tag}" for j in plan(exp, pilot_seeds(exp)) if j.phase != "fps" and not j.out.is_file()})


def status(exp: dict, workers: int) -> None:
    jobs = plan(exp)
    for phase in ("fps", "timing", "diagnostic"):
        pj = [j for j in jobs if j.phase == phase]
        print(f"{phase:<11} {sum(j.out.is_file() for j in pj):>5} done / {len(pj):>5}")
    walls: dict[tuple, list[float]] = {}
    for j in jobs:
        if j.phase != "fps" and j.out.is_file():
            walls.setdefault((j.name, j.tag), []).append(json.loads(j.out.read_text())["wall_time_sec"])
    remaining = sum(sum(walls[(j.name, j.tag)]) / len(walls[(j.name, j.tag)])
                    for j in jobs if j.phase != "fps" and not j.out.is_file() and (j.name, j.tag) in walls)
    unknown = sum(1 for j in jobs if j.phase != "fps" and not j.out.is_file() and (j.name, j.tag) not in walls)
    left = remaining / max(1, workers)
    shown = f"{left / 3600:.1f} h" if left >= 3600 else f"{left / 60:.0f} min"
    print(f"estimated remaining: {shown} on {workers} workers"
          + (f" (+{unknown} jobs without a finished run to estimate from)" if unknown else ""))


def check_pairs(exp: dict) -> int:
    n = bad = 0
    for j in plan(exp):
        if j.phase != "diagnostic" or not j.out.is_file():
            continue
        twin = Job("timing", j.name, j.seed, {**j.config, "mode": "fec"}, j.overrides, j.max_fraction, j.algorithm)
        if not twin.out.is_file():
            continue
        n += 1
        try:
            measurement.same_evolution(json.loads(twin.out.read_text()), json.loads(j.out.read_text()))
        except ValueError as exc:
            bad += 1
            print(f"FAIL {exc}")
    print(f"check: {n - bad} of {n} timing/diagnostic pairs evolved identically")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["plan", "pilot", "run", "status", "check"])
    p.add_argument("experiment", type=Path)
    p.add_argument("--workers", type=int, help="at most this many (default: the experiment's hive.workers)")
    p.add_argument("--max-busy", type=float, help="skip cores used more than this at start (0-1)")
    args = p.parse_args(argv)
    exp = experiment.load(args.experiment)
    workers = args.workers or exp["hive"]["workers"]
    max_busy = args.max_busy if args.max_busy is not None else exp["hive"]["max_busy"]
    if args.command == "plan":
        jobs = plan(exp)
        for phase in ("fps", "timing", "diagnostic"):
            pj = [j for j in jobs if j.phase == phase]
            print(f"{phase:<11} {len(pj):>5} jobs, {sum(not j.out.is_file() for j in pj):>5} to run")
        return 0
    if args.command == "status":
        status(exp, workers or len(physical_cores()))
        return 0
    if args.command == "check":
        return check_pairs(exp)
    if args.command == "run":
        missing = pilot_missing(exp)
        if missing:
            print(f"FAIL pilot incomplete for {len(missing)} config(s), e.g. {missing[:3]}; run `pilot` first")
            return 1
    cores = free_cores(workers, max_busy=max_busy)
    if not cores:
        print("FAIL no idle physical core")
        return 1
    seeds = pilot_seeds(exp) if args.command == "pilot" else exp["seeds"]
    return run_phases(plan(exp, seeds), cores)


if __name__ == "__main__":
    sys.exit(main())
