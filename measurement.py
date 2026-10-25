"""Measurement stage: the one definition of every reported metric.

Per run (recorded by the evaluators, see ge.BaselineEvaluator and fec.FECEvaluator):
  evaluation time  sum over every training evaluation of the time (perf_counter) from the evaluator
                   receiving an individual until its fitness value is ready. Baseline: the full
                   evaluation. FEC: fingerprint evaluation + key + lookup, plus on a miss the full
                   evaluation and the store. Never: fake-hit re-checks, statistics, test evaluations.
  effort           sum over training evaluations of operator calls in the phenotype x rows evaluated
                   (fingerprint rows for a fingerprint, training rows for a full evaluation); every operator
                   weighs 1. effort_by_operator keeps the same sums per operator ("full", "sample"), so S_E
                   under other operator weights needs no new runs (effort_with_weights).
  generation time  sum of the per-generation times; wall time = run start to end (FEC: + sampling).

Per dataset and FEC configuration (metrics(), runs paired by seed):
  S_T       mean baseline evaluation time / mean FEC evaluation time                    (paper)
  dAcc      mean FEC test accuracy - mean baseline test accuracy; per pair FEC_k - base_k (paper)
            (accuracy = 1 - test error rate)
  S_E       mean baseline effort / mean FEC effort
  S_gen     mean baseline generation-time sum / mean FEC generation-time sum
  S_wall    mean baseline wall time / mean FEC wall time; a FEC run's wall time includes the time
            to sample its fingerprint rows also when the farthest-point picks were computed in a
            separate job and read from the cache (fps_from_cache): their recorded sampling time is added
  hit rate  total hits / (total hits + total misses) over the FEC runs
  fake-hit ratio  total fake hits / total hits over the diagnostic runs
"""
from __future__ import annotations

import re
import time

import numpy as np

from operators import OPERATORS

_CALL = re.compile(r"\b(" + "|".join(sorted(OPERATORS, key=len, reverse=True)) + r")\(")


def operator_calls(phenotype: str) -> int:
    return len(_CALL.findall(phenotype))


def operator_counts(phenotype: str) -> dict[str, int]:
    """Calls of each operator in the phenotype; the values sum to operator_calls(phenotype)."""
    out: dict[str, int] = {}
    for name in _CALL.findall(phenotype):
        out[name] = out.get(name, 0) + 1
    return out


def effort_with_weights(measures: dict, weights: dict[str, float], default: float = 1.0) -> float:
    """A run's effort with operator weights (operators not in `weights` weigh `default`), from its
    effort_by_operator; all weights 1 gives the recorded effort."""
    if "effort_by_operator" not in measures:
        raise ValueError("run has no effort_by_operator (recorded from 2026-10-24); rerun it to reweight")
    return float(sum(weights.get(op, default) * v for part in measures["effort_by_operator"].values()
                     for op, v in part.items()))


# ---------------------------------------------------------------- per-generation diagnostics

_FEATURE = re.compile(r"\bx\[(\d+)\]|\bx(\d+)\b")       # GE x[i], GP x<i>


def features_used(phenotype: str) -> set[int]:
    return {int(a or b) for a, b in _FEATURE.findall(phenotype)}


def _mean(values) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def generation_extras(population: list, valid: list, best, cumulative: set, algorithm: str) -> dict:
    """Population diagnostics of one generation (the reference implementation grape/algorithms._ge_pheno_extra_fields and
    gp_algorithms._extra_gen_fields): read from the individuals only, so the evolution is unchanged.
    `valid` = individuals with a fitness; `cumulative` = the run's set of phenotypes seen (updated)."""
    from ge import text_of
    texts = [text_of(ind) for ind in valid]
    current = set(texts)
    novel = current - cumulative
    cumulative |= current
    if algorithm == "GE":
        depth = [ind.depth for ind in valid]
        terminals = [ind.terminals for ind in valid]
    else:
        depth = [ind.height for ind in valid]
        terminals = [sum(1 for node in ind if node.arity == 0) for ind in valid]
    best_text = text_of(best) if best is not None else ""
    out = {"unique_phenotypes": len(current), "novel_phenotypes": len(novel),
           "cumulative_phenotypes": len(cumulative),
           "avg_depth": _mean(depth), "avg_terminals": _mean(terminals),
           "avg_operators": _mean([operator_calls(t) for t in texts]),
           "avg_pheno_len": _mean([len(t) for t in texts]),
           "avg_operators_used": _mean([len(set(_CALL.findall(t))) for t in texts]),
           "avg_features_used": _mean([len(features_used(t)) for t in texts]),
           "fitness_diversity": len({ind.fitness.values[0] for ind in valid}) / len(population),
           "best_terminals": 0 if best is None else (best.terminals if algorithm == "GE" else
                                                      sum(1 for node in best if node.arity == 0)),
           "best_operators": operator_calls(best_text), "best_features_used": len(features_used(best_text)),
           "best_pheno_len": len(best_text)}
    if algorithm == "GE":
        out.update({"avg_genome_length": _mean([len(ind.genome) for ind in valid]),
                    "avg_used_codons": _mean([ind.used_codons for ind in valid]),
                    "structural_diversity": len({str(ind.structure) for ind in population}) / len(population)})
    return out


class Meter:
    """Evaluation time, number of training evaluations and effort of one run."""

    def __init__(self) -> None:
        self.eval_time_sec = 0.0
        self.n_evals = 0
        self.effort_full = 0.0
        self.effort_sample = 0.0
        self.effort_ops = {"full": {}, "sample": {}}     # operator -> calls x rows
        self.n_full = 0
        self.n_sample = 0
        self._t0 = 0.0

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        """The fitness value is ready: everything after this is not evaluation time."""
        self.eval_time_sec += time.perf_counter() - self._t0
        self.n_evals += 1

    def _add(self, part: str, phenotype: str, n_rows: int) -> int:
        """Called after stop(): never evaluation time."""
        ops, total = self.effort_ops[part], 0
        for name, k in operator_counts(phenotype).items():
            ops[name] = ops.get(name, 0) + k * n_rows
            total += k
        return total * n_rows

    def add_full(self, phenotype: str, n_rows: int) -> None:
        self.effort_full += self._add("full", phenotype, n_rows)
        self.n_full += 1

    def add_sample(self, phenotype: str, n_rows: int) -> None:
        self.effort_sample += self._add("sample", phenotype, n_rows)
        self.n_sample += 1

    def as_dict(self) -> dict:
        return {"eval_time_sec": self.eval_time_sec, "n_evals": self.n_evals,
                "effort": self.effort_full + self.effort_sample, "effort_full": self.effort_full,
                "effort_sample": self.effort_sample, "n_full_evals": self.n_full, "n_sample_evals": self.n_sample,
                "effort_by_operator": {p: dict(sorted(d.items())) for p, d in self.effort_ops.items()}}


class PhaseClock:
    """Per-phase time of fitness evaluations, opt-in. A lap clock: begin() reads the clock when the evaluator
    receives a candidate; each lap(name) adds the time since the previous read to that phase, so the phases of a
    candidate tile the interval exactly. Totals per run and per generation (generation_stats resets the latter).
    Draws no random numbers and touches no fitness value."""

    def __init__(self, probe: bool = True) -> None:
        self.times: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.gen_times: dict[str, float] = {}
        self.gen_counts: dict[str, int] = {}
        self.n_laps = 0
        self._last = 0.0
        self.clock_ns_per_lap = probe_ns_per_lap() if probe else float("nan")

    def begin(self) -> None:
        self._last = time.perf_counter()

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        d = now - self._last
        self._last = now
        self.times[name] = self.times.get(name, 0.0) + d
        self.gen_times[name] = self.gen_times.get(name, 0.0) + d
        self.n_laps += 1

    def count(self, name: str) -> None:
        self.counts[name] = self.counts.get(name, 0) + 1
        self.gen_counts[name] = self.gen_counts.get(name, 0) + 1

    def generation_stats(self) -> dict:
        out = {**{f"ph_t_{k}": v for k, v in self.gen_times.items()}, **{f"ph_{k}": v for k, v in self.gen_counts.items()}}
        self.gen_times, self.gen_counts = {}, {}
        return out

    def as_dict(self, eval_time_sec: float) -> dict:
        """Run totals, the clock cost and the part of the evaluation time no phase covers (begin and the Meter's
        own clock reads)."""
        return {**{f"t_{k}": v for k, v in sorted(self.times.items())}, **dict(sorted(self.counts.items())),
                "n_laps": self.n_laps, "clock_ns_per_lap": self.clock_ns_per_lap,
                "t_unaccounted": eval_time_sec - sum(self.times.values())}


def probe_ns_per_lap(n: int = 100_000) -> float:
    """Mean cost of one lap on this machine (ns), measured on a throw-away clock."""
    c = PhaseClock(probe=False)
    c.begin()
    t0 = time.perf_counter()
    for _ in range(n):
        c.lap("probe")
    return (time.perf_counter() - t0) / n * 1e9


# ---------------------------------------------------------------- pair check and metrics

_NOT_EVOLUTION = {"generation_time", "fec_fake_hits", "fec_fitness_diff", "fec_abs_fitness_diff",
                  "fec_max_abs_fitness_diff"}          # diagnostic-only columns


def same_evolution(a: dict, b: dict) -> None:
    """Raise if two runs (a timing and a diagnostic run) did not evolve identically."""
    problems = [k for k in ("dataset", "seed", "params", "best_phenotype", "n_train_evals") if a[k] != b[k]]
    if a.get("fec", {}).get("fraction") != b.get("fec", {}).get("fraction"):
        problems.append("fraction")
    for k in ("hits", "misses", "n_full_evals"):
        if a["measures"].get(k, a.get("fec", {}).get(k)) != b["measures"].get(k, b.get("fec", {}).get(k)):
            problems.append(k)
    rows = lambda r: [{k: v for k, v in g.items() if k not in _NOT_EVOLUTION} for g in r["generations"]]
    if rows(a) != rows(b):
        problems.append("per-generation statistics")
    if problems:
        raise ValueError(f"{a['dataset']} seed {a['seed']}: runs did not evolve identically: {problems}")


def _by_seed(runs: list[dict]) -> dict:
    out = {r["seed"]: r for r in runs}
    if len(out) != len(runs):
        raise ValueError("two runs with the same seed")
    return out


def wall_time(run: dict) -> float:
    """The run's wall time; for farthest-point picks read from the cache (computed once per seed in a
    separate job), plus their recorded sampling time, which the run itself did not spend."""
    f = run.get("fec") or {}
    return run["wall_time_sec"] + (f.get("sampling_time_sec", 0.0) if f.get("fps_from_cache") else 0.0)


def metrics(baseline: list[dict], fec: list[dict], diagnostic: list[dict] | None = None) -> dict:
    """The metrics of one dataset and FEC configuration, as defined in the module docstring."""
    b, f = _by_seed(baseline), _by_seed(fec)
    if set(b) != set(f):
        raise ValueError(f"baseline and FEC seeds differ: {sorted(set(b) ^ set(f))}")
    seeds = sorted(b)
    mean = lambda runs, key: float(np.mean([runs[s][key] if key in runs[s] else runs[s]["measures"][key] for s in seeds]))
    acc = lambda runs, s: 1.0 - runs[s]["best_test_error"]
    pairs = [acc(f, s) - acc(b, s) for s in seeds]
    hits = sum(f[s]["fec"]["hits"] for s in seeds)
    misses = sum(f[s]["fec"]["misses"] for s in seeds)
    out = {"n_runs": len(seeds), "seeds": seeds,
           "S_T": mean(b, "eval_time_sec") / mean(f, "eval_time_sec"),
           "dAcc": float(np.mean(pairs)), "dAcc_pairs": pairs,
           "S_E": mean(b, "effort") / mean(f, "effort"),
           "S_gen": mean(b, "generation_time_sec") / mean(f, "generation_time_sec"),
           "S_wall": mean(b, "wall_time_sec") / float(np.mean([wall_time(f[s]) for s in seeds])),
           "hit_rate": hits / (hits + misses) if hits + misses else float("nan"),
           "sampling_time_sec": float(np.mean([f[s]["fec"]["sampling_time_sec"] for s in seeds]))}
    if diagnostic:
        d = _by_seed(diagnostic)
        if set(d) != set(f):
            raise ValueError("diagnostic and timing seeds differ")
        for s in seeds:
            same_evolution(f[s], d[s])
        d_hits = sum(d[s]["fec"]["hits"] for s in seeds)
        out["fake_hit_ratio"] = sum(d[s]["fec"]["fake_hits"] for s in seeds) / d_hits if d_hits else float("nan")
    return out
