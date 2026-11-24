"""Instrumented FEC diagnostic runs: what wrong cached fitness values do to the search.

A diagnostic run with --instrument evolves exactly as the stored timing run of the same seed (recording only: no
random draw, no fitness value changed) and also records, per generation, how the fitness selection used differs from
the true full-training fitness across the population, how many tournaments a wrong value decided, whether the elite
carries a wrong value, plus memory, prediction hashes, a shadow exact cache, selection pressure, and event logs (hit
log, cache-entry log, fake-hit detail, population snapshots). Definitions: specs/019-instrumented-diagnostics/
data-model.md. Every new generation column starts with 'ins_'.

Runs live under setup 'instrumented' (or 'instrumented_<hash>' with parameter overrides) and are checked against two
stored twins: the timing run (mode fec) and the diagnostic run (mode fec_diagnostic) of the same identity.

Commands:
  python instrument.py check [GE|GP]      twin-check every instrumented run on this machine
  python instrument.py summary [GE|GP]    per configuration: runs, twins ok, overhead ratio, changed-tournament share
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from operator import attrgetter
from pathlib import Path

import numpy as np
import pandas as pd
from deap import tools
from scipy.stats import rankdata, spearmanr

import datasets

ROOT = datasets.ROOT
SETUP = "instrumented"
SNAPSHOT_GENS = (0, 10, 20, 30)
LOG_KINDS = ("hits", "entries", "fakes", "snapshots")
NAN = float("nan")


# ---------------------------------------------------------------- evaluation helpers (same values as ge.error_of)

def full_eval(run, points) -> tuple[float, np.ndarray | None]:
    """(error rate, class labels) of a compiled program on (x, y): the value is computed exactly as ge.error_of
    (same dtype conversions, same ge.error_rate), the labels with the rule error_rate uses. NaN, None if invalid."""
    import ge
    out = None if run is None else run(np.asarray(points[0], dtype=np.float64))
    if out is None:
        return NAN, None
    return ge.error_rate(out, np.asarray(points[1], dtype=np.int64)), labels_of(out)


def labels_of(out: np.ndarray) -> np.ndarray:
    """Class labels of program outputs, as ge.error_rate reads them: binary outputs (all 0/1) -> out > 0,
    otherwise the output is the class itself."""
    binary = np.all((out == 0) | (out == 1))
    return ((out > 0) if binary else out).astype(np.int64)


def label_hash(labels: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(labels, dtype=np.int64).tobytes(), digest_size=16).hexdigest()


def true_of(ind) -> float:
    if not hasattr(ind, "fec_true"):
        raise ValueError("instrumented run: an individual has no true fitness (fec_true); was it evaluated?")
    return float(ind.fec_true)


def _true_key(ind) -> float:
    """Sort key for the correct tournament winner: fitness is minimised, so larger -true is better; a NaN true
    fitness (valid on the fingerprint, invalid on all rows) counts as the worst."""
    t = true_of(ind)
    return -math.inf if math.isnan(t) else -t


# ---------------------------------------------------------------- selection

def select_tournament(individuals, k, tournsize, rec=None, fit_attr="fitness"):
    """DEAP's tools.selTournament, step for step (same tools.selRandom draws, same max with the first of ties), plus
    the recorder's counterfactual: the winner the true fitness would pick from the same contestants."""
    chosen = []
    for _ in range(k):
        aspirants = tools.selRandom(individuals, tournsize)
        winner = max(aspirants, key=attrgetter(fit_attr))
        chosen.append(winner)
        if rec is not None:
            rec.tournament(aspirants, winner)
    if rec is not None:
        rec.parents(chosen)
    return chosen


# ---------------------------------------------------------------- population measures

def population_measures(valid: list) -> dict:
    """ins_n_valid, ins_n_wrong, ins_spearman(+_undefined), ins_rank_shift_mean/max over the valid programs."""
    used = np.array([ind.fitness.values[0] for ind in valid], dtype=float)
    true = np.array([true_of(ind) for ind in valid], dtype=float)
    wrong = ~((used == true) | (np.isnan(used) & np.isnan(true)))
    out = {"ins_n_valid": len(valid), "ins_n_wrong": int(wrong.sum())}
    finite = ~np.isnan(true)                            # a NaN true fitness has no rank to compare
    u, t = used[finite], true[finite]
    if len(u) < 2 or np.ptp(u) == 0 or np.ptp(t) == 0:
        out.update(ins_spearman=NAN, ins_spearman_undefined=1)
    else:
        out.update(ins_spearman=float(spearmanr(u, t).statistic), ins_spearman_undefined=0)
    if wrong.any():
        t = np.where(np.isnan(true), np.inf, true)
        shift = np.abs(rankdata(used, method="average") - rankdata(t, method="average"))[wrong]
        out.update(ins_rank_shift_mean=float(shift.mean()), ins_rank_shift_max=float(shift.max()))
    else:
        out.update(ins_rank_shift_mean=NAN, ins_rank_shift_max=NAN)
    return out


def elite_measures(best) -> dict:
    if best is None:
        return {"ins_elite_used": NAN, "ins_elite_true": NAN, "ins_elite_wrong": 0}
    u, t = float(best.fitness.values[0]), true_of(best)
    return {"ins_elite_used": u, "ins_elite_true": t, "ins_elite_wrong": int(u != t)}


def cache_bytes(*tables: dict) -> int:
    """Bytes of the cache's own tables: each dict plus its keys and values."""
    return int(sum(sys.getsizeof(t) + sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in t.items())
                   for t in tables))


def _q(v: list, q: float) -> float:
    return float(np.quantile(v, q)) if v else NAN


# ---------------------------------------------------------------- the recorder

class Recorder:
    """Collects the instrumented measures of one run. The evaluator calls on_* per evaluation; the evolution loop
    calls begin_generation, the selection wrapper calls tournament/parents, and finish_generation returns the
    generation's ins_ columns (and takes the snapshot)."""

    def __init__(self, last_gen: int, classes) -> None:
        self.last_gen = last_gen
        self.classes = [int(c) for c in classes]
        self.gen = 0
        self.hits_log: list[tuple] = []
        self.entries: list[dict] = []
        self.fakes_log: list[dict] = []
        self.snapshots: list[tuple] = []
        self.exact_seen: set[str] = set()
        self.fingerprint_class_counts: dict = {}
        self.totals = dict.fromkeys(("ins_n_wrong", "ins_tourn_changed", "ins_tournaments", "ins_hits_same_pred",
                                     "ins_hits_equal_fit_only", "ins_exact_hits", "ins_exact_lookups"), 0)
        self.elite_wrong_generations = 0
        self.unchanged = None
        self._reset()

    def _reset(self) -> None:
        self.n_tourn = self.n_changed = 0
        self.gaps: list[float] = []
        self.distinct_parents = None
        self.same_pred = self.equal_fit_only = 0
        self.fake_err: list[float] = []
        self.exact_lookups = self.exact_hits = self.exact_ns = 0

    def begin_generation(self, gen: int) -> None:
        self.gen = gen
        self.unchanged = None
        self._reset()

    # selection ---------------------------------------------------------------
    def tournament(self, aspirants: list, winner) -> None:
        correct = max(aspirants, key=_true_key)
        self.n_tourn += 1
        tw, tc = true_of(winner), true_of(correct)
        tw = math.inf if math.isnan(tw) else tw
        if tw > tc:                                     # the real winner is truly worse: a decision changed
            self.n_changed += 1
            self.gaps.append(tw - tc if math.isfinite(tw) else NAN)

    def parents(self, chosen: list) -> None:
        self.distinct_parents = len({id(x) for x in chosen})

    # evaluations ---------------------------------------------------------------
    def exact_lookup(self, text: str) -> None:
        t0 = time.perf_counter_ns()
        seen = text in self.exact_seen
        self.exact_ns += time.perf_counter_ns() - t0
        self.exact_lookups += 1
        self.exact_hits += seen
        self.exact_seen.add(text)

    def new_entry(self, fitness: float, size: int, label_hash_: str | None) -> int:
        self.entries.append({"entry": len(self.entries), "created_gen": self.gen, "fitness": fitness, "size": size,
                             "reuses": 0, "_hash": label_hash_})
        return len(self.entries) - 1

    def on_hit(self, entry: int, used: float, true: float, same_program: bool, new_hash: str | None,
               size_new: int) -> tuple[bool, bool]:
        """Log a hit; returns (fake, same predictions)."""
        e = self.entries[entry]
        e["reuses"] += 1
        fake = not (used == true or (math.isnan(used) and math.isnan(true)))
        same_pred = new_hash is not None and new_hash == e["_hash"]
        if fake:
            self.fake_err.append(used - true)
        elif same_pred:
            self.same_pred += 1
        else:
            self.equal_fit_only += 1
        self.hits_log.append((self.gen, used, true, int(fake), int(same_program), int(same_pred), size_new, e["size"],
                              self.gen - e["created_gen"]))
        return fake, same_pred

    def on_fake_detail(self, entry: int, used: float, true: float, new_labels, entry_labels, y) -> None:
        row = {"gen": self.gen, "entry": entry, "used": used, "true": true}
        if new_labels is None or entry_labels is None:
            row["n_disagree"] = NAN
            row.update({f"disagree_class_{c}": NAN for c in self.classes})
        else:
            diff = new_labels != entry_labels
            row["n_disagree"] = int(diff.sum())
            row.update({f"disagree_class_{c}": int((diff & (y == c)).sum()) for c in self.classes})
        self.fakes_log.append(row)

    # generation end ---------------------------------------------------------------
    def finish_generation(self, gen: int, population: list, valid: list, best, cache_size_bytes: int) -> dict:
        out = population_measures(valid)
        out.update(elite_measures(best))
        first = gen == 0
        out.update(ins_tournaments=NAN if first else self.n_tourn, ins_tourn_changed=NAN if first else self.n_changed,
                   ins_tourn_changed_share=NAN if first or not self.n_tourn else self.n_changed / self.n_tourn,
                   ins_tourn_gap_mean=float(np.nanmean(self.gaps)) if self.gaps else NAN,
                   ins_tourn_gap_max=float(np.nanmax(self.gaps)) if self.gaps else NAN,
                   ins_distinct_parents=NAN if first else self.distinct_parents,
                   ins_unchanged_offspring=NAN if first else self.unchanged)
        a = [abs(x) for x in self.fake_err]
        out.update(ins_fake_err_median=_q(a, 0.5), ins_fake_err_p90=_q(a, 0.9), ins_fake_err_max=max(a, default=NAN),
                   ins_fake_err_signed_mean=float(np.mean(self.fake_err)) if self.fake_err else NAN,
                   ins_fake_err_signed_median=_q(self.fake_err, 0.5),
                   ins_hits_same_pred=self.same_pred, ins_hits_equal_fit_only=self.equal_fit_only,
                   ins_exact_lookups=self.exact_lookups, ins_exact_hits=self.exact_hits,
                   ins_exact_lookup_ns=self.exact_ns / self.exact_lookups if self.exact_lookups else NAN,
                   ins_cache_bytes=cache_size_bytes)
        for k in self.totals:
            v = out.get(k)
            self.totals[k] += 0 if v is None or (isinstance(v, float) and math.isnan(v)) else int(v)
        self.elite_wrong_generations += out["ins_elite_wrong"]
        self.last_cache_bytes = cache_size_bytes
        if gen in SNAPSHOT_GENS or gen == self.last_gen:
            ids = {id(x) for x in valid}
            for i, ind in enumerate(population):
                used = ind.fitness.values[0] if ind.fitness.valid else NAN
                text = "" if getattr(ind, "invalid", False) else _text(ind)
                self.snapshots.append((gen, i, text, used, true_of(ind) if hasattr(ind, "fec_true") else NAN,
                                       int(id(ind) in ids)))
        return out

    # run end ---------------------------------------------------------------
    def run_fields(self, peak_rss_bytes: int) -> dict:
        return {"peak_rss_bytes": peak_rss_bytes, "cache_bytes_final": getattr(self, "last_cache_bytes", None),
                "fingerprint_class_counts": self.fingerprint_class_counts,
                "totals": {**self.totals, "elite_wrong_generations": self.elite_wrong_generations}}

    def logs(self) -> dict:
        return {"hits": pd.DataFrame(self.hits_log, columns=["gen", "used", "true", "fake", "same_program",
                                                             "same_pred", "size_new", "size_entry", "entry_age"]),
                "entries": pd.DataFrame([{k: v for k, v in e.items() if k != "_hash"} for e in self.entries],
                                        columns=["entry", "created_gen", "fitness", "size", "reuses"]),
                "fakes": pd.DataFrame(self.fakes_log, columns=["gen", "entry", "used", "true", "n_disagree",
                                                               *[f"disagree_class_{c}" for c in self.classes]]),
                "snapshots": pd.DataFrame(self.snapshots, columns=["gen", "index", "phenotype", "used", "true",
                                                                   "valid"])}


def _text(ind) -> str:
    return ind.phenotype if hasattr(ind, "phenotype") else str(ind)


def peak_rss_bytes() -> int:
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024     # Linux reports KB


# ---------------------------------------------------------------- twin checks

DIAG_GEN_FIELDS = ("fec_fake_hits", "fec_fitness_diff", "fec_abs_fitness_diff", "fec_max_abs_fitness_diff")


CORE_GEN_FIELDS = ("fec_hits", "fec_misses", "min", "avg", "std", "max", "invalid", "best_nodes",
                   "unique_phenotypes", "novel_phenotypes", "cumulative_phenotypes")


def compare(inst: dict, twin: dict, diagnostic: bool) -> list[str]:
    """Fields in which an instrumented run differs from a stored twin (empty: identical evolution). Identity, best
    program, evaluations, hits, misses, full evaluations, and every per-generation column both runs have (except
    timing, diagnostics and ins_*; the core columns must be among them): stored twins made before a column existed
    (e.g. cap_hits_* before 015) are compared on the columns they have. Diagnostic twin: additionally the fake-hit
    totals and the per-generation fake-hit columns."""
    import measurement
    bad = [k for k in ("dataset", "seed", "params", "best_phenotype", "n_train_evals") if inst.get(k) != twin.get(k)]
    if (inst.get("fec") or {}).get("fraction") != (twin.get("fec") or {}).get("fraction"):
        bad.append("fraction")
    for k in ("hits", "misses", "n_full_evals"):
        if inst["measures"].get(k, inst.get("fec", {}).get(k)) != twin["measures"].get(k, twin.get("fec", {}).get(k)):
            bad.append(k)
    ga, gb = inst.get("generations", []), twin.get("generations", [])
    if len(ga) != len(gb) or not ga:
        bad.append("generations.length")
    else:
        keys = (set(ga[0]) & set(gb[0])) - set(measurement._NOT_EVOLUTION) - {k for k in ga[0] if k.startswith("ins_")}
        missing = [k for k in CORE_GEN_FIELDS if k not in keys]
        if missing:
            bad.append(f"generations.columns missing {missing}")
        bad += sorted({f"generations.{k}" for x, y in zip(ga, gb) for k in keys if not _same(x.get(k), y.get(k))})
    if diagnostic:
        if (inst.get("fec") or {}).get("fake_hits") != (twin.get("fec") or {}).get("fake_hits"):
            bad.append("fake_hits")
        ga, gb = inst.get("generations", []), twin.get("generations", [])
        for k in DIAG_GEN_FIELDS:
            if len(ga) != len(gb) or any(not _same(x.get(k), y.get(k)) for x, y in zip(ga, gb)):
                bad.append(f"generations.{k}")
    return bad


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))


def twin_paths(path: Path, record: dict) -> dict:
    """{"timing": path, "diagnostic": path} of the stored twins: same place, setup without the 'instrumented'
    prefix; the timing twin's folder is fec_<sampler>_frac<f> instead of fec_diagnostic_..."""
    setup = record["setup"]
    twin_setup = "default" if setup == SETUP else setup[len(SETUP) + 1:]
    parts = list(path.parts)
    parts[parts.index(setup)] = twin_setup
    diag = Path(*parts)
    return {"diagnostic": diag, "timing": diag.parent.parent / diag.parent.name.replace("fec_diagnostic_", "fec_", 1)
            / diag.name}


def check_twin(path: Path) -> dict:
    """Compare an instrumented run with both stored twins; write the outcome into the run record; on any
    difference rename the file to .mismatch. Returns {"path", "ok", "mismatches", "twins"}."""
    path = Path(path)
    record = json.loads(path.read_text())
    bad, twins = [], {}
    for kind, tp in twin_paths(path, record).items():
        twins[kind] = str(tp.relative_to(ROOT)) if tp.is_relative_to(ROOT) else str(tp)
        if not tp.is_file():
            bad.append(f"{kind} twin missing")
        else:
            bad += [f"{kind}: {m}" for m in compare(record, json.loads(tp.read_text()), kind == "diagnostic")]
    result = {"ok": not bad, "mismatches": bad, "twins": twins}
    record.setdefault("instrument", {})["twin_check"] = result
    path.write_text(json.dumps(record, indent=1) + "\n")
    if bad:
        path.rename(path.with_suffix(".mismatch"))
    return {"path": str(path), **result}


# ---------------------------------------------------------------- store tables and CLI

def load_runs(algorithm: str | None = None, root: Path = ROOT) -> list[Path]:
    algos = [algorithm.lower()] if algorithm else ["ge", "gp"]
    return sorted(p for a in algos for s in (root / "results" / a).glob(f"{SETUP}*") for p in s.glob("*/*/seed*.json"))


def _identity(r: dict) -> dict:
    f = r.get("fec") or {}
    return {"algorithm": r["algorithm"], "dataset": r["dataset"], "sampler": f.get("sampler"),
            "fraction": round(float(f.get("fraction", NAN)), 2), "seed": r["seed"], "setup": r["setup"]}


def _diag_wall(path: Path, r: dict) -> float:
    tp = twin_paths(path, r)["diagnostic"]
    return json.loads(tp.read_text())["wall_time_sec"] if tp.is_file() else NAN


AGE_WINDOWS = (5, 10)          # a cache that keeps only the entries of the last N generations


def hit_summary(path: Path, ins: dict) -> dict:
    """Per run, from its hit log: hits by a different program with the correct fitness, split into those that
    predicted every training row as the stored program did (`hits_same_pred`) and the others (`hits_equal_fit_only`,
    equal fitness, different predictions); and the hits whose cache entry was at most N generations old."""
    name = (ins.get("files") or {}).get("hits")
    cols = ["log_hits", "log_hits_same_pred", "log_hits_equal_fit_only", *[f"log_hits_age_le{n}" for n in AGE_WINDOWS]]
    if not name or not path.with_name(name).is_file():
        return dict.fromkeys(cols, NAN)
    h = pd.read_csv(path.with_name(name), usecols=["fake", "same_program", "same_pred", "entry_age"])
    b = h[(h.fake == 0) & (h.same_program == 0)]
    return dict(zip(cols, [len(h), int(b.same_pred.sum()), int((b.same_pred == 0).sum()),
                           *[int((h.entry_age <= n).sum()) for n in AGE_WINDOWS]]))


def tables(algorithm: str | None = None, root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(instrumented_runs, instrumented_generations)."""
    runs, gens = [], []
    for p in load_runs(algorithm, root):
        r = json.loads(p.read_text())
        ins, f = r.get("instrument") or {}, r.get("fec") or {}
        ident = _identity(r)
        dw = _diag_wall(p, r)
        runs.append({**ident, "code_commit": r.get("code_commit"), "table_checksum": r.get("table_checksum"),
                     "hits": f.get("hits"), "misses": f.get("misses"), "fake_hits": f.get("fake_hits"),
                     "hit_rate": f.get("hit_rate"), "cache_size": f.get("cache_size"),
                     "wall_time_sec": r.get("wall_time_sec"), "diag_wall_time_sec": dw,
                     "overhead_ratio": r.get("wall_time_sec", NAN) / dw if dw and not math.isnan(dw) else NAN,
                     "peak_rss_bytes": ins.get("peak_rss_bytes"), "cache_bytes_final": ins.get("cache_bytes_final"),
                     **{f"total_{k.removeprefix('ins_')}": v for k, v in (ins.get("totals") or {}).items()},
                     **{f"fp_class_{k}": v for k, v in (ins.get("fingerprint_class_counts") or {}).items()},
                     "twin_ok": (ins.get("twin_check") or {}).get("ok"), **hit_summary(p, ins)})
        gens += [{**ident, **g} for g in r["generations"]]
    order = ["algorithm", "dataset", "fraction", "seed"]
    return (pd.DataFrame(runs).sort_values(order).reset_index(drop=True) if runs else pd.DataFrame(),
            pd.DataFrame(gens).sort_values(order + ["gen"]).reset_index(drop=True) if gens else pd.DataFrame())


def summary(algorithm: str | None = None) -> pd.DataFrame:
    runs, gens = tables(algorithm)
    if runs.empty:
        return runs
    share = gens.groupby(["algorithm", "dataset", "fraction"])["ins_tourn_changed_share"].median()
    g = runs.groupby(["algorithm", "dataset", "fraction"])
    out = g.agg(runs=("seed", "size"), twins_ok=("twin_ok", "sum"), overhead_median=("overhead_ratio", "median"),
                overhead_max=("overhead_ratio", "max"), peak_rss_mb=("peak_rss_bytes", lambda v: v.max() / 2**20))
    return out.join(share.rename("changed_share_median")).reset_index()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in ("check", "summary"):
        print(__doc__)
        return 1
    algo = argv[1] if len(argv) > 1 else None
    if argv[0] == "check":
        res = [check_twin(p) for p in load_runs(algo)]
        bad = [r for r in res if not r["ok"]]
        for r in bad:
            print(f"MISMATCH {r['path']}: {', '.join(r['mismatches'])}")
        print(f"twin check: {len(res) - len(bad)} of {len(res)} instrumented runs reproduce both stored twins")
        return 1 if bad else 0
    with pd.option_context("display.width", 180, "display.max_rows", 300):
        print(summary(algo).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
