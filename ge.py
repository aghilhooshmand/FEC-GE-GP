"""GE evolution stage (baseline): one seeded run of grammatical evolution on one dataset.

A run: split the rows 80/20 by the seed, z-score with the training rows' numbers, evolve
with the dataset's grammar on the training rows, report each generation's training error
and the best individual's test error. Fitness = error rate (prediction > 0 -> class 1).

The loop is the reference implementation's grape/algorithms.ge_eaSimpleWithElitism (+ varAnd), ported with the same
order of random draws and fewer report fields; grape/grape.py is vendored unchanged. The fitness
is passed in as `evaluate`, so the FEC feature can replace it.

Writes  results/ge/<setup>/<name>/<config>/seed<seed>.json  one file per run (git-ignored);
                                         <setup> = 'default' or a hash of the parameter overrides
        grammars/init_depth.csv                         minimum initial tree depth per grammar

Commands:
  python ge.py depth <names> | --all                    find and record the minimum initial depth
  python ge.py fairness <names> | --all                 feature-type fairness of the grammar
  python ge.py run <name> --seed S [--param KEY=VALUE ...] [--fraction F --sampler S [--diagnostic]]
                                                        one baseline run, or FEC with a fraction
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from deap import base, creator, tools
from sklearn.model_selection import train_test_split

import datasets
import grammar as grammar_stage
import measurement
import preprocessing
from datasets import DatasetError
from grape import grape
from operators import OPERATORS

PARAMS = {                                  # the reference implementation config.json COMMON_CONFIG (GE)
    "population": 1000, "generations": 30, "test_size": 0.2,
    "p_crossover": 0.8, "p_mutation": 0.05, "tournsize": 4, "elite_size": 1, "halloffame_size": 1,
    "min_init_depth": None,                 # from grammars/init_depth.csv
    "max_init_depth": 7, "max_tree_depth": 35, "codon_size": 255,
    "codon_consumption": "lazy", "genome_representation": "list", "max_genome_length": None,
}
def setup_id(overrides: dict | None) -> str:
    """Directory name for a parameter setup: 'default' without overrides, else a short hash of them."""
    if not overrides:
        return "default"
    return "p" + hashlib.sha256(json.dumps(overrides, sort_keys=True).encode()).hexdigest()[:8]


def parse_overrides(items: list[str]) -> dict:
    """['population=200', 'p_mutation=0.01'] -> {...}; names must be in PARAMS."""
    out = {}
    for item in items or []:
        key, _, value = item.partition("=")
        if key not in PARAMS or key == "min_init_depth":
            raise DatasetError("params", f"unknown or fixed parameter {key!r}; choose from {sorted(PARAMS)}")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


DEPTH_CANDIDATES = (3, 5, 7, 9)             # the reference implementation: (3, 5); 7, 9 added for multiclass grammars
DEPTH_CHECK = {"population": 1000, "max_init_depth": 7, "seed": 0}
DEPTH_COLUMNS = ("name", "grammar_checksum", "min_init_depth", "code_commit")

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", grape.Individual, fitness=creator.FitnessMin)


# ---------------------------------------------------------------- fitness

EVAL_ERRORS = (FloatingPointError, ZeroDivisionError, OverflowError, MemoryError, IndexError)


def checked(call, x: np.ndarray) -> np.ndarray | None:
    """Run `call()` (an individual on x, features × rows) under the one validity rule for every mode
    and every number of rows: the output must be real with exactly one value per
    row, so a constant expression is invalid even on a 1-row fingerprint; listed errors -> None."""
    try:
        with np.errstate(all="ignore"):
            out = call()
    except EVAL_ERRORS:
        return None
    if not np.isrealobj(out):
        return None
    out = np.asarray(out, dtype=np.float64)
    return out if out.ndim == 1 and out.shape[0] == x.shape[1] else None


def compile_ge(ind):
    """Compile a GE individual once: a function x -> outputs or None, or None if it is invalid.
    The evaluators call this once per evaluation and reuse it for the fingerprint and the full data."""
    if getattr(ind, "invalid", False):
        return None
    code = compile(ind.phenotype, "<phenotype>", "eval")
    return lambda x: checked(lambda: eval(code, {"np": np, "x": x, **OPERATORS}), x)


def predict(phenotype: str, x: np.ndarray) -> np.ndarray | None:
    """The phenotype's outputs on x, or None if invalid (see `checked`)."""
    return checked(lambda: eval(phenotype, {"np": np, "x": x, **OPERATORS}), x)


def error_rate(prediction: np.ndarray, y: np.ndarray) -> float:
    """Binary (<bool> root, outputs 0/1): prediction > 0 -> class 1. Multiclass (argmax
    root): the output is the class itself, -1 for a tie, which never equals a label."""
    binary = np.all((prediction == 0) | (prediction == 1))
    labels = (prediction > 0).astype(int) if binary else prediction
    return 1 - np.mean(np.equal(y, labels))


def text_of(ind) -> str:
    """The individual's phenotype text: GE's mapped phenotype, or a GP tree's string."""
    return ind.phenotype if hasattr(ind, "phenotype") else str(ind)


def predict_ge(ind, x: np.ndarray) -> np.ndarray | None:
    run = compile_ge(ind)
    return None if run is None else run(x)


def error_of(run, points) -> float:
    """Error rate of a compiled individual on (x, y); NaN when invalid."""
    out = None if run is None else run(np.asarray(points[0], dtype=np.float64))
    return float("nan") if out is None else error_rate(out, np.asarray(points[1], dtype=np.int64))


def fitness_compiled(compile_fn):
    """Fitness for any engine from its compile step (compiled once per call)."""
    return lambda ind, points: (error_of(compile_fn(ind), points),)


def fitness(ind, points) -> tuple[float]:
    """Error rate of an individual on (x: features × rows, y); NaN when it can't be evaluated
    (the reference implementation util_ge.baseline_fitness_eval, binary path)."""
    if getattr(ind, "invalid", False):
        return (float("nan"),)
    prediction = predict(ind.phenotype, np.asarray(points[0], dtype=np.float64))
    if prediction is None:
        return (float("nan"),)
    return (error_rate(prediction, np.asarray(points[1], dtype=np.int64)),)


class BaselineEvaluator:
    """A fitness function (GE's by default) with the measurement of evaluation time and effort
    of every training evaluation."""

    def __init__(self, compile_fn=None, phases=None) -> None:
        from measurement import Meter
        self.meter = Meter()
        self.compile_fn = compile_fn or compile_ge
        self.fitness_fn = fitness_compiled(self.compile_fn)
        self.phases = phases                            # measurement.PhaseClock or None

    def __call__(self, ind, points) -> tuple[float]:
        self.meter.start()
        if self.phases:                                 # the same two steps as fitness_fn, timed apart
            ph = self.phases
            ph.begin()
            run = self.compile_fn(ind)
            ph.lap("compile")
            value = (error_of(run, points),)
            ph.lap("full_eval" if run is not None and not np.isnan(value[0]) else "invalid")
            ph.count("n_full" if run is not None and not np.isnan(value[0]) else "n_invalid")
        else:
            value = self.fitness_fn(ind, points)
        self.meter.stop()
        if not getattr(ind, "invalid", False):
            self.meter.add_full(text_of(ind), len(points[1]))
        return value


# ---------------------------------------------------------------- variation with a retry cap

CAP_ATTEMPTS = 50       # GRAPE retries crossover/mutation until depth <= max; multiclass trees made that endless
CAP_HITS = {"crossover": 0, "mutation": 0}          # give-ups in the current generation (reset by evolve)
_MAPPED = ("genome", "phenotype", "nodes", "depth", "used_codons", "invalid", "n_wraps", "structure")


def _saved(ind) -> dict:
    """What GRAPE's reMap overwrites, copied; restored on a give-up (fitness is never touched)."""
    return {a: copy.deepcopy(getattr(ind, a)) for a in _MAPPED}


def _restored(ind, saved: dict):
    for a, v in saved.items():
        setattr(ind, a, v)
    return ind


def crossover_capped(parent0, parent1, bnf_grammar, max_depth, codon_consumption,
                     genome_representation="list", max_genome_length=None):
    """grape.crossover_onepoint with the same draws per attempt, at most CAP_ATTEMPTS attempts; then
    both parents unchanged (genome, phenotype, fitness). GRAPE's reMap writes into the individual it is
    given, so a give-up restores the parents' saved state."""
    if genome_representation != "list":
        raise ValueError("Only 'list' representation is implemented")
    c0 = len(parent0.genome) if parent0.invalid else min(len(parent0.genome), parent0.used_codons)
    c1 = len(parent1.genome) if parent1.invalid else min(len(parent1.genome), parent1.used_codons)
    g0, g1 = parent0.genome.copy(), parent1.genome.copy()
    s0, s1 = _saved(parent0), _saved(parent1)
    for _ in range(CAP_ATTEMPTS):
        point0, point1 = random.randint(1, c0), random.randint(1, c1)
        new0 = grape.reMap(parent0, g0[0:point0] + g1[point1:], bnf_grammar, max_depth, codon_consumption)
        new1 = grape.reMap(parent1, g1[0:point1] + g0[point0:], bnf_grammar, max_depth, codon_consumption)
        if not (new0.depth > max_depth or new1.depth > max_depth):
            if max_genome_length:
                new0.invalid = new0.invalid or len(new0.genome) > max_genome_length
                new1.invalid = new1.invalid or len(new1.genome) > max_genome_length
            del new0.fitness.values, new1.fitness.values
            return new0, new1
    CAP_HITS["crossover"] += 1
    return _restored(parent0, s0), _restored(parent1, s1)


def mutation_capped(ind, mut_probability, codon_size, bnf_grammar, max_depth, codon_consumption,
                    max_genome_length=None):
    """grape.mutation_int_flip_per_codon with the same draws per attempt (mutations accumulate over
    attempts, as in GRAPE), at most CAP_ATTEMPTS attempts; then the individual unchanged."""
    n = len(ind.genome) if ind.invalid else min(len(ind.genome), ind.used_codons)
    saved = _saved(ind)
    genome = copy.deepcopy(ind.genome)
    mutated = False
    for _ in range(CAP_ATTEMPTS):
        for i in range(n):
            if random.random() < mut_probability:
                genome[i] = random.randint(0, codon_size)
                mutated = True
        new = grape.reMap(ind, genome, bnf_grammar, max_depth, codon_consumption)
        if not new.depth > max_depth:
            if max_genome_length and len(new.genome) > max_genome_length:
                new.invalid = True
            if mutated:
                del new.fitness.values
            return new,
    CAP_HITS["mutation"] += 1
    return _restored(ind, saved),


# ---------------------------------------------------------------- the loop (ported)

def _var_and(population, toolbox, p):
    """GRAPE algorithms.varAnd: one-point crossover on pairs, then per-codon mutation."""
    offspring = [toolbox.clone(ind) for ind in population]
    for i in range(1, len(offspring), 2):
        if random.random() < p["p_crossover"]:
            offspring[i - 1], offspring[i] = toolbox.mate(
                offspring[i - 1], offspring[i], toolbox.grammar, p["max_tree_depth"],
                p["codon_consumption"], p["genome_representation"], p["max_genome_length"])
    for i in range(len(offspring)):
        offspring[i], = toolbox.mutate(offspring[i], p["p_mutation"], p["codon_size"], toolbox.grammar,
                                       p["max_tree_depth"], p["codon_consumption"], p["max_genome_length"])
    return offspring


def _row(gen, population, valid, invalid, hof, fitness_test, seconds) -> dict:
    fits = np.array([ind.fitness.values[0] for ind in population], dtype=float)
    best = hof.items[0] if len(hof) else None
    return {"gen": gen, "invalid": invalid,
            "min": float(np.nanmin(fits)), "avg": float(np.nanmean(fits)),
            "std": float(np.nanstd(fits)), "max": float(np.nanmax(fits)),
            "fitness_test": None if math.isnan(fitness_test) else float(fitness_test),   # None at gen 0
            "best_nodes": best.nodes if best else 0, "best_depth": best.depth if best else 0,
            "avg_nodes": float(np.mean([ind.nodes for ind in valid])) if valid else 0.0,
            "generation_time": seconds,
            "best_phenotype": text_of(best) if best else None}     # moved out of the rows by finish_run


def individual_rows(gen: int, population: list, outcome: dict) -> list[dict]:
    """The individuals file: one row per individual and generation. `outcome` maps
    id(individual) -> hit / miss / invalid (FEC) or "" (baseline) for those evaluated this generation;
    the others were kept (elites)."""
    rows = []
    for i, ind in enumerate(population):
        is_ge = hasattr(ind, "genome")
        fit = ind.fitness.values[0] if ind.fitness.valid else float("nan")
        rows.append({"gen": gen, "index": i, "phenotype": "" if is_ge and ind.invalid else text_of(ind),
                     "fitness": None if math.isnan(fit) else fit,
                     "nodes": ind.nodes if is_ge else len(ind), "depth": ind.depth if is_ge else ind.height,
                     "fec": outcome.get(id(ind), "kept")})
    return rows


def evolve(grammar, points_train, points_test, p: dict, seed: int, evaluate=fitness, individuals: list | None = None,
           capped: bool = True):
    """Seed, create the population, run the generations. Returns (rows, hall of fame, n_evals).
    `individuals`: a list to fill with individual_rows (the optional individuals file)."""
    toolbox = base.Toolbox()
    toolbox.grammar = grammar
    toolbox.register("select", tools.selTournament, tournsize=p["tournsize"])
    # GRAPE's operators with a retry cap; capped=False: GRAPE's own (no cap), as in runs made before 015
    toolbox.register("mate", crossover_capped if capped else grape.crossover_onepoint)
    toolbox.register("mutate", mutation_capped if capped else grape.mutation_int_flip_per_codon)
    n_evals = 0

    def evaluate_train(ind):
        nonlocal n_evals
        n_evals += 1
        return evaluate(ind, points_train)

    random.seed(seed)
    np.random.seed(seed)
    population = grape.sensible_initialisation(
        creator.Individual, pop_size=p["population"], bnf_grammar=grammar, min_init_depth=p["min_init_depth"],
        max_init_depth=p["max_init_depth"], codon_size=p["codon_size"],
        codon_consumption=p["codon_consumption"], genome_representation=p["genome_representation"])
    hof = tools.HallOfFame(p["halloffame_size"])
    rows, seen = [], set()                              # seen: phenotypes so far (novelty)
    for gen in range(p["generations"] + 1):
        start = time.time()
        CAP_HITS.update(crossover=0, mutation=0)
        if gen == 0:
            todo = [ind for ind in population if not ind.fitness.valid]
        else:
            pool = valid if valid else population
            offspring = _var_and(toolbox.select(pool, len(population) - p["elite_size"]), toolbox, p)
            todo = [ind for ind in offspring if not ind.fitness.valid]
        outcome = {}
        for ind in todo:
            ind.fitness.values = evaluate_train(ind)
            outcome[id(ind)] = getattr(evaluate, "last_outcome", "")
        if gen > 0:
            population[:] = offspring
            for i in range(min(p["elite_size"], len(hof.items))):
                population.append(hof.items[i])
        valid0 = [ind for ind in population if not ind.invalid]
        valid = [ind for ind in valid0 if not math.isnan(ind.fitness.values[0])]
        hof.update(valid)
        fitness_test = fitness(hof.items[0], points_test)[0] if gen > 0 and len(hof) else float("nan")
        row = _row(gen, population, valid, len(population) - len(valid0), hof, fitness_test, time.time() - start)
        row.update(cap_hits_crossover=CAP_HITS["crossover"], cap_hits_mutation=CAP_HITS["mutation"])
        row.update(measurement.generation_extras(population, valid, hof.items[0] if len(hof) else None,
                                                 seen, "GE"))
        if hasattr(evaluate, "generation_stats"):      # FEC counters for this generation
            row.update(evaluate.generation_stats())
        rows.append(row)
        if individuals is not None:
            individuals += individual_rows(gen, population, outcome)
    return rows, hof, n_evals


# ---------------------------------------------------------------- a run

def split(X, y, seed: int, test_size: float, zscore: bool, feature_info: list[dict]):
    """the reference implementation's split, then option A z-score; returns points as (x: features × rows, y)."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
    if zscore:
        params = preprocessing.fit_zscore(X_tr, feature_info)
        X_tr, X_te = preprocessing.apply_zscore(X_tr, params), preprocessing.apply_zscore(X_te, params)
    return [X_tr.T, y_tr], [X_te.T, y_te]


PHASE_SETUP = "phase_timing"        # phase-timed runs live under their own setup, never over a stored run


def twin_record(algorithm: str, name: str, seed: int, overrides: dict, fec: dict) -> dict | None:
    """The stored run a phase-timed run must reproduce; None when there is none (the twin check then fails
    the run with 'twin missing')."""
    path = datasets.ROOT / "results" / algorithm.lower() / setup_id(overrides) / name / \
        f"fec_{fec.get('sampler', 'farthest_point')}_frac{float(fec['fraction']):g}" / f"seed{seed}.json"
    return json.loads(path.read_text()) if path.is_file() else None


def phase_setup(overrides: dict | None) -> str:
    """Setup of a phase-timed run: 'phase_timing', or 'phase_timing_<hash>' with parameter overrides; its stored
    twin is the run with the same overrides under setup_id(overrides)."""
    return PHASE_SETUP if not overrides else f"{PHASE_SETUP}_{setup_id(overrides)}"


def start_run(name: str, seed: int, p: dict, overrides: dict, fec: dict | None, compile_fn,
              phase_timing: bool = False) -> dict:
    """The steps every engine shares: checks, split + z-score (option A), FEC sampling and evaluator.
    FEC run k uses the same seed as baseline run k (paired, option 1)."""
    import fec as fec_stage
    record = datasets.dataset_record(name)
    if not 2 <= record["n_classes"] <= grammar_stage.MAX_CLASSES:
        raise DatasetError(name, f"{record['n_classes']} classes; runs cover 2..{grammar_stage.MAX_CLASSES}")
    X, y, info = datasets.load_table(name)
    gpath = grammar_stage.grammar_dir() / f"{name}.bnf"
    if not gpath.is_file():
        raise DatasetError(name, f"no grammar; run `python grammar.py build {name}`")
    train, test = split(X, y, seed, p["test_size"], True, info)
    if phase_timing and (not fec or fec.get("diagnostic")):
        raise DatasetError(name, "phase timing is for FEC timing runs only (--fraction, no --diagnostic)")
    from measurement import PhaseClock
    clock = PhaseClock() if phase_timing else None
    run = {"record": record, "gpath": gpath, "train": train, "test": test, "start": time.perf_counter(),
           "evaluate": BaselineEvaluator(compile_fn), "fec_fields": {}, "mode": "baseline", "phases": clock}
    if fec:
        n = fec_stage.sample_size(float(fec["fraction"]), len(train[1]))
        sampler = fec.get("sampler", "farthest_point")
        cache = fec_stage.load_fps_cache(name, seed, record["checksum"], len(train[1]), setup_id(overrides)) \
            if fec.get("fps_cache") and sampler == "farthest_point" else None
        rows_picked, sampling_sec = fec_stage.sample_rows(sampler, train[0], train[1], n, seed, cache)
        run["evaluate"] = fec_stage.FECEvaluator(train[0][:, rows_picked], bool(fec.get("diagnostic")), compile_fn,
                                                 phases=clock)
        run["mode"] = "fec_diagnostic" if fec.get("diagnostic") else "fec"
        run["fec_fields"] = {"sampler": sampler, "fraction": float(fec["fraction"]), "sample_size": n,
                             "sampling_time_sec": sampling_sec, "fps_from_cache": cache is not None,
                             "sample_rows": [int(i) for i in rows_picked]}    # positions in this run's training split
    return run


def finish_run(algorithm: str, name: str, seed: int, p: dict, overrides: dict, run: dict, rows: list,
               best, n_evals: int, compile_fn) -> dict:
    """The run record every engine writes (, measurements of 007)."""
    if run["mode"] != "baseline":
        run["fec_fields"].update(run["evaluate"].summary())
    best_per_gen = [g.pop("best_phenotype", None) for g in rows]   # own list: not a generations.csv column
    clock = run.get("phases")
    out = {"algorithm": algorithm, "dataset": name, "seed": seed, "mode": run["mode"],
           "setup": phase_setup(overrides) if clock else setup_id(overrides),
            "overrides": overrides, "params": p, "fec": run["fec_fields"],
            "table_checksum": run["record"]["checksum"], "corpus_version": datasets.corpus_version(),
            "grammar_checksum": datasets.sha256_file(run["gpath"]), "code_commit": datasets.git_commit(),
            "n_train": len(run["train"][1]), "n_test": len(run["test"][1]), "generations": rows,
            "best_phenotype": text_of(best), "best_train_error": float(best.fitness.values[0]),
            "best_test_error": error_of(compile_fn(best), run["test"]), "n_train_evals": n_evals,
            "best_phenotype_per_generation": best_per_gen,       # hall-of-fame best after each generation (0..G)
            "measures": run["evaluate"].meter.as_dict(),
            "generation_time_sec": float(sum(r["generation_time"] for r in rows)),
            "wall_time_sec": time.perf_counter() - run["start"]}
    if clock:
        out["phases"] = clock.as_dict(out["measures"]["eval_time_sec"])
    return out


def run(name: str, seed: int, params: dict | None = None, fec: dict | None = None,
        save_individuals: bool = False, phase_timing: bool = False) -> dict:
    """One GE run: baseline, or FEC with fec={"fraction", "sampler", "diagnostic", "fps_cache"}.
    save_individuals: keep every individual of every generation (save_result writes them beside the run)."""
    overrides = dict(params or {})
    p = {**PARAMS, **overrides}
    if p["min_init_depth"] is None:
        p["min_init_depth"] = recorded_depth(name)
    r = start_run(name, seed, p, overrides, fec, compile_ge, phase_timing)
    capped = True
    if phase_timing:                    # reproduce the stored twin, made with or without the retry cap
        twin = twin_record("GE", name, seed, overrides, fec)
        capped = twin is None or "cap_attempts" in twin
    individuals = [] if save_individuals else None
    rows, hof, n_evals = evolve(grape.Grammar(str(r["gpath"])), r["train"], r["test"], p, seed, r["evaluate"], individuals,
                                capped)
    result = finish_run("GE", name, seed, p, overrides, r, rows, hof.items[0], n_evals, compile_ge)
    result["cap_attempts"] = CAP_ATTEMPTS if capped else None
    if individuals is not None:
        result["individuals"] = individuals
    return result


INDIVIDUAL_COLUMNS = ("gen", "index", "phenotype", "fitness", "nodes", "depth", "fec")


def save_result(result: dict) -> Path:
    tag = "baseline" if result["mode"] == "baseline" else \
        f"{result['mode']}_{result['fec']['sampler']}_frac{result['fec']['fraction']:g}"
    path = datasets.ROOT / "results" / result["algorithm"].lower() / result["setup"] / result["dataset"] / tag / \
        f"seed{result['seed']}.json"
    if result["setup"].startswith(PHASE_SETUP) and path.exists():
        raise DatasetError(result["dataset"], f"{path} exists; phase-timing runs are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    individuals = result.pop("individuals", None)
    if individuals is not None:                         # optional: beside the run, never in the JSON
        ipath = path.with_name(f"seed{result['seed']}.individuals.csv.gz")
        pd.DataFrame(individuals, columns=INDIVIDUAL_COLUMNS).to_csv(ipath, index=False)
        result["individuals_file"] = ipath.name
    path.write_text(json.dumps(result, indent=1) + "\n")
    return path


# ---------------------------------------------------------------- minimum initial depth

def depth_path() -> Path:
    return grammar_stage.grammar_dir() / "init_depth.csv"


def find_min_init_depth(grammar_file: Path) -> int:
    """Smallest candidate depth giving no invalid individual at population 1000 (seeded)."""
    g = grape.Grammar(str(grammar_file))
    for d in DEPTH_CANDIDATES:
        random.seed(DEPTH_CHECK["seed"])
        np.random.seed(DEPTH_CHECK["seed"])
        try:
            pop = grape.sensible_initialisation(creator.Individual, DEPTH_CHECK["population"], g, d,
                                                max(DEPTH_CHECK["max_init_depth"], d), 255, "lazy", "list")
        except Exception:
            continue
        if not any(ind.invalid for ind in pop):
            return d
    raise DatasetError(grammar_file.stem, f"no depth in {DEPTH_CANDIDATES} gives 0 invalid individuals")


def record_depth(name: str) -> int:
    gpath = grammar_stage.grammar_dir() / f"{name}.bnf"
    d = find_min_init_depth(gpath)
    datasets.upsert_csv_row(depth_path(), DEPTH_COLUMNS, {
        "name": name, "grammar_checksum": datasets.sha256_file(gpath), "min_init_depth": d,
        "code_commit": datasets.git_commit()})
    return d


def recorded_depth(name: str) -> int:
    df = pd.read_csv(depth_path(), dtype=str) if depth_path().is_file() else pd.DataFrame(columns=DEPTH_COLUMNS)
    hit = df[df["name"] == name]
    gpath = grammar_stage.grammar_dir() / f"{name}.bnf"
    if hit.empty or hit.iloc[0]["grammar_checksum"] != datasets.sha256_file(gpath):
        raise DatasetError(name, f"no minimum depth for the current grammar; run `python ge.py depth {name}`")
    return int(hit.iloc[0]["min_init_depth"])


# ---------------------------------------------------------------- feature-type fairness

FAIRNESS = {"populations": 30, "population": 1000, "max_init_depth": 7}
FAIRNESS_COLUMNS = ("name", "grammar_checksum", "n_numeric", "n_boolean", "ratio_mean", "ratio_ci_low",
                    "ratio_ci_high", "p_numeric_mean", "p_boolean_mean", "code_commit")


def fairness(name: str) -> dict:
    """Feature-type fairness of a grammar: in each of 30 seeded initial
    populations, P(feature appears in an individual) per feature; ratio = mean over boolean features /
    mean over numeric features; reported as the mean ratio with a 95% CI over the populations."""
    import re
    gpath = grammar_stage.grammar_dir() / f"{name}.bnf"
    text = gpath.read_text()
    def rule_features(rule: str) -> list[int]:          # the rule and its sub-rules (<rule_0>, ...; grammar.py)
        return [int(i) for line in text.splitlines() if line.startswith(rule)
                for i in re.findall(r"x\[(\d+)\]", line)]
    num, boo = rule_features("<num_feature"), rule_features("<bool_feature")
    g, depth = grape.Grammar(str(gpath)), recorded_depth(name)
    ratios, pn, pb = [], [], []
    for k in range(FAIRNESS["populations"]):
        random.seed(k)
        np.random.seed(k)
        pop = grape.sensible_initialisation(creator.Individual, FAIRNESS["population"], g, depth,
                                            max(FAIRNESS["max_init_depth"], depth), 255, "lazy", "list")
        used = [set(map(int, re.findall(r"x\[(\d+)\]", ind.phenotype))) for ind in pop if not ind.invalid]
        p = {i: sum(i in u for u in used) / len(used) for i in num + boo}
        if num:
            pn.append(np.mean([p[i] for i in num]))
        if boo:
            pb.append(np.mean([p[i] for i in boo]))
        if num and boo:
            ratios.append(pb[-1] / pn[-1] if pn[-1] else float("inf"))
    row = {"name": name, "grammar_checksum": datasets.sha256_file(gpath), "n_numeric": len(num), "n_boolean": len(boo),
           "p_numeric_mean": float(np.mean(pn)) if pn else None, "p_boolean_mean": float(np.mean(pb)) if pb else None,
           "code_commit": datasets.git_commit()}
    if ratios:
        m, half = float(np.mean(ratios)), 1.96 * float(np.std(ratios, ddof=1)) / np.sqrt(len(ratios))
        row.update(ratio_mean=m, ratio_ci_low=m - half, ratio_ci_high=m + half)
    datasets.upsert_csv_row(grammar_stage.grammar_dir() / "fairness.csv", FAIRNESS_COLUMNS, row)
    return row


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("depth")
    d.add_argument("names", nargs="*")
    d.add_argument("--all", action="store_true")
    fa = sub.add_parser("fairness")
    fa.add_argument("names", nargs="*")
    fa.add_argument("--all", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                   help="override a GE parameter (repeatable), e.g. population=200")
    r.add_argument("--fraction", type=float, help="FEC sample fraction (omit for baseline)")
    r.add_argument("--diagnostic", action="store_true", help="FEC diagnostic mode: count fake hits")
    r.add_argument("--sampler", default="farthest_point", help="farthest_point | random | stratified")
    r.add_argument("--fps-cache", action="store_true", help="use results/fps_cache picks (python fec.py prepare)")
    r.add_argument("--save-individuals", action="store_true",
                   help="also write every individual of every generation (large; small datasets)")
    r.add_argument("--phase-timing", action="store_true",
                   help="time every phase of each evaluation (FEC timing runs only; setup phase_timing)")
    args = p.parse_args(argv)
    try:
        if args.command == "depth":
            for name in (list(datasets.load_recipes()) if args.all else args.names):
                print(f"{name:<24} min_init_depth {record_depth(name)}")
            return 0
        if args.command == "fairness":
            for name in (list(datasets.load_recipes()) if args.all else args.names):
                r = fairness(name)
                shown = f"{r['ratio_mean']:.2f} [{r['ratio_ci_low']:.2f}, {r['ratio_ci_high']:.2f}]" if "ratio_mean" in r else "single type"
                print(f"{name:<24} boolean/numeric appearance ratio {shown}")
            return 0
        overrides = parse_overrides(args.param)
        fec_opts = {"fraction": args.fraction, "diagnostic": args.diagnostic, "sampler": args.sampler,
                    "fps_cache": args.fps_cache} if args.fraction else None
        result = run(args.name, args.seed, overrides, fec_opts, args.save_individuals, args.phase_timing)
        path = save_result(result)
        print(f"{args.name} seed {args.seed}: train error {result['best_train_error']:.4f}, "
              f"test error {result['best_test_error']:.4f}, {result['wall_time_sec']:.1f} s -> {path}")
        print(f"best: {result['best_phenotype']}")
        if result["fec"]:
            print("fec: " + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in result["fec"].items()))
        if args.phase_timing:                           # must reproduce the stored twin
            import phase_timing
            t = phase_timing.check_twin(path)
            print(f"twin check: {'ok' if t['ok'] else 'MISMATCH ' + ', '.join(t['mismatches'])} ({t['twin']})")
            if not t["ok"]:
                print(f"FAIL {args.name} seed {args.seed}: phase-timed run differs from its stored twin")
                return 1
        return 0
    except DatasetError as exc:
        print(f"FAIL {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
