"""Typed GP stage: strongly typed tree GP whose search space is exactly the dataset's GE grammar.

Types number and boolean (gp.PrimitiveSetTyped), root boolean, and the grammar's pieces:
  less_than_or_equal, greater_than_or_equal   number, number -> boolean     (<bool> comparisons)
  and_, or_ / not_                            boolean(s)     -> boolean
  add, sub, mul, pdiv                         number, number -> number      (<num>)
  inputs: the grammar's numeric features (number) and boolean features (boolean)
  constant: the grammar's draw - sign, integer part 0..K-1, two digits (003's range ±K)
The functions are operators.py's, the same objects GE evaluates. the reference implementation's GP was untyped (any
function on any value, constants 0.00-99.99), so it searched a space the grammar can't express.

Tree generation is DEAP's half-and-half, except that where a boolean leaf is needed and the set
has no boolean terminal (numeric-only datasets), a comparison of two number terminals is placed:
one level deeper, as the grammar's <bool> must end in a comparison. The loop is the reference implementation's
gp_algorithms.gp_ea_simple_with_elitism (elites skip variation), with selection and the hall of
fame over individuals that have a fitness (not NaN), as in GE.

Commands: python typed_gp.py run <name> --seed S [--param K=V ...] [--fraction F --sampler S
          [--diagnostic] [--fps-cache]]                    writes results/gp/<setup>/<name>/...
"""
from __future__ import annotations

import argparse
import json
import math
import operator
import random
import sys
import time
from inspect import isclass

import numpy as np
from deap import algorithms, base, creator, gp, tools

import datasets
import ge
import grammar as grammar_stage
import measurement
import operators as ops
import preprocessing
from datasets import DatasetError

NUM, BOOL = float, bool


class Score:
    """Type tag (multiclass, 013): a class score; only `where` and `argmaxK` take it, never arithmetic."""


class Klass:
    """Type tag (multiclass): the predicted class; the root of a multiclass tree."""


TYPE_ORDER = (NUM, BOOL, Score, Klass)


def cx_one_point(ind1, ind2):
    """DEAP gp.cxOnePoint (typed), with the common types in a fixed order. DEAP draws from
    list(set(types)), whose order follows the types' memory addresses: it differs between machines
    (float, bool on one machine; bool, float on another) and processes (Score, Klass), so a seed did not fix the run.
    The same random draws as DEAP otherwise; TYPE_ORDER is the laptop's order of float, bool."""
    if len(ind1) < 2 or len(ind2) < 2:
        return ind1, ind2
    types1, types2 = {}, {}
    for idx, node in enumerate(ind1[1:], 1):
        types1.setdefault(node.ret, []).append(idx)
    for idx, node in enumerate(ind2[1:], 1):
        types2.setdefault(node.ret, []).append(idx)
    common = [t for t in TYPE_ORDER if t in types1 and t in types2]
    if common:
        type_ = random.choice(common)
        slice1 = ind1.searchSubtree(random.choice(types1[type_]))
        slice2 = ind2.searchSubtree(random.choice(types2[type_]))
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]
    return ind1, ind2
PARAMS = {                                  # the reference implementation config.json GP_SPECIFIC_CONFIG + COMMON_CONFIG
    "population": 1000, "generations": 30, "test_size": 0.2,
    "p_crossover": 0.8, "p_mutation": 0.05,  # mutation: per individual (DEAP varAnd), not per codon
    "tournsize": 4, "elite_size": 1, "halloffame_size": 1,
    "min_init_depth": 3, "max_init_depth": 7, "max_height": 35,
}

if not hasattr(creator, "FitnessMinGP"):
    creator.create("FitnessMinGP", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "IndividualGP"):
    creator.create("IndividualGP", gp.PrimitiveTree, fitness=creator.FitnessMinGP)


# ---------------------------------------------------------------- the primitive set

def _const_pm1():
    return _draw(1)


def _const_pm2():
    return _draw(2)


def _const_pm3():
    return _draw(3)


def _draw(k: int) -> float:
    """The grammar's <constant>: -<h>.<c><c> | +<h>.<c><c>, <h> in 0..k-1, two digits."""
    sign = random.choice((-1, 1))
    h = random.randrange(k)
    d = random.randrange(100)
    return round(sign * (h + d / 100), 2) + 0.0        # + 0.0 turns -0.0 into 0.0


CONSTANTS = {1: _const_pm1, 2: _const_pm2, 3: _const_pm3}


def _score_const() -> float:
    """The multiclass grammar's <score_constant> ::= <c>.<c>: two digits, 0.0 .. 9.9."""
    return round(random.randrange(10) + random.randrange(10) / 10, 1)


def primitive_set(name: str) -> tuple[gp.PrimitiveSetTyped, list[int]]:
    """The dataset's typed set and the column positions of its arguments (grammar order)."""
    record = datasets.dataset_record(name)
    numeric, boolean = grammar_stage.grammar_features(record)
    args = sorted(numeric + boolean)
    k_classes = record["n_classes"]
    if k_classes > grammar_stage.MAX_CLASSES:
        raise DatasetError(name, f"{k_classes} classes; at most {grammar_stage.MAX_CLASSES}")
    multiclass = k_classes > 2
    pset = gp.PrimitiveSetTyped("MAIN", [NUM if i in numeric else BOOL for i in args], Klass if multiclass else BOOL)
    pset.renameArguments(**{f"ARG{j}": f"x{i}" for j, i in enumerate(args)})
    if numeric:
        for f in (ops.add, ops.sub, ops.mul, ops.pdiv):
            pset.addPrimitive(f, [NUM, NUM], NUM)
        for f in (ops.less_than_or_equal, ops.greater_than_or_equal):
            pset.addPrimitive(f, [NUM, NUM], BOOL)
        rng = preprocessing.constant_range(name)
        k = round(rng[1] + 0.01)
        pset.addEphemeralConstant(f"const_pm{k}", CONSTANTS[k], NUM)
    pset.addPrimitive(ops.and_, [BOOL, BOOL], BOOL)
    pset.addPrimitive(ops.or_, [BOOL, BOOL], BOOL)
    pset.addPrimitive(ops.not_, [BOOL], BOOL)
    if multiclass:                               # the grammar's top: argmaxK(<score> x K), where, <c>.<c>
        pset.addPrimitive(getattr(ops, f"argmax{k_classes}"), [Score] * k_classes, Klass)
        pset.addPrimitive(ops.where, [BOOL, Score, Score], Score)
        pset.addEphemeralConstant("score_const", _score_const, Score)
    return pset, args


# ---------------------------------------------------------------- typed generation (DEAP + leaf fallback)

def generate(pset, min_, max_, condition, type_=None):
    """DEAP gp.generate, except that a leaf of a type without terminals becomes a primitive whose
    arguments all have terminals (a comparison of two number terminals)."""
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_, False)]                 # (depth, type, must be a leaf)
    while stack:
        depth, type_, forced = stack.pop()
        leaf = forced or condition(height, depth)
        if forced and not pset.terminals[type_]:
            raise DatasetError("gp", f"no {type_.__name__} terminal for a forced leaf")
        if leaf and pset.terminals[type_]:
            term = random.choice(pset.terminals[type_])
            if isclass(term):
                term = term()
            expr.append(term)
            continue
        prims = pset.primitives[type_]
        if leaf:                                   # no terminal of this type: shallowest way to end
            prims = [p for p in prims if all(pset.terminals[a] for a in p.args)]
        if not prims:
            raise DatasetError("gp", f"no way to build a {type_.__name__} node")
        prim = random.choice(prims)
        expr.append(prim)
        for arg in reversed(prim.args):         # a fallback's arguments must end the branch
            stack.append((depth + 1, arg, leaf))
    return expr


def gen_full(pset, min_, max_, type_=None):
    return generate(pset, min_, max_, lambda height, depth: depth == height, type_)


def gen_grow(pset, min_, max_, type_=None):
    def condition(height, depth):
        return depth == height or (depth >= min_ and random.random() < pset.terminalRatio)
    return generate(pset, min_, max_, condition, type_)


def gen_half_and_half(pset, min_, max_, type_=None):
    return random.choice((gen_grow, gen_full))(pset, min_, max_, type_)


# ---------------------------------------------------------------- evaluation and the loop

def make_compile(pset, args: list[int]):
    """compile_fn for the shared evaluators: a tree compiled once -> x -> outputs or None
    (ge.checked's validity rule)."""
    def compile_tree(ind):
        func = gp.compile(ind, pset)
        return lambda x: ge.checked(lambda: func(*[x[i] for i in args]), x)
    return compile_tree


def make_predict(pset, args: list[int]):
    """predict(tree, x) = compile, then run (tests and one-off use)."""
    comp = make_compile(pset, args)
    return lambda ind, x: comp(ind)(x)


def evolve(pset, compile_fn, points_train, points_test, p: dict, seed: int, evaluate, individuals: list | None = None):
    """the reference implementation gp_ea_simple_with_elitism (selection and hall of fame over valid individuals)."""
    toolbox = base.Toolbox()
    toolbox.register("expr", gen_half_and_half, pset=pset, min_=p["min_init_depth"], max_=p["max_init_depth"])
    toolbox.register("individual", tools.initIterate, creator.IndividualGP, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=p["tournsize"])
    toolbox.register("mate", cx_one_point)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    limit = gp.staticLimit(key=operator.attrgetter("height"), max_value=p["max_height"])
    toolbox.decorate("mate", limit)
    toolbox.decorate("mutate", limit)
    test_fitness = ge.fitness_compiled(compile_fn)
    n_evals = 0

    random.seed(seed)
    np.random.seed(seed)
    population = toolbox.population(n=p["population"])
    hof = tools.HallOfFame(p["halloffame_size"])
    rows, seen = [], set()                              # seen: phenotypes so far (novelty)
    for gen in range(p["generations"] + 1):
        start = time.time()
        if gen > 0:
            pool = valid or population
            offspring = algorithms.varAnd(toolbox.select(pool, len(population) - len(hof.items)), toolbox,
                                          p["p_crossover"], p["p_mutation"])
        else:
            offspring = population
        outcome = {}
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = evaluate(ind, points_train)
                outcome[id(ind)] = getattr(evaluate, "last_outcome", "")
                n_evals += 1
        if gen > 0:
            offspring.extend(hof.items)                       # elites skip variation
            population[:] = offspring
        valid = [ind for ind in population if not math.isnan(ind.fitness.values[0])]
        hof.update(valid)
        fits = np.array([ind.fitness.values[0] for ind in population])
        best = hof.items[0] if len(hof) else None
        row = {"gen": gen, "invalid": int(np.isnan(fits).sum()), "min": float(np.nanmin(fits)),
               "avg": float(np.nanmean(fits)), "std": float(np.nanstd(fits)), "max": float(np.nanmax(fits)),
               "fitness_test": float(test_fitness(best, points_test)[0]) if gen > 0 and best else None,
               "best_nodes": len(best) if best else 0, "best_depth": best.height if best else 0,
               "avg_nodes": float(np.mean([len(i) for i in valid])) if valid else 0.0,
               "generation_time": time.time() - start,
               "best_phenotype": ge.text_of(best) if best else None}   # moved out of the rows by ge.finish_run
        row.update(measurement.generation_extras(population, valid, hof.items[0] if len(hof) else None,
                                                 seen, "GP"))
        if hasattr(evaluate, "generation_stats"):
            row.update(evaluate.generation_stats())
        rows.append(row)
        if individuals is not None:
            individuals += ge.individual_rows(gen, population, outcome)
    return rows, hof, n_evals


def run(name: str, seed: int, params: dict | None = None, fec: dict | None = None,
        save_individuals: bool = False, phase_timing: bool = False) -> dict:
    """One typed-GP run: baseline, or FEC (same options as ge.run)."""
    overrides = dict(params or {})
    bad = [k for k in overrides if k not in PARAMS]
    if bad:
        raise DatasetError(name, f"unknown GP parameter(s) {bad}; choose from {sorted(PARAMS)}")
    p = {**PARAMS, **overrides}
    pset, args = primitive_set(name)
    compile_fn = make_compile(pset, args)
    r = ge.start_run(name, seed, p, overrides, fec, compile_fn, phase_timing)
    individuals = [] if save_individuals else None
    rows, hof, n_evals = evolve(pset, compile_fn, r["train"], r["test"], p, seed, r["evaluate"], individuals)
    result = ge.finish_run("GP", name, seed, p, overrides, r, rows, hof.items[0], n_evals, compile_fn)
    if individuals is not None:
        result["individuals"] = individuals
    return result


def parse_overrides(items: list[str]) -> dict:
    out = {}
    for item in items or []:
        key, _, value = item.partition("=")
        if key not in PARAMS:
            raise DatasetError("params", f"unknown GP parameter {key!r}; choose from {sorted(PARAMS)}")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    r.add_argument("--fraction", type=float)
    r.add_argument("--diagnostic", action="store_true")
    r.add_argument("--sampler", default="farthest_point")
    r.add_argument("--fps-cache", action="store_true")
    r.add_argument("--save-individuals", action="store_true")
    r.add_argument("--phase-timing", action="store_true")
    args = p.parse_args(argv)
    try:
        fec_opts = {"fraction": args.fraction, "diagnostic": args.diagnostic, "sampler": args.sampler,
                    "fps_cache": args.fps_cache} if args.fraction else None
        result = run(args.name, args.seed, parse_overrides(args.param), fec_opts, args.save_individuals,
                     args.phase_timing)
        path = ge.save_result(result)
        print(f"{args.name} GP seed {args.seed}: train error {result['best_train_error']:.4f}, "
              f"test error {result['best_test_error']:.4f}, {result['wall_time_sec']:.1f} s -> {path}")
        print(f"best: {result['best_phenotype']}")
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
