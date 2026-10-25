import json
import math
import subprocess

import numpy as np
import pytest

import datasets as ds
import ge
from grape import grape


class Ind:
    def __init__(self, phenotype, invalid=False):
        self.phenotype, self.invalid = phenotype, invalid


def test_fitness_rules():
    x = np.array([[1.0, -1.0, 2.0, -2.0]])
    y = np.array([1, 0, 1, 1])
    assert ge.fitness(Ind("greater_than_or_equal(x[0],+0.00)"), [x, y])[0] == 0.25
    assert math.isnan(ge.fitness(Ind("x[0]", invalid=True), [x, y])[0])
    assert math.isnan(ge.fitness(Ind("x[5]"), [x, y])[0])                 # IndexError -> NaN
    assert math.isnan(ge.fitness(Ind("+1.00"), [x, y])[0])                 # wrong length -> NaN
    assert ge.fitness(Ind("pdiv(x[0],sub(x[0],x[0]))"), [x, y])[0] == 0.25  # pdiv by 0 -> 1 -> all class 1


def small_run(name, seed, zscore):
    X, y, info = ds.load_table(name)
    train, test = ge.split(X, y, seed, 0.2, zscore, info)
    p = {**ge.PARAMS, "population": 100, "generations": 5, "min_init_depth": ge.recorded_depth(name)}
    rows, hof, _ = ge.evolve(grape.Grammar(str(ds.ROOT / "grammars" / f"{name}.bnf")), train, test, p, seed)
    for r in rows:
        r.pop("generation_time")                   # wall-clock, not part of the run's outcome
    return rows, hof.items[0].phenotype


@pytest.mark.parametrize("name", ["appendicitis", "prnn_crabs"])
def test_same_seed_same_run(name):
    assert small_run(name, 30, True) == small_run(name, 30, True)


def test_error_rate_binary_and_multiclass():
    y = np.array([0, 1, 2, 2])
    assert ge.error_rate(np.array([0.0, 1.0, 2.0, -1.0]), y) == 0.25          # tie (-1) counts wrong
    assert ge.error_rate(np.array([1.0, 0.0, 1.0, 1.0]), np.array([1, 0, 1, 0])) == 0.25   # binary unchanged


# ---------------------------------------------------------------- retry-capped variation

def _population(name="appendicitis", n=40, seed=3):
    import random
    from deap import creator
    g = grape.Grammar(str(ge.grammar_stage.grammar_dir() / f"{name}.bnf"))
    random.seed(seed)
    pop = grape.sensible_initialisation(creator.Individual, pop_size=n, bnf_grammar=g,
                                        min_init_depth=ge.recorded_depth(name), max_init_depth=7,
                                        codon_size=255, codon_consumption="lazy", genome_representation="list")
    for i, ind in enumerate(pop):
        ind.fitness.values = (i / n,)
    return g, pop


def test_capped_operators_equal_grape_below_the_cap():
    """Same random state in, same offspring out (genome, phenotype, depth) as GRAPE's own operators."""
    import random
    from deap import base
    g, pop = _population()
    clone = base.Toolbox().clone if hasattr(base.Toolbox(), "clone") else __import__("copy").deepcopy
    for k in range(0, len(pop) - 1, 2):
        for ours, theirs, args in [
                (ge.crossover_capped, grape.crossover_onepoint, (g, 35, "lazy", "list", None)),
                (lambda a, b, *r: ge.mutation_capped(a, 0.05, 255, *r) + (b,),
                 lambda a, b, *r: grape.mutation_int_flip_per_codon(a, 0.05, 255, *r) + (b,), (g, 35, "lazy", None))]:
            a, b = clone(pop[k]), clone(pop[k + 1])
            c, d = clone(pop[k]), clone(pop[k + 1])
            random.seed(k); out1 = ours(a, b, *args)
            random.seed(k); out2 = theirs(c, d, *args)
            assert [(i.genome, i.phenotype, i.depth, i.fitness.valid) for i in out1] == \
                   [(i.genome, i.phenotype, i.depth, i.fitness.valid) for i in out2]


def test_give_up_leaves_individuals_unchanged(monkeypatch):
    """A depth limit nothing can meet: after CAP_ATTEMPTS the parents come back as they were."""
    import copy
    g, pop = _population()
    a, b = copy.deepcopy(pop[0]), copy.deepcopy(pop[1])
    before = [(i.genome[:], i.phenotype, i.fitness.values) for i in (a, b)]
    ge.CAP_HITS.update(crossover=0, mutation=0)
    out = ge.crossover_capped(a, b, g, 1, "lazy")
    assert [(i.genome, i.phenotype, i.fitness.values) for i in out] == before and ge.CAP_HITS["crossover"] == 1
    m, = ge.mutation_capped(copy.deepcopy(pop[2]), 0.5, 255, g, 1, "lazy")
    assert (m.genome, m.phenotype, m.fitness.values) == (pop[2].genome, pop[2].phenotype, pop[2].fitness.values)
    assert ge.CAP_HITS["mutation"] == 1


def test_binary_run_records_zero_cap_hits():
    r = ge.run("appendicitis", 0, params={"population": 50, "generations": 3})
    assert r["cap_attempts"] == ge.CAP_ATTEMPTS == 50
    assert all(g["cap_hits_crossover"] == 0 and g["cap_hits_mutation"] == 0 for g in r["generations"])
