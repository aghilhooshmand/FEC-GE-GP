"""Tests for typed_gp.py: small datasets and tiny populations only (memory)."""
import re

import numpy as np
import pytest
from deap import gp

import datasets as ds
import fec
import ge
import grammar
import measurement
import typed_gp as tg

TINY = {"population": 20, "generations": 2}


def grammar_parts(name):
    text = (grammar.grammar_dir() / f"{name}.bnf").read_text()
    rule = lambda r: re.search(rf"^{r} ::= (.*)$", text, re.M)
    feats = lambda r: [int(i) for i in re.findall(r"x\[(\d+)\]", rule(r).group(1))] if rule(r) else []
    return text, feats("<num_feature>"), feats("<bool_feature>")


@pytest.mark.parametrize("name", list(ds.load_recipes()))
def test_primitive_set_equals_grammar(name):
    """SC-001, for every dataset: reads only the records, no table."""
    if not (grammar.grammar_dir() / f"{name}.bnf").is_file():
        pytest.skip(f"{name}: no grammar on this machine (large datasets are recreated on the compute server only)")
    text, numeric, boolean = grammar_parts(name)
    pset, args = tg.primitive_set(name)
    types = dict(zip(args, pset.ins))
    assert sorted(numeric + boolean) == args
    assert all(types[i] is tg.NUM for i in numeric) and all(types[i] is tg.BOOL for i in boolean)
    prims = {p.name: (tuple(a.__name__ for a in p.args), p.ret.__name__) for v in pset.primitives.values() for p in v}
    want = {"and_": (("bool", "bool"), "bool"), "or_": (("bool", "bool"), "bool"), "not_": (("bool",), "bool")}
    if numeric:
        want.update({f: (("float", "float"), "float") for f in ("add", "sub", "mul", "pdiv")})
        want.update({f: (("float", "float"), "bool") for f in ("less_than_or_equal", "greater_than_or_equal")})
    k = ds.dataset_record(name)["n_classes"]
    if k > 2:                                                   # multiclass top
        want[f"argmax{k}"] = (("Score",) * k, "Klass")
        want["where"] = (("bool", "Score", "Score"), "Score")
    assert prims == want and pset.ret is (tg.Klass if k > 2 else tg.BOOL)
    for f in want:                                              # every primitive is used by the grammar too
        assert f + "(" in text


def test_constants_are_the_grammars():
    for k in (1, 2, 3):
        allowed = set(ge.preprocessing.candidate_constants(k))
        draws = {tg.CONSTANTS[k]() for _ in range(5000)}
        assert draws <= allowed and len(draws) > 0.6 * len(allowed)


@pytest.mark.parametrize("name", ["appendicitis", "prnn_crabs"])
def test_generated_trees_are_typed_and_bounded(name):
    pset, args = tg.primitive_set(name)
    X, y, info = ds.load_table(name)
    predict = tg.make_predict(pset, args)
    x = X.T
    for _ in range(500):
        tree = gp.PrimitiveTree(tg.gen_half_and_half(pset, 3, 7))
        assert tree.height <= 8                                  # at most one level over max (the fallback)
        out = predict(tree, x)
        assert out is None or (out.shape == (x.shape[1],) and set(np.unique(out)) <= {0.0, 1.0})


@pytest.mark.parametrize("name", ["appendicitis", "prnn_crabs"])
def test_same_seed_same_run(name):
    a, b = tg.run(name, 30, TINY), tg.run(name, 30, TINY)
    strip = lambda r: [{k: v for k, v in g.items() if k != "generation_time"} for g in r["generations"]]
    assert strip(a) == strip(b) and a["best_phenotype"] == b["best_phenotype"]


def test_fec_timing_equals_diagnostic():
    t = tg.run("prnn_crabs", 30, TINY, {"fraction": 0.1, "sampler": "random"})
    d = tg.run("prnn_crabs", 30, TINY, {"fraction": 0.1, "sampler": "random", "diagnostic": True})
    measurement.same_evolution(t, d)
    assert t["algorithm"] == "GP" and t["fec"]["hits"] + t["fec"]["misses"] == t["n_train_evals"]


def test_constant_tree_is_invalid():
    pset, args = tg.primitive_set("prnn_crabs")
    predict = tg.make_predict(pset, args)
    lte = next(p for p in pset.primitives[tg.BOOL] if p.name == "less_than_or_equal")
    c = next(t for t in pset.terminals[tg.NUM] if isinstance(t, type))
    tree = gp.PrimitiveTree([lte, c(), c()])
    X, _, _ = ds.load_table("prnn_crabs")
    ev = fec.FECEvaluator(X.T[:, :1], compile_fn=tg.make_compile(pset, args))
    assert predict(tree, X.T) is None and ev.compile(tree)(X.T[:, :1]) is None       # invalid on 1 row too


def test_run_does_not_depend_on_the_process():
    """A seed fixes the run in any process (hash seed, type addresses): DEAP's crossover drew the type
    from an unordered set; typed_gp.cx_one_point uses a fixed order."""
    import os
    import subprocess
    import sys
    code = ("import typed_gp; r = typed_gp.run('iris', 30, {'population': 20, 'generations': 2}); "
            "print(r['best_phenotype'], r['n_train_evals'])")
    out = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ds.ROOT,
                          env={**os.environ, "PYTHONHASHSEED": h, "PYTHONPATH": str(ds.ROOT)}).stdout
           for h in ("1", "2", "3", "4")}
    assert len(out) == 1
