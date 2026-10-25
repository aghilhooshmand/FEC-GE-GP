"""Multiclass on iris only, tiny populations."""
import re

import pytest

import datasets as ds
import fec
import ge
import grammar
import measurement
import typed_gp as tg
from datasets import DatasetError
from grape import grape

TINY = {"population": 20, "generations": 2}


def test_iris_grammar_is_top_rules_plus_binary_lines():
    text = (grammar.grammar_dir() / "iris.bnf").read_text()
    lines = text.splitlines()
    assert lines[0] == "<start> ::= argmax3(<score>,<score>,<score>)"
    assert lines[1:4] == list(grammar.MULTICLASS_TOP)
    numeric, boolean = grammar.grammar_features(ds.dataset_record("iris"))
    assert "\n".join(lines[4:]) + "\n" == grammar.grammar_text(numeric, boolean, 2)    # binary lines unchanged
    assert grammar.DIGITS in lines                                                    # <c> shared with the constants
    assert grape.Grammar(str(grammar.grammar_dir() / "iris.bnf")).start_rule == "<start>"


def test_multiclass_without_numeric_features_still_has_digits():
    text = grammar.grammar_text([], [0, 1], None, n_classes=3)
    assert grammar.DIGITS in text.splitlines() and "<constant>" not in text
    import re
    used = set(re.findall(r"<\w+>", text)); defined = set(re.findall(r"^(<\w+>) ::=", text, re.M))
    assert used <= defined


def test_more_than_twenty_classes_refused():
    with pytest.raises(DatasetError, match="at most 20"):
        grammar.grammar_text([0, 1], [], 2, n_classes=21)


def test_argmax_up_to_twenty():
    from operators import OPERATORS
    import numpy as np
    scores = [np.zeros(2)] * 19 + [np.ones(2)]
    assert list(OPERATORS["argmax20"](*scores)) == [19, 19]
    assert OPERATORS["argmax13"].__name__ == "argmax13"


def test_ten_class_grammar_is_argmax10_plus_binary_lines():
    lines = grammar.grammar_text([0, 1, 2], [3], 2, n_classes=10).splitlines()
    assert lines[0] == "<start> ::= argmax10(" + ",".join(["<score>"] * 10) + ")"
    assert lines[1:4] == list(grammar.MULTICLASS_TOP)
    assert "\n".join(lines[4:]) + "\n" == grammar.grammar_text([0, 1, 2], [3], 2)


@pytest.mark.parametrize("engine", [ge, tg])
def test_tiny_runs_and_timing_equals_diagnostic(engine):
    base = engine.run("iris", 30, TINY)
    t = engine.run("iris", 30, TINY, {"fraction": 0.1, "sampler": "stratified"})
    d = engine.run("iris", 30, TINY, {"fraction": 0.1, "sampler": "stratified", "diagnostic": True})
    measurement.same_evolution(t, d)
    assert base["best_phenotype"].startswith("argmax3(") and 0 <= base["best_test_error"] <= 1
    assert t["fec"]["hits"] + t["fec"]["misses"] <= t["n_train_evals"]     # GE: unmapped individuals skip the cache


def test_gp_trees_return_classes():
    pset, args = tg.primitive_set("iris")
    X, _, _ = ds.load_table("iris")
    predict = tg.make_predict(pset, args)
    from deap import gp
    for _ in range(200):
        out = predict(gp.PrimitiveTree(tg.gen_half_and_half(pset, 3, 7)), X.T)
        assert out is None or set(out.tolist()) <= {-1.0, 0.0, 1.0, 2.0}


@pytest.mark.parametrize("engine", [ge, tg])
@pytest.mark.parametrize("name, k", [("dry_bean", 7)])
def test_tiny_runs_up_to_ten_classes(engine, name, k):
    """K = 6..10 run like any other dataset; timing and diagnostic evolve identically."""
    if not (ds.DATA_DIR / f"{name}.csv").is_file():
        pytest.skip(f"needs data/{name}.csv: python datasets.py recreate {name}")
    base = engine.run(name, 30, TINY)
    t = engine.run(name, 30, TINY, {"fraction": 0.05, "sampler": "farthest_point"})
    d = engine.run(name, 30, TINY, {"fraction": 0.05, "sampler": "farthest_point", "diagnostic": True})
    measurement.same_evolution(t, d)
    assert base["best_phenotype"].startswith(f"argmax{k}(") and 0 <= base["best_test_error"] <= 1
