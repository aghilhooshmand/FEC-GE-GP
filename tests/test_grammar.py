import importlib.util
import json
import re

import pytest

import datasets as ds
import grammar as gr
from datasets import DatasetError


def test_numeric_only():
    g = gr.grammar_text([0, 1], [], 3)
    assert "<bool_feature>" not in g and "<h> ::= 0|1|2\n" in g and g.endswith("<c> ::= 0|1|2|3|4|5|6|7|8|9\n")


def test_boolean_only():
    g = gr.grammar_text([], [0, 1], None)
    assert g == "<bool> ::= and_(<bool>,<bool>)|or_(<bool>,<bool>)|not_(<bool>)|<bool_feature>\n" \
                "<bool_feature> ::= x[0]|x[1]\n"


def test_constant_rules_and_no_comments():
    assert "<constant> ::= -0.<c><c>|+0.<c><c>\n<c>" in gr.grammar_text([0], [], 1)
    assert "<h> ::= 0|1\n" in gr.grammar_text([0], [], 2)
    for k in (1, 2, 3):
        assert not any(line.startswith("#") for line in gr.grammar_text([0], [1], k).splitlines())


def test_more_than_twenty_classes_rejected(monkeypatch):
    monkeypatch.setattr(ds, "dataset_record", lambda name: {"n_classes": 21})
    with pytest.raises(DatasetError, match="grammars cover 2..20"):
        gr.build("x")


def test_wide_feature_rule_is_split_into_equal_sub_rules():
    assert "<num_feature> ::= " + "|".join(f"x[{i}]" for i in range(256)) in gr.grammar_text(list(range(256)), [], 3)
    lines = gr.grammar_text([], list(range(287)), None).splitlines()
    assert "<bool_feature> ::= <bool_feature_0>|<bool_feature_1>" in lines
    subs = [l for l in lines if l.startswith("<bool_feature_")]
    assert [l.count("x[") for l in subs] == [143, 144]
    assert sorted(int(i) for l in subs for i in re.findall(r"x\[(\d+)\]", l)) == list(range(287))


def test_existing_grammars_rebuild_byte_identical():
    """raising the class limit leaves every recorded grammar unchanged."""
    import datasets
    corpus = datasets.read_corpus()["datasets"]
    for path in sorted(gr.grammar_dir().glob("*.bnf")):
        rec = corpus.get(path.stem)
        if rec is None:
            continue
        numeric, boolean = gr.grammar_features(rec)
        rng = gr.preprocessing.constant_range(path.stem)
        k = None if rng is None else round(rng[1] + 0.01)          # as in grammar.build
        assert gr.grammar_text(numeric, boolean, k, rec["n_classes"]) == path.read_text(), path.stem
