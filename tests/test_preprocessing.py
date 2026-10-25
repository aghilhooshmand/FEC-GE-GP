import hashlib

import numpy as np
import pandas as pd
import pytest

import datasets as ds
import preprocessing as pp
from datasets import DatasetError

INFO = [{"name": "a", "type": "numeric"}, {"name": "b", "type": "boolean"}, {"name": "c", "type": "numeric"}]


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "DATA_DIR", tmp_path)
    return tmp_path


def make_dataset(data_dir, name, df):
    """Recreate df (with a `class` column) through 001 as a manual recipe."""
    d = data_dir / "raw" / name
    d.mkdir(parents=True)
    df.to_csv(d / f"{name}.csv", index=False)
    recipe = {"source": "manual", "id": f"{name}.csv", "url": "u", "source_label": "class", "retrieved": "2026-10-02",
              "source_checksum": hashlib.sha256((d / f"{name}.csv").read_bytes()).hexdigest()}
    recipes = ds.yaml.safe_load((data_dir / "recipes.yaml").read_text()) if (data_dir / "recipes.yaml").exists() else {}
    (data_dir / "recipes.yaml").write_text(ds.yaml.safe_dump({**(recipes or {}), name: recipe}))
    ds.recreate(name, recipe)


# ---------------------------------------------------------------- US1: constant range

def test_candidates():
    for k, n, top in [(1, 199, 0.99), (2, 399, 1.99), (3, 599, 2.99)]:
        c = pp.candidate_constants(k)
        assert len(c) == n and c[0] == -top and c[-1] == top and 0.0 in c


def test_pick_winner():
    t = [{"coverage_min": 0.6, "degenerate_rate": 0.0}, {"coverage_min": 0.96, "degenerate_rate": 0.2},
         {"coverage_min": 0.99, "degenerate_rate": 0.4}]
    assert pp.pick_winner(t) is t[1]
    low = [{"coverage_min": 0.9, "degenerate_rate": 0.3}, {"coverage_min": 0.9, "degenerate_rate": 0.1}]
    assert pp.pick_winner(low) is low[1]


def test_measure_boolean_only_and_constant(data_dir):
    make_dataset(data_dir, "bools", pd.DataFrame({"b": [0, 1, 1, 0], "class": [0, 1, 0, 1]}))
    assert pp.measure("bools")["winner"] == "none"
    rng = np.random.default_rng(0)
    make_dataset(data_dir, "num", pd.DataFrame({"a": rng.normal(size=200), "k": [5.0] * 200,
                                                "class": rng.integers(0, 2, 200)}))
    row = pp.measure("num")
    assert row["constant_columns"] == "k" and row["winner"] in ("pm2", "pm3")


def test_record_updates_one_row(data_dir):
    rng = np.random.default_rng(1)
    for name in ("a", "b"):
        make_dataset(data_dir, name, pd.DataFrame({"x": rng.normal(size=50), "class": rng.integers(0, 2, 50)}))
    pp.write_row(pp.measure("a"))
    pp.write_row(pp.measure("b"))
    before = pp.load_ranges().set_index("name").loc["a"].to_dict()
    pp.write_row(pp.measure("b"))
    assert pp.load_ranges().set_index("name").loc["a"].to_dict() == before
    assert pp.constant_range("a") == (-2.99, 2.99) or pp.constant_range("a") == (-1.99, 1.99)
    assert pp.check() == 0


# ---------------------------------------------------------------- US2: z-score per run

def test_fit_apply_zscore():
    rng = np.random.default_rng(2)
    X = np.column_stack([rng.normal(5, 3, 100), rng.integers(0, 2, 100), rng.normal(-1, 0.5, 100)])
    train, test = X[:80], X[80:]
    params = pp.fit_zscore(train, INFO)
    zt = pp.apply_zscore(train, params)
    for j in (0, 2):
        assert abs(zt[:, j].mean()) < 1e-12 and abs(zt[:, j].std() - 1) < 1e-12
    assert (zt[:, 1] == train[:, 1]).all()
    ztest = pp.apply_zscore(test, params)
    expect = (test[:, 0] - train[:, 0].mean()) / train[:, 0].std()
    assert np.array_equal(ztest[:, 0], expect) and (ztest[:, 1] == test[:, 1]).all()
    again = pp.fit_zscore(train, INFO)
    assert all(np.array_equal(params[k], again[k]) for k in params)


def test_zscore_constant_and_errors():
    X = np.array([[1.0, 0, 3.0], [1.0, 1, 4.0], [1.0, 0, 5.0]])
    z = pp.apply_zscore(np.array([[7.0, 1, 4.0]]), pp.fit_zscore(X, INFO))
    assert z[0, 0] == 0.0 and z[0, 2] == 0.0
    with pytest.raises(DatasetError):
        pp.fit_zscore(X[:, :2], INFO)
    with pytest.raises(DatasetError):
        pp.apply_zscore(X[:, :2], pp.fit_zscore(X, INFO))


def test_split_coverage_counts_gaps_with_a_constant(data_dir):
    make_dataset(data_dir, "low", pd.DataFrame({"k": [0, 0, 1, 1, 2, 2, 5, 5] * 5, "class": [0, 1] * 20}))
    pp.write_row(pp.measure("low"))
    z = np.unique(pp.zscore_column(np.array([0, 0, 1, 1, 2, 2, 5, 5] * 5, dtype=float)))
    consts = pp.candidate_constants(int(round(pp.constant_range("low")[1] + 0.01)))
    reached = sum(any(a <= c < b for c in consts) for a, b in zip(z[:-1], z[1:]))   # gaps a constant falls in
    assert pp.split_coverage("low") == reached / 3
    assert set(pp.winners_by_target("low").values()) <= {"pm1", "pm2", "pm3"}
