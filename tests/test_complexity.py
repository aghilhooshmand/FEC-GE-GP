import csv
import hashlib

import numpy as np
import pandas as pd
import pytest

import complexity as cx
import datasets as ds
from datasets import DatasetError


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "DATA_DIR", tmp_path)
    (tmp_path / "recipes.yaml").write_text("{}\n")
    return tmp_path


def small_dataset(data_dir, name="m", n=60, seed=1):
    """A manual-recipe dataset recreated by 3 numeric features, 2 classes."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.integers(0, 5, n) * 1.0,
                       "class": rng.integers(0, 2, n)})
    d = data_dir / "raw" / name
    d.mkdir(parents=True)
    df.to_csv(d / f"{name}.csv", index=False)
    recipe = {"source": "manual", "id": f"{name}.csv", "url": "u", "source_label": "class", "retrieved": "2026-10-02",
              "source_checksum": hashlib.sha256((d / f"{name}.csv").read_bytes()).hexdigest()}
    (data_dir / "recipes.yaml").write_text(ds.yaml.safe_dump({name: recipe}))
    ds.recreate(name, recipe)
    return recipe


# ---------------------------------------------------------------- foundation (T004)

def test_write_rows_replaces_one(data_dir):
    cx.write_rows([{"name": "a", "origin": "computed", "pycol_F1": 0.5}])
    cx.write_rows([{"name": "b", "origin": "computed"}])
    before_b = cx.complexity_row("b")
    cx.write_rows([{"name": "a", "origin": "computed", "pycol_F1": 0.25}])
    df = cx.load_complexity()
    assert sorted(df["name"]) == ["a", "b"] and cx.complexity_row("a")["pycol_F1"] == "0.25"
    assert cx.complexity_row("b") == before_b
    with pytest.raises(DatasetError, match="unknown record column"):
        cx.write_rows([{"name": "a", "bogus": 1}])


def test_load_rejects_other_columns(data_dir):
    (data_dir / "complexity.csv").write_text("name,extra\nx,1\n")
    with pytest.raises(DatasetError, match="columns differ"):
        cx.load_complexity()


def test_presets():
    assert [len(cx.PRESETS[p]) for p in ("cheap_minimal", "cheap", "standard", "all")] == [7, 24, 26, 29]
    for metrics in cx.PRESETS.values():
        assert list(metrics) == [m for m in cx.METRICS if m in metrics]


# ---------------------------------------------------------------- US1: import (T005)

def write_source(path, rows):
    cols = list(cx.SOURCE_COLUMNS) + ["pycol_matrix_dtype", "pycol_heom_parallel", "profile_url"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


@pytest.fixture
def source(tmp_path_factory, monkeypatch):
    path = tmp_path_factory.mktemp("src") / "summary.csv"
    monkeypatch.setattr(cx, "source_provenance", lambda s: ("abc123", "2026-07-22"))
    monkeypatch.setattr(cx, "REFERENCE_INPUT_DIR", path.parent)
    rows = [{"dataset_file": f"d{i}.csv", "n_rows_used": "10.0", "n_features_after_encoding": "3.0",
             "pycol_metrics_preset": "cheap" if i else "", "pycol_F1": "0.1234567890123456",
             "pycol_N4": "0.5", "pycol_matrix_dtype": "float64"} for i in range(3)]
    write_source(path, rows)
    return path


def test_import_copies_values(data_dir, source):
    assert cx.import_reference(source) == 3
    r = cx.complexity_row("d1")
    assert r["pycol_F1"] == "0.1234567890123456" and r["preset"] == "cheap" and r["origin"] == "DataComplexity"
    assert cx.complexity_row("d0")["preset"] == "unknown" and r["matches_table"] == "no recipe"
    assert r["source_commit"] == "abc123" and r["computed_at"] == "2026-07-22" and r["pycol_T1"] == ""
    before = (data_dir / "complexity.csv").read_bytes()
    cx.import_reference(source)
    assert (data_dir / "complexity.csv").read_bytes() == before


def test_import_errors(data_dir, source):
    df = pd.read_csv(source, dtype=str, keep_default_na=False)
    df.drop(columns="pycol_N1").to_csv(source, index=False)
    with pytest.raises(DatasetError, match="misses column"):
        cx.import_reference(source)
    assert not (data_dir / "complexity.csv").exists()
    pd.concat([df, df.iloc[:1]]).to_csv(source, index=False)
    with pytest.raises(DatasetError, match="repeated"):
        cx.import_reference(source)


def test_import_keeps_computed_rows(data_dir, source):
    cx.write_rows([{"name": "d1", "origin": "computed", "pycol_F1": 0.9}])
    cx.import_reference(source)
    assert cx.complexity_row("d1")["origin"] == "computed" and cx.complexity_row("d1")["pycol_F1"] == "0.9"


# ---------------------------------------------------------------- US2: compute (T008)

REAL_INPUT = cx.REFERENCE_INPUT_DIR


def test_seeded_repeatable():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(50, 3)), rng.integers(0, 2, 50)
    a, _ = cx.compute_metrics("t", x, y, ["N4", "MRCA", "F1"])
    b, _ = cx.compute_metrics("t", x, y, ["N4", "MRCA", "F1"])
    assert a == b


def test_size_rule(data_dir, monkeypatch):
    small_dataset(data_dir, n=100)
    monkeypatch.setattr(cx, "MAX_ROWS_DISTANCE", 50)
    row = cx.compute("m", "cheap")
    assert row["skipped"].endswith(": n_rows > 50") and "N1" in row["skipped"]
    stored = cx.complexity_row("m")
    assert stored["pycol_F1"] != "" and stored["pycol_N1"] == ""


def test_resume_equals_uninterrupted(data_dir, monkeypatch):
    small_dataset(data_dir)
    full = cx.compute("m", "cheap_minimal")
    (data_dir / "complexity.csv").unlink()                                  # start over, then interrupt at F3
    real = cx._pycol_shell

    def interrupting_shell(*a):
        comp = real(*a)

        def boom():
            raise KeyboardInterrupt
        comp.F3 = boom
        return comp
    monkeypatch.setattr(cx, "_pycol_shell", interrupting_shell)
    with pytest.raises(KeyboardInterrupt):
        cx.compute("m", "cheap_minimal")
    partial = cx.complexity_row("m")
    assert partial["pycol_F1"] != "" and partial["pycol_F3"] == "" and partial["pycol_F4"] == ""
    monkeypatch.setattr(cx, "_pycol_shell", real)
    resumed = cx.compute("m", "cheap_minimal")
    for m in cx.PRESETS["cheap_minimal"]:
        assert float(resumed[f"pycol_{m}"]) == float(full[f"pycol_{m}"]), m


def test_failed_metric_recorded(data_dir, monkeypatch):
    small_dataset(data_dir)
    real = cx._pycol_shell

    def failing_shell(*a):
        comp = real(*a)

        def bad():
            raise ValueError("broken")
        comp.F2 = bad
        return comp
    monkeypatch.setattr(cx, "_pycol_shell", failing_shell)
    cx.compute("m", "cheap_minimal")
    row = cx.complexity_row("m")
    assert row["pycol_F2"] == "" and "F2: ValueError: broken" in row["failed"] and row["pycol_F1"] != ""


def test_refuses_to_overwrite(data_dir):
    small_dataset(data_dir)
    cx.write_rows([{"name": "m", "origin": "DataComplexity", "preset": "cheap"}])
    with pytest.raises(DatasetError, match="imported row exists"):
        cx.compute("m", "cheap_minimal")
    cx.compute("m", "cheap_minimal", replace=True)
    with pytest.raises(DatasetError, match="use --replace"):
        cx.compute("m", "cheap")
    assert cx.check() == 0


def test_check_reports_stale(data_dir):
    small_dataset(data_dir)
    cx.compute("m", "cheap_minimal")
    cx.write_rows([{**cx.complexity_row("m"), "table_checksum": "0" * 64}])
    assert cx.check() == 1


def test_minmax_constant_column():
    x = np.array([[1.0, 5.0], [3.0, 5.0], [2.0, 5.0]])
    assert cx.minmax(x).tolist() == [[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]]
