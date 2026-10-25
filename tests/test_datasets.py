"""Tests for datasets.py. Offline by default; `-m network` runs the download tests."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import datasets as ds
from datasets import DatasetError

REPO_DATA = Path(ds.__file__).resolve().parent / "data"


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """An empty data/ directory; every path in datasets.py is derived from DATA_DIR at call time."""
    monkeypatch.setattr(ds, "DATA_DIR", tmp_path)
    return tmp_path


def manual_recipe(data_dir, name, df, **extra):
    """Write df as a manually downloaded CSV and return its recipe."""
    d = data_dir / "raw" / name
    d.mkdir(parents=True)
    path = d / f"{name}.csv"
    df.to_csv(path, index=False)
    recipe = {"source": "manual", "id": f"{name}.csv", "url": f"https://example.org/{name}",
              "source_label": "class", "retrieved": "2026-10-02",
              "source_checksum": hashlib.sha256(path.read_bytes()).hexdigest()}
    recipe.update(extra)
    return recipe


def small():
    return pd.DataFrame({"a": [0.5, 1.5, 2.5, 3.5], "b": [0, 1, 1, 0], "class": ["no", "yes", "no", "yes"]})


def base_recipe(**extra):
    r = {"source": "manual", "id": "x.csv", "url": "u", "source_label": "class", "source_checksum": "0"}
    r.update(extra)
    return r


# ---------------------------------------------------------------- foundation (T010)

@pytest.mark.parametrize("extra, message", [
    ({"source_label": "nope"}, "label column"),
    ({"drop": [{"column": "zz", "reason": "r"}]}, "drop: column"),
    ({"rename": {"zz": "c"}}, "rename: column"),
    ({"rename": {"a": "target"}}, "a feature is named 'target'"),
    ({"label_map": {"no": 0}}, "not in the map"),
    ({"label_map": {"no": 0, "yes": 2}}, "onto 0..K-1"),
])
def test_apply_recipe_errors(extra, message):
    with pytest.raises(DatasetError, match=message):
        ds.apply_recipe("t", base_recipe(**extra), small())


def test_single_class_after_cleaning():
    raw = small()
    raw.loc[raw["class"] == "yes", "a"] = np.nan
    with pytest.raises(DatasetError, match="only one class"):
        ds.apply_recipe("t", base_recipe(), raw)


def test_missing_rows_dropped_and_counted_default_map_sorted():
    raw = small()
    raw.loc[0, "a"] = np.nan
    df, changes, label_map = ds.apply_recipe("t", base_recipe(), raw)
    assert len(df) == 3
    assert {"action": "drop_missing_rows", "detail": "rows with any missing value", "effect": "1 rows"} in changes
    assert label_map == {"no": 0, "yes": 1}
    assert list(df.columns) == ["a", "b", "target"] and df["target"].dtype == np.int64


def test_float32_cast_to_float64_full_repr(data_dir):
    raw = pd.DataFrame({"a": np.array([-4.4008059, 1.0], dtype=np.float32), "class": [0, 1]})
    df, changes, _ = ds.apply_recipe("t", base_recipe(), raw)
    assert df["a"].dtype == np.float64
    assert {"action": "cast", "detail": "float32 → float64", "effect": "1 columns"} in changes
    ds.write_table("t", df)
    assert "-4.400805950164795" in (data_dir / "t.csv").read_text()


def test_corpus_update_touches_one_entry(data_dir):
    ds.update_corpus_entry("a", {"checksum": "1", "x": 1})
    ds.update_corpus_entry("b", {"checksum": "2"})
    version = ds.corpus_version()
    ds.update_corpus_entry("b", {"checksum": "2", "y": 2})
    assert ds.corpus_version() == version
    corpus = ds.read_corpus()
    assert corpus["datasets"]["a"] == {"checksum": "1", "x": 1}
    ds.update_corpus_entry("b", {"checksum": "3"})
    assert ds.corpus_version() != version


def test_write_table_deterministic(data_dir):
    df, _, _ = ds.apply_recipe("t", base_recipe(), small())
    assert ds.write_table("t", df) == ds.write_table("t", df)


def test_validate_recipe():
    with pytest.raises(DatasetError, match="unknown recipe key"):
        ds.validate_recipe("t", base_recipe(lable="x"))
    r = base_recipe()
    del r["source_checksum"]
    with pytest.raises(DatasetError, match="needs source_checksum"):
        ds.validate_recipe("t", r)
    with pytest.raises(DatasetError, match="unknown source"):
        ds.validate_recipe("t", base_recipe(source="kaggle"))


def test_types_override():
    """A recipe's `types` entry replaces a source-declared type and records the change with its reason."""
    good = [{"column": "V3", "type": "numeric", "reason": "0-10 levels like the others"}]
    ds.validate_recipe("t", base_recipe(types=good))
    for bad in ([{"column": "V3", "type": "nominal", "reason": "x"}], [{"column": "V3", "type": "numeric"}]):
        with pytest.raises(DatasetError, match="types entries"):
            ds.validate_recipe("t", base_recipe(types=bad))
    declared, changes = ds.override_types({"V3": "nominal", "V4": "numeric"}, {"types": good})
    assert declared == {"V3": "numeric", "V4": "numeric"}
    assert changes[0]["action"] == "override_type" and "nominal -> numeric" in changes[0]["detail"]
    assert ds.override_types({"V3": "nominal"}, {}) == ({"V3": "nominal"}, [])


# ---------------------------------------------------------------- US1: verify, compare-reference (T011)

def test_verify_reports_every_problem(data_dir, capsys):
    recipes = {"t": manual_recipe(data_dir, "t", small()), "never_built": base_recipe(retrieved="2026-10-02")}
    ds.recreate("t", recipes["t"])
    assert ds.verify(["t", "never_built"], recipes) == 1          # never_built: no table
    (data_dir / "t.csv").write_text((data_dir / "t.csv").read_text().replace("0.5", "0.6"))
    (data_dir / "stray.csv").write_text("a,target\n1,0\n")
    ds.update_corpus_entry("orphan", {"checksum": "x"})
    assert ds.verify(["t", "never_built"], recipes) == 4
    out = capsys.readouterr().out
    assert "t: checksum" in out and "stray: table without a record" in out and "orphan: record without a recipe" in out


def test_compare_reference_pmlb_minmax(data_dir, tmp_path_factory):
    recipe = manual_recipe(data_dir, "t", small())
    ds.recreate("t", recipe)
    ref_dir = tmp_path_factory.mktemp("ref")
    ref = pd.read_csv(data_dir / "t.csv")
    ref["a"] = (ref["a"] - ref["a"].min()) / (ref["a"].max() - ref["a"].min())
    ref.to_csv(ref_dir / "t.csv", index=False)
    assert ds.compare_reference("t", {**recipe, "source": "pmlb"}, ref_dir)
    assert not ds.compare_reference("t", recipe, ref_dir)          # exact comparison for non-PMLB
    ref.loc[0, "target"] = 1 - ref.loc[0, "target"]
    ref.to_csv(ref_dir / "t.csv", index=False)
    assert not ds.compare_reference("t", {**recipe, "source": "pmlb"}, ref_dir)


@pytest.mark.network
def test_recreate_appendicitis_from_pmlb(data_dir):
    recipes = ds.yaml.safe_load((REPO_DATA / "recipes.yaml").read_text())
    entry = ds.recreate("appendicitis", recipes["appendicitis"])
    assert (entry["n_rows"], entry["n_features"]) == (106, 7)
    assert entry["class_counts"] == {"0": 85, "1": 21}
    assert entry["checksum"] == ds.sha256_file(REPO_DATA / "appendicitis.csv")


@pytest.mark.network
def test_unknown_openml_id(data_dir):
    with pytest.raises(DatasetError, match="OpenML"):
        ds.recreate("nothing", {"source": "openml", "id": 999999999, "url": "u", "source_label": "class"})
    assert not (data_dir / "nothing.csv").exists() and "nothing" not in ds.read_corpus()["datasets"]


# ---------------------------------------------------------------- US2: manual downloads (T020)

def test_manual_download_ok(data_dir):
    entry = ds.recreate("m", manual_recipe(data_dir, "m", small()))
    assert entry["n_rows"] == 4 and (data_dir / "m.csv").is_file()


def test_manual_file_missing(data_dir):
    recipe = manual_recipe(data_dir, "m", small())
    (data_dir / "raw" / "m" / "m.csv").unlink()
    with pytest.raises(DatasetError, match=r"m\.csv.*https://example.org/m.*expected sha256 " + recipe["source_checksum"]):
        ds.recreate("m", recipe)
    assert not (data_dir / "m.csv").exists() and "m" not in ds.read_corpus()["datasets"]


def test_manual_file_changed(data_dir):
    recipe = manual_recipe(data_dir, "m", small())
    path = data_dir / "raw" / "m" / "m.csv"
    path.write_text(path.read_text() + "9.5,1,no\n")
    with pytest.raises(DatasetError, match="source changed: expected sha256 " + recipe["source_checksum"]):
        ds.recreate("m", recipe)
    assert not (data_dir / "m.csv").exists() and "m" not in ds.read_corpus()["datasets"]


# ---------------------------------------------------------------- US3: provenance (T023)

def test_classify_features():
    df = pd.DataFrame({"bin": [0, 1, 0, 1], "num2": [0.0, 1.0, 1.0, 0.0], "const": [3.0] * 4,
                       "x": [0.1, 0.2, 0.3, 0.4], "target": [0, 1, 0, 1]})
    out = {f["name"]: f for f in ds.classify_features("t", df, {"bin": "boolean", "num2": "numeric"})}
    assert out["bin"]["type"] == "boolean" and out["bin"]["flags"] == []
    assert out["num2"]["type"] == "numeric" and out["num2"]["flags"] == ["type_disagreement"]
    assert out["const"]["flags"] == ["constant"] and out["x"]["type"] == "numeric"


def test_classify_features_errors():
    df = pd.DataFrame({"n3": [1, 2, 3, 1], "target": [0, 1, 0, 1]})
    with pytest.raises(DatasetError, match="nominal with 3 values"):
        ds.classify_features("t", df, {"n3": "nominal"})
    df = pd.DataFrame({"b": [1, 2, 1, 2], "target": [0, 1, 0, 1]})
    with pytest.raises(DatasetError, match="not 0/1; add a feature_maps"):
        ds.classify_features("t", df, {"b": "boolean"})
    raw = pd.DataFrame({"txt": ["a", "b", "c", "a"], "class": [0, 1, 0, 1]})
    with pytest.raises(DatasetError, match="non-numeric values"):
        ds.apply_recipe("t", base_recipe(), raw)


def test_record_has_every_field(data_dir):
    raw = pd.concat([small(), small().iloc[:2]], ignore_index=True)
    entry = ds.recreate("m", manual_recipe(data_dir, "m", raw, notes="n"))
    fields = {"name", "source", "id", "url", "retrieved", "source_checksum", "original_shape", "changes",
              "label_map", "features", "n_rows", "n_features", "n_numeric", "n_boolean", "n_classes",
              "class_counts", "duplicate_rows", "notes", "checksum", "code_commit"}
    assert fields <= set(entry)
    assert entry["duplicate_rows"] == 2 and entry["n_boolean"] == 1 and entry["n_numeric"] == 1
    assert json.loads((data_dir / "corpus.json").read_text())["datasets"]["m"] == entry


# ---------------------------------------------------------------- US4: difficulty (T028)

def test_difficulty_repeatable(data_dir):
    crabs = pd.read_csv(REPO_DATA / "prnn_crabs.csv").rename(columns={"target": "class"})
    ds.recreate("crabs", manual_recipe(data_dir, "crabs", crabs))
    first, second = ds.measure_difficulty("crabs"), ds.measure_difficulty("crabs")
    for key in ("tree_depth2", "logreg", "gradient_boosting"):
        assert 0 <= first[key]["mean"] <= 1 and len(first[key]["folds"]) == 3
        assert first[key] == second[key]
    s = first["settings"]
    assert "n_splits=3" in s["cv"] and set(s["models"]) == {"tree_depth2", "logreg", "gradient_boosting"}
    assert s["sklearn_version"] and s["code_commit"]


def test_load_table_checks_checksum(data_dir):
    ds.recreate("m", manual_recipe(data_dir, "m", small()))
    X, y, info = ds.load_table("m")
    assert X.shape == (4, 2) and y.tolist() == [0, 1, 0, 1] and info[1] == {"name": "b", "type": "boolean"}
    shutil.copy(data_dir / "m.csv", data_dir / "backup")
    (data_dir / "m.csv").write_text((data_dir / "m.csv").read_text().replace("0.5", "0.7"))
    with pytest.raises(DatasetError, match="does not match"):
        ds.load_table("m")


def test_sparse_columns_densified():
    raw = pd.DataFrame({"a": pd.arrays.SparseArray([0.0, 1.5, 0.0, 2.5]),
                        "class": pd.arrays.SparseArray([0.0, 1.0, 0.0, 1.0])})
    df, changes, label_map = ds.apply_recipe("t", base_recipe(), raw)
    assert label_map == {"0.0": 0, "1.0": 1} and df["a"].tolist() == [0.0, 1.5, 0.0, 2.5]
    assert changes[0]["action"] == "densify"


def test_one_hot_in_place():
    raw = pd.DataFrame({"a": [0.5, 1.5, 2.5, 3.5], "day": ["mon", "tue", "wed", "mon"], "z": [1.0] * 4,
                        "class": [0, 1, 0, 1]})
    df, changes, _ = ds.apply_recipe("t", base_recipe(one_hot=["day"]), raw)
    assert list(df.columns) == ["a", "day=mon", "day=tue", "day=wed", "z", "target"]
    assert df["day=mon"].tolist() == [1, 0, 0, 1] and df["day=mon"].dtype == np.int64
    assert {"action": "one_hot", "detail": "day: 3 values", "effect": "3 columns"} in changes
    out = {f["name"]: f for f in ds.classify_features("t", df, {"day=mon": None})}
    assert out["day=tue"]["type"] == "boolean"
    with pytest.raises(DatasetError, match="one_hot: column"):
        ds.apply_recipe("t", base_recipe(one_hot=["nope"]), raw)


def test_recipes_duplicate_key_rejected(data_dir):
    (data_dir / "recipes.yaml").write_text("a:\n  source: pmlb\n  source: openml\n")
    with pytest.raises(DatasetError, match="duplicate key"):
        ds.load_recipes()


def test_verify_ignores_record_files(data_dir, capsys):
    (data_dir / "constant_ranges.csv").write_text("name\n")
    (data_dir / "complexity.csv").write_text("name\n")
    assert ds.verify([], {}) == 0


# ---------------------------------------------------------------- balance, zip source

def unbalanced():
    return pd.DataFrame({"a": np.arange(10, dtype=float), "class": list("xxxxxxyyyz")})


def test_balance_equal_classes_source_order_and_recorded():
    recipe = base_recipe(balance={"method": "undersample", "seed": 0})
    df, changes, _ = ds.apply_recipe("t", recipe, unbalanced())
    assert df["target"].value_counts().tolist() == [1, 1, 1]
    assert df["a"].is_monotonic_increasing                       # rows kept in source order
    change = next(c for c in changes if c["action"] == "balance")
    assert change["detail"] == 'undersample seed 0; before {"0": 6, "1": 3, "2": 1}' and change["effect"] == "10 → 3 rows"
    again, _, _ = ds.apply_recipe("t", recipe, unbalanced())
    assert again.equals(df)


def test_balance_other_seed_other_rows():
    pick = lambda seed: ds.apply_recipe("t", base_recipe(balance={"method": "undersample", "seed": seed}),
                                        pd.DataFrame({"a": np.arange(40.0), "class": ["x"] * 30 + ["y"] * 10}))[0]
    assert len(pick(0)) == 20 and not pick(0).equals(pick(1))


@pytest.mark.parametrize("balance", [{"method": "oversample", "seed": 0}, {"method": "undersample"},
                                     {"method": "undersample", "seed": "0"}, {"method": "undersample", "seed": True},
                                     "undersample"])
def test_balance_bad_values_refused(balance):
    with pytest.raises(DatasetError, match="balance must be"):
        ds.validate_recipe("t", base_recipe(balance=balance))


def zip_recipe(data_dir, name, df, member="inner/data.csv"):
    import zipfile
    d = data_dir / "raw" / name
    d.mkdir(parents=True)
    with zipfile.ZipFile(d / "source.zip", "w") as z:
        z.writestr(member, df.to_csv(index=False))
    return {"source": "zip", "id": "inner/data.csv", "download": "https://example.org/x.zip",
            "url": "https://example.org/x", "source_label": "class", "retrieved": "2026-10-05",
            "source_checksum": hashlib.sha256((d / "source.zip").read_bytes()).hexdigest()}


def test_zip_source_reads_member(data_dir):
    entry = ds.recreate("z", zip_recipe(data_dir, "z", small()))
    assert entry["n_rows"] == 4 and entry["source"] == "zip" and (data_dir / "z.csv").is_file()


def test_zip_member_missing(data_dir):
    recipe = zip_recipe(data_dir, "z", small(), member="other.csv")
    recipe["source_checksum"] = ds.raw_checksum("z", recipe)
    with pytest.raises(DatasetError, match="no member 'inner/data.csv'"):
        ds.recreate("z", recipe)


def test_download_only_with_zip():
    with pytest.raises(DatasetError, match="`download`"):
        ds.validate_recipe("t", {"source": "zip", "id": "a.csv", "url": "u", "source_label": "c"})
    with pytest.raises(DatasetError, match="`download`"):
        ds.validate_recipe("t", base_recipe(download="https://example.org/x.zip"))
