"""Datasets stage: recreate every dataset from its recipe, record its provenance, measure difficulty.

Reads   data/recipes.yaml          hand-written recipes, one per dataset (source, id, label, changes)
        data/raw/<name>/           downloaded source files (git-ignored), a manually downloaded file,
                                   or a zip archive (source zip: `download` URL, `id` = member CSV)
Writes  data/<name>.csv            the table: features in source order, then `target` (0..K-1)
        data/corpus.json           one provenance entry per dataset, written only by this module

Commands (python datasets.py <command> <names...> | --all):
  recreate           fetch (or reuse the cached raw files), apply the recipe, write table + record
  verify             check table checksums against the record, and recipes/tables/records that lack each other
  difficulty         3-fold balanced accuracy of a depth-2 tree, logistic regression, gradient boosting
  compare-reference  check the tables against the reference implementation's copies

Later stages use only load_table(), dataset_record() and corpus_version().
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
import urllib.error
import urllib.request
import warnings
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REFERENCE_DIR = ROOT.parent / "reference" / "data"
LABEL = "target"
RECORD_FILES = ("complexity.csv", "constant_ranges.csv")   # per-dataset records kept in data/, not tables

SOURCES = ("openml", "uci", "pmlb", "manual", "zip")
REQUIRED_KEYS = ("source", "id", "url", "source_label")
OPTIONAL_KEYS = ("source_checksum", "retrieved", "label_map", "drop", "rename", "one_hot", "feature_maps", "types", "notes",
                 "balance", "download")
PMLB_URL = "https://github.com/EpistasisLab/pmlb/raw/master/datasets/{id}/{file}"


class DatasetError(Exception):
    """Every failure of this stage; the message always starts with the dataset name."""

    def __init__(self, name: str, message: str):
        super().__init__(f"{name}: {message}")


# ---------------------------------------------------------------- files, checksums, recipes, corpus

def raw_dir(name: str) -> Path:
    return DATA_DIR / "raw" / name


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_recipes() -> dict:
    path = DATA_DIR / "recipes.yaml"
    if not path.is_file():
        raise DatasetError("recipes", f"{path} not found")
    return yaml.load(path.read_text(encoding="utf-8"), Loader=_NoDuplicatesLoader) or {}


class _NoDuplicatesLoader(yaml.SafeLoader):
    """safe_load that raises on a repeated key (plain YAML silently keeps the last one)."""

    def construct_mapping(self, node, deep=False):
        keys = [self.construct_object(k, deep=deep) for k, _ in node.value]
        repeated = sorted({str(k) for k in keys if keys.count(k) > 1})
        if repeated:
            raise DatasetError("recipes", f"duplicate key(s) {repeated} near line {node.start_mark.line + 1}")
        return super().construct_mapping(node, deep)


def validate_recipe(name: str, recipe: dict) -> None:
    if not isinstance(recipe, dict):
        raise DatasetError(name, "recipe must be a mapping")
    unknown = set(recipe) - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS)
    if unknown:
        raise DatasetError(name, f"unknown recipe key(s) {sorted(unknown)}")
    missing = [k for k in REQUIRED_KEYS if k not in recipe]
    if missing:
        raise DatasetError(name, f"recipe misses {missing}")
    if recipe["source"] not in SOURCES:
        raise DatasetError(name, f"unknown source {recipe['source']!r}, expected one of {SOURCES}")
    if recipe["source"] == "manual" and not recipe.get("source_checksum"):
        raise DatasetError(name, "a manual recipe needs source_checksum (sha256sum of the downloaded file)")
    if (recipe["source"] == "zip") != bool(recipe.get("download")):
        raise DatasetError(name, "`download` (archive URL) goes with source zip, and only with it")
    balance = recipe.get("balance")
    if balance is not None and (not isinstance(balance, dict) or balance.get("method") != "undersample"
                                or set(balance) != {"method", "seed"} or not isinstance(balance["seed"], int)
                                or isinstance(balance["seed"], bool)):
        raise DatasetError(name, f"balance must be {{method: undersample, seed: <int>}}, got {balance!r}")
    for item in recipe.get("drop") or []:
        if not isinstance(item, dict) or not item.get("column") or not item.get("reason"):
            raise DatasetError(name, f"drop entries need a column and a reason, got {item!r}")
    for item in recipe.get("types") or []:
        if not isinstance(item, dict) or set(item) != {"column", "type", "reason"} \
                or item["type"] not in ("numeric", "boolean") or not item["reason"]:
            raise DatasetError(name, f"types entries need column, type (numeric | boolean) and reason, got {item!r}")
    new_names = list((recipe.get("rename") or {}).values())
    if len(new_names) != len(set(new_names)):
        raise DatasetError(name, "rename gives two columns the same new name")


def read_corpus() -> dict:
    path = DATA_DIR / "corpus.json"
    if not path.is_file():
        return {"corpus_version": "", "datasets": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def compute_corpus_version(datasets: dict) -> str:
    lines = "".join(f"{n} {e['checksum']}\n" for n, e in sorted(datasets.items()))
    return hashlib.sha256(lines.encode()).hexdigest()[:12]


def update_corpus_entry(name: str, entry: dict) -> None:
    """Replace only datasets[name] () and recompute corpus_version."""
    corpus = read_corpus()
    corpus["datasets"][name] = entry
    corpus["corpus_version"] = compute_corpus_version(corpus["datasets"])
    path = DATA_DIR / "corpus.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(corpus, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def git_commit() -> str:
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    try:
        commit = git("rev-parse", "HEAD")
        # "-dirty" = uncommitted Python code or pins; data, docs and specs do not count
        return commit + ("-dirty" if git("status", "--porcelain", "--", "*.py", "requirements.txt") else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def upsert_csv_row(path: Path, columns: tuple, row: dict) -> None:
    """Add or replace the row whose `name` equals row["name"] in a per-dataset CSV record; every
    other row is kept exactly as read. Values are written as text (floats as repr)."""
    unknown = set(row) - set(columns)
    if unknown:
        raise DatasetError(row.get("name", "?"), f"unknown record column(s) {sorted(unknown)}")
    text = {c: "" if row.get(c) is None else repr(row[c]) if isinstance(row[c], float) else str(row[c])
            for c in columns}
    df = pd.read_csv(path, dtype=str, keep_default_na=False) if path.is_file() else pd.DataFrame(columns=columns)
    if tuple(df.columns) != tuple(columns):
        raise DatasetError(row["name"], f"{path} columns differ from {columns}")
    df = pd.concat([df[df["name"] != row["name"]], pd.DataFrame([text], columns=columns)], ignore_index=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.sort_values("name", kind="stable").to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(path)


def write_table(name: str, df: pd.DataFrame) -> str:
    """Write features then `target` deterministically; return the file's sha256."""
    if list(df.columns)[-1] != LABEL:
        raise DatasetError(name, f"the last column must be {LABEL!r}")
    path = DATA_DIR / f"{name}.csv"
    tmp = DATA_DIR / f"{name}.csv.tmp"
    df.to_csv(tmp, index=False, lineterminator="\n", encoding="utf-8")
    tmp.replace(path)
    return sha256_file(path)


# ---------------------------------------------------------------- fetching raw source files

def raw_files(name: str, recipe: dict) -> list[Path]:
    """The downloaded files that make up the source, in checksum order."""
    d, i = raw_dir(name), recipe["id"]
    oml = d / "openml" / "org" / "openml" / "www" / "datasets" / str(i)
    # OpenML serves parquet when it has it, else ARFF (e.g. codrna, 351)
    oml_data = oml / "dataset.arff" if (oml / "dataset.arff").is_file() and not (oml / f"dataset_{i}.pq").is_file() \
        else oml / f"dataset_{i}.pq"
    return {
        "openml": [oml_data, oml / "features.xml"],
        "pmlb": [d / f"{i}.tsv.gz", d / "metadata.yaml"],
        "uci": [d / "uci.csv", d / "uci_variables.csv"],
        "manual": [d / str(i)],
        "zip": [d / "source.zip"],
    }[recipe["source"]]


def raw_checksum(name: str, recipe: dict) -> str:
    """sha256 over the listed raw files; for a manual download or a zip archive, the plain `sha256sum`."""
    files = raw_files(name, recipe)
    for path in files:
        if not path.is_file():
            raise DatasetError(name, f"raw file {path} missing")
    if recipe["source"] in ("manual", "zip"):
        return sha256_file(files[0])
    h = hashlib.sha256()
    for path in files:
        h.update(f"{path.relative_to(raw_dir(name))}\n".encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def _openml_dataset(name: str, recipe: dict):
    import openml
    openml.config.set_root_cache_directory(str(raw_dir(name) / "openml"))
    try:
        return openml.datasets.get_dataset(int(recipe["id"]), download_data=True,
                                           download_features_meta_data=True, download_qualities=False)
    except Exception as exc:
        raise DatasetError(name, f"OpenML {recipe['id']}: {exc}") from exc


def _download(name: str, recipe: dict) -> None:
    """Fetch the source files into data/raw/<name>/ (ported from the reference implementation dataset_fetchers.py)."""
    d, source = raw_dir(name), recipe["source"]
    d.mkdir(parents=True, exist_ok=True)
    if source == "pmlb":
        for file in (f"{recipe['id']}.tsv.gz", "metadata.yaml"):
            url = PMLB_URL.format(id=recipe["id"], file=file)
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    (d / file).write_bytes(r.read())
            except (urllib.error.URLError, TimeoutError) as exc:
                raise DatasetError(name, f"PMLB download {url}: {exc}") from exc
    elif source == "openml":
        _openml_dataset(name, recipe)
    elif source == "zip":
        try:
            with urllib.request.urlopen(recipe["download"], timeout=300) as r:
                (d / "source.zip").write_bytes(r.read())
        except (urllib.error.URLError, TimeoutError) as exc:
            raise DatasetError(name, f"download {recipe['download']}: {exc}") from exc
    elif source == "uci":
        from ucimlrepo import fetch_ucirepo
        try:
            ds = fetch_ucirepo(id=int(recipe["id"]))
        except Exception as exc:
            raise DatasetError(name, f"UCI {recipe['id']}: {exc}") from exc
        pd.concat([ds.data.features, ds.data.targets], axis=1).to_csv(d / "uci.csv", index=False, lineterminator="\n")
        ds.variables.to_csv(d / "uci_variables.csv", index=False, lineterminator="\n")
    for path in raw_files(name, recipe):
        if not path.is_file():
            raise DatasetError(name, f"download finished but {path.name} is missing")


def fetch_raw(name: str, recipe: dict, force_download: bool = False) -> pd.DataFrame:
    """Return the source table unchanged, from the raw cache when its checksum matches the recipe."""
    expected = recipe.get("source_checksum")
    files = raw_files(name, recipe)
    if recipe["source"] == "manual":
        if not files[0].is_file():
            raise DatasetError(name, f"file missing: download {recipe['id']} from {recipe['url']} "
                                     f"into {raw_dir(name)}/ (expected sha256 {expected})")
    else:
        cached = all(f.is_file() for f in files)
        if force_download or not cached or (expected and raw_checksum(name, recipe) != expected):
            _download(name, recipe)
    if expected:
        actual = raw_checksum(name, recipe)
        if actual != expected:
            raise DatasetError(name, f"source changed: expected sha256 {expected}, got {actual}")
    return _read_raw(name, recipe)


def _read_raw(name: str, recipe: dict) -> pd.DataFrame:
    source, path = recipe["source"], raw_files(name, recipe)[0]
    if source == "pmlb":
        return pd.read_csv(path, sep="\t", compression="gzip")
    if source == "openml":
        df, *_ = _openml_dataset(name, recipe).get_data()
        return df
    if source == "uci":
        return pd.read_csv(path)
    if source == "zip":
        with zipfile.ZipFile(path) as z:
            if recipe["id"] not in z.namelist():
                raise DatasetError(name, f"zip: no member {recipe['id']!r} in the archive ({z.namelist()})")
            with z.open(recipe["id"]) as f:
                return pd.read_csv(f)
    if path.suffix.lower() != ".csv":
        raise DatasetError(name, f"manual file {path.name}: only .csv is supported")
    return pd.read_csv(path)


def declared_types(name: str, recipe: dict) -> dict:
    """Source-declared type per source column: numeric | boolean | nominal; {} if none."""
    source, files = recipe["source"], raw_files(name, recipe)
    if source == "openml":
        ns = {"oml": "http://openml.org/openml"}
        root = ET.parse(files[1]).getroot()
        return {f.findtext("oml:name", namespaces=ns): f.findtext("oml:data_type", namespaces=ns)
                for f in root.findall("oml:feature", ns)}
    if source == "pmlb":
        to_type = {"continuous": "numeric", "integer": "numeric", "ordinal": "numeric",
                   "binary": "boolean", "categorical": "nominal"}
        meta = yaml.safe_load(files[1].read_text(encoding="utf-8"))
        return {f["name"]: to_type.get(str(f.get("type")).lower()) for f in meta.get("features", [])}
    if source == "uci":
        to_type = {"continuous": "numeric", "integer": "numeric", "binary": "boolean", "categorical": "nominal"}
        v = pd.read_csv(files[1])
        return {row["name"]: to_type.get(str(row["type"]).lower()) for _, row in v.iterrows()}
    return {}


# ---------------------------------------------------------------- applying the recipe

def override_types(declared: dict, recipe: dict) -> tuple[dict, list]:
    """The recipe's `types` entries replace a source-declared type (e.g. a numeric column the source tags
    categorical); returns the new declared types and one change per override, with its reason."""
    declared, changes = dict(declared), []
    for item in recipe.get("types") or []:
        before = declared.get(item["column"])
        declared[item["column"]] = item["type"]
        changes.append({"action": "override_type", "detail": f"{item['column']}: declared {before} -> "
                        f"{item['type']} ({item['reason']})", "effect": "1 column"})
    return declared, changes


def _value_map(name: str, what: str, values, mapping: dict | None) -> dict:
    """Map each distinct value (compared as str) through `mapping`, or sorted values -> 0..K-1."""
    distinct = sorted(pd.unique(values).tolist())
    if mapping is None:
        return {str(v): i for i, v in enumerate(distinct)}
    mapping = {str(k): int(v) for k, v in mapping.items()}
    uncovered = [v for v in distinct if str(v) not in mapping]
    if uncovered:
        raise DatasetError(name, f"{what}: values {uncovered} are not in the map {mapping}")
    return {str(v): mapping[str(v)] for v in distinct}


def apply_recipe(name: str, recipe: dict, raw: pd.DataFrame) -> tuple[pd.DataFrame, list, dict]:
    """Apply the recipe's changes in a fixed order; pure, no I/O. Returns (table, changes, label_map)."""
    df, changes, label = raw.copy(), [], recipe["source_label"]
    sparse = [c for c in df.columns if isinstance(df[c].dtype, pd.SparseDtype)]
    for col in sparse:                     # sparse ARFF from OpenML (e.g. codrna): same values, dense storage
        df[col] = df[col].sparse.to_dense()
    if sparse:
        changes.append({"action": "densify", "detail": "sparse columns stored densely, values unchanged",
                        "effect": f"{len(sparse)} columns"})
    if label not in df.columns:
        raise DatasetError(name, f"label column {label!r} not in source columns {list(df.columns)}")
    for item in recipe.get("drop") or []:
        if item["column"] not in df.columns:
            raise DatasetError(name, f"drop: column {item['column']!r} does not exist (stale recipe?)")
        df = df.drop(columns=item["column"])
        changes.append({"action": "drop_column", "detail": f"{item['column']}: {item['reason']}", "effect": "1 column"})
    rename = recipe.get("rename") or {}
    for old in rename:
        if old not in df.columns:
            raise DatasetError(name, f"rename: column {old!r} does not exist (stale recipe?)")
    if rename:
        df = df.rename(columns=rename)
        label = rename.get(label, label)
        changes.append({"action": "rename_columns", "detail": json.dumps(rename), "effect": f"{len(rename)} columns"})
    if LABEL in df.columns and label != LABEL:
        raise DatasetError(name, f"a feature is named {LABEL!r}; rename it in the recipe")
    n = len(df)
    df = df.dropna(axis=0, how="any").reset_index(drop=True)
    changes.append({"action": "drop_missing_rows", "detail": "rows with any missing value", "effect": f"{n - len(df)} rows"})

    label_map = _value_map(name, "label_map", df[label], recipe.get("label_map"))
    if sorted(set(label_map.values())) != list(range(len(set(label_map.values())))) \
            or len(set(label_map.values())) != len(label_map):
        raise DatasetError(name, f"label_map {label_map} is not a one-to-one map onto 0..K-1")
    if len(label_map) < 2:
        raise DatasetError(name, f"only one class after cleaning ({list(label_map)})")
    y = df[label].map(lambda v: label_map[str(v)]).astype("int64")
    changes.append({"action": "encode_label", "detail": json.dumps(label_map), "effect": f"{len(label_map)} classes"})
    df = df.drop(columns=label)

    if recipe.get("balance"):                     # every class cut to the smallest one's size
        seed, labels = recipe["balance"]["seed"], y.to_numpy()
        before = {int(c): int((labels == c).sum()) for c in sorted(set(labels))}
        n, rng = min(before.values()), np.random.default_rng(seed)
        keep = np.sort(np.concatenate([rng.choice(np.flatnonzero(labels == c), n, replace=False) for c in before]))
        df, y = df.iloc[keep].reset_index(drop=True), y.iloc[keep].reset_index(drop=True)   # source order kept
        changes.append({"action": "balance", "detail": f"undersample seed {seed}; before {json.dumps(before)}",
                        "effect": f"{len(labels)} → {len(keep)} rows"})

    for col in recipe.get("one_hot") or []:
        if col not in df.columns:
            raise DatasetError(name, f"one_hot: column {col!r} does not exist (stale recipe?)")
        values = sorted(pd.unique(df[col]).tolist(), key=str)
        dummies = pd.DataFrame({f"{col}={v}": (df[col] == v).astype("int64") for v in values})
        at = list(df.columns).index(col)          # dummies take the original column's place
        df = pd.concat([df.iloc[:, :at], dummies, df.iloc[:, at + 1:]], axis=1)
        changes.append({"action": "one_hot", "detail": f"{col}: {len(values)} values", "effect": f"{len(values)} columns"})

    for col, mapping in (recipe.get("feature_maps") or {}).items():
        if col not in df.columns:
            raise DatasetError(name, f"feature_maps: column {col!r} does not exist")
        m = _value_map(name, f"feature_maps[{col}]", df[col], mapping)
        df[col] = df[col].map(lambda v: m[str(v)])
        changes.append({"action": "map_feature", "detail": f"{col}: {json.dumps(m)}", "effect": f"{len(df)} rows"})

    casts = {}
    for col in df.columns:
        before = df[col].dtype
        if isinstance(before, pd.CategoricalDtype) or before == object:
            try:
                df[col] = pd.to_numeric(df[col].astype(str) if isinstance(before, pd.CategoricalDtype) else df[col])
            except (ValueError, TypeError) as exc:
                raise DatasetError(name, f"feature {col!r} has non-numeric values; only numeric and boolean "
                                         f"features are supported (map or drop it in the recipe)") from exc
        if pd.api.types.is_bool_dtype(df[col]) or pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype("int64")
        else:
            df[col] = df[col].astype("float64")
        if df[col].dtype != before:
            casts.setdefault(f"{before} → {df[col].dtype}", []).append(col)
    for change, cols in casts.items():
        changes.append({"action": "cast", "detail": change, "effect": f"{len(cols)} columns"})
    df[LABEL] = y.to_numpy()
    return df, changes, label_map


def classify_features(name: str, df: pd.DataFrame, declared: dict) -> list[dict]:
    """numeric | boolean per feature, with declared/inferred types and flags; booleans cast to int64 0/1."""
    out = []
    for col in df.columns[:-1]:
        n_values = df[col].nunique()
        inferred = "boolean" if n_values == 2 else "numeric"
        d = declared.get(col)
        if d == "nominal":
            if n_values > 2:
                raise DatasetError(name, f"feature {col!r} is nominal with {n_values} values; "
                                         "only numeric and boolean features are supported")
            final = "boolean"
        else:
            final = d or inferred
        flags = []
        if n_values == 1:
            flags.append("constant")
        elif d is not None and {"nominal": "boolean"}.get(d, d) != inferred:
            flags.append("type_disagreement")
        if final == "boolean":
            if not set(df[col].unique()) <= {0, 1}:
                raise DatasetError(name, f"boolean feature {col!r} has values {sorted(df[col].unique())}, "
                                         "not 0/1; add a feature_maps entry")
            df[col] = df[col].astype("int64")
        out.append({"name": col, "type": final, "declared": d, "inferred": inferred, "flags": flags})
    return out


# ---------------------------------------------------------------- commands

def recreate(name: str, recipe: dict, force_download: bool = False) -> dict:
    validate_recipe(name, recipe)
    raw = fetch_raw(name, recipe, force_download)
    rename = recipe.get("rename") or {}
    declared = {rename.get(k, k): v for k, v in declared_types(name, recipe).items()}
    df, changes, label_map = apply_recipe(name, recipe, raw)
    unknown = [i["column"] for i in recipe.get("types") or [] if i["column"] not in df.columns[:-1]]
    if unknown:
        raise DatasetError(name, f"types: column(s) {unknown} do not exist (stale recipe?)")
    declared, overrides = override_types(declared, recipe)
    changes += overrides
    features = classify_features(name, df, declared)
    checksum = write_table(name, df)

    source_checksum = recipe.get("source_checksum") or raw_checksum(name, recipe)
    retrieved = str(recipe.get("retrieved") or datetime.date.today().isoformat())
    if not recipe.get("source_checksum") or not recipe.get("retrieved"):
        print(f"{name:<16} NOTE add to data/recipes.yaml: source_checksum: {source_checksum}  retrieved: {retrieved}")
    counts = df[LABEL].value_counts().sort_index()
    entry = {
        "name": name, "source": recipe["source"], "id": recipe["id"], "url": recipe["url"],
        "retrieved": retrieved, "source_checksum": source_checksum,
        "original_shape": list(raw.shape), "changes": changes, "label_map": label_map,
        "features": features,
        "n_rows": len(df), "n_features": len(features),
        "n_numeric": sum(f["type"] == "numeric" for f in features),
        "n_boolean": sum(f["type"] == "boolean" for f in features),
        "n_classes": len(label_map), "class_counts": {str(k): int(v) for k, v in counts.items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "notes": recipe.get("notes", ""), "checksum": checksum, "code_commit": git_commit(),
    }
    old = read_corpus()["datasets"].get(name, {})
    if "difficulty" in old:
        if old.get("checksum") == checksum:
            entry["difficulty"] = old["difficulty"]
        else:
            print(f"{name:<16} NOTE table changed, its difficulty scores were removed")
    update_corpus_entry(name, entry)
    flags = [f"{f['name']}:{x}" for f in features for x in f["flags"]]
    print(f"{name:<16} recreate  ok  {len(df)}×{len(features)} (+target)  {len(label_map)} classes "
          f"{counts.tolist()}  sha256 {checksum[:10]}…  flags: {', '.join(flags) or 'none'}")
    return entry


def verify(names: list[str], recipes: dict) -> int:
    datasets, problems = read_corpus()["datasets"], []
    for name in names:
        recipe, entry, path = recipes.get(name) or {}, datasets.get(name), DATA_DIR / f"{name}.csv"
        if recipe and not (recipe.get("source_checksum") and recipe.get("retrieved")):
            problems.append(f"{name}: recipe has no source_checksum/retrieved yet")
        if not path.is_file():
            problems.append(f"{name}: no table {path.name}")
        elif entry is None:
            problems.append(f"{name}: table without a record")
        elif sha256_file(path) != entry["checksum"]:
            problems.append(f"{name}: checksum {path.name} {sha256_file(path)[:10]}… != record {entry['checksum'][:10]}…")
    for path in sorted(DATA_DIR.glob("*.csv")):
        if path.name in RECORD_FILES:
            continue
        if path.stem not in datasets and path.stem not in names:
            problems.append(f"{path.stem}: table without a record")
    for name in sorted(set(datasets) - set(recipes)):
        problems.append(f"{name}: record without a recipe")
    for p in problems:
        print(f"FAIL {p}")
    print(f"verify: {len(problems)} problem(s)")
    return len(problems)


def compare_reference(name: str, recipe: dict, reference_dir: Path) -> bool:
    """Equivalence with the reference implementation's copy; PMLB copies were min-max scaled per column."""
    ref_path = Path(reference_dir) / f"{name}.csv"
    if not ref_path.is_file():
        print(f"{name:<16} compare   no reference ({ref_path})")
        return True
    if not (DATA_DIR / f"{name}.csv").is_file():
        raise DatasetError(name, f"no table {name}.csv to compare; run `python datasets.py recreate {name}`")
    ours, ref = pd.read_csv(DATA_DIR / f"{name}.csv"), pd.read_csv(ref_path)
    notes = []
    dropped = [d["column"] for d in recipe.get("drop") or [] if d["column"] in ref.columns]
    if dropped:                            # the reference kept columns our recipe drops
        ref = ref.drop(columns=dropped)
        notes.append(f"reference without dropped {dropped}")
    if LABEL in ref.columns and sorted(pd.unique(ref[LABEL]).tolist(), key=str) != list(range(ref[LABEL].nunique())):
        codes = {v: i for i, v in enumerate(sorted(pd.unique(ref[LABEL]).tolist(), key=str))}
        ref[LABEL] = ref[LABEL].map(codes)  # the reference kept text / 1..K labels: same sorted rule as ours
        notes.append(f"reference labels mapped {codes}")
    problem = None
    if ours.shape != ref.shape:
        problem = f"shape {ours.shape} != reference {ref.shape}"
    elif list(ours.columns) != list(ref.columns):
        problem = f"columns {list(ours.columns)} != reference {list(ref.columns)}"
    elif not (ours[LABEL].to_numpy() == ref[LABEL].to_numpy()).all():
        problem = f"target differs first at row {int(np.argmax(ours[LABEL].to_numpy() != ref[LABEL].to_numpy()))}"
    else:
        for col in ours.columns[:-1]:
            x, r = ours[col].to_numpy(float), ref[col].to_numpy(float)
            if recipe["source"] == "pmlb":
                span = x.max() - x.min()
                x = (x - x.min()) / span if span else np.zeros_like(x)
                bad = ~np.isclose(x, r, rtol=0, atol=1e-12)
            else:
                bad = x != r
            if bad.any():
                i = int(np.argmax(bad))
                problem = f"column {col!r} differs first at row {i}: {x[i]!r} vs reference {r[i]!r}"
                break
    note = f"  ({'; '.join(notes)})" if notes else ""
    print(f"{name:<16} compare   {'ok' if problem is None else 'FAIL ' + problem}{note}")
    return problem is None


def measure_difficulty(name: str) -> dict:
    """Stratified 3-fold balanced accuracy of three reference models, same for every dataset (R6)."""
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    X, y, _ = load_table(name)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    models = {
        "tree_depth2": DecisionTreeClassifier(max_depth=2, random_state=0),
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "gradient_boosting": HistGradientBoostingClassifier(random_state=0, early_stopping=False),
    }
    result = {"settings": {"cv": "StratifiedKFold(n_splits=3, shuffle=True, random_state=0)",
                           "scoring": "balanced_accuracy",
                           "models": {k: repr(m) for k, m in models.items()},
                           "code_commit": git_commit(), "sklearn_version": sklearn.__version__}}
    for key, model in models.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scores = cross_val_score(model, X, y, cv=folds, scoring="balanced_accuracy")
        for w in caught:
            print(f"{name:<16} WARNING {key}: {w.message}")
        result[key] = {"mean": float(scores.mean()), "folds": [float(s) for s in scores]}
    return result


# ---------------------------------------------------------------- interface for later stages

def dataset_record(name: str) -> dict:
    entry = read_corpus()["datasets"].get(name)
    if entry is None:
        raise DatasetError(name, "no record in data/corpus.json; run `python datasets.py recreate`")
    return entry


def corpus_version() -> str:
    return read_corpus()["corpus_version"]


def load_table(name: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """(X float rows × features, y int 0..K-1 from `target`, [{name, type}]); checksum-checked."""
    entry, path = dataset_record(name), DATA_DIR / f"{name}.csv"
    if not path.is_file():
        raise DatasetError(name, f"table {path} missing; run `python datasets.py recreate {name}`")
    if sha256_file(path) != entry["checksum"]:
        raise DatasetError(name, f"table {path.name} does not match its recorded checksum")
    df = pd.read_csv(path)
    feature_info = [{"name": f["name"], "type": f["type"]} for f in entry["features"]]
    return df.drop(columns=LABEL).to_numpy(dtype=float), df[LABEL].to_numpy(dtype=np.int64), feature_info


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["recreate", "verify", "difficulty", "compare-reference"])
    p.add_argument("names", nargs="*")
    p.add_argument("--all", action="store_true", help="every dataset in data/recipes.yaml")
    p.add_argument("--force-download", action="store_true", help="ignore the raw cache (recreate)")
    p.add_argument("--reference-dir", type=Path, default=REFERENCE_DIR, help="the reference implementation data/ (compare-reference)")
    args = p.parse_args(argv)
    recipes = load_recipes()
    names = list(recipes) if args.all or (args.command == "verify" and not args.names) else args.names
    if not names:
        p.error("give dataset names or --all")
    unknown = [n for n in names if n not in recipes]
    if unknown:
        print(f"FAIL no recipe for {unknown}")
        return 1
    if args.command == "verify":
        return 1 if verify(names, recipes) else 0
    failed = 0
    for name in names:
        try:
            if args.command == "recreate":
                recreate(name, recipes[name], args.force_download)
            elif args.command == "difficulty":
                entry = dataset_record(name)
                entry["difficulty"] = measure_difficulty(name)
                update_corpus_entry(name, entry)
                d = entry["difficulty"]
                print(f"{name:<16} difficulty  tree2 {d['tree_depth2']['mean']:.3f}  logreg {d['logreg']['mean']:.3f}"
                      f"  gb {d['gradient_boosting']['mean']:.3f}")
            elif not compare_reference(name, recipes[name], args.reference_dir):
                failed += 1
        except DatasetError as exc:
            print(f"FAIL {exc}")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
