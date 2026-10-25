"""PyCol dataset complexity: import DataComplexity's results, compute new ones the same way.

Reads   ../DataComplexity/results/datasets_complexity_summary.csv   reference rows (import only)
        ../reference/data/<name>.csv                                the reference's input tables
        data/<name>.csv, data/corpus.json                           001 tables and records
Writes  data/complexity.csv                                         one row per dataset (committed)

Commands (python complexity.py <command> ...):
  import              copy the 126 reference rows unchanged, with their origin
  compute <names>     PyCol metrics on the min-max scaling of each table; saved after every metric
  check               stale rows (table changed) and recipes without a row
  compare-reference   the ported code on the reference's inputs must reproduce its values (1e-12)

Ported from ../DataComplexity (complexity_core.py, pycol_heom.py, commit 0d9ff30), keeping its
method: min-max scaled input, float32 features, HEOM distance, nanmean of multi-valued metrics.
Differences: every metric is seeded (N4 and MRCA are random); above MAX_ROWS_DISTANCE rows only
the metrics without a distance matrix are computed. Decisions: docs/decisions/002-dataset-complexity.md.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import datasets
from datasets import DatasetError

ROOT = datasets.ROOT
REFERENCE_CSV = ROOT.parent / "DataComplexity" / "results" / "datasets_complexity_summary.csv"
REFERENCE_INPUT_DIR = ROOT.parent / "reference" / "data"

METRICS = ("F1", "F1v", "F2", "F3", "F4", "input_noise", "R_value", "deg_overlap", "N3", "SI", "N4",
           "kDN", "D3_value", "CM", "N1", "T1", "Clust", "ONB", "LSC", "DBC", "N2", "NSG", "ICSV",
           "MRCA", "C1", "C2", "purity", "neighbourhood_separability", "borderline")
DISTANCE_FREE = frozenset({"F1", "F1v", "F2", "F3", "F4", "input_noise", "purity"})
NEED_UNNORM = frozenset({"T1", "NSG", "ICSV"})
RANDOM_METRICS = frozenset({"N4", "MRCA"})
PRESETS = {
    "cheap_minimal": tuple(m for m in METRICS if m in DISTANCE_FREE),
    "cheap": tuple(m for m in METRICS if m not in NEED_UNNORM | {"ONB", "DBC"}),
    "standard": tuple(m for m in METRICS if m not in NEED_UNNORM),
    "all": METRICS,
}
MAX_ROWS_DISTANCE = 40_000
SEED = 0
TOLERANCE = 1e-12
COLUMNS = ("name", "origin", "source_commit", "computed_at", "pycol_version", "preset", "input",
           "input_checksum", "input_scaling", "matches_table", "table_checksum", "n_rows", "n_features",
           "seed", "max_rows_distance", "skipped", "failed", "seconds", "notes") \
    + tuple(f"pycol_{m}" for m in METRICS)
SOURCE_COLUMNS = ("dataset_file", "n_rows_used", "n_features_after_encoding", "pycol_metrics_preset",
                  "pycol_metrics_failed") + tuple(f"pycol_{m}" for m in METRICS)


# ---------------------------------------------------------------- the record

def record_path() -> Path:
    return datasets.DATA_DIR / "complexity.csv"


def load_complexity() -> pd.DataFrame:
    """The whole record as text (values are never re-formatted on rewrite)."""
    if not record_path().is_file():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(record_path(), dtype=str, keep_default_na=False)
    if tuple(df.columns) != COLUMNS:
        raise DatasetError("complexity", f"{record_path()} columns differ from {COLUMNS}")
    return df


def write_rows(rows: list[dict]) -> None:
    """Add or replace the rows with these names; every other row is kept exactly as read ()."""
    for row in rows:
        unknown = set(row) - set(COLUMNS)
        if unknown:
            raise DatasetError(row.get("name", "?"), f"unknown record column(s) {sorted(unknown)}")
    new = {r["name"]: {c: _text(r.get(c)) for c in COLUMNS} for r in rows}
    df = load_complexity()
    df = df[~df["name"].isin(new)]
    df = pd.concat([df, pd.DataFrame(list(new.values()), columns=COLUMNS)], ignore_index=True)
    df = df.sort_values("name", kind="stable")
    tmp = record_path().with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(record_path())


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def complexity_row(name: str) -> dict:
    df = load_complexity()
    hit = df[df["name"] == name]
    if hit.empty:
        raise DatasetError(name, "no row in data/complexity.csv")
    return hit.iloc[0].to_dict()


# ---------------------------------------------------------------- import (US1)

def source_provenance(source: Path) -> tuple[str, str]:
    """(commit, commit date) of the reference CSV in its own git repository."""
    def git(*args):
        return subprocess.run(["git", "-C", str(source.parent), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        return git("rev-parse", "HEAD"), git("log", "-1", "--format=%cs", "--", source.name)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise DatasetError("import", f"{source} is not in a git repository; can't record its commit") from exc


def import_reference(source: Path = REFERENCE_CSV) -> int:
    source = Path(source)
    if not source.is_file():
        raise DatasetError("import", f"reference CSV {source} not found")
    src = pd.read_csv(source, dtype=str, keep_default_na=False)
    missing = [c for c in SOURCE_COLUMNS if c not in src.columns]
    if missing:
        raise DatasetError("import", f"source misses column(s) {missing}")
    repeated = sorted(set(src["dataset_file"][src["dataset_file"].duplicated()]))
    if repeated:
        raise DatasetError("import", f"dataset(s) repeated in the source: {repeated}")
    commit, date = source_provenance(source)
    recipes = datasets.load_recipes()
    existing = load_complexity()
    computed = set(existing.loc[existing["origin"] == "computed", "name"])
    rows, matching = [], 0
    for _, s in src.iterrows():
        name = Path(s["dataset_file"]).stem
        if name in computed:
            print(f"{name:<24} import    kept the computed row")
            continue
        ref_file = REFERENCE_INPUT_DIR / s["dataset_file"]
        matches, table_checksum = "no recipe", ""
        if name in recipes:
            matches = "no"
            try:
                if ref_file.is_file() and datasets.compare_reference(name, recipes[name], REFERENCE_INPUT_DIR):
                    matches, table_checksum = "yes", datasets.dataset_record(name)["checksum"]
            except DatasetError as exc:
                print(f"{name:<24} import    NOTE not linked: {exc}")
        matching += matches == "yes"
        rows.append({
            "name": name, "origin": "DataComplexity", "source_commit": commit, "computed_at": date,
            "pycol_version": "1.0.4", "preset": s["pycol_metrics_preset"] or "unknown",
            "input": f"reference/data/{s['dataset_file']}",
            "input_checksum": datasets.sha256_file(ref_file) if ref_file.is_file() else "",
            "input_scaling": "minmax", "matches_table": matches, "table_checksum": table_checksum,
            "n_rows": s["n_rows_used"], "n_features": s["n_features_after_encoding"],
            "failed": s["pycol_metrics_failed"], "notes": "N4, MRCA random (unseeded in source)",
            **{f"pycol_{m}": s[f"pycol_{m}"] for m in METRICS},
        })
    write_rows(rows)
    print(f"imported {len(rows)} rows ({matching} match our tables)")
    return len(rows)


# ---------------------------------------------------------------- HEOM and PyCol (ported)

def _available_bytes() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def _heom_rows(args) -> tuple[int, np.ndarray, np.ndarray | None]:
    """HEOM rows i_start..i_end for numeric features (pycol_heom._heom_row_sq, missing values removed)."""
    i_start, i_end, x, span, unnorm = args
    n = x.shape[0]
    d = np.zeros((i_end - i_start, n))
    u = np.zeros((i_end - i_start, n)) if unnorm else None
    for r, i in enumerate(range(i_start, i_end)):
        dist_sq, unnorm_sq = np.zeros(n), np.zeros(n)
        for k in range(x.shape[1]):
            diff = np.abs(x[:, k] - x[i, k])
            dist_sq += diff ** 2 if span[k] == 0.0 else (diff / span[k]) ** 2
            if unnorm:
                unnorm_sq += diff ** 2
        dist_sq[i] = 0.0
        d[r] = np.sqrt(dist_sq)
        if unnorm:
            unnorm_sq[i] = 0.0
            u[r] = np.sqrt(unnorm_sq)
    return i_start, d, u


def heom_matrices(name: str, x: np.ndarray, *, unnorm: bool, n_jobs: int = 1):
    """PyCol's HEOM matrices (normalized, optionally unnormalized), as DataComplexity built them."""
    n = x.shape[0]
    needed = n * n * 8 * (2 if unnorm else 1)
    if needed > 0.7 * _available_bytes():
        raise DatasetError(name, f"distance matrix needs {needed / 1e9:.1f} GB, more than 70% of available "
                                 f"memory ({_available_bytes() / 1e9:.1f} GB); preset cheap_minimal needs none")
    span = (x.max(axis=0) - x.min(axis=0)).astype(float)
    chunk = max(1, -(-n // max(1, n_jobs)))
    tasks = [(s, min(s + chunk, n), x, span, unnorm) for s in range(0, n, chunk)]
    dist = np.zeros((n, n))
    und = np.zeros((n, n)) if unnorm else None
    if n_jobs <= 1:
        results = map(_heom_rows, tasks)
    else:
        with ProcessPoolExecutor(n_jobs) as pool:
            results = list(pool.map(_heom_rows, tasks))
    for s, d, u in results:
        dist[s:s + len(d)] = d
        if unnorm:
            und[s:s + len(u)] = u
    return dist, und


def _pycol_shell(x: np.ndarray, y: np.ndarray, dist: np.ndarray, unnorm: np.ndarray):
    """A PyCol Complexity object built from arrays (complexity_core._init_pycol_complexity_shell)."""
    from pycol_complexity import complexity as pycol
    comp = pycol.Complexity.__new__(pycol.Complexity)
    comp.X, comp.y = np.array(x), np.array(y)
    comp.classes = np.unique(comp.y)
    comp.meta = comp.is_categorical(comp.X)
    comp.dist_matrix, comp.unnorm_dist_matrix = dist, unnorm
    comp.class_count = np.array([float((comp.y == c).sum()) for c in comp.classes])
    comp.class_inxs = [np.where(comp.y == c)[0] for c in comp.classes]
    comp.sphere_inst_count_T1, comp.sphere_tuple_ONB = [], []
    comp.metrics = {"feature": {}, "struct": {}, "instance": {}, "multi": {}}
    return comp


def compute_metrics(name: str, x: np.ndarray, y: np.ndarray, metrics, *, n_jobs: int = 1,
                    done: dict | None = None, on_metric=None) -> tuple[dict, dict]:
    """PyCol values for `metrics` (seeded per metric); `on_metric(metric, value, error)` after each."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    x = np.asarray(x, dtype=np.float32).astype(float)       # DataComplexity's prepare_xy: float32
    y = pd.factorize(np.asarray(y), sort=True)[0].astype(int)
    values, failed = dict(done or {}), {}
    todo = [m for m in metrics if m not in values]
    empty = np.zeros((0, 0))
    groups = [[m for m in todo if m in DISTANCE_FREE], [m for m in todo if m not in DISTANCE_FREE]]
    for group, needs_matrix in zip(groups, (False, True)):
        if not group:
            continue
        if needs_matrix:
            dist, und = heom_matrices(name, x, unnorm=bool(NEED_UNNORM & set(group)), n_jobs=n_jobs)
            comp = _pycol_shell(x, y, dist, und if und is not None else empty)
        else:
            comp = _pycol_shell(x, y, empty, empty)
        for metric in group:
            np.random.seed(SEED)
            random.seed(SEED)
            error = None
            try:
                v = getattr(comp, metric)()
                arr = np.asarray(v, dtype=float)
                values[metric] = float(arr) if arr.ndim == 0 else (float(np.nanmean(arr)) if arr.size else None)
            except Exception as exc:                          # recorded, never a silent value
                values[metric], error = None, f"{type(exc).__name__}: {exc}"
                failed[metric] = error
            if on_metric:
                on_metric(metric, values[metric], error)
    return values, failed


# ---------------------------------------------------------------- compute (US2), check, compare

def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(axis=0), x.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return np.where(hi > lo, (x - lo) / span, 0.0)


def compute(name: str, preset: str = "cheap", *, n_jobs: int = 1, replace: bool = False) -> dict:
    if preset not in PRESETS:
        raise DatasetError(name, f"unknown preset {preset!r}; choose one of {list(PRESETS)}")
    X, y, _ = datasets.load_table(name)
    checksum = datasets.dataset_record(name)["checksum"]
    df = load_complexity()
    old = df[df["name"] == name].iloc[0].to_dict() if (df["name"] == name).any() else None
    done, seconds = {}, 0.0
    if old and not replace:
        if old["origin"] != "computed":
            raise DatasetError(name, "an imported row exists; use --replace to compute over it")
        if old["preset"] != preset or old["input_checksum"] != checksum:
            raise DatasetError(name, f"a computed row exists with preset {old['preset']} and checksum "
                                     f"{old['input_checksum'][:10]}…; use --replace")
        done = {m: float(old[f"pycol_{m}"]) for m in METRICS if old[f"pycol_{m}"] != ""}
        seconds = float(old["seconds"] or 0)
    metrics = list(PRESETS[preset])
    skipped = ""
    if len(X) > MAX_ROWS_DISTANCE:
        too_big = [m for m in metrics if m not in DISTANCE_FREE]
        metrics = [m for m in metrics if m in DISTANCE_FREE]
        skipped = f"{','.join(too_big)}: n_rows > {MAX_ROWS_DISTANCE}" if too_big else ""
    row = {"name": name, "origin": "computed", "source_commit": datasets.git_commit(),
           "computed_at": datetime.date.today().isoformat(),
           "pycol_version": importlib.metadata.version("pycol-complexity"), "preset": preset,
           "input": f"data/{name}.csv", "input_checksum": checksum, "input_scaling": "minmax",
           "matches_table": "yes", "table_checksum": checksum, "n_rows": len(X), "n_features": X.shape[1],
           "seed": SEED, "max_rows_distance": MAX_ROWS_DISTANCE, "skipped": skipped,
           "failed": old["failed"] if old and not replace else "", "seconds": seconds, "notes": "",
           **{f"pycol_{m}": v for m, v in done.items()}}
    failures = dict(item.split(": ", 1) for item in row["failed"].split("; ") if item)
    start, last = time.time(), [time.time()]

    def save(metric, value, error):
        row[f"pycol_{metric}"] = value
        if error:
            failures[metric] = error
        row["failed"] = "; ".join(f"{m}: {e}" for m, e in failures.items())
        row["seconds"] = round(seconds + time.time() - start, 1)
        write_rows([row])
        shown = f"FAILED {error}" if error else ("empty" if value is None else f"{value:.6g}")
        print(f"{name:<24} {metric:<28} {shown}  ({time.time() - last[0]:.1f} s)")
        last[0] = time.time()

    write_rows([row])
    if skipped:
        print(f"{name:<24} skipped {skipped}")
    compute_metrics(name, minmax(X), y, metrics, n_jobs=n_jobs, done=done, on_metric=save)
    row["seconds"] = round(seconds + time.time() - start, 1)
    write_rows([row])
    print(f"{name:<24} compute   done  {len(metrics)} metrics ({len(done)} resumed), {len(failures)} failed, "
          f"{row['seconds']} s")
    return row


def check() -> int:
    df, recipes, problems = load_complexity(), datasets.load_recipes(), []
    for _, r in df.iterrows():
        current = r["table_checksum"]          # our 001 checksum: computed rows, and linked imported rows
        if not current:
            continue
        try:
            now = datasets.dataset_record(r["name"])["checksum"]
        except DatasetError:
            problems.append(f"{r['name']}: row refers to a table with no record")
            continue
        if now != current:
            problems.append(f"{r['name']}: stale, computed on {current[:10]}…, table is now {now[:10]}…")
    for name in recipes:
        if name not in set(df["name"]):
            problems.append(f"{name}: recipe without a complexity row")
        elif (df.loc[df["name"] == name, "matches_table"] != "yes").all():
            problems.append(f"{name}: only a row for a different table (matches_table != yes)")
    for p in problems:
        print(f"FAIL {p}")
    print(f"check: {len(problems)} problem(s)")
    return len(problems)


def compare_reference(name: str) -> bool:
    """the ported code on the reference's own input reproduces its non-random values."""
    ref = complexity_row(name)
    if ref["origin"] != "DataComplexity" or ref["preset"] not in PRESETS:
        raise DatasetError(name, f"needs an imported row with a known preset, got {ref['origin']}/{ref['preset']}")
    path = REFERENCE_INPUT_DIR / f"{name}.csv"
    if not path.is_file():
        raise DatasetError(name, f"reference input {path} not found")
    df = pd.read_csv(path)
    ours, _ = compute_metrics(name, df.drop(columns=datasets.LABEL).to_numpy(), df[datasets.LABEL].to_numpy(),
                              PRESETS[ref["preset"]])
    ok = True
    for m in PRESETS[ref["preset"]]:
        theirs = float(ref[f"pycol_{m}"]) if ref[f"pycol_{m}"] != "" else None
        if m in RANDOM_METRICS:
            status = f"random (not compared): {ours[m]} vs {theirs}"
        elif ours[m] is None or theirs is None:
            status = "ok" if ours[m] is None and theirs is None else f"DIFF {ours[m]} vs {theirs}"
        else:
            status = "ok" if abs(ours[m] - theirs) <= TOLERANCE else f"DIFF {ours[m]!r} vs {theirs!r}"
        ok &= status == "ok" or status.startswith("random")
        print(f"{name:<24} {m:<28} {status}")
    print(f"{name:<24} compare-reference {'ok' if ok else 'FAIL'}")
    return ok


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import")
    imp.add_argument("--source", type=Path, default=REFERENCE_CSV)
    comp = sub.add_parser("compute")
    comp.add_argument("names", nargs="+")
    comp.add_argument("--preset", default="cheap", choices=list(PRESETS))
    comp.add_argument("--n-jobs", type=int, default=1)
    comp.add_argument("--replace", action="store_true")
    sub.add_parser("check")
    cmp_ = sub.add_parser("compare-reference")
    cmp_.add_argument("names", nargs="+")
    args = p.parse_args(argv)
    try:
        if args.command == "import":
            import_reference(args.source)
            return 0
        if args.command == "check":
            return 1 if check() else 0
    except DatasetError as exc:
        print(f"FAIL {exc}")
        return 1
    failed = 0
    for name in args.names:
        try:
            if args.command == "compute":
                compute(name, args.preset, n_jobs=args.n_jobs, replace=args.replace)
            elif not compare_reference(name):
                failed += 1
        except DatasetError as exc:
            print(f"FAIL {exc}")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
