"""Preprocessing stage: z-score per run, and each dataset's grammar constant range.

Z-score (option A): each run fits mean and std of its numeric columns on its training rows
(fit_zscore) and applies them to its training and test rows (apply_zscore). Booleans are never
scaled. Nothing is stored.

Constant range: for each dataset, z-score its numeric columns on all rows (selection only), test
the signed ranges ±1 / ±2 / ±3 (constants -0.99..+0.99, -1.99..+1.99, -2.99..+2.99 in 0.01 steps)
and pick the narrowest whose coverage_min >= 0.95 (else the highest coverage_min, then the lowest
degenerate rate). Ported from the reference implementation select_zscore_constant_range.py / select_best_preprocessing.py.

Writes  data/constant_ranges.csv   one row per dataset (committed), read by the grammar stage

Commands (python preprocessing.py <command> <names...> | --all):
  ranges             choose and record the constant range
  check              stale rows (table changed) and recipes without a row
  compare-reference  winner and measurements vs the reference implementation's selection report (1e-9)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import datasets
from datasets import DatasetError

COVERAGE_TARGET = 0.95
CANDIDATES = {"pm1": 1, "pm2": 2, "pm3": 3}                    # narrowest first
MEASURES = ("coverage_min", "coverage_mean", "degenerate_rate")
COLUMNS = ("name", "table_checksum", "n_numeric", "n_boolean") \
    + tuple(f"{c}_{m}" for c in CANDIDATES for m in MEASURES) \
    + ("winner", "constant_min", "constant_max", "fallback", "coverage_target", "constant_columns", "code_commit")
REFERENCE_REPORT = datasets.ROOT.parent / "reference" / "selection_report.json"
TOLERANCE = 1e-9


# ---------------------------------------------------------------- z-score

def zscore_column(v: np.ndarray) -> np.ndarray:
    """(v - mean) / std, population std; a constant column becomes 0 (the reference implementation _zscore)."""
    mean, std = float(np.mean(v)), float(np.std(v))
    return np.zeros_like(v, dtype=float) if std == 0 else (v - mean) / std


def fit_zscore(X_train: np.ndarray, feature_info: list[dict]) -> dict:
    """Per-column mean and std of the numeric columns, from a run's training rows only."""
    if X_train.shape[1] != len(feature_info):
        raise DatasetError("zscore", f"{X_train.shape[1]} columns but {len(feature_info)} features")
    numeric = np.array([f["type"] == "numeric" for f in feature_info])
    cols = [np.ascontiguousarray(X_train[:, j], dtype=float) for j in range(X_train.shape[1])]
    return {"numeric": numeric,                     # per column, exactly as zscore_column computes it
            "mean": np.array([float(np.mean(c)) for c in cols]), "std": np.array([float(np.std(c)) for c in cols])}


def apply_zscore(X: np.ndarray, params: dict) -> np.ndarray:
    """Z-score the numeric columns of any rows with a run's training parameters; booleans unchanged."""
    if X.shape[1] != len(params["numeric"]):
        raise DatasetError("zscore", f"{X.shape[1]} columns but parameters for {len(params['numeric'])}")
    out = np.array(X, dtype=float)
    for j in np.flatnonzero(params["numeric"]):
        std = params["std"][j]
        out[:, j] = 0.0 if std == 0 else (out[:, j] - params["mean"][j]) / std
    return out


# ---------------------------------------------------------------- constant range

def candidate_constants(k: int) -> list[float]:
    """Every value the signed <constant> rule with integer part 0..k-1 produces."""
    mags = [round(i / 100, 2) for i in range(100 * k)]
    return sorted({-m for m in mags} | set(mags))


def pick_winner(tested: list[dict], target: float = COVERAGE_TARGET) -> dict:
    """tested is ordered narrowest -> widest (the reference implementation pick_winner)."""
    for t in tested:
        if t["coverage_min"] >= target:
            return t
    return max(tested, key=lambda t: (t["coverage_min"], -t["degenerate_rate"]))


def measure(name: str) -> dict:
    X, _, info = datasets.load_table(name)
    numeric = [j for j, f in enumerate(info) if f["type"] == "numeric"]
    row = {"name": name, "table_checksum": datasets.dataset_record(name)["checksum"],
           "n_numeric": len(numeric), "n_boolean": len(info) - len(numeric),
           "coverage_target": COVERAGE_TARGET, "code_commit": datasets.git_commit()}
    if not numeric:
        return {**row, "winner": "none", "fallback": "no", "constant_columns": ""}
    columns = {info[j]["name"]: np.sort(zscore_column(X[:, j])) for j in numeric}
    n = len(X)
    tested = []
    for cand, k in CANDIDATES.items():
        consts = np.array(candidate_constants(k))
        lo, hi = consts[0], consts[-1]
        shares, dead = [], 0
        for v in columns.values():
            shares.append(float(np.mean((v >= lo) & (v <= hi))))
            frac = np.searchsorted(v, consts, side="right") / n          # = mean(v <= c), same float
            dead += int(np.sum((frac < 0.01) | (frac > 0.99)))
        tested.append({"candidate": cand, "lo": lo, "hi": hi, "coverage_min": min(shares),
                       "coverage_mean": float(np.mean(shares)), "degenerate_rate": dead / (len(columns) * len(consts))})
    win = pick_winner(tested)
    for t in tested:
        for m in MEASURES:
            row[f"{t['candidate']}_{m}"] = t[m]
    constant = [c for c, v in columns.items() if v[0] == v[-1]]
    return {**row, "winner": win["candidate"], "constant_min": win["lo"], "constant_max": win["hi"],
            "fallback": "no" if win["coverage_min"] >= COVERAGE_TARGET else "yes",
            "constant_columns": ";".join(constant)}


def split_coverage(name: str) -> float | None:
    """Smallest, over numeric columns (z-scored on all rows), of: the number of different non-trivial
    splits `column <= c` the range's constants produce / the most they could produce
    (min(distinct values - 1, number of constants)). Near 1 for a continuous column the constants
    span; for a low-cardinality column, the share of its possible splits some constant reaches.
    Complements coverage_min, which misjudges low-cardinality columns."""
    rng = constant_range(name)
    if rng is None:
        return None
    k = round(rng[1] + 0.01)
    consts = np.array(candidate_constants(k))
    X, _, info = datasets.load_table(name)
    shares = []
    for j, f in enumerate(info):
        if f["type"] != "numeric":
            continue
        u = np.unique(zscore_column(X[:, j]))
        if len(u) < 2:
            continue
        cut = np.unique(np.searchsorted(u, consts, side="right"))    # values <= c, per constant
        splits = int(np.sum((cut > 0) & (cut < len(u))))               # neither all nor none
        shares.append(splits / min(len(u) - 1, len(consts)))
    return min(shares) if shares else None


def winners_by_target(name: str, targets=(0.90, 0.92, 0.93, 0.94, 0.95, 0.96)) -> dict:
    """The winning range at other coverage targets, from the recorded measures (sensitivity)."""
    r = load_ranges().set_index("name").loc[name]
    if r["winner"] == "none":
        return {t: "none" for t in targets}
    tested = [{"candidate": c, **{m: float(r[f"{c}_{m}"]) for m in MEASURES}} for c in CANDIDATES]
    return {t: pick_winner(tested, t)["candidate"] for t in targets}


# ---------------------------------------------------------------- record

def record_path() -> Path:
    return datasets.DATA_DIR / "constant_ranges.csv"


def load_ranges() -> pd.DataFrame:
    if not record_path().is_file():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(record_path(), dtype=str, keep_default_na=False)
    if tuple(df.columns) != COLUMNS:
        raise DatasetError("constant_ranges", f"{record_path()} columns differ from {COLUMNS}")
    return df


def write_row(row: dict) -> None:
    """Add or replace one dataset's row; every other row is kept exactly as read."""
    datasets.upsert_csv_row(record_path(), COLUMNS, row)


def constant_range(name: str) -> tuple[float, float] | None:
    """The grammar stage's input: (lo, hi), or None for a dataset without numeric features."""
    df = load_ranges()
    hit = df[df["name"] == name]
    if hit.empty:
        raise DatasetError(name, "no constant range; run `python preprocessing.py ranges`")
    r = hit.iloc[0]
    if r["table_checksum"] != datasets.dataset_record(name)["checksum"]:
        raise DatasetError(name, "constant range is stale (table changed); rerun `ranges`")
    return None if r["winner"] == "none" else (float(r["constant_min"]), float(r["constant_max"]))


def check() -> int:
    df, problems = load_ranges(), []
    for _, r in df.iterrows():
        try:
            if r["table_checksum"] != datasets.dataset_record(r["name"])["checksum"]:
                problems.append(f"{r['name']}: stale, table changed since its range was chosen")
        except DatasetError:
            problems.append(f"{r['name']}: row without a dataset record")
    problems += [f"{n}: recipe without a constant range" for n in datasets.load_recipes() if n not in set(df["name"])]
    for p in problems:
        print(f"FAIL {p}")
    print(f"check: {len(problems)} problem(s)")
    return len(problems)


def compare_reference(name: str, report: Path = REFERENCE_REPORT) -> bool:
    """same winner and measurements as the reference implementation's selection report."""
    rows = {r.get("dataset_file", "")[:-4]: r for r in json.loads(Path(report).read_text())}
    ref = rows.get(name)
    tested = {c["constant_style"].replace("signed_", ""): c for c in (ref or {}).get("combos_tested", [])
              if str(c.get("constant_style", "")).startswith("signed_pm")}
    if not tested:
        print(f"{name:<24} compare   no reference")
        return True
    ours = load_ranges().set_index("name").loc[name]
    problems = []
    if ours["winner"] != ref["winner"]["constant_style"].replace("signed_", ""):
        problems.append(f"winner {ours['winner']} vs {ref['winner']['constant_style']}")
    for cand in CANDIDATES:
        for m in MEASURES:
            a, b = float(ours[f"{cand}_{m}"]), float(tested[cand][m])
            if abs(a - b) > TOLERANCE:
                problems.append(f"{cand} {m} {a!r} vs {b!r}")
    print(f"{name:<24} compare   {'ok' if not problems else 'FAIL ' + '; '.join(problems)}")
    return not problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["ranges", "check", "compare-reference"])
    p.add_argument("names", nargs="*")
    p.add_argument("--all", action="store_true")
    args = p.parse_args(argv)
    if args.command == "check":
        return 1 if check() else 0
    names = list(datasets.load_recipes()) if args.all else args.names
    if not names:
        p.error("give dataset names or --all")
    failed = 0
    for name in names:
        try:
            if args.command == "ranges":
                row = measure(name)
                write_row(row)
                detail = "no numeric features" if row["winner"] == "none" else (
                    f"{row['winner']}  [" + ", ".join(f"{c}: cov_min {row[f'{c}_coverage_min']:.3f} "
                                                     f"deg {row[f'{c}_degenerate_rate']:.3f}" for c in CANDIDATES) + "]"
                    + ("  FALLBACK" if row["fallback"] == "yes" else ""))
                print(f"{name:<24} range     {detail}")
            elif not compare_reference(name):
                failed += 1
        except DatasetError as exc:
            print(f"FAIL {exc}")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
