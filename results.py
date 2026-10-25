"""Results store and postprocessing: runs -> tables for the paper and the app.

Writes (all derived, rebuilt from results/ and the records; git-ignored):
  results/store/runs.csv         one row per run (every accepted run under results/ge/ and results/gp/)
  results/store/generations.csv  one row per run and generation (every recorded column, 014)
  results/store/metrics.csv      per (setup, dataset, sampler, fraction): measurement.metrics + paired tests
  results/store/datasets.csv     per recipe: the properties used to choose datasets

Commands: python results.py integrate | metrics | properties | all
Rules: a dataset's rows are rebuilt as a whole; one dataset's runs must all be on one table
checksum; metrics only from measurement.py; paired Wilcoxon tests on
paired seeds (option 1), pairs checked identical first. Accuracy equivalence:
paired Wilcoxon TOST within +-EQUIVALENCE_MARGIN (p_equivalence; small p = shown comparable) and the
Bayesian correlated t-test with the same margin as rope (bayes_worse / bayes_rope / bayes_better).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import analysis
import complexity
import datasets
import ge
import grammar
import measurement
import preprocessing

ROOT = datasets.ROOT
STORE = ROOT / "results" / "store"
MIN_PAIRS_FOR_TEST = 6
# equivalence margin for test accuracy: FEC is "comparable" to
# the baseline when TOST shows the paired difference within +-1 percentage point
EQUIVALENCE_MARGIN = 0.01
# difficulty class: gradient-boosting 3-fold balanced
# accuracy (001 difficulty) below 0.80 = hard, below 0.88 = medium, otherwise simple
DIFFICULTY_CUTS = (0.80, 0.88)
BEST_COLUMNS = ("best_depth", "best_terminals", "best_operators", "best_features_used", "best_pheno_len")


def run_row(r: dict) -> dict:
    f = r.get("fec") or {}
    m = r["measures"]
    return {"algorithm": r.get("algorithm", "GE"), "setup": r["setup"], "dataset": r["dataset"], "mode": r["mode"],
            "sampler": f.get("sampler", ""),
            "fraction": f.get("fraction", np.nan), "seed": r["seed"],
            "test_accuracy": 1 - r["best_test_error"], "train_accuracy": 1 - r["best_train_error"],
            "eval_time_sec": m["eval_time_sec"], "n_evals": m["n_evals"], "effort": m["effort"],
            "n_full_evals": m["n_full_evals"], "generation_time_sec": r["generation_time_sec"],
            "wall_time_sec": r["wall_time_sec"], "sample_size": f.get("sample_size", np.nan),
            "sampling_time_sec": f.get("sampling_time_sec", np.nan), "hits": f.get("hits", np.nan),
            "misses": f.get("misses", np.nan), "same_phenotype_hits": f.get("same_phenotype_hits", np.nan),
            "fake_hits": f.get("fake_hits", np.nan), "hit_rate": f.get("hit_rate", np.nan),
            "cache_size": f.get("cache_size", np.nan), "best_nodes": r["generations"][-1]["best_nodes"],
            **{k: r["generations"][-1].get(k, np.nan) for k in BEST_COLUMNS},     # 014; older runs: blank
            "best_phenotype": r["best_phenotype"], "table_checksum": r["table_checksum"],
            "corpus_version": r["corpus_version"], "grammar_checksum": r["grammar_checksum"],
            "code_commit": r["code_commit"]}


def load_runs(setup: str | None = None, dataset: str | None = None) -> list[dict]:
    """Every stored run except the phase-timed ones (setup phase_timing*), which have their own table and must
    never enter runs.csv, generations.csv or the metrics."""
    pattern = f"{setup or '*'}/{dataset or '*'}/*/seed*.json"
    return [json.loads(p.read_text()) for algo in ("ge", "gp") for p in sorted((ROOT / "results" / algo).glob(pattern))
            if not p.relative_to(ROOT / "results" / algo).parts[0].startswith("phase_timing")]


def integrate() -> pd.DataFrame:
    """Rebuild runs.csv and generations.csv from every accepted run (per dataset, checksum-guarded)."""
    runs = load_runs()
    rows, gens = [], []
    for r in runs:
        rows.append(run_row(r))
        key = {"algorithm": r.get("algorithm", "GE"), "setup": r["setup"], "dataset": r["dataset"], "mode": r["mode"],
               "sampler": (r.get("fec") or {}).get("sampler", ""), "fraction": (r.get("fec") or {}).get("fraction", np.nan),
               "seed": r["seed"]}
        gens += [{**key, **g} for g in r["generations"]]
    df = pd.DataFrame(rows)
    if df.empty:
        raise datasets.DatasetError("results", "no runs under results/ge/; fetch an experiment first")
    mixed = df.groupby("dataset")["table_checksum"].nunique()
    if (mixed > 1).any():
        raise datasets.DatasetError("results", f"runs on two tables for {list(mixed[mixed > 1].index)}: "
                                               "remove the stale runs")
    STORE.mkdir(parents=True, exist_ok=True)
    df.to_csv(STORE / "runs.csv", index=False)
    pd.DataFrame(gens).to_csv(STORE / "generations.csv", index=False)
    print(f"integrate: {len(df)} runs, {df['dataset'].nunique()} dataset(s), {df['setup'].nunique()} setup(s)")
    import phase_timing                                        # phase-timed runs, own table
    pt = phase_timing.table()
    if not pt.empty:
        pt.to_csv(STORE / "phase_timing.csv", index=False)
        print(f"integrate: {len(pt)} phase-timed run(s) -> phase_timing.csv")
    return df


def _paired_p(a: list[float], b: list[float]) -> float:
    if len(a) < MIN_PAIRS_FOR_TEST or np.allclose(a, b):
        return np.nan
    return float(wilcoxon(a, b).pvalue)


def _tost_p(a: list[float], b: list[float], margin: float = EQUIVALENCE_MARGIN) -> float:
    """Paired Wilcoxon TOST: H0 = |median(b - a)| >= margin; p = the larger of the two one-sided p-values."""
    if len(a) < MIN_PAIRS_FOR_TEST:
        return np.nan
    d = np.asarray(b, float) - np.asarray(a, float)
    return float(max(wilcoxon(d + margin, alternative="greater").pvalue,
                     wilcoxon(d - margin, alternative="less").pvalue))


def _sd(values) -> float:
    v = [x for x in values if np.isfinite(x)]
    return float(np.std(v, ddof=1)) if len(v) > 1 else np.nan


def spread(b: list[dict], t: list[dict], d: list[dict]) -> dict:
    """sd over the seeds (ddof 1) beside each metric: of the per-seed ratios baseline k / FEC k for the
    speedups (the metric itself is a ratio of means), of the paired differences for dAcc, over runs otherwise."""
    val = lambda r, k: r[k] if k in r else r["measures"][k]
    ratio = lambda key, fec=val: [val(x, key) / fec(y, key) if fec(y, key) else np.nan for x, y in zip(b, t)]
    acc = lambda r: 1 - r["best_test_error"]
    return {"S_T_sd": _sd(ratio("eval_time_sec")), "S_E_sd": _sd(ratio("effort")),
            "S_gen_sd": _sd(ratio("generation_time_sec")),
            "S_wall_sd": _sd(ratio("wall_time_sec", lambda r, k: measurement.wall_time(r))),
            "dAcc_sd": _sd([acc(y) - acc(x) for x, y in zip(b, t)]),
            "baseline_accuracy_sd": _sd([acc(x) for x in b]), "fec_accuracy_sd": _sd([acc(y) for y in t]),
            "hit_rate_sd": _sd([y["fec"]["hits"] / (y["fec"]["hits"] + y["fec"]["misses"])
                                if y["fec"]["hits"] + y["fec"]["misses"] else np.nan for y in t]),
            "fake_hit_ratio_sd": _sd([r["fec"]["fake_hits"] / r["fec"]["hits"] if r["fec"]["hits"] else np.nan
                                      for r in d])}


def metrics_table() -> pd.DataFrame:
    """One row per (setup, dataset, sampler, fraction), from the runs on disk."""
    by: dict[tuple, dict[str, list]] = {}
    for r in load_runs():
        f = r.get("fec") or {}
        key = (r.get("algorithm", "GE"), r["setup"], r["dataset"])
        by.setdefault(key, {}).setdefault((r["mode"], f.get("sampler", ""), f.get("fraction")), []).append(r)
    rows = []
    for (algorithm, setup, name), groups in sorted(by.items()):
        base = groups.get(("baseline", "", None), [])
        for (mode, sampler, fraction), timing in sorted(groups.items(), key=lambda kv: str(kv[0])):
            if mode != "fec":
                continue
            diag = groups.get(("fec_diagnostic", sampler, fraction), [])
            seeds = {r["seed"] for r in base} & {r["seed"] for r in timing} & {r["seed"] for r in diag}
            pick = lambda runs: sorted((r for r in runs if r["seed"] in seeds), key=lambda r: r["seed"])
            b, t, d = pick(base), pick(timing), pick(diag)
            if not seeds:
                continue
            m = measurement.metrics(b, t, d)
            acc_b, acc_t = [1 - r["best_test_error"] for r in b], [1 - r["best_test_error"] for r in t]
            bayes = analysis.bayes_correlated_t(acc_b, acc_t, EQUIVALENCE_MARGIN, np.mean([r["n_test"] for r in b]),
                                                np.mean([r["n_train"] for r in b]))
            hits = sum(r["fec"]["hits"] for r in t)
            rows.append({"algorithm": algorithm, "setup": setup, "dataset": name, "sampler": sampler, "fraction": fraction,
                         "n_runs": m["n_runs"], "S_T": m["S_T"], "dAcc": m["dAcc"], "S_E": m["S_E"],
                         "S_gen": m["S_gen"], "S_wall": m["S_wall"], "hit_rate": m["hit_rate"],
                         "fake_hit_ratio": m["fake_hit_ratio"],
                         "duplicate_hit_share": sum(r["fec"]["same_phenotype_hits"] for r in t) / hits if hits else np.nan,
                         "sampling_time_sec": m["sampling_time_sec"],
                         "p_eval_time": _paired_p([r["measures"]["eval_time_sec"] for r in b],
                                                  [r["measures"]["eval_time_sec"] for r in t]),
                         "p_accuracy": _paired_p(acc_b, acc_t), "p_equivalence": _tost_p(acc_b, acc_t),
                         "bayes_worse": bayes[0], "bayes_rope": bayes[1], "bayes_better": bayes[2],
                         "baseline_accuracy": float(np.mean(acc_b)), "fec_accuracy": float(np.mean(acc_t)),
                         **spread(b, t, d)})
    df = pd.DataFrame(rows)
    STORE.mkdir(parents=True, exist_ok=True)
    df.to_csv(STORE / "metrics.csv", index=False)
    print(f"metrics: {len(df)} configuration(s)")
    return df


def difficulty_class(gb: float) -> str:
    if pd.isna(gb):
        return ""
    return "hard" if gb < DIFFICULTY_CUTS[0] else ("medium" if gb < DIFFICULTY_CUTS[1] else "simple")


def properties_table(full: bool = False) -> pd.DataFrame:
    """Per recipe: everything shown on the app's dataset page. Incremental: a dataset's row is reused
    when its table, range and grammar are unchanged (split coverage would otherwise read every table,
    the largest ones included); `full=True` recomputes all."""
    corpus = datasets.read_corpus()["datasets"]
    old_path = STORE / "datasets.csv"
    old = pd.read_csv(old_path).set_index("dataset") if old_path.is_file() and not full else pd.DataFrame()
    ranges = preprocessing.load_ranges().set_index("name")
    fairness_path = grammar.grammar_dir() / "fairness.csv"
    fair = pd.read_csv(fairness_path).set_index("name") if fairness_path.is_file() else pd.DataFrame()
    pycol = complexity.load_complexity()
    linked = set(pycol.loc[pycol["matches_table"] == "yes", "name"])
    rows = []
    for name in datasets.load_recipes():
        if name not in corpus:                    # recipe written, table not built yet
            print(f"properties: {name} has a recipe but no record yet (python datasets.py recreate {name}); skipped")
            continue
        e = corpus[name]
        stamp = f"{e['checksum']}|{ranges.loc[name, 'winner'] if name in ranges.index else ''}|" \
                f"{fair.loc[name, 'grammar_checksum'] if name in fair.index else ''}"
        # rows from before stamps existed were built from the current tables: reused once, stamped
        if name in old.index and ("stamp" not in old.columns or old.loc[name, "stamp"] == stamp):
            rows.append({"dataset": name, **old.loc[name].to_dict(), "stamp": stamp})
            continue
        counts = np.array(list(e["class_counts"].values()), dtype=float)
        d = e.get("difficulty", {})
        r = ranges.loc[name] if name in ranges.index else None
        row = {"dataset": name, "source": e["source"], "rows": e["n_rows"], "features": e["n_features"],
               "numeric": e["n_numeric"], "boolean": e["n_boolean"], "classes": e["n_classes"],
               "minority_share": counts.min() / counts.sum(), "duplicate_share": e["duplicate_rows"] / e["n_rows"],
               "constant_features": sum("constant" in f["flags"] for f in e["features"]),
               "difficulty_tree2": d.get("tree_depth2", {}).get("mean"), "difficulty_logreg": d.get("logreg", {}).get("mean"),
               "difficulty_gb": d.get("gradient_boosting", {}).get("mean"),
               "range": None if r is None else r["winner"],
               "coverage_min": None if r is None or r["winner"] == "none" else float(r[f"{r['winner']}_coverage_min"]),
               "degenerate_rate": None if r is None or r["winner"] == "none" else float(r[f"{r['winner']}_degenerate_rate"]),
               "split_coverage": preprocessing.split_coverage(name),
               "min_init_depth": ge.recorded_depth(name),
               "fairness_ratio": fair.loc[name, "ratio_mean"] if name in fair.index else None,
               "fairness_ci_low": fair.loc[name, "ratio_ci_low"] if name in fair.index else None,
               "fairness_ci_high": fair.loc[name, "ratio_ci_high"] if name in fair.index else None,
               "pycol_rows": name in linked or (pycol["name"] == name).any() and
               (pycol.loc[pycol["name"] == name, "origin"] == "computed").any(),
               "notes": e.get("notes", ""), "stamp": stamp}
        row.update({f"range_at_{int(t * 100)}": w for t, w in preprocessing.winners_by_target(name).items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    df["difficulty_class"] = [difficulty_class(v) for v in df["difficulty_gb"]]
    STORE.mkdir(parents=True, exist_ok=True)
    df.to_csv(STORE / "datasets.csv", index=False)
    print(f"properties: {len(df)} dataset(s)")
    return df


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["integrate", "metrics", "properties", "all"])
    p.add_argument("--full", action="store_true", help="properties: recompute every dataset (reads every table)")
    args = p.parse_args(argv)
    try:
        if args.command in ("integrate", "all"):
            integrate()
        if args.command in ("metrics", "all"):
            metrics_table()
        if args.command in ("properties", "all"):
            properties_table(full=args.full)
        return 0
    except datasets.DatasetError as exc:
        print(f"FAIL {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
