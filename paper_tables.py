"""Every table, figure and number of the paper, written from the results store, so that no result in the paper is
typed by hand.

Reads results/store/ (metrics.csv, datasets.csv, runs.csv, generations.csv, phase_timing.csv and the instrumented
runs; plain or .csv.gz) and the experiment files that define the paper's datasets. The statistics (Holm,
Benjamini-Hochberg, Bayesian signed-rank, Wilcoxon, Spearman) come from analysis.py and scipy. A paper dataset
without results is written with a visible \\RES mark, never dropped.

Output, under the output folder (default: paper/, the published ones; FEC_PAPER_DIR overrides):
    tables/          LaTeX tables, numbers.tex (one macro per number in the text), the main-text figures (PDF)
                     and claims_check.txt (every statement of the text checked against the data: OK or FAIL)
    supplementary/   Tables S1-S20 (CSV), Figures S1-S5 (PDF), INDEX.md and FileS1_data_dictionary.md

    python paper_tables.py                  write everything into the output folder
    python paper_tables.py --out DIR        write into DIR/tables and DIR/supplementary instead
    python paper_tables.py --check          rebuild in a temporary folder and compare with the output folder
                                            (exit 1 and list the files that differ)
"""
from __future__ import annotations

import argparse
import json
import os
import itertools
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import datasets
import diagrams
import experiment
import figstyle as fs
import results

ROOT = datasets.ROOT
# output folder for the paper's tables and figures (default: paper/, which holds the published ones);
# FEC_PAPER_DIR overrides
PAPER = Path(os.environ.get("FEC_PAPER_DIR", ROOT / "paper"))
TABLES = PAPER / "tables"
SUPPLEMENTARY = PAPER / "supplementary"
EXPERIMENTS = ROOT / "experiments"

# The experiments whose datasets form the paper corpus. Internal names only:
# the paper describes datasets by task type and feature types, never by experiment.
# 17 datasets, the ones run with GE and GP; binary_rest (11 more) left out of this paper
PAPER_EXPERIMENTS = ("wine_quality", "line1_hard", "line1_medium", "line1_simple", "line2_simple", "line2_hard")
MULTICLASS_EXPERIMENTS = ("line2_simple", "line2_hard")
CLASS_ORDER = {"hard": 0, "medium": 1, "simple": 2}
SETUP = "default"
METRIC_COLUMNS = ("algorithm", "setup", "dataset", "sampler", "fraction", "n_runs", "S_T", "S_T_sd", "dAcc", "S_E",
                  "S_wall", "hit_rate", "fake_hit_ratio", "sampling_time_sec", "p_eval_time", "bayes_worse",
                  "bayes_rope", "bayes_better", "baseline_accuracy", "fec_accuracy")
DATASET_COLUMNS = ("dataset", "source", "rows", "numeric", "boolean", "classes", "difficulty_gb", "difficulty_class",
                   "range", "coverage_min", "degenerate_rate", "fairness_ratio", "min_init_depth")


class TableError(Exception):
    pass


def _f(store: Path, name: str) -> Path:
    """A results-store file, plain or gzipped (the public release keeps the large ones as .csv.gz)."""
    plain = Path(store) / name
    return plain if plain.is_file() or not plain.with_name(name + ".gz").is_file() else plain.with_name(name + ".gz")


# ---------------------------------------------------------------- inputs

def paper_corpus(experiments_dir: Path = EXPERIMENTS) -> pd.DataFrame:
    """The paper's datasets: one row each, with the experiment it comes from and its task type."""
    rows = []
    for name in PAPER_EXPERIMENTS:
        path = experiments_dir / f"{name}_ge.yaml"
        if not path.is_file():
            raise TableError(f"experiment file missing: {path}")
        for ds in experiment.load(path)["datasets"]:
            rows.append({"dataset": ds, "experiment": name,
                         "task": "multiclass" if name in MULTICLASS_EXPERIMENTS else "binary"})
    corpus = pd.DataFrame(rows)
    dup = corpus[corpus.duplicated("dataset")]
    if len(dup):
        raise TableError(f"dataset in more than one paper experiment: {sorted(dup.dataset)}")
    return corpus


def _require(df: pd.DataFrame, columns, what: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise TableError(f"{what} lacks column(s): {', '.join(missing)}")


def load_store(store: Path = results.STORE, only=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(metrics of setup 'default', dataset properties) for the datasets in `only` (all if None); loud on a
    missing file, a missing column, or a dataset/algorithm under more than one setup."""
    paths = {"metrics": _f(store, "metrics.csv"), "datasets": _f(store, "datasets.csv")}
    for what, path in paths.items():
        if not path.is_file():
            raise TableError(f"store file missing: {path} (run `python results.py all`)")
    metrics = pd.read_csv(paths["metrics"])
    props = pd.read_csv(paths["datasets"])
    _require(metrics, METRIC_COLUMNS, "metrics.csv")
    _require(props, DATASET_COLUMNS, "datasets.csv")
    if only is not None:
        metrics = metrics[metrics.dataset.isin(set(only))]
        props = props[props.dataset.isin(set(only))]
    setups = metrics.groupby(["dataset", "algorithm"])["setup"].nunique()
    mixed = setups[setups > 1]
    if len(mixed):
        raise TableError("dataset/algorithm with rows under more than one setup: "
                         + ", ".join(f"{d}/{a}" for d, a in mixed.index))
    return with_holm(metrics[metrics.setup == SETUP].reset_index(drop=True)), props


# The speed tests are corrected for multiple testing. One family = one algorithm and
# sampler over every loaded dataset and fraction (paper: 17 datasets x 3 fractions = 51 tests); Holm decides,
# Benjamini-Hochberg is reported as a check and the uncorrected count beside it.
SPEED_FAMILY = ["algorithm", "sampler"]


def with_holm(metrics: pd.DataFrame) -> pd.DataFrame:
    """metrics with p_eval_holm and p_eval_bh: p_eval_time adjusted within each SPEED_FAMILY group. Done once on the
    whole loaded store, never on a subset, so that every table uses the same family."""
    from analysis import bh_fdr, holm
    m = metrics.copy()
    g = m.groupby(SPEED_FAMILY).p_eval_time
    m["p_eval_holm"] = g.transform(lambda p: holm(p.to_numpy()))
    m["p_eval_bh"] = g.transform(lambda p: bh_fdr(p.to_numpy()))
    return m


def _speed(p, ratio) -> np.ndarray:
    sig = p <= ALPHA
    return np.where(sig & (ratio > 1), "faster", np.where(sig & (ratio < 1), "slower", ""))


def ordered_corpus(corpus: pd.DataFrame, props: pd.DataFrame) -> pd.DataFrame:
    """Corpus joined to its properties, ordered: binary then multiclass, by rows. Datasets without properties go last in their task group, by name."""
    df = corpus.merge(props, on="dataset", how="left")
    df["_task"] = (df.task == "multiclass").astype(int)
    df["_rows"] = df.rows.fillna(float("inf"))
    return df.sort_values(["_task", "_rows", "dataset"], kind="stable").drop(columns=["_task", "_rows"]).reset_index(drop=True)


# ---------------------------------------------------------------- output

def header(sources: str) -> str:
    return f"% written by paper_tables.py from {sources}; code {datasets.git_commit()}; do not edit by hand\n"


def write(out_dir: Path, name: str, body: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(body, encoding="utf-8")
    return path


def same_content(new: Path, old: Path) -> bool:
    """Equal files, ignoring the provenance header line of a .tex table (it names the code commit, which
    changes with every commit while the table does not)."""
    if not old.is_file():
        return False
    if new.suffix not in (".tex", ".csv", ".md", ".txt"):
        return new.read_bytes() == old.read_bytes()
    a, b = new.read_text(encoding="utf-8"), old.read_text(encoding="utf-8")
    if new.suffix == ".tex":
        strip = lambda t: "\n".join(l for l in t.splitlines() if not l.startswith("% written by paper_tables.py"))
        a, b = strip(a), strip(b)
    return a == b


def build(out_tables: Path, out_supp: Path, min_rows: int = 0, store: Path = results.STORE) -> list[Path]:
    """Write every table; returns the files written. Table functions are added per task."""
    corpus = paper_corpus()
    metrics, props = load_store(store, only=corpus.dataset)
    main = metrics[metrics.sampler == MAIN_SAMPLER].reset_index(drop=True)
    written = []
    for fn in BUILDERS:
        written += fn(corpus=corpus, metrics=main, all_metrics=metrics, props=props, out_tables=out_tables,
                      out_supp=out_supp, min_rows=min_rows, manuscript=PAPER / "FEC_manuscript_final.tex")
    return written


# ---------------------------------------------------------------- formatting

def _missing(n: int, mark: str) -> list[str]:
    return [f"\\RES{{{mark}}}"] * n


def _num(x, digits: int) -> str:
    return "--" if pd.isna(x) else f"{x:.{digits}f}"


def _rows(x) -> str:
    return "--" if pd.isna(x) else f"{int(x):,}".replace(",", "{,}")


def _tt(name: str) -> str:
    return "\\texttt{" + name.replace("_", "\\_") + "}"


SOURCE_LABEL = {"openml": "OpenML", "pmlb": "PMLB", "uci": "UCI", "zip": "UCI"}


# Tables taken from the Grinsztajn et al. benchmark on OpenML whose original data is a UCI dataset (checked on
# OpenML and UCI, 2026-10-13). The table names the UCI origin first and the balanced OpenML version used.
UCI_ORIGIN = {"wine_quality": 186, "bank_marketing": 222, "MagicTelescope": 159, "Diabetes130US": 296, "covertype": 31}
UCI_URL = "https://archive.ics.uci.edu/dataset/{}"


def source_link(name: str) -> str:
    """Source of a dataset as a hyperlink to its page, from its provenance record (data/corpus.json):
    "OpenML 44091", "UCI 602", "PMLB"; a UCI origin, if any, comes first: "UCI 186 / OpenML 44091"."""
    r = datasets.dataset_record(name)
    url, src = r.get("url"), r.get("source")
    if not url or src not in SOURCE_LABEL:
        raise TableError(f"{name}: provenance record has no usable source/url")
    m = re.search(r"/(?:d|dataset)/(\d+)", url)
    label = SOURCE_LABEL[src] + (f" {m.group(1)}" if m else "")
    used = "\\href{" + url + "}{" + label + "}"
    uci = UCI_ORIGIN.get(name)
    return used if uci is None else "\\href{" + UCI_URL.format(uci) + "}{UCI " + str(uci) + "} / " + used
RANGE_LABEL = {"pm1": "$\\pm 0.99$", "pm2": "$\\pm 1.99$", "pm3": "$\\pm 2.99$"}


def _table(label: str, caption: str, spec: str, head: str, body: list[str], sources: str,
           size: str = "\\small", colsep: str | None = None, overhang: bool = True) -> str:
    return (header(sources)
            + "\\begin{table}[H]\n" + ("\\begin{adjustwidth}{-\\extralength}{0cm}\n" if overhang else "") + "\\caption{" + caption + "}\n"
            + "\\label{" + label + "}\n" + size + "\n"
            + (f"\\setlength{{\\tabcolsep}}{{{colsep}}}\n" if colsep else "") + "\\centering\n"
            + "\\begin{tabular}{" + spec + "}\n\\toprule\n" + head + " \\\\\n\\midrule\n"
            + " \\\\\n".join(body) + " \\\\\n\\bottomrule\n\\end{tabular}\n" + ("\\end{adjustwidth}\n" if overhang else "") + "\\end{table}\n")


# ---------------------------------------------------------------- corpus

def _balance_change(name: str) -> dict | None:
    try:
        changes = datasets.dataset_record(name).get("changes", [])
    except datasets.DatasetError:
        return None
    for ch in changes:
        if ch["action"] == "balance":
            return ch
    return None


def preprocessing_rows(names) -> pd.DataFrame:
    """Table S5: per dataset, what the recorded recipe did between the source and the table used."""
    out = []
    for n in names:
        try:
            rec = datasets.dataset_record(n)
        except datasets.DatasetError:                          # a dataset without a recipe (synthetic store)
            continue
        ch = rec.get("changes", [])
        pick = lambda act: [c for c in ch if c["action"] == act]
        miss = pick("drop_missing_rows")
        bal = _balance_change(n)
        out.append({
            "dataset": n, "source": rec.get("source"), "url": rec.get("url"),
            "rows_source": rec["original_shape"][0], "columns_source": rec["original_shape"][1],
            "missing_value_rows_removed": int(miss[0]["effect"].split()[0]) if miss else 0,
            "columns_dropped": "; ".join(c["detail"] for c in pick("drop_column")) or "none",
            "one_hot": "; ".join(c["detail"] for c in pick("one_hot")) or "none",
            "type_changes": "; ".join(f"{c['action']}: {c['detail']}" for c in pick("cast") + pick("override_type")) or "none",
            "label_encoding": "; ".join(c["detail"] for c in pick("encode_label")),
            "balancing": bal["effect"] if bal else "none (as published)",
            "duplicate_rows_kept": rec.get("duplicate_rows"), "rows_used": rec.get("n_rows"),
            "numeric_features": rec.get("n_numeric"), "boolean_features": rec.get("n_boolean"),
            "classes": rec.get("n_classes"), "checksum": rec.get("checksum")})
    return pd.DataFrame(out)


def baseline_seconds(store: Path = results.STORE) -> pd.DataFrame:
    """Median evaluation time per baseline run (s), per dataset and algorithm (setup default); empty without runs."""
    path = _f(store, "runs.csv")
    if not path.is_file():
        return pd.DataFrame(columns=["dataset", "GE", "GP"])
    r = pd.read_csv(path, usecols=["algorithm", "setup", "dataset", "mode", "eval_time_sec"])
    r = r[(r.setup == SETUP) & (r["mode"] == "baseline")]
    return r.groupby(["dataset", "algorithm"]).eval_time_sec.median().unstack("algorithm").reset_index()


def _secs(v) -> str:
    return "--" if pd.isna(v) else _int(round(float(v)))


def corpus_tables(corpus, props, out_tables, **_) -> list[Path]:
    """corpus.tex (who the datasets are) and grammar_props.tex (constant range, coverage, degenerate
    rate, feature-type fairness, initial depth). A paper dataset without properties: \\RES{missing}."""
    df = ordered_corpus(corpus, props)
    missing = sorted(df[df.rows.isna()].dataset)
    if missing:
        print(f"datasets without properties in the store (\\RES{{missing}}): {', '.join(missing)}")
    body1, body2 = [], []
    for _, r in df.iterrows():
        if pd.isna(r.rows):
            body1.append(_tt(r.dataset) + " & " + r.task + " & " + source_link(r.dataset)
                         + " & \\multicolumn{5}{c}{\\RES{not yet in the store}}")
            body2.append(_tt(r.dataset) + " & \\multicolumn{5}{c}{\\RES{not yet in the store}}")
            continue
        body1.append(" & ".join([_tt(r.dataset), r.task, source_link(r.dataset), _rows(r.rows),
                                 str(int(r.numeric)),
                                 str(int(r.boolean)), str(int(r.classes)), _num(r.difficulty_gb, 3)]))
        body2.append(" & ".join([_tt(r.dataset), RANGE_LABEL.get(r["range"], str(r["range"])),
                                 _num(r.coverage_min, 3), _num(r.degenerate_rate, 3), _num(r.fairness_ratio, 2),
                                 str(int(r.min_init_depth))]))
    t1 = _table("tab:corpus",
                f"The {len(df)} datasets, ordered by rows within each task, with rows and features as used in the "
                "experiments. Num., Bool.: numeric and boolean features; Cl.: classes; GB acc.: balanced accuracy "
                "of gradient boosting (Section~\\ref{sec:exp:datasets}).",
                "lllrrrrr", "Dataset & Task & Source & Rows & Num. & Bool. & Cl. & GB acc.",
                body1, "results/store/datasets.csv")
    t2 = _table("tab:grammar-props",
                "Grammar properties per dataset: range of the signed constants, smallest share of a numeric "
                "column's standardised values inside that range (coverage), share of (column, constant) pairs "
                "that split off fewer than 1\\% or more than 99\\% of the rows (degenerate rate), ratio of the "
                "appearance probability of boolean to numeric features in initial populations (fairness; -- "
                "when all features are numeric), and minimum initial derivation depth.",
                "lrrrrr", "Dataset & Constants & Coverage & Degenerate & Fairness & Init. depth",
                body2, "results/store/datasets.csv")
    return [write(out_tables, "corpus.tex", t1), write(out_tables, "grammar_props.tex", t2)]


# ---------------------------------------------------------------- results

# Datasets left out of the main-text results (reported in Tables S2/S3 only). Empty: every dataset is shown.
MAIN_TEXT_EXCLUDED: tuple = ()
SAMPLER_ORDER = ("farthest_point", "random", "stratified")
# The main results use random fingerprint rows; farthest-point and stratified are compared in their own subsection and
# reported in full in the supplementary material.
MAIN_SAMPLER = "random"
SAMPLER_LABEL = {"farthest_point": "FPS", "random": "random", "stratified": "stratified"}
ALPHA = 0.05
THRESHOLD = 0.95                    # Bayesian verdict
ROPE_PP = 100 * results.EQUIVALENCE_MARGIN   # region of practical equivalence for dAcc, percentage points
OUTCOME_DEFS = ("Faster: significantly less evaluation time than the baseline (paired Wilcoxon, $p \\leq 0.05$ \\rev{after Holm "
                "correction over the algorithm's 51 configurations}); "
                "comparable: posterior probability $\\geq 0.95$ that $|\\Delta\\mathrm{Acc}| \\leq 1$ pp (Bayesian "
                "correlated $t$-test); both: faster and comparable. ")
SENSITIVITY = {"RopeTwo": (0.02, THRESHOLD), "RopeThree": (0.03, THRESHOLD),
               "ThrNinety": (results.EQUIVALENCE_MARGIN, 0.90)}
FULL_COLUMNS = ("algorithm", "dataset", "fraction", "n_runs", "S_T", "S_T_sd", "p_eval_time", "p_eval_holm",
                "p_eval_bh", "speed", "speed_raw", "dAcc", "bayes_worse", "bayes_rope", "bayes_better", "verdict",
                "hit_rate", "fake_hit_ratio", "S_wall", "S_wall_sd", "p_wall", "p_wall_holm", "wall_speed",
                "eval_saved_s",
                "wall_saved_s")


def verdict(worse, rope, better) -> str:
    if any(pd.isna(v) for v in (worse, rope, better)):
        return ""
    for name, p in (("equivalent", rope), ("worse", worse), ("better", better)):
        if p >= THRESHOLD:
            return name
    return "undecided"


def annotated(metrics: pd.DataFrame) -> pd.DataFrame:
    """Metrics with the speed mark (faster / slower / '': paired Wilcoxon, Holm-adjusted p <= ALPHA over the
    SPEED_FAMILY; speed_raw and speed_bh: the same mark from the uncorrected and the
    Benjamini-Hochberg p) and the Bayesian verdict; sorted. Needs p_eval_holm (with_holm, done in load_store)."""
    _require(metrics, ("p_eval_holm", "p_eval_bh"), "metrics (call with_holm on the whole store first)")
    df = metrics.copy()
    df["speed"] = _speed(df.p_eval_holm, df.S_T)
    df["speed_raw"] = _speed(df.p_eval_time, df.S_T)
    df["speed_bh"] = _speed(df.p_eval_bh, df.S_T)
    df["verdict"] = [verdict(w, r, b) for w, r, b in zip(df.bayes_worse, df.bayes_rope, df.bayes_better)]
    df["_s"] = df.sampler.map({s: i for i, s in enumerate(SAMPLER_ORDER)})
    return df.sort_values(["algorithm", "dataset", "_s", "fraction"], kind="stable").drop(columns="_s").reset_index(drop=True)


def main_text_datasets(corpus, props, min_rows: int) -> list[str]:
    df = ordered_corpus(corpus, props)
    keep = ~df.dataset.isin(MAIN_TEXT_EXCLUDED)
    if min_rows:
        keep &= ~(df.rows < min_rows)                # datasets without properties stay (marked later)
    return list(df[keep].dataset)


def chosen_rows(ann: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, sampler): the smallest fraction significantly faster, else the 1 % row marked n.s."""
    out = []
    for _, g in ann.groupby(["dataset", "sampler"], sort=False):
        g = g.sort_values("fraction", kind="stable")
        fast = g[g.speed == "faster"]
        row = (fast.iloc[0] if len(fast) else g.iloc[0]).copy()
        row["significant"] = bool(len(fast))
        out.append(row)
    return pd.DataFrame(out)


def _mark(speed: str) -> str:
    return {"faster": "$^{\\blacktriangle}$", "slower": "$^{\\blacktriangledown}$"}.get(speed, "")


def _pp(x) -> str:
    if pd.isna(x):
        return "--"
    v = round(100 * x, 2)
    return "0.00" if v == 0 else f"{v:+.2f}"


def _longtable(label: str, caption: str, spec: str, head: str, body: list[str], sources: str) -> str:
    return (header(sources) + "\\begin{small}\n\\setlength{\\LTleft}{-\\extralength}\\setlength{\\LTright}{0pt plus 1fill}\n"
            + "\\begin{longtable}{" + spec + "}\n"
            + "\\caption{" + caption + "}\\label{" + label + "}\\\\\n\\toprule\n" + head + " \\\\\n\\midrule\n\\endfirsthead\n"
            + "\\toprule\n" + head + " \\\\\n\\midrule\n\\endhead\n\\bottomrule\n\\endfoot\n"
            + " \\\\\n".join(body) + " \\\\\n\\end{longtable}\n\\end{small}\n")


def results_tables(corpus, metrics, props, out_tables, min_rows=0, **_) -> list[Path]:
    """results_ge.tex / results_gp.tex: one row per main-text dataset x sampler (data-model.md)."""
    ann = annotated(metrics)
    keep = main_text_datasets(corpus, props, min_rows)
    written = []
    for algo in ("GE", "GP"):
        a = ann[(ann.algorithm == algo) & ann.dataset.isin(keep)]
        chosen = chosen_rows(a) if len(a) else pd.DataFrame()
        body, running = [], []
        for ds in keep:
            sub = chosen[chosen.dataset == ds] if len(chosen) else chosen
            if not len(sub):
                running.append(ds)
                body.append(_tt(ds) + " & \\multicolumn{8}{c}{\\RES{running}}")
                continue
            sub = sub.assign(_s=sub.sampler.map({s: i for i, s in enumerate(SAMPLER_ORDER)})).sort_values("_s", kind="stable")
            for i, (_, r) in enumerate(sub.iterrows()):
                frac = f"{r.fraction:.2f}" if r.significant else f"{r.fraction:.2f} (n.s.)"
                body.append(" & ".join([_tt(ds) if i == 0 else "", SAMPLER_LABEL[r.sampler], frac,
                                        f"{r.S_T:.2f}{_mark(r.speed)} $\\pm$ {r.S_T_sd:.2f}", _pp(r.dAcc),
                                        _num(r.bayes_rope, 2), r.verdict or "--", _num(r.S_E, 1),
                                        _num(r.sampling_time_sec, 2) if r.sampler == "farthest_point" else "--"]))
        if running:
            print(f"{algo}: datasets without results (\\RES{{running}}): {', '.join(running)}")
        excluded = (", ".join(_tt(d) for d in MAIN_TEXT_EXCLUDED)
                    + f" are reported in Table~S{2 if algo == 'GE' else 3} only.") if MAIN_TEXT_EXCLUDED else ""
        rows_note = f" Datasets with fewer than {min_rows:,} rows are left out." if min_rows else ""
        caption = (f"{algo}: per dataset and sampler, the smallest fingerprint fraction $\\rho$ at which FEC is "
                   "significantly faster (paired Wilcoxon, \\rev{Holm over the algorithm's 51 configurations}, "
                   f"$\\alpha={ALPHA}$); when none is, the 1\\% configuration is shown and marked n.s. $S_T$: mean "
                   "$\\pm$ sd over 30 paired runs ($^{\\blacktriangle}$ faster, $^{\\blacktriangledown}$ slower). "
                   "$\\Delta$Acc in percentage points. P(eq.): posterior probability that $|\\Delta\\mathrm{Acc}|"
                   f"\\leq 1$ pp (Bayesian correlated $t$-test); verdict at probability $\\geq {THRESHOLD}$. "
                   "$S_E$: effort speedup. Samp.: farthest-point sampling time in seconds. "
                   f"{excluded}{rows_note}")
        body_tex = _longtable(f"tab:results-{algo.lower()}", caption, "llrrrrlrr",
                              "Dataset & Sampler & $\\rho$ & $S_T$ & $\\Delta$Acc & P(eq.) & Verdict & $S_E$ & Samp.",
                              body, "results/store/metrics.csv")
        written.append(write(out_tables, f"results_{algo.lower()}.tex", body_tex))
    return written


def summary_table(corpus, metrics, props, out_tables, min_rows=0, **_) -> list[Path]:
    """summary_by_config.tex: per algorithm x configuration over the main-text datasets with results."""
    ann = annotated(metrics)
    keep = main_text_datasets(corpus, props, min_rows)
    ann = ann[ann.dataset.isin(keep)]
    shown = [k for k in ("equivalent", "worse", "better", "undecided")   # worse/better only if any occurs
             if k in ("equivalent", "undecided") or (ann.verdict == k).any()]
    body = []
    for algo in ("GE", "GP"):
        if not (ann.algorithm == algo).any():
            continue
        body.append(("\\midrule\n" if body else "") + f"\\multicolumn{{6}}{{l}}{{\\textbf{{{algo}}}}}")
        for s in SAMPLER_ORDER:
            for f in sorted(ann.fraction.unique()):
                g = ann[(ann.algorithm == algo) & (ann.sampler == s) & (ann.fraction == f)]
                if not len(g):
                    continue
                q1, med, q3 = g.S_T.quantile([0.25, 0.5, 0.75])
                v = g.verdict.value_counts()
                body.append(" & ".join([f"{f:.2f}", str(len(g)),
                                        f"{med:.2f} [{q1:.2f}, {q3:.2f}]",
                                        f"{(g.speed == 'faster').sum()}/{(g.speed == 'slower').sum()}",
                                        f"{_pp(g.dAcc.median())} [{_pp(g.dAcc.quantile(0.25))}, {_pp(g.dAcc.quantile(0.75))}]",
                                        "/".join(str(v.get(k, 0)) for k in shown)]))
    absent = [k for k in ("worse", "better") if k not in shown]
    none_note = f" No configuration was judged {' or '.join(absent)}." if absent else ""
    t = _table("tab:summary",
               "Summary per algorithm and fingerprint fraction $\\rho$ (random fingerprint rows) over the datasets "
               "with results ($n$). $S_T$ and "
               "$\\Delta$Acc: median [interquartile range] over the $n$ datasets of the per-dataset values "
               "($\\Delta$Acc in percentage points). Faster/slower: datasets with a significant difference in evaluation "
               "time (paired Wilcoxon, \\rev{Holm-adjusted} $p \\leq 0.05$ \\rev{over the algorithm's 51 configurations}), out of $n$. "
               f"Verdicts: number of the $n$ datasets judged {' / '.join(shown)} by the Bayesian correlated "
               f"$t$-test (probability $\\geq {THRESHOLD}$).{none_note}",
               "rrlrll", "$\\rho$ & $n$ & Median $S_T$ & Faster/slower & Median $\\Delta$Acc & "
               + "/".join(k.capitalize() if i == 0 else k for i, k in enumerate(shown)),
               body, "results/store/metrics.csv")
    return [write(out_tables, "summary_by_config.tex", t)]


GEGP_MEASURES = (("S_T", "$S_T$", 1, 2), ("hit_rate", "hit rate", 1, 2),
                 ("fake_hit_ratio", "fake-hit ratio", 1, 2), ("dAcc", "$\\Delta$Acc (pp)", 100, 2))
GEGP_KEY = {"S_T": "ST", "hit_rate": "Hit", "fake_hit_ratio": "Fake", "dAcc": "Dacc"}


def gegp_pairs(ann: pd.DataFrame) -> pd.DataFrame:
    """GE and GP side by side, one row per (dataset, fraction) with results for both (main sampler)."""
    w = ann.pivot_table(index=["dataset", "fraction"], columns="algorithm",
                        values=[m for m, *_ in GEGP_MEASURES]).dropna()
    return w if len(w) and {"GE", "GP"} <= set(w.columns.get_level_values(1)) else pd.DataFrame()


def gegp_compare(ann: pd.DataFrame) -> dict:
    """Per measure, one pair per dataset (the mean over its fractions, so that the correlated fractions of a dataset
    are not counted as independent observations): medians of GE and GP, how many datasets GP is higher / lower in,
    paired Wilcoxon p over the datasets, and the same test within each fraction (p_frac) as a robustness check."""
    from analysis import wilcoxon_paired
    w = gegp_pairs(ann)
    if w.empty:
        return {}
    per = w.groupby(level=0).mean()
    fracs = sorted(set(w.index.get_level_values(1)))
    out = {"n": len(per), "datasets": len(per), "pairs": len(w)}
    for col, _l, scale, _d in GEGP_MEASURES:
        ge, gp = per[(col, "GE")], per[(col, "GP")]
        out[col] = {"GE": scale * ge.median(), "GP": scale * gp.median(), "gp_higher": int((gp > ge).sum()),
                    "gp_lower": int((gp < ge).sum()), "p": float(wilcoxon_paired(ge, gp)["p"]),
                    "p_frac": [float(wilcoxon_paired(w[(col, "GE")].xs(f, level=1), w[(col, "GP")].xs(f, level=1))["p"])
                               for f in fracs],
                    "gp_higher_frac": [int((w[(col, "GP")].xs(f, level=1) > w[(col, "GE")].xs(f, level=1)).sum())
                                       for f in fracs]}
    return out


# What a higher value of each GE-vs-GP measure means, for the reader of Table 7
GEGP_HIGHER_IS = {"S_T": "faster evaluation", "hit_rate": "more cache reuse",
                  "fake_hit_ratio": "more wrong hits", "dAcc": "less accuracy lost"}


def gegp_result(r: dict, n: int) -> str:
    """Plain-language outcome of one measure: which algorithm is higher, if the dataset-level test is significant."""
    if not r["p"] <= ALPHA:                                     # also when p is undefined (all pairs tied)
        return "no significant difference"
    return "GP higher" if r["gp_higher"] > n / 2 else "GE higher"


def gegp_same_each_fraction(r: dict, n: int) -> bool:
    """True when the test within every fraction gives the same outcome (significance and direction) as over the datasets."""
    sig = r["p"] <= ALPHA
    gp_up = r["gp_higher"] > n / 2
    return all((pf <= ALPHA) == sig and (not sig or (hf > n / 2) == gp_up)
               for pf, hf in zip(r["p_frac"], r["gp_higher_frac"]))


def ge_vs_gp_table(corpus, metrics, props, out_tables, min_rows=0, **_) -> list[Path]:
    """ge_vs_gp.tex: GE against GP on the datasets with results for both (main sampler), one pair per dataset (mean
    over the fractions): what a higher value means, the medians, in how many datasets GP is higher, the Wilcoxon p
    over the datasets, the outcome in words, and whether every fraction on its own gives the same outcome."""
    keep = main_text_datasets(corpus, props, min_rows)
    v = gegp_compare(annotated(metrics[metrics.dataset.isin(keep)]))
    if not v:
        body = ["\\multicolumn{7}{c}{\\RES{no dataset with results for both algorithms}}"]
    else:
        n = v["n"]
        body = [" & ".join([label, GEGP_HIGHER_IS[col], f"${_num(v[col]['GE'], d)}$", f"${_num(v[col]['GP'], d)}$",
                            f"{v[col]['gp_higher']} of {n}", f"${_p(v[col]['p']).lstrip('=')}$",
                            gegp_result(v[col], n), "yes" if gegp_same_each_fraction(v[col], n) else "no"])
                for col, label, _s, d in GEGP_MEASURES]
    nd = f"{v['n']} datasets" if v else "no datasets"
    t = _table("tab:ge-vs-gp",
               f"GE against GP with random fingerprint rows ({nd} with results for both). Each dataset gives one "
               "value per algorithm, the mean over the three fingerprint fractions. GE, GP: median over the datasets. "
               "GP $>$ GE: datasets in which GP's value is higher. $p$: paired Wilcoxon signed-rank test over the "
               "datasets; a difference is called significant when $p \\leq 0.05$. Same per fraction: the test "
               "repeated within the 1\\%, 5\\% and 10\\% fractions separately gives the same result.",
               "llrrrrll", "Measure & Higher value means & GE & GP & GP $>$ GE & $p$ & Result & Same per fraction",
               body, "results/store/metrics.csv")
    return [write(out_tables, "ge_vs_gp.tex", t)]


def gegp_equivalent(ann: pd.DataFrame, props: pd.DataFrame, store: Path = results.STORE) -> pd.DataFrame:
    """§6.7: the dataset-fraction configurations in which FEC kept the accuracy of both GE and GP (both verdicts
    equivalent), with each algorithm's S_T, hit rate, hit/miss size ratio and work avoided, and the size group of
    the table (small / mid / large, cut at SMALL_BELOW and LARGE_ABOVE rows)."""
    if not (_f(store, "generations.csv")).is_file() or "classes" not in props:
        return pd.DataFrame()
    x = _work_main(ann, props, store).merge(props[["dataset", "rows"]], on="dataset", how="left", suffixes=("", "_p"))
    if "rows_p" in x:
        x = x.drop(columns="rows_p")
    cols = ["S_T", "hit_rate", "hit_size_ratio", "work_avoided", "verdict", "rows"]
    ge = x[x.algorithm == "GE"].set_index(["dataset", "fraction"])[cols]
    gp = x[x.algorithm == "GP"].set_index(["dataset", "fraction"])[cols]
    j = ge.join(gp, lsuffix="_GE", rsuffix="_GP", how="inner")
    j = j[(j.verdict_GE == "equivalent") & (j.verdict_GP == "equivalent")].copy()
    j["group"] = ["small" if r < SMALL_BELOW else "large" if r > LARGE_ABOVE else "mid" for r in j.rows_GE]
    return j


def gegp_equivalent_numbers(ann: pd.DataFrame, props: pd.DataFrame, store: Path = results.STORE) -> dict:
    j = gegp_equivalent(ann, props, store)
    if j.empty:
        return {}
    out = {"GEGPEqN": str(len(j))}
    rng = lambda v, pct=False, d=2: (f"{100 * v.min():.0f}", f"{100 * v.max():.0f}") if pct else (f"{v.min():.{d}f}", f"{v.max():.{d}f}")
    for grp, key in (("mid", "Mid"), ("large", "Large")):
        g = j[j.group == grp]
        if g.empty:
            continue
        names = sorted(set(g.index.get_level_values(0)), key=lambda n: (props.set_index("dataset").rows.get(n, 0), n))
        out[f"GEGPEq{key}Names"], out[f"GEGPEq{key}N"] = _names(names), str(len(g))
        out[f"GEGPEq{key}RatioMin"], out[f"GEGPEq{key}RatioMax"] = rng(g.S_T_GP / g.S_T_GE)
        for alg in ("GE", "GP"):
            out[f"GEGPEq{key}Hit{alg}Min"], out[f"GEGPEq{key}Hit{alg}Max"] = rng(g[f"hit_rate_{alg}"])
            out[f"GEGPEq{key}Size{alg}Min"], out[f"GEGPEq{key}Size{alg}Max"] = rng(g[f"hit_size_ratio_{alg}"], pct=True)
            out[f"GEGPEq{key}Work{alg}Min"], out[f"GEGPEq{key}Work{alg}Max"] = rng(g[f"work_avoided_{alg}"], pct=True)
            out[f"GEGPEq{key}ST{alg}Min"], out[f"GEGPEq{key}ST{alg}Max"] = rng(g[f"S_T_{alg}"])
    return out


def gegp_equivalent_claims(ann: pd.DataFrame, props: pd.DataFrame, store: Path = results.STORE) -> list:
    j = gegp_equivalent(ann, props, store)
    if j.empty:
        return []
    mid, large = j[j.group == "mid"], j[j.group == "large"]
    return [("gegp-eq/mid-gp-faster", "§6.7: where both kept their accuracy, GP was faster on every mid-sized table, its "
             "hits skipping programs larger than its misses (ratio > 1) while GE's were less than half the size of a miss",
             len(mid) > 0 and (mid.S_T_GP > mid.S_T_GE).all() and (mid.hit_size_ratio_GP > 1).all()
             and (mid.hit_size_ratio_GE < 0.5).all()),
            ("gegp-eq/large-ge-faster", "§6.7: where both kept their accuracy on the large tables, GE was faster in every "
             "configuration and GP hit the cache less often",
             len(large) > 0 and (large.S_T_GE > large.S_T_GP).all() and (large.hit_rate_GE > large.hit_rate_GP).all()),
            ("gegp-eq/work-orders-speed", "§6.7: in every configuration where both kept their accuracy, the algorithm whose "
             "hits avoided more work had the larger S_T",
             bool(((j.work_avoided_GP > j.work_avoided_GE) == (j.S_T_GP > j.S_T_GE)).all()))]


def gegp_numbers(ann: pd.DataFrame) -> dict:
    v = gegp_compare(ann)
    if not v:
        return {}
    out = {"GEGPPairs": str(v["pairs"]), "GEGPDatasets": str(v["datasets"])}
    for col, _l, _s, d in GEGP_MEASURES:
        k = GEGP_KEY[col]
        fmt = (lambda x: f"{x:+.2f}") if col == "dAcc" else (lambda x: f"{x:.2f}")
        out[f"GEGP{k}GE"], out[f"GEGP{k}GP"] = fmt(v[col]["GE"]), fmt(v[col]["GP"])
        out[f"GEGP{k}Higher"], out[f"GEGP{k}P"] = str(v[col]["gp_higher"]), _p(v[col]["p"])
        out[f"GEGP{k}Lower"] = str(v[col]["gp_lower"])
        out[f"GEGP{k}PMin"] = _p(min(v[col]["p_frac"]))
    shared = set(gegp_pairs(ann).index.get_level_values(0))
    for algo in ("GE", "GP"):
        x = ann[(ann.algorithm == algo) & ann.dataset.isin(shared)]
        out[f"GEGPBoth{algo}"] = f"{100 * ((x.speed == 'faster') & (x.verdict == 'equivalent')).mean():.0f}"
    return out


def gegp_claims(ann: pd.DataFrame) -> list[tuple[str, str, bool]]:
    v = gegp_compare(ann)
    if not v:
        return []
    shared = set(gegp_pairs(ann).index.get_level_values(0))
    both = {a: ((x.speed == "faster") & (x.verdict == "equivalent")).mean()
            for a, x in ann[ann.dataset.isin(shared)].groupby("algorithm")}
    n = v["n"]
    return [("gegp/same-speed", "§6.6: GE and GP gained the same speed (paired S_T difference not significant)",
             v["S_T"]["p"] > ALPHA),
            ("gegp/fewer-hits", "§6.6: GP had fewer hits (lower in most pairs, p <= 0.05)",
             v["hit_rate"]["gp_higher"] < n / 2 and v["hit_rate"]["p"] <= ALPHA),
            ("gegp/more-fakes", "§6.6: GP's cache was wrong more often (fake-hit ratio higher, p <= 0.05)",
             v["fake_hit_ratio"]["gp_higher"] > n / 2 and v["fake_hit_ratio"]["p"] <= ALPHA),
            ("gegp/accuracy-cost", "§6.6: GP's accuracy difference was lower (worse) on most datasets",
             v["dAcc"]["gp_higher"] < n / 2 and v["dAcc"]["GP"] < v["dAcc"]["GE"]),
            ("gegp/accuracy-ns", "§6.6: 'but the difference was not significant', over the datasets and within every "
             "fraction", v["dAcc"]["p"] > ALPHA and min(v["dAcc"]["p_frac"]) > ALPHA),
            ("gegp/fewer-both", "§6.6: fewer GP configurations were both faster and comparable on the shared datasets",
             both.get("GP", 1) < both.get("GE", 0))]


def ge_vs_gp_figure(corpus, metrics, props, out_tables, min_rows=0, **_) -> list[Path]:
    """fig_ge_vs_gp.pdf: GE and GP on the shared datasets, one panel per measure, box with one point per dataset
    (the mean over its fractions)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keep = main_text_datasets(corpus, props, min_rows)
    w = gegp_pairs(annotated(metrics[metrics.dataset.isin(keep)]))
    if w.empty:
        return []
    w = w.groupby(level=0).mean()                                  # one point per dataset, as in the test
    rng = np.random.default_rng(0)
    fs.use()
    fig, axes = plt.subplots(1, len(GEGP_MEASURES), figsize=(6.3, 2.5))
    colours = (SERIES_COLOUR, SECOND_COLOUR)
    for ax, (col, label, scale, _d) in zip(axes, GEGP_MEASURES):
        vals = [scale * w[(col, a)].to_numpy() for a in ("GE", "GP")]
        jit = [k + rng.uniform(-0.08, 0.08, len(vals[0])) for k in (1, 2)]
        bp = ax.boxplot(vals, widths=0.45, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, linewidth=1.2), whiskerprops=dict(color=INK_SECONDARY, linewidth=0.8),
                        capprops=dict(color=INK_SECONDARY, linewidth=0.8), boxprops=dict(linewidth=0.8, color=INK_SECONDARY))
        for patch, colr in zip(bp["boxes"], colours):
            patch.set_facecolor(colr)
            patch.set_alpha(0.25)
        for k, (v_, colr) in enumerate(zip(vals, colours)):
            ax.scatter(jit[k], v_, s=7, color=colr, edgecolor="white", linewidth=0.3, zorder=3)
        ref = {"S_T": 1.0, "dAcc": 0.0}.get(col)
        if ref is not None:
            ax.axhline(ref, color=INK_SECONDARY, linestyle=(0, (4, 3)), linewidth=0.8, zorder=0)
        if col in ("hit_rate", "fake_hit_ratio"):
            ax.set_ylim(0, 1)
        ax.set_xticks([1, 2], ["GE", "GP"])
        ax.set_title(label, fontsize=8, color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    out_tables.mkdir(parents=True, exist_ok=True)
    path = out_tables / "fig_ge_vs_gp.pdf"
    fs.save(fig, path)
    plt.close(fig)
    return [path]


# ---------------------------------------------------------------- outcome per dataset (the outcome-map figure)

GROUP_WORD = {"all": "all", "some": "some", "none": "none"}


def outcome_counts(ann: pd.DataFrame) -> pd.DataFrame:
    """Per dataset of one algorithm: how many of its configurations are significantly faster (Holm), slower,
    comparable (Bayesian equivalent), worse, and both faster and comparable; group all / some / none by 'both'."""
    g = ann.assign(both=(ann.speed == "faster") & (ann.verdict == "equivalent"))
    t = g.groupby("dataset").agg(configs=("speed", "size"), faster=("speed", lambda s: int((s == "faster").sum())),
                                 slower=("speed", lambda s: int((s == "slower").sum())),
                                 comparable=("verdict", lambda s: int((s == "equivalent").sum())),
                                 worse=("verdict", lambda s: int((s == "worse").sum())), both=("both", "sum"))
    t["both"] = t["both"].astype(int)
    t["group"] = ["all" if b == c else "none" if b == 0 else "some" for b, c in zip(t.both, t.configs)]
    return t


def outcome_tables(corpus, metrics, props, out_tables, min_rows=0, **_) -> list[Path]:
    """outcomes_ge.tex / outcomes_gp.tex: one row per dataset (ordered by rows), the counts of outcome_counts."""
    keep = main_text_datasets(corpus, props, min_rows)
    ann = annotated(metrics[metrics.dataset.isin(keep)])
    rows_of = props.set_index("dataset")["rows"]
    order = sorted(keep, key=lambda d: (rows_of.get(d, float("inf")), d))
    written = []
    for algo in ("GE", "GP"):
        t = outcome_counts(ann[ann.algorithm == algo]) if (ann.algorithm == algo).any() else pd.DataFrame()
        body = []
        for d in order:
            rows = _rows(rows_of.get(d)) if d in rows_of.index else "--"
            if d not in t.index:
                body.append(f"{_tt(d)} & {rows} & \\multicolumn{{6}}{{c}}{{\\RES{{running}}}}")
                continue
            r = t.loc[d]
            body.append(" & ".join([_tt(d), rows, str(r.faster), str(r.slower), str(r.comparable), str(r.worse),
                                    str(r.both), GROUP_WORD[r.group]]))
        tab = _table(f"tab:outcomes-{algo.lower()}",
                     f"{algo}: outcome of the nine FEC configurations of each dataset, ordered by number of rows. "
                     "Faster/slower: significant difference in evaluation time (paired Wilcoxon, Holm over the nine "
                     "configurations, $\\alpha=0.05$). Comparable/worse: posterior probability $\\geq 0.95$ that "
                     "$|\\Delta\\mathrm{Acc}| \\leq 1$ pp, or that FEC is worse by more than 1 pp (Bayesian correlated "
                     "$t$-test). Both: faster and comparable. Group: whether all nine, some or none of the "
                     "configurations are both.",
                     "lrrrrrrl", "Dataset & Rows & Faster & Slower & Comparable & Worse & Both & Group", body,
                     "results/store/metrics.csv")
        written.append(write(out_tables, f"outcomes_{algo.lower()}.tex", tab))
    return written


# ---------------------------------------------------------------- outcome map as a table (main sampler)

def _sd(v: float, digits: int = 2) -> str:
    return "" if pd.isna(v) else f"\\,{{\\scriptsize$\\pm${v:.{digits}f}}}"


HIT_KINDS = (("structural", "structurally equivalent"), ("behavioural", "fitness-equivalent"),
             ("nonequiv", "not equivalent"))
# Table S12 column names: the internal key 'behavioural' is the paper's (b) fitness-equivalent group
S12_NAMES = {"behavioural_mean": "fitness_equivalent_mean", "behavioural_sd": "fitness_equivalent_sd"}

_SPLIT_CACHE: dict = {}
AGE_WINDOWS = (5, 10)                    # A cache that keeps only entries of the last N generations


def hit_split(keep=None, store: Path = results.STORE) -> pd.DataFrame:
    """The (b) fitness-equivalent hits split by their predictions, and the hits a cache limited to the last N
    generations would keep, from the instrumented diagnostic runs (instrumented_runs.csv, one row per run, summed from
    its hit log). These runs evolve exactly as the stored diagnostic runs of the same seed (twin check). Per algorithm,
    dataset and fraction, summed over the seeds: hits; `beh` = fitness-equivalent hits whose program predicted every
    training row as the stored program did (behaviour-equivalent); `fitonly` = the others (equal fitness, different
    predictions); `seeds`. Empty when no instrumented run exists."""
    import instrument
    cols = ["algorithm", "dataset", "fraction", "seed", "hits", "beh", "fitonly", *[f"age_le{n}" for n in AGE_WINDOWS]]
    if store in _SPLIT_CACHE:
        s = _SPLIT_CACHE[store]
    elif not (_f(store, "instrumented_runs.csv")).is_file():
        s = pd.DataFrame(columns=cols)
    else:
        r = pd.read_csv(_f(store, "instrumented_runs.csv"))
        r = r[(r.setup == instrument.SETUP) & (r.twin_ok == True) & (r.sampler == MAIN_SAMPLER)  # noqa: E712
              & r.log_hits.notna()]
        s = _SPLIT_CACHE[store] = pd.DataFrame({
            "algorithm": r.algorithm, "dataset": r.dataset, "fraction": r.fraction.round(2), "seed": r.seed,
            "hits": r.log_hits.astype(int), "beh": r.log_hits_same_pred.astype(int),
            "fitonly": r.log_hits_equal_fit_only.astype(int),
            **{f"age_le{n}": r[f"log_hits_age_le{n}"].astype(int) for n in AGE_WINDOWS}}, columns=cols)
    s = s[s.seed.isin(set(PAPER_SEEDS))]
    if keep is not None:
        s = s[s.dataset.isin(keep)]
    return s.groupby(["algorithm", "dataset", "fraction"]).agg(seeds=("seed", "size"), hits=("hits", "sum"),
                                                               beh=("beh", "sum"), fitonly=("fitonly", "sum"),
                                                               **{f"age_le{n}": (f"age_le{n}", "sum") for n in AGE_WINDOWS}) \
        .assign(beh_share=lambda d: d.beh / (d.beh + d.fitonly)).reset_index()


def hit_kinds(keep=None, store: Path = results.STORE) -> pd.DataFrame:
    """Every hit of a diagnostic run (random rows, setup default) is one of: (a) structurally
    equivalent - the same program as the one stored under the key (same phenotype; always the correct fitness),
    (b) fitness-equivalent - a different program with the same full-training fitness (judged by fitness, not by
    the outputs), (c) not equivalent - a different program whose full-training fitness differs (the diagnostic fake
    hit). Per run: shares of all hits; columns algorithm, dataset, fraction, seed, hits, structural, behavioural,
    nonequiv."""
    r = pd.read_csv(_f(store, "runs.csv"), usecols=["algorithm", "setup", "dataset", "mode", "sampler", "fraction", "seed",
                                                  "hits", "same_phenotype_hits", "fake_hits"])
    r = r[(r.setup == SETUP) & (r["mode"] == "fec_diagnostic") & (r.sampler == MAIN_SAMPLER) & (r.hits > 0)]
    if keep is not None:
        r = r[r.dataset.isin(keep)]
    if (r.fake_hits > r.hits - r.same_phenotype_hits).any():
        raise TableError("a run has more fake hits than hits by a different program")
    return pd.DataFrame({"algorithm": r.algorithm, "dataset": r.dataset, "fraction": r.fraction.round(2),
                         "seed": r.seed, "hits": r.hits, "structural": r.same_phenotype_hits / r.hits,
                         "behavioural": (r.hits - r.same_phenotype_hits - r.fake_hits) / r.hits,
                         "nonequiv": r.fake_hits / r.hits})


def hit_kinds_table(keep, props) -> pd.DataFrame:
    """Table S12: per algorithm, dataset (ordered by rows) and fraction, mean and sd over the seeds of the shares of
    (a) structurally, (b) fitness- and (c) not equivalent hits, and the mean number of hits per run."""
    h = hit_kinds(keep)
    g = h.groupby(["algorithm", "dataset", "fraction"])
    t = g.agg(runs=("seed", "size"), hits_per_run=("hits", "mean"),
              **{f"{k}_mean": (k, "mean") for k, _ in HIT_KINDS}, **{f"{k}_sd": (k, "std") for k, _ in HIT_KINDS})
    t = t.reset_index().assign(rows=lambda d: d.dataset.map(props.set_index("dataset")["rows"]))
    cols = ["algorithm", "dataset", "rows", "fraction", "runs", "hits_per_run"] + \
        [f"{k}_{s}" for k, _ in HIT_KINDS for s in ("mean", "sd")]
    sp = hit_split(keep)[["algorithm", "dataset", "fraction", "seeds", "beh_share"]] \
        .rename(columns={"seeds": "split_runs", "beh_share": "behaviour_equivalent_share"})
    t = t.merge(sp, on=["algorithm", "dataset", "fraction"], how="left").astype({"split_runs": "Int64"})
    cols += ["split_runs", "behaviour_equivalent_share"]
    return t.sort_values(["algorithm", "rows", "dataset", "fraction"], kind="stable")[cols].round(4).rename(columns=S12_NAMES)


# ---------------------------------------------------------------- exact and behavioural repeats

_NOVEL_CACHE: dict = {}


def novel_by_run(store: Path = results.STORE) -> pd.DataFrame:
    """Per run (setup default; baseline runs and diagnostic runs of the main sampler): the number of distinct valid
    programs evaluated in the run, i.e. the sum over generations of novel_phenotypes (measurement.generation_extras)."""
    path = _f(store, "generations.csv")
    if path in _NOVEL_CACHE:
        return _NOVEL_CACHE[path]
    if not path.is_file():
        raise TableError(f"store file missing: {path} (run `python results.py all`)")
    cols = ["algorithm", "setup", "dataset", "mode", "sampler", "fraction", "seed", "novel_phenotypes"]
    parts = [c[(c.setup == SETUP) & ((c["mode"] == "baseline")
                                     | ((c["mode"] == "fec_diagnostic") & (c.sampler == MAIN_SAMPLER)))]
             for c in pd.read_csv(path, usecols=cols, chunksize=500_000)]
    g = pd.concat(parts).assign(fraction=lambda d: d.fraction.round(2))
    out = g.groupby(["algorithm", "dataset", "mode", "fraction", "seed"], dropna=False).novel_phenotypes.sum() \
        .rename("novel").reset_index()
    _NOVEL_CACHE[path] = out
    return out


def repeats(keep=None, store: Path = results.STORE) -> pd.DataFrame:
    """What a cache keyed on the program itself (GP tree / GE phenotype as a string) would find,
    against FEC. Per diagnostic run (main sampler): the evaluations that reached the cache (hits + misses) are a
    first evaluation of a distinct program (a miss, or a hit that only behavioural matching can give: `new`) or an
    exact repeat of a program evaluated earlier in the run (always a hit: `exact` = evaluations - distinct programs);
    fake hits as counted by the diagnostic check (an exact repeat of a program that got a fake hit gets the same wrong
    value). struct = hit rate of a structural cache, which never changes evolution: in the baseline run of the same
    seed, (valid evaluations - distinct programs) / valid evaluations. Both counts were checked against an exact
    counter on appendicitis and prnn_crabs (equal; GE baselines: a program with NaN fitness is not counted as
    distinct, so struct can be overcounted slightly, by at most about 2 points there)."""
    nov = novel_by_run(store)
    r = pd.read_csv(_f(store, "runs.csv"), usecols=["algorithm", "setup", "dataset", "mode", "sampler", "fraction", "seed",
                                                  "n_full_evals", "hits", "misses", "fake_hits"])
    r = r[r.setup == SETUP].assign(fraction=lambda d: d.fraction.round(2))
    if keep is not None:
        r = r[r.dataset.isin(keep)]
    b = r[r["mode"] == "baseline"].merge(nov[nov["mode"] == "baseline"][["algorithm", "dataset", "seed", "novel"]],
                                         on=["algorithm", "dataset", "seed"])
    b = b.assign(struct=(b.n_full_evals - b.novel) / b.n_full_evals)[["algorithm", "dataset", "seed", "struct"]]
    d = r[(r["mode"] == "fec_diagnostic") & (r.sampler == MAIN_SAMPLER)].merge(
        nov[nov["mode"] == "fec_diagnostic"].drop(columns="mode"), on=["algorithm", "dataset", "fraction", "seed"])
    d = d.assign(evals=d.hits + d.misses).assign(exact=lambda x: x.evals - x.novel).assign(new=lambda x: x.hits - x.exact)
    if (d.exact < 0).any() or (d.new < 0).any() or (b.struct < 0).any():
        raise TableError("repeat counts out of range: distinct programs exceed the evaluations or the hits")
    d = d.merge(b, on=["algorithm", "dataset", "seed"])
    return d[["algorithm", "dataset", "fraction", "seed", "evals", "hits", "exact", "new", "fake_hits", "struct"]] \
        .rename(columns={"fake_hits": "fake"})


REPEAT_MEASURES = (("struct", "structural cache hit rate (baseline runs)"), ("hit_rate", "FEC hit rate"),
                   ("exact_rate", "exact repeats / evaluations"), ("new_rate", "hits on new programs / evaluations"),
                   ("fake_rate", "fake hits / evaluations"), ("new_share", "hits on new programs / hits"))


def repeat_rates(keep=None, store: Path = results.STORE) -> pd.DataFrame:
    """Per run: the shares of REPEAT_MEASURES."""
    d = repeats(keep, store)
    return d.assign(hit_rate=d.hits / d.evals, exact_rate=d.exact / d.evals, new_rate=d.new / d.evals,
                    fake_rate=d.fake / d.evals, new_share=d.new / d.hits.where(d.hits > 0))


def repeats_summary(keep, props) -> pd.DataFrame:
    """Per algorithm, task (binary / multiclass) and fraction: the median over datasets of each dataset's mean."""
    per = repeat_rates(keep).groupby(["algorithm", "dataset", "fraction"])[[k for k, _ in REPEAT_MEASURES]].mean()
    per = per.reset_index().assign(task=lambda d: np.where(
        d.dataset.map(props.set_index("dataset")["classes"]) > 2, "multiclass", "binary"))
    return per.groupby(["algorithm", "task", "fraction"])[[k for k, _ in REPEAT_MEASURES]].median().reset_index() \
        .assign(n=per.groupby(["algorithm", "task", "fraction"]).size().values)


def hit_kinds_figure(corpus, metrics, props, out_tables, min_rows=0, **_) -> list[Path]:
    """fig_hit_kinds.pdf (§6): per dataset (ordered by rows), GE top, GP bottom, at
    REPEAT_FRACTION: the hits split into (a) structurally equivalent, (b) fitness-equivalent and (c) not
    equivalent (fake), mean share over the seeds (hit_kinds)."""
    if not (_f(results.STORE, "runs.csv")).is_file() or "classes" not in props:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs.use()
    keep = main_text_datasets(corpus, props, min_rows)
    h = hit_kinds(set(keep))
    per = h[np.isclose(h.fraction, REPEAT_FRACTION)].groupby(["algorithm", "dataset"])[[k for k, _ in HIT_KINDS]].mean()
    order = props.set_index("dataset").loc[[d for d in keep if d in set(h.dataset)], "rows"].sort_values(kind="stable").index
    colours = {"structural": SERIES_COLOUR, "behavioural": THIRD_COLOUR, "nonequiv": SECOND_COLOUR}
    names = {"structural": "(a) structurally equivalent", "behavioural": "(b) fitness-equivalent",
             "nonequiv": "(c) not equivalent (fake hit)"}
    # (b) split by predictions where the instrumented runs recorded them (hit_split); elsewhere unsplit
    sp = hit_split(set(keep))
    sp = sp[np.isclose(sp.fraction, REPEAT_FRACTION)].set_index(["algorithm", "dataset"]).beh_share
    measured = {d for d in order if all((a, d) in sp.index for a in ("GE", "GP"))}
    partial = bool(measured) and len(measured) < len(order)        # mark the measured ones only if not all are
    fs.use()
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 4.8), sharex=True)
    x = np.arange(len(order))
    for ax, algo in zip(axes, ("GE", "GP")):
        d = per.loc[algo].reindex(order)
        bottom = np.zeros(len(order))
        for k, _ in HIT_KINDS:
            if k == "behavioural" and measured:
                share = np.array([sp.get((algo, ds), np.nan) for ds in order])
                has = ~np.isnan(share)
                same, rest = d[k].values * np.where(has, share, 0), d[k].values * np.where(has, 1 - share, 0)
                ax.bar(x, same, bottom=bottom, width=0.72, color=THIRD_COLOUR, edgecolor="white", linewidth=1.0,
                       label="(b) fitness-equivalent: same predictions", zorder=2)
                ax.bar(x, rest, bottom=bottom + same, width=0.72, color=THIRD_COLOUR, alpha=0.4, edgecolor="white",
                       linewidth=1.0, label="(b) fitness-equivalent: equal fitness only", zorder=2)
                if partial:
                    ax.bar(x, np.where(has, 0, d[k].values), bottom=bottom, width=0.72, color=THIRD_COLOUR,
                           hatch="////", edgecolor="white", linewidth=1.0,
                           label="(b) fitness-equivalent: predictions not recorded", zorder=2)
            else:
                ax.bar(x, d[k], bottom=bottom, width=0.72, color=colours[k], edgecolor="white", linewidth=1.0,
                       label=names[k], zorder=2)
            bottom += d[k].values
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"{algo}: share of hits")
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    multi = props.set_index("dataset")["classes"] > 2                # multiclass tables marked with an asterisk
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([d.replace("_information", "") + ("*" if multi.get(d, False) else "")
                             + ("$^\\dagger$" if partial and d in measured else "") for d in order],
                            rotation=45, ha="right", fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3 if len(labels) <= 3 else 2, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.95 if len(labels) <= 3 else 0.90))
    path = out_tables / "fig_hit_kinds.pdf"
    fs.save(fig, path)
    plt.close(fig)
    return [path]


REPEAT_FRACTION = 0.05                   # the fraction the figure quotes; Table S12 gives all three


def repeats_numbers(keep, props) -> dict:
    """§6: hit rate of a structural cache in the baseline runs, median over datasets per task (GE)
    and the range over datasets (GP)."""
    s = repeats_summary(keep, props)
    per = repeat_rates(keep).groupby(["algorithm", "dataset"])[["struct"]].mean().reset_index()
    at = lambda task: s[(s.algorithm == "GE") & (s.task == task)].struct.iloc[0]
    gp = per[per.algorithm == "GP"].struct
    return {"RepGEBinStruct": f"{100 * at('binary'):.0f}", "RepGEMultiStruct": f"{100 * at('multiclass'):.0f}",
            "RepGPStructMin": f"{100 * gp.min():.0f}", "RepGPStructMax": f"{100 * gp.max():.0f}"}


def hit_kind_task(keep, props) -> tuple[pd.DataFrame, dict]:
    """§5: per algorithm and dataset at REPEAT_FRACTION, the mean share of each kind of hit;
    medians per task and the Spearman correlation of each share with the number of rows."""
    from scipy.stats import spearmanr
    h = hit_kinds(keep)
    per = h[np.isclose(h.fraction, REPEAT_FRACTION)].groupby(["algorithm", "dataset"])[[k for k, _ in HIT_KINDS]] \
        .mean().reset_index()
    pr = props.set_index("dataset")
    per = per.assign(task=np.where(per.dataset.map(pr["classes"]) > 2, "multiclass", "binary"),
                     rows=per.dataset.map(pr["rows"]))
    rho = {(a, k): spearmanr(x.rows, x[k]) for a, x in per.groupby("algorithm") for k, _ in HIT_KINDS}
    return per, rho


def _pooled_kind(keep, algo: str, fraction: float, kind: str) -> float:
    h = hit_kinds(keep)
    x = h[(h.algorithm == algo) & np.isclose(h.fraction, fraction)]
    return float((x[kind] * x.hits).sum() / x.hits.sum())


_OPS_CACHE: dict = {}


def baseline_ops(keep, store: Path = results.STORE) -> pd.Series:
    """Per (algorithm, dataset): mean operator calls per program in the baseline runs (avg_operators over generations
    and seeds)."""
    path = _f(store, "generations.csv")
    if path not in _OPS_CACHE:
        cols = ["algorithm", "setup", "dataset", "mode", "avg_operators"]
        g = pd.concat([c[(c.setup == SETUP) & (c["mode"] == "baseline")]
                       for c in pd.read_csv(path, usecols=cols, chunksize=500_000)])
        _OPS_CACHE[path] = g.groupby(["algorithm", "dataset"]).avg_operators.mean()
    s = _OPS_CACHE[path]
    return s[s.index.get_level_values("dataset").isin(keep)]


def behavioural_per_eval(keep, props) -> pd.DataFrame:
    """Per algorithm, task and fraction: median over datasets of the fitness-equivalent hits per evaluation,
    i.e. the full evaluations a cache matching programs exactly would have run although the fitness was the same."""
    h, r = hit_kinds(keep), repeat_rates(keep)
    m = h.merge(r[["algorithm", "dataset", "fraction", "seed", "hit_rate"]], on=["algorithm", "dataset", "fraction", "seed"])
    m = m.assign(beh=m.behavioural * m.hit_rate,
                 task=np.where(m.dataset.map(props.set_index("dataset")["classes"]) > 2, "multiclass", "binary"))
    return m.groupby(["algorithm", "task", "dataset", "fraction"]).beh.mean().groupby(
        ["algorithm", "task", "fraction"]).median()


NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
                10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen",
                17: "seventeen"}


def split_numbers(keep) -> dict:
    """§5.4: the fitness-equivalent hits split by predictions at REPEAT_FRACTION, over
    the datasets the instrumented runs cover. Per algorithm: range over datasets of the behaviour-equivalent share of
    the fitness-equivalent hits, and the largest share of all hits that had equal fitness only (and its dataset).
    SplitPilot is a red placeholder while any configuration has fewer than 30 seeds."""
    sp = hit_split(keep)
    sp = sp[np.isclose(sp.fraction, REPEAT_FRACTION)]
    both = sorted(set(sp[sp.algorithm == "GE"].dataset) & set(sp[sp.algorithm == "GP"].dataset))
    out = {"SplitN": str(len(both)), "SplitNWord": NUMBER_WORDS.get(len(both), str(len(both)))}
    split_def = ("the fitness-equivalent hits are split into those whose program predicted every training row as the "
                 "stored program did (same predictions) and those with equal fitness only")
    n_all = len(set(keep))                                     # the datasets of the figure
    out["SplitCaption"] = ("" if not both else f"{split_def[0].upper()}{split_def[1:]}." if len(both) >= n_all
                           else f"On the datasets marked $\\dagger$, {split_def}; elsewhere the predictions were not "
                                "recorded.")
    if not both:
        out["SplitPilot"] = "\\RES{no instrumented runs yet}"
        for a in ("GE", "GP"):
            for k in ("BehMin", "BehMax", "FitOnlyMax"):
                out[f"{a}Split{k}"] = "\\RES{--}"
            out[f"{a}SplitFitOnlyMaxDS"] = "\\RES{--}"
        return out
    sp = sp[sp.dataset.isin(both)]
    lo, hi = int(sp.seeds.min()), int(sp.seeds.max())
    out["SplitPilot"] = "" if lo >= 30 else f"\\RES{{pilot: {lo}{'' if lo == hi else f'--{hi}'} seeds per configuration}}"
    for a in ("GE", "GP"):
        x = sp[sp.algorithm == a].assign(fitonly_hits=lambda d: d.fitonly / d.hits)
        out[f"{a}SplitBehMin"], out[f"{a}SplitBehMax"] = f"{100 * x.beh_share.min():.0f}", f"{100 * x.beh_share.max():.0f}"
        top = x.loc[x.fitonly_hits.idxmax()]
        out[f"{a}SplitFitOnlyMax"], out[f"{a}SplitFitOnlyMaxDS"] = f"{100 * top.fitonly_hits:.0f}", _tt(top.dataset)
    return out


def hit_kind_numbers(keep, props) -> dict:
    per, rho = hit_kind_task(keep, props)
    med = per.groupby(["algorithm", "task"])[[k for k, _ in HIT_KINDS]].median()
    out = {}
    be = behavioural_per_eval(keep, props)
    for a in ("GE", "GP"):
        for task, tk in (("binary", "Bin"), ("multiclass", "Multi")):
            out[f"{a}BehEval{tk}"] = f"{100 * be.loc[(a, task, REPEAT_FRACTION)]:.0f}"
    ops = baseline_ops(keep).loc["GE"]
    multi = props.set_index("dataset")["classes"].reindex(ops.index) > 2
    out["GEOpsBin"], out["GEOpsMulti"] = f"{ops[~multi].median():.1f}", f"{ops[multi].median():.1f}"
    for a in ("GE", "GP"):
        for task, tk in (("binary", "Bin"), ("multiclass", "Multi")):
            for k, _ in HIT_KINDS:
                out[f"{a}Kind{k.capitalize()}{tk}"] = f"{100 * med.loc[(a, task), k]:.0f}"
        for k, _ in HIT_KINDS:
            r = rho[(a, k)]
            out[f"{a}RhoKind{k.capitalize()}Rows"], out[f"{a}RhoKind{k.capitalize()}RowsP"] = \
                f"{r.statistic:.2f}", _p(r.pvalue)
    return out


def hit_kind_claims(keep, props) -> list[tuple[str, str, bool]]:
    per, rho = hit_kind_task(keep, props)
    med = per.groupby(["algorithm", "task"])[[k for k, _ in HIT_KINDS]].median()
    return [
        ("kinds/multi-behavioural", "§5 (points 2, 4): on multiclass tables most hits were fitness-equivalent, "
         "more than on binary tables, with fewer structural and fake hits (both algorithms, medians at 5%)",
         all(med.loc[(a, "multiclass"), "behavioural"] > max(0.5, med.loc[(a, "binary"), "behavioural"])
             and med.loc[(a, "multiclass"), "structural"] < med.loc[(a, "binary"), "structural"]
             and med.loc[(a, "multiclass"), "nonequiv"] < med.loc[(a, "binary"), "nonequiv"] for a in ("GE", "GP"))),
        ("kinds/fake-falls-rows", "§5 (points 2, 4): the share of fake hits fell with the number of rows in both "
         "algorithms (Spearman < 0, p <= 0.05)",
         all(rho[(a, "nonequiv")].statistic < 0 and rho[(a, "nonequiv")].pvalue <= ALPHA for a in ("GE", "GP"))),
        ("kinds/gp-behavioural-rises-rows", "§5 (points 2, 4): in GP the behavioural share rose with rows (p <= 0.05)",
         rho[("GP", "behavioural")].statistic > 0 and rho[("GP", "behavioural")].pvalue <= ALPHA),
        ("kinds/structural-little-change", "§5 (points 2, 4): 'the share of structurally equivalent hits changed little' "
         "with the fraction (pooled, 1% vs 10%: within 6 points)",
         all(abs(_pooled_kind(keep, a, 0.1, "structural") - _pooled_kind(keep, a, 0.01, "structural")) <= 0.06
             for a in ("GE", "GP"))),
        ("kinds/beh-eval-multi", "§5.4: fitness-equivalent hits replaced more full evaluations on multiclass "
         "than on binary tables (both algorithms, every fraction; medians)",
         all(behavioural_per_eval(keep, props).loc[(a, "multiclass", f)] > behavioural_per_eval(keep, props).loc[(a, "binary", f)]
             for a in ("GE", "GP") for f in (0.01, 0.05, 0.1))),
        ("kinds/ge-multi-larger", "§5.4: in GE, multiclass programs were several times larger than binary ones "
         "(median operator calls, baseline runs, at least 3x)",
         (lambda o, mc: o[mc].median() >= 3 * o[~mc].median())(baseline_ops(keep).loc["GE"],
             props.set_index("dataset")["classes"].reindex(baseline_ops(keep).loc["GE"].index) > 2)),
        ("kinds/structural-not-rows", "§5 (points 2, 4): the structural share was not associated with rows (p > 0.05)",
         all(rho[(a, "structural")].pvalue > ALPHA for a in ("GE", "GP"))),
    ]


def repeats_claims(keep, props) -> list[tuple[str, str, bool]]:
    s = repeats_summary(keep, props)
    per = repeat_rates(keep).groupby(["algorithm", "dataset", "fraction"])[[k for k, _ in REPEAT_MEASURES]].mean() \
        .reset_index()
    ge_b, ge_m = s[(s.algorithm == "GE") & (s.task == "binary")], s[(s.algorithm == "GE") & (s.task == "multiclass")]
    return [
        ("rep/gp-few-repeats", "§6 (point 2): a structural cache would hit on 2-14% of GP's evaluations (< 15% on "
         "every dataset)", bool((per[per.algorithm == "GP"].struct < 0.15).all())),
        ("rep/ge-binary-more", "§6 (point 2): GE repeats programs more on binary than on multiclass tables",
         bool((ge_m.struct.values < ge_b.struct.values).all())),
        ("rep/ge-binary-much", "§5.4: in GE on binary tables exact repeats alone accounted for much of the hit rate "
         "(median structural-cache rate >= half of the median FEC hit rate at every fraction)",
         bool((ge_b.struct >= 0.5 * ge_b.hit_rate).all())),
        ("rep/structural-fewer", "§6 (point 2): 'far fewer than FEC's hits' (structural cache hit rate below FEC's hit "
         "rate on every dataset and fraction)", bool((per.struct < per.hit_rate).all())),
    ]


# ---------------------------------------------------------------- how large the fake hits are

_SEV_CACHE: dict = {}


def severity_generations(store: Path = results.STORE) -> pd.DataFrame:
    """Per generation of the diagnostic runs (setup default, main sampler) that had at least one fake hit: the sums
    needed for the size of the fitness errors. fec.py records per generation the mean of (cached - full) and of
    |cached - full| over all hits (correct hits add 0) and the largest |cached - full|; so the error per fake hit is
    |mean| x hits / fake hits. Fitness is the training error rate, so errors are in percentage points x 100."""
    path = _f(store, "generations.csv")
    if path in _SEV_CACHE:
        return _SEV_CACHE[path]
    cols = ["algorithm", "setup", "dataset", "mode", "sampler", "fraction", "seed", "gen", "fec_hits",
            "fec_fake_hits", "fec_fitness_diff", "fec_abs_fitness_diff", "fec_max_abs_fitness_diff"]
    g = pd.concat([c[(c.setup == SETUP) & (c["mode"] == "fec_diagnostic") & (c.sampler == MAIN_SAMPLER)
                     & (c.fec_fake_hits > 0)] for c in pd.read_csv(path, usecols=cols, chunksize=500_000)])
    g = g.assign(fraction=g.fraction.round(2), abs_sum=g.fec_abs_fitness_diff * g.fec_hits,
                 sgn_sum=g.fec_fitness_diff * g.fec_hits, worst=g.fec_max_abs_fitness_diff)
    _SEV_CACHE[path] = g
    return g


def _severity(x: pd.DataFrame) -> pd.Series:
    fk = x.fec_fake_hits.sum()
    return pd.Series({"fake_hits": int(fk), "mean_error_pp": 100 * x.abs_sum.sum() / fk,
                      "mean_signed_pp": 100 * x.sgn_sum.sum() / fk, "worst_median_pp": 100 * x.worst.median(),
                      "worst_p90_pp": 100 * x.worst.quantile(0.9), "worst_max_pp": 100 * x.worst.max(),
                      "gens_worst_over_1pp": (x.worst > 0.01).mean(), "gens_worst_over_5pp": (x.worst > 0.05).mean()})


def fake_severity(keep, by=("algorithm", "dataset", "fraction")) -> pd.DataFrame:
    g = severity_generations()
    g = g[g.dataset.isin(keep)]
    cols = list(by) + ["fake_hits", "mean_error_pp", "mean_signed_pp", "worst_median_pp", "worst_p90_pp", "worst_max_pp",
                       "gens_worst_over_1pp", "gens_worst_over_5pp"]          # as _severity
    if g.empty:                                                # no fake hits (e.g. a small test store)
        return pd.DataFrame(columns=cols)
    return g.groupby(list(by))[["fec_fake_hits", "abs_sum", "sgn_sum", "worst"]].apply(_severity).reset_index()


def severity_supp_table(keep, props) -> pd.DataFrame:
    """Table S13: fake_severity per algorithm, dataset (ordered by rows) and fraction."""
    t = fake_severity(keep).assign(rows=lambda d: d.dataset.map(props.set_index("dataset")["rows"]))
    cols = ["algorithm", "dataset", "rows", "fraction"] + [c for c in t.columns if c not in
                                                           ("algorithm", "dataset", "rows", "fraction")]
    return t.sort_values(["algorithm", "rows", "dataset", "fraction"], kind="stable")[cols].round(4)


def _small_pct(v: float) -> str:
    """A share in percent; a non-zero share below 0.05 is shown as <0.1, not as 0.0."""
    return "0" if v == 0 else ("$<$0.1" if v < 0.05 else f"{v:.1f}")


def severity_table(corpus, metrics, props, out_tables, min_rows=0, **_) -> list[Path]:
    """fake_severity.tex (§5.5): the size of the fitness errors of fake hits, per algorithm and
    fraction over all tables; per table in Table S13."""
    if not (_f(results.STORE, "generations.csv")).is_file():
        return []
    s = fake_severity(set(main_text_datasets(corpus, props, min_rows)), ("algorithm", "fraction"))
    body = []
    for i, r in enumerate(s.itertuples()):
        lead = ("\\addlinespace[3pt]" if r.algorithm == "GP" and r.fraction == 0.01 else "") + \
            (r.algorithm if r.fraction == 0.01 else "")
        body.append(" & ".join([lead, FRACTION_WORD[round(r.fraction, 2)], f"{r.mean_error_pp:.2f}",
                                f"{r.worst_median_pp:.2f}", f"{r.worst_p90_pp:.1f}", f"{100 * r.gens_worst_over_1pp:.0f}",
                                _small_pct(100 * r.gens_worst_over_5pp), f"{r.worst_max_pp:.1f}"]))
    tex = _table("tab:fake-severity",
                 "How far the fitness of a fake hit was from its true value, over all tables (random fingerprint "
                 "rows, diagnostic runs, 30 seeds; generations with at least one fake hit). Mean error: mean "
                 "absolute difference between the cached and the full-training-set error rate per fake hit. Worst: "
                 "the largest difference in a generation, as median and 90th percentile over the generations, the "
                 "share of generations in which it exceeded 1 and 5 percentage points, and the largest value "
                 "observed. Differences in percentage points (pp). Per table in Table~S13.",
                 "llrrrrrr", "Alg. & Fraction & Mean error (pp) & Worst, median (pp) & Worst, 90th pct (pp) & "
                 "$>$1 pp (\\%) & $>$5 pp (\\%) & Largest (pp)",
                 body, "results/store/generations.csv", colsep="4pt")
    return [write(out_tables, "fake_severity.tex", tex)]


def severity_numbers(keep, props) -> dict:
    from scipy.stats import spearmanr
    s = fake_severity(keep, ("algorithm", "fraction")).set_index(["algorithm", "fraction"])
    d = fake_severity(keep)
    small = list(props.set_index("dataset").loc[sorted(keep), "rows"].sort_values(kind="stable").index[:3])
    out = {}
    for a in ("GE", "GP"):
        for f, w in ((0.01, "One"), (0.05, "Five"), (0.1, "Ten")):
            r = s.loc[(a, f)]
            out[f"{a}SevMean{w}"] = f"{r.mean_error_pp:.2f}"
            out[f"{a}SevNinety{w}"] = f"{r.worst_p90_pp:.1f}"
            out[f"{a}SevMax{w}"] = f"{r.worst_max_pp:.1f}"
            out[f"{a}SevOverOne{w}"] = f"{100 * r.gens_worst_over_1pp:.0f}"
        per = d[d.algorithm == a].groupby("dataset").mean_error_pp.mean()
        rho = spearmanr(per.index.map(props.set_index("dataset")["rows"]), per.values)
        out[f"{a}RhoSevRows"], out[f"{a}RhoSevRowsP"] = f"{rho.statistic:.2f}", _p(rho.pvalue)
        out[f"{a}SevSignedOne"] = f"{s.loc[(a, 0.01)].mean_signed_pp:.2f}"
    out["SevMeanMaxHigh"] = f"{max(s.loc[(a, f)].mean_error_pp for a in ('GE', 'GP') for f in (0.05, 0.1)):.2f}"
    x = d[d.dataset.isin(small) & np.isclose(d.fraction, 0.01)]
    out["SevSmallOverFiveMin"] = f"{100 * x.gens_worst_over_5pp.min():.0f}"
    out["SevSmallNames"] = _names(small)
    r = pd.read_csv(_f(results.STORE, "runs.csv"), usecols=["setup", "mode", "sampler", "fraction", "dataset", "sample_size"])
    r = r[(r.setup == SETUP) & (r["mode"] == "fec") & (r.sampler == MAIN_SAMPLER) & np.isclose(r.fraction, 0.01)
          & r.dataset.isin(small)]
    out["SevSmallRowsMin"], out["SevSmallRowsMax"] = f"{int(r.sample_size.min())}", f"{int(r.sample_size.max())}"
    rest = d[~(d.dataset.isin(small) & np.isclose(d.fraction, 0.01))]   # everything but the small tables at 1%
    out["SevRestMax"] = f"{rest.worst_max_pp.max():.1f}"
    out["SevRestOverFiveMax"] = f"{100 * rest.gens_worst_over_5pp.max():.1f}"
    return out


def severity_claims(keep, props, ann) -> list[tuple[str, str, bool]]:
    from scipy.stats import spearmanr
    d = fake_severity(keep)
    small = list(props.set_index("dataset").loc[sorted(keep), "rows"].sort_values(kind="stable").index[:3])
    out = []
    for a in ("GE", "GP"):
        per = d[d.algorithm == a].groupby("dataset").mean_error_pp.mean()
        rho = spearmanr(per.index.map(props.set_index("dataset")["rows"]), per.values)
        out.append((f"sev/{a}/falls-rows", f"§5.5 {a}: error per fake hit fell with rows (rho < 0, p <= 0.05)",
                    rho.statistic < 0 and rho.pvalue <= ALPHA))
        x = d[d.algorithm == a].set_index(["dataset", "fraction"]).mean_error_pp
        out.append((f"sev/{a}/falls-fraction", f"§5.5 {a}: on every table the error per fake hit fell from 1% to 5% "
                    "to 10%", all(x[(ds, 0.01)] > x[(ds, 0.05)] > x[(ds, 0.1)] for ds in set(d.dataset))))
    pooled = fake_severity(keep, ("algorithm", "fraction"))
    out.append(("sev/abstract-small", "Abstract: at 5% and 10% a fake hit changed fitness by at most 0.15 pp on average "
                "(both algorithms)", bool((pooled[~np.isclose(pooled.fraction, 0.01)].mean_error_pp <= 0.15).all())))
    rest = d[~(d.dataset.isin(small) & np.isclose(d.fraction, 0.01))]
    out.append(("sev/large-only-small-1pct", "§5.5: errors above 5 pp were common only at 1% on the three smallest "
                "tables (>= 75% of their generations) and rare elsewhere (<= 2% of generations, never above 7 pp)",
                bool((d[d.dataset.isin(small) & np.isclose(d.fraction, 0.01)].gens_worst_over_5pp >= 0.75).all()
                     and (rest.gens_worst_over_5pp <= 0.02).all() and (rest.worst_max_pp <= 7).all())))
    small_ann = ann[ann.dataset.isin(small)]
    out.append(("sev/small-never-equivalent", "§5.5: the three smallest tables were never confirmed equivalent in "
                "accuracy, at any fraction, in either algorithm", bool((small_ann.verdict != "equivalent").all())))
    return out


def seconds_saved(store: Path = results.STORE) -> pd.DataFrame:
    """Per (algorithm, dataset, fraction), main sampler, setup default: seconds saved per run by FEC = mean baseline
    time - mean FEC timing-run time, for fitness evaluation (eval_saved_s) and the whole run (wall_saved_s);
    negative = FEC slower."""
    path = _f(store, "runs.csv")
    if not path.is_file():
        return pd.DataFrame(columns=["algorithm", "dataset", "fraction", "eval_saved_s", "wall_saved_s"])
    r = pd.read_csv(path, usecols=["algorithm", "setup", "dataset", "mode", "sampler", "fraction", "eval_time_sec",
                                   "wall_time_sec"])
    r = r[r.setup == SETUP]
    b = r[r["mode"] == "baseline"].groupby(["algorithm", "dataset"])[["eval_time_sec", "wall_time_sec"]].mean()
    f = r[(r["mode"] == "fec") & (r.sampler == MAIN_SAMPLER)].assign(fraction=lambda d: d.fraction.round(2)) \
        .groupby(["algorithm", "dataset", "fraction"])[["eval_time_sec", "wall_time_sec"]].mean().reset_index()
    f = f.join(b, on=["algorithm", "dataset"], rsuffix="_base")
    return f.assign(eval_saved_s=f.eval_time_sec_base - f.eval_time_sec,
                    wall_saved_s=f.wall_time_sec_base - f.wall_time_sec)[
        ["algorithm", "dataset", "fraction", "eval_saved_s", "wall_saved_s"]]


def _saved(v) -> str:
    """'(+0.64 s)' / '(+28.9 m)' / '(+2.9 h)': time saved per run in the unit that fits (s below a minute, m below
    an hour, else h); negative = FEC slower."""
    if v is None or pd.isna(v):
        return ""
    a = abs(v)
    if a < 60:
        num, unit = (f"{a:.2f}" if a < 1 else f"{a:.1f}"), "s"
    elif a < 3600:
        num, unit = f"{a / 60:.1f}", "m"
    else:
        num, unit = f"{a / 3600:.1f}", "h"
    sign = "" if float(num) == 0 else ("+" if v > 0 else "-")
    return f"\\,{{\\scriptsize({sign}{num}\\,{unit})}}"


def _dur(v) -> str:
    """A signed duration in the unit that fits, as in Table 5 ('0.64 s', '5.8 min', '2.9 h'); for the text."""
    a = abs(v)
    txt = (f"{a:.2f}~s" if a < 1 else f"{a:.1f}~s" if a < 60 else f"{a / 60:.1f}~min" if a < 3600
           else f"{a / 3600:.1f}~h")
    return ("$-$" if v < 0 else "") + txt


def _st_cell(r, seconds: bool = True) -> str:
    v = f"{r.S_T:.2f}{_sd(r.S_T_sd)}{_saved(getattr(r, 'eval_saved_s', None)) if seconds else ''}"
    if r.speed == "faster":
        return f"\\textbf{{$\\blacktriangle$\\,{v}}}"
    if r.speed == "slower":
        return f"$\\blacktriangledown$\\,{v}"
    return v


_WALL_CACHE: dict = {}


def wall_tests(store: Path = results.STORE) -> pd.DataFrame:
    """Per (algorithm, dataset, fraction), main sampler: paired Wilcoxon p of the whole-run wall times of the
    baseline and FEC runs (timing runs, paired by seed), the same test as for evaluation time."""
    from analysis import wilcoxon_paired
    path = _f(store, "runs.csv")
    if path in _WALL_CACHE:
        return _WALL_CACHE[path]
    r = pd.read_csv(path, usecols=["algorithm", "setup", "dataset", "mode", "sampler", "fraction", "seed",
                                   "wall_time_sec"])
    r = r[r.setup == SETUP]
    base = r[r["mode"] == "baseline"][["algorithm", "dataset", "seed", "wall_time_sec"]]
    fec = r[(r["mode"] == "fec") & (r.sampler == MAIN_SAMPLER)].assign(fraction=lambda d: d.fraction.round(2))
    x = fec.merge(base, on=["algorithm", "dataset", "seed"], suffixes=("", "_base"))
    x = x[x.dataset.isin(set(paper_corpus().dataset))]                 # Holm family = the paper's 51 tests
    out = pd.DataFrame([{"algorithm": a, "dataset": d, "fraction": f,
                         "p_wall": float(wilcoxon_paired(g.wall_time_sec_base, g.wall_time_sec)["p"])}
                        for (a, d, f), g in x.groupby(["algorithm", "dataset", "fraction"])])
    if not len(out):
        out = pd.DataFrame(columns=["algorithm", "dataset", "fraction", "p_wall"])
    from analysis import holm
    out["p_wall_holm"] = (out.groupby("algorithm").p_wall.transform(lambda p: holm(p.to_numpy()))
                          if len(out) else pd.Series(dtype=float))
    _WALL_CACHE[path] = out
    return out


def with_wall(ann: pd.DataFrame, store: Path = results.STORE) -> pd.DataFrame:
    """ann with p_wall, p_wall_holm and wall_speed (faster / slower / ''), marked like the evaluation-time speed
    (Holm over the algorithm's 51 paper tests)."""
    if not (_f(store, "runs.csv")).is_file() or ann.empty:
        return ann.assign(p_wall=np.nan, p_wall_holm=np.nan, wall_speed="")
    x = ann.assign(fraction=ann.fraction.round(2)).merge(wall_tests(store), on=["algorithm", "dataset", "fraction"],
                                                          how="left")
    x["wall_speed"] = _speed(x.p_wall_holm, x.S_wall)
    return x


def _wall_cell(r, seconds: bool = True, sd: bool = True) -> str:
    v = (f"{r.S_wall:.2f}{_sd(r.S_wall_sd) if sd else ''}"
         f"{_saved(getattr(r, 'wall_saved_s', None)) if seconds else ''}")
    if r.wall_speed == "faster":
        return f"\\textbf{{$\\blacktriangle$\\,{v}}}"
    if r.wall_speed == "slower":
        return f"$\\blacktriangledown$\\,{v}"
    return v


def _acc_cell(r) -> str:
    v = f"{_pp(r.dAcc)}{_sd(100 * r.dAcc_sd)}"
    if r.verdict == "equivalent":
        return f"\\textbf{{$\\approx$\\,{v}}}"
    if r.verdict == "worse":                                   # its own symbol: the triangle marks slower speed
        return f"$\\times$\\,{v}"
    return v


TWO_LINE_NAMES = {"Run_or_walk_information": ("Run_or_walk_", "information")}   # Table 5: frees column width


def _tt_two_lines(name: str, bold: bool = False) -> str:
    """A dataset name for a table cell; the names in TWO_LINE_NAMES are split over two lines (top-aligned)."""
    wrap = (lambda t: "\\textbf{" + t + "}") if bold else (lambda t: t)
    if name not in TWO_LINE_NAMES:
        return wrap(_tt(name))
    return "\\begin{tabular}[t]{@{}l@{}}" + " \\\\ ".join(wrap(_tt(part)) for part in TWO_LINE_NAMES[name]) \
        + "\\end{tabular}"


def outcome_map_tables(corpus, metrics, props, out_tables, min_rows=0, **_) -> list[Path]:
    """outmap.tex: one table for GE and GP (the table form of the outcome maps). Per dataset (ordered by rows) a GE
    line and a GP line with S_T and dAcc (mean +- sd) at each fraction of the main sampler, marked as in the maps;
    a dataset with no result for either algorithm gets one 'running' line."""
    keep = main_text_datasets(corpus, props, min_rows)
    ann = with_wall(annotated(metrics[metrics.dataset.isin(keep)]))
    sv = seconds_saved()                                    # seconds saved beside each speedup
    if len(sv):
        ann = ann.assign(_fr=ann.fraction.round(2)).merge(sv.rename(columns={"fraction": "_fr"}),
                                                          on=["algorithm", "dataset", "_fr"], how="left")
    p = props.set_index("dataset")
    order = sorted(keep, key=lambda d: (p["rows"].get(d, float("inf")), d))
    fracs = sorted(ann.fraction.round(2).unique()) or [0.01, 0.05, 0.10]
    idx = {(r.algorithm, r.dataset, round(float(r.fraction), 2)): r for r in ann.itertuples()}
    n = 3 * len(fracs)                                  # S_T, S_wall and dAcc per fraction
    both = {(r.algorithm, r.dataset) for r in ann.itertuples()       # some fraction faster and equivalent
            if r.speed == "faster" and r.verdict == "equivalent"}
    body = []                                           # one block per algorithm, no Alg. column
    for algo in ("GE", "GP"):
        body.append(("\\midrule\n" if body else "") + f"\\multicolumn{{{3 + n}}}{{l}}{{\\textbf{{{algo}}}}}")
        for d in order:
            rows = _rows(p["rows"].get(d)) if d in p.index else "--"
            gb = _num(p["difficulty_gb"].get(d), 2) if d in p.index else "--"
            name = _tt_two_lines(d, bold=(algo, d) in both)       # long names on two lines
            if not any((algo, d, f) in idx for f in fracs):
                body.append(" & ".join([name, rows, gb, f"\\multicolumn{{{n}}}{{c}}{{\\RES{{running}}}}"]))
                continue
            cells = []
            for f in fracs:
                r = idx.get((algo, d, f))
                cells += ([_st_cell(r, seconds=False), _wall_cell(r, seconds=False), _acc_cell(r)] if r is not None
                          else ["--", "--", "--"])
            body.append(" & ".join([name, rows, gb] + cells))
    head_top = (" & & & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{FRACTION_WORD[f]}}}" for f in fracs)
                + " \\\\\n" + "".join(f"\\cmidrule(lr){{{4 + 3 * i}-{6 + 3 * i}}}" for i in range(len(fracs))) + "\n")
    head = head_top + "Dataset & Rows & GB & " + " & ".join(["$S_T$ & $S_\\mathrm{wall}$ & $\\Delta$Acc"] * len(fracs))
    worse = (ann.verdict == "worse").any()                         # mention x only if it occurs
    t = _table("tab:outmap",
               "GE and GP with random fingerprint rows at each fingerprint fraction: evaluation-time speedup $S_T$ and "
               "accuracy difference $\\Delta$Acc (FEC minus baseline, percentage points). $S_T$ (Equation~(\\ref{eq:st})), "
               "\\rev{equal to the ratio of the mean times, $\\pm$ the standard deviation of the 30 per-run ratios "
               "$T_r^{\\mathrm{base}}/T_r^{\\mathrm{FEC}}$}; \\rev{$S_\\mathrm{wall}$: the same ratio for the time of "
               "the whole run, $\\pm$ the standard deviation of the 30 per-run ratios}; "
               "$\\Delta$Acc: mean $\\pm$ sd of the 30 paired differences. "
               "$\\blacktriangle$/$\\blacktriangledown$: significantly faster/slower (paired Wilcoxon, \\rev{Holm-adjusted} "
               "$p \\leq 0.05$ \\rev{over the algorithm's 51 configurations}); "
               "$\\approx$: practically equivalent"
               + (", $\\times$: worse" if worse else "")
               + f" (Bayesian correlated $t$-test, probability $\\geq {THRESHOLD}$, region $\\pm 1$ pp). "
               "Dataset name in bold: FEC was both significantly faster and practically equivalent at one or more "
               "fractions, for that algorithm. GB: balanced accuracy of gradient boosting. "
               "\\rev{The seconds saved per run}: Tables S2 (GE) and S3 (GP).",
               "lrr" + "rrr" * len(fracs), head, body, "results/store/metrics.csv", size="\\footnotesize",
               colsep="2.4pt", overhang=True)                # portrait, full width
    return [write(out_tables, "outmap.tex", t)]


# ---------------------------------------------------------------- numbers for the results text

FRACTION_WORD = {0.01: "1\\%", 0.05: "5\\%", 0.10: "10\\%"}
SAMPLER_WORD = {"farthest_point": "farthest-point", "random": "random", "stratified": "stratified"}


def _cfg(r) -> str:
    return f"{SAMPLER_WORD[r.sampler]} {FRACTION_WORD[round(float(r.fraction), 2)]}"


def _names(names) -> str:
    names = [_tt(n) for n in names]
    if not names:
        return "none"
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _int(n) -> str:
    return f"{int(n):,}".replace(",", "{,}")


def all_faster_threshold(g: pd.DataFrame, col: str = "speed") -> tuple[str, float] | None:
    """(dataset, rows) of the smallest table such that every dataset with at least that many rows is
    significantly faster in all of its configurations (by `col`: Holm `speed` or uncorrected `speed_raw`);
    None if no such table."""
    per = g.groupby("dataset").agg(rows=("rows", "first"), all_fast=(col, lambda s: (s == "faster").all()))
    per = per.sort_values("rows", kind="stable")
    for i in range(len(per)):
        if per.all_fast.iloc[i:].all():
            return per.index[i], float(per.rows.iloc[i])
    return None


def holm_above_raw_threshold(g: pd.DataFrame) -> pd.DataFrame:
    """Configurations of one algorithm on the tables from the uncorrected all-faster threshold upwards (empty if none)."""
    thr = all_faster_threshold(g, "speed_raw")
    return g[g.rows >= thr[1]] if thr else g.iloc[:0]


def holm_size_numbers(g: pd.DataFrame, p: str) -> dict:
    """§5 and §6.2: the size threshold before and after the Holm correction. `g`: one algorithm,
    annotated, with rows. Above = from the uncorrected all-faster table upwards; Missed = those not faster after Holm."""
    out = {}
    thr = all_faster_threshold(g, "speed_raw")
    out[p + "ThresholdDatasetRaw"] = _tt(thr[0]) if thr else "none"
    out[p + "ThresholdRowsRaw"] = _int(thr[1]) if thr else "--"
    above = holm_above_raw_threshold(g)
    miss = above[above.speed != "faster"].sort_values(["rows", "fraction"], kind="stable")
    out[p + "AboveN"], out[p + "AboveFast"] = str(len(above)), str(int((above.speed == "faster").sum()))
    cfg = [f"{_tt(r.dataset)} at {100 * r.fraction:.0f}\\%" for r in miss.itertuples()]
    out[p + "AboveMissed"] = "none" if not cfg else cfg[0] if len(cfg) == 1 else ", ".join(cfg[:-1]) + " and " + cfg[-1]
    out[p + "AboveMissedSTMin"] = f"{miss.S_T.min():.2f}" if len(miss) else "--"
    out[p + "AboveMissedSTMax"] = f"{miss.S_T.max():.2f}" if len(miss) else "--"
    small = list(g.groupby("dataset").rows.first().sort_values(kind="stable").index[:N_ENDS])
    sm = g[g.dataset.isin(small)]
    n_all = int(sm.groupby("dataset").speed.apply(lambda s: (s == "faster").all()).sum())
    out[p + "SmallAllFast"] = ("none", "one", "two", "three")[n_all] if n_all <= 3 else str(n_all)
    out[p + "SmallSlowerTenNames"] = _names(sorted(set(sm[(sm.fraction.round(2) == 0.10)
                                                          & (sm.speed == "slower")].dataset)))
    return out


_RUNS_CACHE: dict = {}


def _runs(path: Path) -> pd.DataFrame:
    """runs.csv (accuracy columns only), read once per path."""
    if path not in _RUNS_CACHE:
        _RUNS_CACHE[path] = pd.read_csv(path, usecols=["algorithm", "setup", "dataset", "mode", "sampler", "fraction",
                                                       "seed", "test_accuracy"])
    return _RUNS_CACHE[path]


def probabilities_under(g: pd.DataFrame, rope: float, threshold: float, props: pd.DataFrame,
                        runs_path: Path | None = None, rho: float | None = None) -> pd.DataFrame:
    """Per configuration of g: the posterior probabilities of worse / within the ROPE / better and the Bayesian
    verdict (equivalent / worse / better / undecided, priority as verdict()) under another ROPE (a fraction, e.g.
    0.02) and threshold. The stored rope (EQUIVALENCE_MARGIN) reuses the stored probabilities; another rope
    recomputes the correlated t-test from the per-seed accuracies in runs.csv (split sizes from the rows and test
    size 0.2, as the runs did). `rho`: another correlation between the runs than Nadeau and Bengio's
    n_test / (n_test + n_train) (Table S15); None = theirs. NaN and '' when the runs are not in the store."""
    import analysis
    decide = lambda w, r, b: "" if any(pd.isna(v) for v in (w, r, b)) else next(
        (name for name, q in (("equivalent", r), ("worse", w), ("better", b)) if q >= threshold), "undecided")
    cols = ["p_worse", "p_rope", "p_better"]
    if rho is None and abs(rope - results.EQUIVALENCE_MARGIN) < 1e-12:
        out = pd.DataFrame({"p_worse": g.bayes_worse.to_numpy(), "p_rope": g.bayes_rope.to_numpy(),
                            "p_better": g.bayes_better.to_numpy()}, index=g.index)
        out["verdict"] = [decide(w, r, b_) for w, r, b_ in zip(out.p_worse, out.p_rope, out.p_better)]
        return out
    path = runs_path or _f(results.STORE, "runs.csv")
    if not path.is_file():
        return pd.DataFrame({**{c: np.nan for c in cols}, "verdict": ""}, index=g.index)
    r = _runs(path)
    base = r[r["mode"] == "baseline"].set_index(["algorithm", "setup", "dataset", "seed"])["test_accuracy"]
    fec = r[r["mode"] == "fec"]
    rows = props.set_index("dataset")["rows"]
    out = []
    for row in g.itertuples():
        f = fec[(fec.algorithm == row.algorithm) & (fec.setup == row.setup) & (fec.dataset == row.dataset)
                & (fec.sampler == row.sampler) & np.isclose(fec.fraction, row.fraction)]
        bacc = base.reindex([(row.algorithm, row.setup, row.dataset, s_) for s_ in f.seed]).to_numpy()
        ok = ~np.isnan(bacc)
        if ok.sum() < 2 or row.dataset not in rows.index:
            out.append((np.nan, np.nan, np.nan, ""))
            continue
        n_test = int(np.ceil(0.2 * rows[row.dataset]))
        n_tr = rows[row.dataset] - n_test
        if rho is not None:                     # n_test / n_train = rho / (1 - rho): same variance factor
            n_test, n_tr = rho, 1 - rho
        w, rp, bt = analysis.bayes_correlated_t(bacc[ok], f.test_accuracy.to_numpy()[ok], rope, n_test, n_tr)
        out.append((w, rp, bt, decide(w, rp, bt)))
    return pd.DataFrame(out, columns=cols + ["verdict"], index=g.index)


def verdicts_under(g: pd.DataFrame, rope: float, threshold: float, props: pd.DataFrame,
                   runs_path: Path | None = None) -> pd.Series:
    """Per configuration of g: the Bayesian verdict under another ROPE and threshold (see probabilities_under)."""
    return probabilities_under(g, rope, threshold, props, runs_path).verdict


def equivalent_under(g: pd.DataFrame, rope: float, threshold: float, props: pd.DataFrame,
                     runs_path: Path | None = None) -> pd.Series:
    """1.0 / 0.0 per configuration: practically equivalent under that ROPE and threshold (NaN without the runs)."""
    v = verdicts_under(g, rope, threshold, props, runs_path)
    return v.map(lambda x: np.nan if x == "" else float(x == "equivalent"))


# Supplementary Tables S7-S11: the accuracy verdicts of Table 5 under every other combination of
# ROPE (+-1, +-2, +-3 pp) and threshold (0.95, 0.90); Table 5 itself is +-1 pp and 0.95.
SENSITIVITY_TABLES = [("Table S7", 0.01, 0.90), ("Table S8", 0.02, 0.95), ("Table S9", 0.02, 0.90),
                      ("Table S10", 0.03, 0.95), ("Table S11", 0.03, 0.90)]


def sensitivity_name(sid: str, rope: float, thr: float) -> str:
    return f"TableS{sid.split('S')[1]}_accuracy_rope{round(100 * rope)}pp_p{round(100 * thr)}.csv"


def sensitivity_table(metrics: pd.DataFrame, props: pd.DataFrame, rope: float, thr: float) -> pd.DataFrame:
    """As Table 5's accuracy columns: per algorithm and fraction (random rows), the number of datasets, median and
    interquartile range of dAcc (pp) over them, and how many are equivalent / worse / better / undecided."""
    g = metrics[metrics.sampler == MAIN_SAMPLER].copy()
    g["verdict"] = verdicts_under(g, rope, thr, props).to_numpy()
    rows = []
    for (algo, frac), x in g.groupby(["algorithm", g.fraction.round(2)]):
        d = 100 * x.dAcc
        v = x.verdict.value_counts()
        rows.append({"algorithm": algo, "fraction": frac, "datasets": x.dataset.nunique(),
                     "median_dAcc_pp": round(d.median(), 2), "q1_dAcc_pp": round(d.quantile(0.25), 2),
                     "q3_dAcc_pp": round(d.quantile(0.75), 2), "rope_pp": round(100 * rope),
                     "threshold": thr, **{k: int(v.get(k, 0)) for k in ("equivalent", "worse", "better", "undecided")}})
    return pd.DataFrame(rows)


def sensitivity_detail(metrics: pd.DataFrame, props: pd.DataFrame, rope: float, thr: float) -> pd.DataFrame:
    """Tables S7-S11: as the accuracy columns of Table 5, per algorithm, dataset (ordered by rows) and fraction
    (random rows): dAcc mean and sd of the 30 paired differences (pp), the posterior probabilities of worse /
    within the ROPE / better, and the verdict, under that ROPE and threshold."""
    g = metrics[metrics.sampler == MAIN_SAMPLER].copy()
    pr = probabilities_under(g, rope, thr, props)
    rows_of = props.set_index("dataset")["rows"]
    out = pd.DataFrame({"algorithm": g.algorithm, "dataset": g.dataset, "rows": g.dataset.map(rows_of).astype("Int64"),
                        "fraction": g.fraction.round(2), "dAcc_pp": (100 * g.dAcc).round(2),
                        "dAcc_sd_pp": (100 * g.get("dAcc_sd", pd.Series(np.nan, index=g.index))).round(2), "rope_pp": round(100 * rope), "threshold": thr,
                        "p_worse": pr.p_worse.round(3), "p_rope": pr.p_rope.round(3),
                        "p_better": pr.p_better.round(3), "verdict": pr.verdict})
    return out.sort_values(["algorithm", "rows", "dataset", "fraction"], kind="stable").reset_index(drop=True)


RELATIVE_ROPES = (0.05, 0.10, 0.20)      # Table S14 (supervisor review, 2026-10-25): ROPE = r x baseline error
RELATIVE_MAIN = 0.10                     # the one reported in the text


def relative_rope_detail(metrics: pd.DataFrame, props: pd.DataFrame, thr: float = THRESHOLD) -> pd.DataFrame:
    """Table S14: per algorithm, dataset and fraction (random rows), the accuracy verdict with the predefined +-1 pp
    ROPE and with ROPEs relative to the baseline's error, r x (100 - baseline accuracy %) pp for r in
    RELATIVE_ROPES; same test, per-seed data and threshold as Table 5 (probabilities_under), only the ROPE differs."""
    g = metrics[metrics.sampler == MAIN_SAMPLER].copy()
    rows_of = props.set_index("dataset")["rows"]
    err = 1 - g.baseline_accuracy
    out = pd.DataFrame({"algorithm": g.algorithm, "dataset": g.dataset, "rows": g.dataset.map(rows_of).astype("Int64"),
                        "fraction": g.fraction.round(2), "baseline_accuracy": g.baseline_accuracy.round(4),
                        "dAcc_pp": (100 * g.dAcc).round(2),
                        "verdict_rope1pp": probabilities_under(g, results.EQUIVALENCE_MARGIN, thr, props).verdict})
    for r in RELATIVE_ROPES:
        tag = f"rel{round(100 * r)}"
        out[f"rope_{tag}_pp"] = (100 * r * err).round(2)
        out[f"verdict_{tag}"] = [probabilities_under(g.loc[[i]], r * err.loc[i], thr, props).verdict.iloc[0]
                                 for i in g.index]
    return out.sort_values(["algorithm", "rows", "dataset", "fraction"], kind="stable").reset_index(drop=True)


# Table S15: the accuracy verdicts under other correlations between the 30 runs. The test
# uses Nadeau and Bengio's rho = n_test / (n_test + n_train) = 0.2 (80/20 random splits); rho cannot be estimated.
CORRELATIONS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
RHO_WORD = {0.0: "Zero", 0.1: "One", 0.2: "Two", 0.3: "Three", 0.4: "Four", 0.5: "Five"}


def correlation_detail(metrics: pd.DataFrame, props: pd.DataFrame, thr: float = THRESHOLD) -> pd.DataFrame:
    """Table S15: per algorithm, dataset and fraction (random rows), Delta Acc and, for every rho in CORRELATIONS,
    P(worse), P(within +-1 pp) and the verdict; same test, data, ROPE and threshold as Table 5, only rho differs."""
    g = metrics[metrics.sampler == MAIN_SAMPLER].copy()
    rows_of = props.set_index("dataset")["rows"]
    out = pd.DataFrame({"algorithm": g.algorithm, "dataset": g.dataset, "rows": g.dataset.map(rows_of).astype("Int64"),
                        "fraction": g.fraction.round(2), "dAcc_pp": (100 * g.dAcc).round(2)})
    for rho in CORRELATIONS:
        pr = probabilities_under(g, results.EQUIVALENCE_MARGIN, thr, props, rho=rho)
        tag = f"rho{rho:.1f}"
        out[f"p_worse_{tag}"], out[f"p_rope_{tag}"] = pr.p_worse.round(3), pr.p_rope.round(3)
        out[f"verdict_{tag}"] = pr.verdict
    return out.sort_values(["algorithm", "rows", "dataset", "fraction"], kind="stable").reset_index(drop=True)


PAPER_SEEDS = range(30, 60)            # paired seeds 30-59


def split_overlap(props: pd.DataFrame, datasets, seeds=PAPER_SEEDS, test_size: float = 0.2) -> pd.DataFrame:
    """Per dataset, how many rows the splits of two seeds share, over every pair of seeds. The
    split is ge.split's train_test_split(test_size, random_state=seed) without stratification, so it depends only on
    the number of rows: rebuilt from row indices. test_* = share of a test set also in the other seed's test set;
    train_* = the same for the training sets."""
    from itertools import combinations
    from sklearn.model_selection import train_test_split
    rows = props.set_index("dataset")["rows"]
    out = []
    for ds in datasets:
        n = int(rows[ds])
        te = {s: set(train_test_split(np.arange(n), test_size=test_size, random_state=s)[1].tolist()) for s in seeds}
        n_te = len(next(iter(te.values())))
        t = [len(te[a] & te[b]) / n_te for a, b in combinations(seeds, 2)]
        tr = [(n - len(te[a] | te[b])) / (n - n_te) for a, b in combinations(seeds, 2)]
        out.append({"dataset": ds, "rows": n, "test_mean": np.mean(t), "test_min": min(t), "test_max": max(t),
                    "train_mean": np.mean(tr), "train_min": min(tr), "train_max": max(tr)})
    return pd.DataFrame(out)


def correlation_numbers(metrics: pd.DataFrame, props: pd.DataFrame) -> dict:
    """Macros of §4.4 / §5: equivalent and worse counts per rho; the configurations judged
    worse when the runs are taken as independent (rho = 0) with their P(worse) under the test's rho = 0.2; the
    baseline's run-to-run spread of test accuracy (median sd over the datasets, pp)."""
    out = {}
    t = correlation_detail(metrics, props)
    for algo in ("GE", "GP"):
        x = t[t.algorithm == algo]
        if not len(x):
            continue
        for rho in CORRELATIONS:
            out[f"{algo}EquivRho{RHO_WORD[rho]}"] = str(int((x[f"verdict_rho{rho:.1f}"] == "equivalent").sum()))
            out[f"{algo}WorseRho{RHO_WORD[rho]}"] = str(int((x[f"verdict_rho{rho:.1f}"] == "worse").sum()))
    w = t[t["verdict_rho0.0"] == "worse"].sort_values(["algorithm", "rows", "fraction"], kind="stable")
    out["RhoZeroWorseN"] = str(len(w))
    cfg = [f"{r.algorithm} on {_tt(r.dataset)} at {100 * r.fraction:.0f}\\%" for r in w.itertuples()]
    out["RhoZeroWorseConfigs"] = "none" if not cfg else cfg[0] if len(cfg) == 1 else ", ".join(cfg[:-1]) + " and " + cfg[-1]
    out["RhoZeroWorseDaccMin"] = f"{-w.dAcc_pp.max():.1f}" if len(w) else "--"     # losses, positive pp
    out["RhoZeroWorseDaccMax"] = f"{-w.dAcc_pp.min():.1f}" if len(w) else "--"
    out["RhoZeroWorsePMin"] = f"{w['p_worse_rho0.2'].min():.2f}" if len(w) else "--"
    out["RhoZeroWorsePMax"] = f"{w['p_worse_rho0.2'].max():.2f}" if len(w) else "--"
    ov = split_overlap(props, sorted(set(metrics.dataset) & set(props.dataset)))
    if len(ov):
        out["SplitTestOverlapMean"] = f"{100 * ov.test_mean.mean():.0f}"
        out["SplitTestOverlapMin"] = f"{100 * ov.test_min.min():.0f}"
        out["SplitTestOverlapMax"] = f"{100 * ov.test_max.max():.0f}"
        out["SplitTrainOverlapMean"] = f"{100 * ov.train_mean.mean():.0f}"
    path = _f(results.STORE, "runs.csv")
    if path.is_file():
        r = _runs(path)
        b = r[(r["mode"] == "baseline") & (r.setup == SETUP) & r.dataset.isin(set(metrics.dataset))]
        for algo in ("GE", "GP"):
            sd = b[b.algorithm == algo].groupby("dataset").test_accuracy.std()
            out[f"{algo}BaseAccSdMedian"] = f"{100 * sd.median():.1f}" if len(sd) else "--"
    return out


def correlation_claims(metrics: pd.DataFrame, props: pd.DataFrame) -> list[tuple[str, str, bool]]:
    """§5: wider ROPEs and other correlations between the runs."""
    out = []
    g = metrics[metrics.sampler == MAIN_SAMPLER]
    for algo in ("GE", "GP"):
        x = g[g.algorithm == algo]
        if not len(x):
            continue
        v = {r: verdicts_under(x, r, THRESHOLD, props) for r in (0.01, 0.02, 0.03)}
        out.append((f"rope/{algo}/no-worse-better", f"§5 {algo}: at none of +-1/2/3 pp was a configuration worse or better",
                    all(not v[r].isin(["worse", "better"]).any() for r in v)))
        out.append((f"rope/{algo}/only-undecided-to-equivalent", f"§5 {algo}: a wider ROPE only moved configurations "
                    "from undecided to equivalent", ((v[0.01] != "equivalent") | (v[0.02] == "equivalent")).all()
                    and ((v[0.02] != "equivalent") | (v[0.03] == "equivalent")).all()))
    t = correlation_detail(metrics, props)
    eq = {(a, rho): int((t[t.algorithm == a][f"verdict_rho{rho:.1f}"] == "equivalent").sum())
          for a in ("GE", "GP") for rho in CORRELATIONS}
    out.append(("rho/fewer-equivalent", "§5: a larger rho gives fewer equivalent configurations (never more)",
                all(eq[(a, r1)] >= eq[(a, r2)] for a in ("GE", "GP") for r1, r2 in zip(CORRELATIONS, CORRELATIONS[1:]))))
    out.append(("rho/none-worse-from-0.2", "§5: no configuration judged worse at any rho from 0.2 to 0.5",
                all(not (t[f"verdict_rho{r:.1f}"] == "worse").any() for r in CORRELATIONS if r >= 0.2)))
    w = t[t["verdict_rho0.0"] == "worse"]
    small = set(props.sort_values("rows", kind="stable").dataset.iloc[:N_ENDS])
    out.append(("rho/zero-worse-small-one", "§5: the configurations worse at rho = 0 are undecided at rho = 0.2 and use "
                "a 1% fingerprint on the smallest tables", len(w) > 0 and (w["verdict_rho0.2"] == "undecided").all()
                and (w.fraction.round(2) == 0.01).all() and set(w.dataset) <= small))
    return out


# ---------------------------------------------------------------- table size or fingerprint size?
# With a fixed fraction the fingerprint grows with the table, so an association with the number of rows mixes the two.
# Each table has three fingerprint sizes (1/5/10 %) at one table size, and the fingerprint sizes of different tables
# overlap, so one model per algorithm and measure, log10 y = a + b_rows log10(rows) + b_fp log10(fingerprint rows) over
# the 51 configurations, separates them. b_fp is the change between fractions within a table (the same
# three fractions on every table); b_rows compares tables at the same fingerprint size; b_rows + b_fp is a tenfold
# larger table at the same fraction. Intervals: 95 % percentile bootstrap over datasets; leave-one-dataset-out range.
SIZE_OUTCOMES = {"S_T": "ST", "hit_rate": "Hit", "fake_hit_ratio": "Fake", "error_per_fake_pp": "Err",
                 "abs_dAcc_pp": "AbsAcc"}
SIZE_LINEAR = {"abs_dAcc_pp"}                       # no log: zero and near-zero differences; coefficient in pp
SIZE_TERMS = {"log_rows": "Rows", "log_fp": "Fp", "rows_same_fraction": "RowsFrac", "work_avoided": "Work"}
SIZE_BOOT, SIZE_SEED = 2000, 1
_SIZE_CACHE: dict = {}


def size_design(metrics: pd.DataFrame, props: pd.DataFrame, store: Path = results.STORE) -> pd.DataFrame:
    """Per algorithm, dataset and fraction (main sampler): rows, fingerprint rows (sample_size recorded by the FEC
    runs), S_T, hit rate, fake-hit ratio, error per fake hit (pp, Table S13), |Delta Acc| (pp) and the share of
    work avoided (Table S4)."""
    g = metrics[metrics.sampler == MAIN_SAMPLER].assign(fraction=lambda d: d.fraction.round(2))
    r = pd.read_csv(_f(store, "runs.csv"), usecols=["algorithm", "setup", "dataset", "mode", "sampler", "fraction",
                                                 "sample_size"])
    r = r[(r.setup == SETUP) & (r["mode"] == "fec") & (r.sampler == MAIN_SAMPLER) & r.dataset.isin(set(g.dataset))]
    if r.empty:                                    # no FEC runs of these datasets in the store (e.g. a test corpus)
        return pd.DataFrame()
    r = r.assign(fraction=lambda d: d.fraction.round(2))
    fp = r.groupby(["algorithm", "dataset", "fraction"]).sample_size
    if (fp.nunique() > 1).any():
        raise TableError("fingerprint size differs between the seeds of one configuration")
    sev = fake_severity(set(g.dataset)).assign(fraction=lambda d: d.fraction.round(2))
    w = hit_work(store)
    w = w[w.sampler == MAIN_SAMPLER].drop(columns="sampler")
    keys = ["algorithm", "dataset", "fraction"]
    x = (g[keys + ["S_T", "hit_rate", "fake_hit_ratio", "dAcc"]]
         .merge(fp.first().rename("fp_rows").reset_index(), on=keys, how="left")
         .merge(sev[keys + ["mean_error_pp"]].rename(columns={"mean_error_pp": "error_per_fake_pp"}), on=keys, how="left")
         .merge(w[keys + ["work_avoided"]], on=keys, how="left")
         .merge(props[["dataset", "rows"]], on="dataset", how="left"))
    x = x.assign(abs_dAcc_pp=(100 * x.dAcc).abs()).drop(columns="dAcc")
    if x[["fp_rows", "rows", "work_avoided", *SIZE_OUTCOMES]].isna().any().any():
        raise TableError("size_design: a configuration lacks fingerprint rows, rows, work or an outcome")
    if (x[[c for c in SIZE_OUTCOMES if c not in SIZE_LINEAR]] <= 0).any().any():
        raise TableError("size_design: a log-scale outcome is zero or negative")
    return x.sort_values(["algorithm", "rows", "dataset", "fraction"], kind="stable").reset_index(drop=True)


def _size_fit(d: pd.DataFrame, y: str, terms: list[str]) -> tuple[np.ndarray, float]:
    """OLS of y (log10 unless linear) on log10 rows, log10 fingerprint rows (and the extra terms): coefficients and R2."""
    cols = {"log_rows": np.log10(d.rows.to_numpy(float)), "log_fp": np.log10(d.fp_rows.to_numpy(float))}
    z = np.column_stack([np.ones(len(d))] + [cols[t] if t in cols else d[t].to_numpy(float) for t in terms])
    v = d[y].to_numpy(float) if y in SIZE_LINEAR else np.log10(d[y].to_numpy(float))
    b = np.linalg.lstsq(z, v, rcond=None)[0]
    e = v - z @ b
    return b[1:], float(1 - (e ** 2).sum() / ((v - v.mean()) ** 2).sum())


def size_model(design: pd.DataFrame) -> pd.DataFrame:
    """Table S16: per algorithm, measure and model, each term's coefficient with its bootstrap interval over datasets
    and leave-one-dataset-out range; 'factor' = 10**coefficient (log measures: the multiplier for a tenfold larger
    table or fingerprint; work_avoided: per 0.1 more of the work avoided)."""
    key = (len(design), float(design.S_T.sum()), float(design.fp_rows.sum()))
    if key in _SIZE_CACHE:
        return _SIZE_CACHE[key]
    models = [(y, ["log_rows", "log_fp"]) for y in SIZE_OUTCOMES] + [("S_T", ["log_rows", "log_fp", "work_avoided"])]
    out = []
    for algo, g in design.groupby("algorithm"):
        ds = g.dataset.unique()
        parts = {d: g[g.dataset == d] for d in ds}
        rng = np.random.default_rng(SIZE_SEED)
        draws = [pd.concat([parts[d] for d in rng.choice(ds, len(ds))]) for _ in range(SIZE_BOOT)]
        for y, terms in models:
            b, r2 = _size_fit(g, y, terms)
            bs = np.array([_size_fit(d, y, terms)[0] for d in draws])
            loo = np.array([_size_fit(g[g.dataset != d], y, terms)[0] for d in ds])
            name = "rows+fp" + ("+work" if "work_avoided" in terms else "")
            show = list(terms) + (["rows_same_fraction"] if name == "rows+fp" else [])
            for t in show:
                pick = (lambda a: a[..., 0] + a[..., 1]) if t == "rows_same_fraction" else (lambda a, i=terms.index(t): a[..., i])
                scale = 0.1 if t == "work_avoided" else 1.0
                c, lo, hi = (scale * v for v in (pick(b), *np.percentile(pick(bs), [2.5, 97.5])))
                out.append({"algorithm": algo, "measure": y, "model": name, "term": t, "coef": c, "ci_low": lo,
                            "ci_high": hi, "loo_min": scale * pick(loo).min(), "loo_max": scale * pick(loo).max(),
                            "factor": np.nan if y in SIZE_LINEAR else 10 ** c,
                            "factor_low": np.nan if y in SIZE_LINEAR else 10 ** lo,
                            "factor_high": np.nan if y in SIZE_LINEAR else 10 ** hi,
                            "r2": r2, "configurations": len(g), "datasets": len(ds)})
    t = pd.DataFrame(out)
    _SIZE_CACHE[key] = t
    return t


def _within_slope(g: pd.DataFrame, y: str) -> float:
    """Slope of log10 y on log10 fingerprint rows within datasets (dataset means removed)."""
    v = pd.Series(np.log10(g[y].to_numpy(float)), index=g.index)
    m = pd.Series(np.log10(g.fp_rows.to_numpy(float)), index=g.index)
    dv, dm = v - v.groupby(g.dataset).transform("mean"), m - m.groupby(g.dataset).transform("mean")
    return float((dv * dm).sum() / (dm ** 2).sum())


def size_overlap(design: pd.DataFrame) -> dict:
    """How far the fingerprint sizes of different tables overlap: per algorithm the same, so GE is used. Tables whose
    range of fingerprint rows (1-10 %) overlaps that of a table at least three times larger or smaller."""
    g = design[design.algorithm == design.algorithm.iloc[0]]
    rng_ = g.groupby("dataset").agg(lo=("fp_rows", "min"), hi=("fp_rows", "max"), rows=("rows", "first"))
    n = sum(any(max(a.lo, b.lo) <= min(a.hi, b.hi) and max(a.rows, b.rows) >= 3 * min(a.rows, b.rows)
                for j, b in rng_.iterrows() if j != i) for i, a in rng_.iterrows())
    return {"overlap": n, "tables": len(rng_), "fp_min": int(g.fp_rows.min()), "fp_max": int(g.fp_rows.max())}


def size_numbers(metrics: pd.DataFrame, props: pd.DataFrame, store: Path = results.STORE) -> dict:
    """Macros \\res<Algo>Size<Measure><Term>[Lo|Hi] (factor per tenfold, or pp for |Delta Acc|, two decimals) and
    \\res<Algo>Size<Measure>RTwo; \\resSizeOverlapN / Tables / FpMin / FpMax."""
    if not (_f(store, "runs.csv")).is_file() or not (_f(store, "generations.csv")).is_file():
        return {}
    d = size_design(metrics, props, store)
    if d.empty:
        return {}
    t = size_model(d)
    out = {}
    for r in t.itertuples():
        k = f"{r.algorithm}Size{SIZE_OUTCOMES[r.measure]}{SIZE_TERMS[r.term]}"
        if r.model == "rows+fp+work":
            k += "Adj"
        lin = r.measure in SIZE_LINEAR
        val, lo, hi = (r.coef, r.ci_low, r.ci_high) if lin else (r.factor, r.factor_low, r.factor_high)
        fmt = (lambda v: f"{v:+.2f}") if lin else (lambda v: f"{v:.2f}")
        out[k], out[k + "Lo"], out[k + "Hi"] = fmt(val), fmt(lo), fmt(hi)
        out[f"{r.algorithm}Size{SIZE_OUTCOMES[r.measure]}RTwo" + ("Adj" if r.model == "rows+fp+work" else "")] = f"{r.r2:.2f}"
    for algo in ("GE", "GP"):
        sub = t[(t.algorithm == algo) & (t.model == "rows+fp") & (t.term == "log_fp")]
        f = sub.set_index("measure")
        if "fake_hit_ratio" in f.index:      # 'a tenfold larger fingerprint cut the fake-hit ratio by X %'
            out[f"{algo}SizeFakeFpCut"] = f"{100 * (1 - f.loc['fake_hit_ratio', 'factor']):.0f}"
            out[f"{algo}SizeErrFpCut"] = f"{100 * (1 - f.loc['error_per_fake_pp', 'factor']):.0f}"
            out[f"{algo}SizeErrFpSlope"] = f"{-f.loc['error_per_fake_pp', 'coef']:.2f}"
    ov = size_overlap(d)
    out["SizeOverlapN"], out["SizeOverlapTables"] = str(ov["overlap"]), str(ov["tables"])
    out["SizeFpMin"], out["SizeFpMax"] = _int(ov["fp_min"]), _int(ov["fp_max"])
    out["SizeBoot"] = _int(SIZE_BOOT)
    return out


def size_claims(metrics: pd.DataFrame, props: pd.DataFrame, store: Path = results.STORE) -> list[tuple[str, str, bool]]:
    """What the joint model says about table size and fingerprint size."""
    if not (_f(store, "runs.csv")).is_file() or not (_f(store, "generations.csv")).is_file():
        return []
    d = size_design(metrics, props, store)
    if d.empty:
        return []
    t = size_model(d).set_index(["algorithm", "measure", "model", "term"])
    out = []
    has = [a for a in ("GE", "GP") if (a, "S_T", "rows+fp", "log_rows") in t.index]
    ci = lambda a, y, term, m="rows+fp": (t.loc[(a, y, m, term), "ci_low"], t.loc[(a, y, m, term), "ci_high"])
    out.append(("size/fake-not-rows", "§5.5: at the same fingerprint size, the table size had no detectable effect on "
                "the fake-hit ratio (95% interval of b_rows contains 0, GE and GP)",
                bool(has) and all(ci(a, "fake_hit_ratio", "log_rows")[0] < 0 < ci(a, "fake_hit_ratio", "log_rows")[1] for a in has)))
    out.append(("size/fake-fp", "§5.5: a larger fingerprint gave fewer fake hits and smaller errors per fake hit "
                "(interval of b_fp below 0, GE and GP)",
                bool(has) and all(ci(a, y, "log_fp")[1] < 0 for a in has for y in ("fake_hit_ratio", "error_per_fake_pp"))))
    out.append(("size/err-near-inverse", "§5.5: the error per fake hit fell about in proportion to the fingerprint size "
                "(b_fp between -1.1 and -0.8, GE and GP)",
                bool(has) and all(-1.1 < t.loc[(a, "error_per_fake_pp", "rows+fp", "log_fp"), "coef"] < -0.8 for a in has)))
    out.append(("size/st-rows", "§5.1/5.2: at the same fingerprint size a larger table gave a larger S_T, also with the "
                "share of work avoided in the model (interval of b_rows above 0, GE and GP)",
                bool(has) and all(ci(a, "S_T", "log_rows")[0] > 0 and ci(a, "S_T", "log_rows", "rows+fp+work")[0] > 0
                                  for a in has)))
    out.append(("size/st-fp", "§5.1/5.2: at the same table size a larger fingerprint gave a smaller S_T (GE and GP)",
                bool(has) and all(ci(a, "S_T", "log_fp")[1] < 0 for a in has)))
    out.append(("size/within", "§4.4: b_fp equals the within-table slope (the same three fractions on every table; fingerprint rounding aside)",
                bool(has) and all(abs(t.loc[(a, y, "rows+fp", "log_fp"), "coef"] - _within_slope(d[d.algorithm == a], y)) < 0.01
                                  for a in has for y in SIZE_OUTCOMES if y not in SIZE_LINEAR)))
    out.append(("size/overlap", "§4.4: every table shares fingerprint sizes with a table at least three times larger "
                "or smaller", size_overlap(d)["overlap"] == size_overlap(d)["tables"]))
    out.append(("size/err-rows", "§5.5: at the same fingerprint size, a larger table gave smaller errors per fake hit "
                "in GP (interval below 0) but not in GE (interval contains 0)",
                "GP" in has and "GE" in has and ci("GP", "error_per_fake_pp", "log_rows")[1] < 0
                and ci("GE", "error_per_fake_pp", "log_rows")[0] < 0 < ci("GE", "error_per_fake_pp", "log_rows")[1]))
    fp = [r for r in t.reset_index().itertuples() if r.model == "rows+fp" and r.term == "log_fp"]
    out.append(("size/fp-stable", "§6.2: the fingerprint-size effects changed little when one dataset was left out "
                "(leave-one-out range inside the bootstrap interval, every measure)",
                bool(fp) and all(r.ci_low <= r.loo_min and r.loo_max <= r.ci_high for r in fp)))
    out.append(("size/rows-wider", "§6.2: the intervals of the table-size effects were wider than those of the "
                "fingerprint-size effects (S_T, hit rate, fake-hit ratio, error per fake hit; GE and GP)",
                bool(has) and all(ci(a, y, "log_rows")[1] - ci(a, y, "log_rows")[0] > ci(a, y, "log_fp")[1] - ci(a, y, "log_fp")[0]
                                  for a in has for y in SIZE_OUTCOMES if y not in SIZE_LINEAR)))
    out.append(("size/acc-fp", "§5.5: a larger fingerprint on the same table gave a smaller |Delta Acc| (interval "
                "below 0) and no effect of table size at the same fingerprint size was detected (interval contains 0), "
                "GE and GP", bool(has) and all(ci(a, "abs_dAcc_pp", "log_fp")[1] < 0
                                               and ci(a, "abs_dAcc_pp", "log_rows")[0] < 0 < ci(a, "abs_dAcc_pp", "log_rows")[1]
                                               for a in has)))
    big = d.sort_values("rows", kind="stable").groupby("algorithm").tail(3 * 3)        # three largest tables, three fractions each
    out.append(("size/large-one-thousands", "§6.1: on the largest tables a 1% fingerprint has thousands of rows "
                "(three largest, >= 1,000)", len(big) > 0 and (big[big.fraction == 0.01].fp_rows >= 1000).all()))
    return out


SIZE_RAMP = ("#86b6ef", "#0d366b")         # dataviz sequential blue, step 250 -> 700 (magnitude: rows)


FRACTION_MARKERS = {0.01: "o", 0.05: "s", 0.1: "^"}   # Figure S4: marker shape per fingerprint fraction


def size_fingerprint_figure(design: pd.DataFrame, out_dir: Path) -> Path:
    """FigureS4_size_vs_fingerprint.pdf: fake-hit ratio (left) and S_T (right) against the fingerprint rows (log),
    one line per dataset through its three fractions, coloured by the number of rows (sequential, colour bar);
    rows GE and GP; model factors in the panel titles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    fs.use()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "FigureS4_size_vs_fingerprint.pdf"
    if design.empty:                               # no runs in the store: a placeholder, as the CSVs' "no data"
        fs.use()
        fig = plt.figure(figsize=(6.3, 1))
        fig.text(0.5, 0.5, "no data", ha="center", va="center", color=INK_SECONDARY)
        fs.save(fig, path)
        plt.close(fig)
        return path
    cmap = LinearSegmentedColormap.from_list("rows", SIZE_RAMP)
    norm = LogNorm(design.rows.min(), design.rows.max())
    t = size_model(design).set_index(["algorithm", "measure", "model", "term"])
    fs.use()
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.2), sharex=True)
    for r, algo in enumerate(("GE", "GP")):
        g = design[design.algorithm == algo]
        for c, (y, label) in enumerate((("fake_hit_ratio", "fake-hit ratio (log scale)"), ("S_T", "$S_T$"))):
            ax = axes[r, c]
            for ds, h in g.groupby("dataset"):
                col = cmap(norm(h.rows.iloc[0]))
                ax.plot(h.fp_rows, h[y], color=col, linewidth=1.2, zorder=2)
                for frac, mk in FRACTION_MARKERS.items():   # marker shape tells the fingerprint fraction
                    k = h[h.fraction.round(2) == frac]
                    ax.scatter(k.fp_rows, k[y], s=20, marker=mk, color=col, edgecolor="white", linewidth=0.5,
                               zorder=3)
            ax.set_xscale("log")
            if y == "fake_hit_ratio":
                ax.set_yscale("log")
            else:
                ax.axhline(1, color=INK_SECONDARY, linewidth=0.8, linestyle="--", zorder=1)
            ax.grid(color=GRID, linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if (algo, y, "rows+fp", "log_rows") in t.index:
                fr, ff = (t.loc[(algo, y, "rows+fp", k)] for k in ("log_rows", "log_fp"))
                fs.title(ax, algo)
                ax.text(0.98, 0.97, f"dataset rows x10: {fr.factor:.2f} [{fr.factor_low:.2f}, {fr.factor_high:.2f}]\n"
                                    f"fingerprint rows x10: {ff.factor:.2f} [{ff.factor_low:.2f}, "
                                    f"{ff.factor_high:.2f}]", transform=ax.transAxes, ha="right", va="top",
                        fontsize=fs.BASE_SIZE - 1, color=INK_SECONDARY, linespacing=1.3,
                        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
            ax.set_ylabel(label)
            if r == 1:
                ax.set_xlabel("fingerprint rows (log scale)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], linestyle="none", marker=mk, markersize=5, color=INK_SECONDARY,
                      label=f"fingerprint {round(frac * 100)}%") for frac, mk in FRACTION_MARKERS.items()]
    handles.append(Line2D([], [], color=INK_SECONDARY, linewidth=1.2, label="one dataset"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 0.86, 1))
    cax = fig.add_axes((0.875, 0.17, 0.015, 0.68))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("dataset size: rows (log scale)", color=INK)
    cb.outline.set_edgecolor(INK_SECONDARY)
    fs.save(fig, path)
    plt.close(fig)
    return path


CACHE_MEASURES = ("S_T", "hit_rate", "fake_hit_ratio", "dAcc")


def dataset_rho(g: pd.DataFrame, y: str, x: str) -> tuple[float, float, int]:
    """Spearman rho, p and n between two measures over datasets, each dataset's mean over its configurations."""
    from scipy.stats import spearmanr
    per = g.groupby("dataset")[[x, y]].mean().dropna()
    if len(per) < 3:
        return float("nan"), float("nan"), len(per)
    rho, pval = spearmanr(per[x], per[y])
    return float(rho), float(pval), len(per)


def _p(pval: float) -> str:
    """'=0.32', '=0.004' or '<0.001', for $p\\res...{}$."""
    return "<0.001" if pval < 0.001 else (f"={pval:.3f}" if pval < 0.01 else f"={pval:.2f}")


# --- choosing the fingerprint size --------------------------------------------------------------
# The worst fake hit of a generation, against the number of fingerprint rows (Table S13 joined with the fingerprint
# rows of the FEC runs). GUIDE_ROWS is the round number of rows from which almost no generation had a worst fake hit
# above 1 pp; GUIDE_FEW the number below which most generations did. Both are read off Table S13 and checked here.
GUIDE_ROWS, GUIDE_FEW = 350, 150
GUIDE_FRACTIONS = (0.01, 0.05, 0.1)


def guide_design(metrics: pd.DataFrame, props: pd.DataFrame, store: Path = results.STORE) -> pd.DataFrame:
    """size_design plus the share of generations whose worst fake hit exceeded 1 pp."""
    if not (_f(store, "runs.csv")).is_file() or not (_f(store, "generations.csv")).is_file():
        return pd.DataFrame()
    d = size_design(metrics, props, store)
    if d.empty:
        return d
    keys = ["algorithm", "dataset", "fraction"]
    sev = fake_severity(set(d.dataset)).assign(fraction=lambda x: x.fraction.round(2))
    d = d.merge(sev[keys + ["gens_worst_over_1pp"]], on=keys, how="left")
    if d.gens_worst_over_1pp.isna().any():
        raise TableError("guide_design: a configuration lacks the share of generations over 1 pp")
    return d


def _guide_bands(d: pd.DataFrame) -> dict:
    """Per dataset, the smallest fraction whose fingerprint reaches GUIDE_ROWS (None if even 10% does not)."""
    fp = d.groupby(["dataset", "fraction"]).fp_rows.max().unstack()
    return {ds: next((f for f in GUIDE_FRACTIONS if fp.loc[ds, f] >= GUIDE_ROWS), None) for ds in fp.index}


def guide_numbers(metrics: pd.DataFrame, props: pd.DataFrame, store: Path = results.STORE) -> dict:
    """Macros \\resGuide...: the row thresholds, the share of generations over 1 pp above and below them, and which
    datasets reach GUIDE_ROWS at 1%, at 5% only, or not even at 10%."""
    d = guide_design(metrics, props, store)
    if d.empty:
        return {}
    many, few = d[d.fp_rows >= GUIDE_ROWS], d[d.fp_rows < GUIDE_FEW]
    band = _guide_bands(d)
    rows = props.set_index("dataset")["rows"]
    one = [ds for ds, f in band.items() if f == 0.01]
    five = [ds for ds, f in band.items() if f == 0.05]
    none = sorted((ds for ds, f in band.items() if f is None), key=lambda ds: rows[ds])
    return {"GuideRows": _int(GUIDE_ROWS), "GuideFew": _int(GUIDE_FEW),
            "GuideOverOneMax": f"{100 * many.gens_worst_over_1pp.max():.1f}",
            "GuideFewOverOneMin": f"{100 * few.gens_worst_over_1pp.min():.0f}",
            "GuideOneRowsMin": _int(rows[one].min()),
            "GuideFiveRowsMin": _int(rows[five].min()), "GuideFiveRowsMax": _int(rows[five].max()),
            "GuideSmallNames": _names(none), "GuideSmallRowsMax": _int(rows[none].max()),
            "GuideSmallTenMax": _int(d[d.dataset.isin(none) & np.isclose(d.fraction, 0.1)].fp_rows.max())}


def guide_claims(metrics: pd.DataFrame, props: pd.DataFrame, store: Path = results.STORE) -> list[tuple[str, str, bool]]:
    """The fingerprint-size guidance of §6.1."""
    d = guide_design(metrics, props, store)
    if d.empty:
        return []
    band = _guide_bands(d)
    rows = props.set_index("dataset")["rows"]
    order = sorted(band, key=lambda ds: rows[ds])
    rank = {None: 3, 0.01: 0, 0.05: 1, 0.1: 2}
    smallest3 = order[:3]
    a = annotated(metrics[metrics.sampler == MAIN_SAMPLER]).assign(fraction=lambda x: x.fraction.round(2))
    small = a[a.dataset.isin(smallest3)]
    st = a.groupby(["algorithm", "dataset", "fraction"]).S_T.mean()
    large = [ds for ds, f in band.items() if f == 0.01]
    return [
        ("guide/small-often-not-faster", "§6.1: on the smallest datasets FEC was often not faster (at least half of "
         "their configurations not significantly faster)", bool((small.speed != "faster").mean() >= 0.5)),
        ("guide/one-pct-usually-fastest", "§6.1: on the datasets where 1% reaches the threshold, 1% had the highest "
         "mean S_T of the three fractions usually (all but at most one dataset per algorithm)",
         all(sum(st[(al, ds, 0.01)] < max(st[(al, ds, 0.05)], st[(al, ds, 0.1)]) for ds in large) <= 1
             for al in ("GE", "GP"))),
        ("guide/many-rows", f"§6.1: with at least {GUIDE_ROWS} fingerprint rows, the worst fake hit exceeded 1 pp in "
         "at most 2% of generations (every configuration, both algorithms)",
         bool((d[d.fp_rows >= GUIDE_ROWS].gens_worst_over_1pp <= 0.02).all())),
        ("guide/few-rows", f"§6.1: with fewer than {GUIDE_FEW} fingerprint rows, the worst fake hit exceeded 1 pp in "
         "more than half of the generations (every configuration)",
         bool((d[d.fp_rows < GUIDE_FEW].gens_worst_over_1pp > 0.5).all())),
        ("guide/bands-by-size", "§6.1: the fraction needed to reach the row threshold falls as datasets grow "
         "(1% for the largest, 5% for mid-sized, none for the smallest)",
         all(rank[band[a]] >= rank[band[b]] for a, b in zip(order, order[1:]))),
        ("guide/no-ten-pct-band", "§6.1: no dataset needs 10% to reach the row threshold (the bands are 1%, 5%, none)",
         0.1 not in band.values()),
        ("guide/small-are-smallest3", "§6.1: the datasets that do not reach the threshold even at 10% are the three "
         "smallest (the ones §6.2 discusses)", sorted(ds for ds, f in band.items() if f is None) == sorted(smallest3)),
    ]


# --- a cost model for when FEC pays off ---------------------------------------------------------
# Per new program, FEC pays the fingerprint step (c, relative to a full evaluation, measured in the phase-timing runs)
# and the full evaluation of the misses (1 - w, w = share of the full-evaluation work the hits avoided, diagnostic
# runs). So S_T ~ 1 / (1 + c - w), with no fitted parameter, and FEC is faster when w > c. The pilot check takes w
# from the first PILOT_GENS generations only, as a user would in a short pilot run.
PILOT_GENS = (5, 10)
_COST_CACHE: dict = {}


def _gen_work(store: Path) -> pd.DataFrame:
    """Hit and miss work (hits x mean hit size, misses x mean miss size, operator calls) per (algorithm, dataset,
    fraction, generation), main sampler, diagnostic runs, summed over seeds; read once."""
    key = (store, "gen_work")
    if key not in _COST_CACHE:
        cols = ["algorithm", "setup", "dataset", "mode", "sampler", "fraction", "gen", "fec_hits", "fec_misses",
                "fec_hit_operators", "fec_miss_operators"]
        g = pd.concat([c[(c.setup == SETUP) & (c["mode"] == "fec_diagnostic") & (c.sampler == MAIN_SAMPLER)
                         & (c.gen >= 1)] for c in pd.read_csv(_f(store, "generations.csv"), usecols=cols,
                                                               chunksize=1_000_000)])
        g = g.assign(fraction=g.fraction.round(2), hw=g.fec_hits * g.fec_hit_operators.fillna(0),
                     mw=g.fec_misses * g.fec_miss_operators.fillna(0))
        _COST_CACHE[key] = g.groupby(["algorithm", "dataset", "fraction", "gen"])[["hw", "mw"]].sum().reset_index()
    return _COST_CACHE[key]


def _work_upto(store: Path, last: int | None) -> pd.DataFrame:
    """w per (algorithm, dataset, fraction), main sampler, from generations 1..last (all when last is None)."""
    if last is None:
        w = hit_work(store)
        w = w[w.sampler == MAIN_SAMPLER][["algorithm", "dataset", "fraction", "work_avoided"]]
        return w.assign(fraction=w.fraction.round(2))
    g = _gen_work(store)
    s = g[g.gen <= last].groupby(["algorithm", "dataset", "fraction"])[["hw", "mw"]].sum()
    return (s.hw / (s.hw + s.mw)).rename("work_avoided").reset_index()


def cost_pilot_curve(design: pd.DataFrame, store: Path = results.STORE) -> pd.DataFrame:
    """Per algorithm and pilot length n (generations 1..n): configurations for which the rule w > c, with w from the
    pilot and c from the whole run, gives the observed side of S_T = 1."""
    keys = ["algorithm", "dataset", "fraction"]
    last = int(_gen_work(store).gen.max())
    rows = []
    for n in range(1, last + 1):
        x = design[keys + ["S_T", "fp_cost"]].merge(_work_upto(store, n), on=keys)
        for a, y in x.groupby("algorithm"):
            rows.append({"algorithm": a, "generations": n,
                         "right": int(((y.work_avoided > y.fp_cost) == (y.S_T > 1)).sum()), "of": len(y)})
    return pd.DataFrame(rows)


def cost_design(metrics: pd.DataFrame, props: pd.DataFrame, store: Path = results.STORE) -> pd.DataFrame:
    """Table S17: per algorithm, dataset and fraction, measured S_T, fingerprint cost c, work avoided w (whole run and
    the pilots), predicted S_T and whether the rule w > c gives the observed side of 1."""
    if not all(_f(store, f).is_file() for f in ("generations.csv", "phase_timing.csv")):
        return pd.DataFrame()
    keys = ["algorithm", "dataset", "fraction"]
    g = metrics[metrics.sampler == MAIN_SAMPLER].assign(fraction=lambda d: d.fraction.round(2))
    x = (g.groupby(keys).S_T.mean().reset_index()
         .merge(eval_overhead(store, set(g.dataset)).rename(columns={"overhead": "fp_cost"}), on=keys)
         .merge(_work_upto(store, None), on=keys))
    for n in PILOT_GENS:
        x = x.merge(_work_upto(store, n).rename(columns={"work_avoided": f"work_avoided_gen1_{n}"}), on=keys)
    if x.empty:
        return x
    if len(x) != len(g.groupby(keys)):
        raise TableError("cost_design: a configuration lacks the fingerprint cost or the work avoided")
    x = x.merge(props[["dataset", "rows"]], on="dataset", how="left")
    x["S_T_predicted"] = 1 / (1 + x.fp_cost - x.work_avoided)
    x["error_pct"] = 100 * (x.S_T_predicted / x.S_T - 1)
    x["rule_right"] = (x.work_avoided > x.fp_cost) == (x.S_T > 1)
    for n in PILOT_GENS:
        x[f"rule_right_gen1_{n}"] = (x[f"work_avoided_gen1_{n}"] > x.fp_cost) == (x.S_T > 1)
    cols = ["algorithm", "dataset", "rows", "fraction", "S_T", "fp_cost", "work_avoided", "S_T_predicted", "error_pct",
            "rule_right"] + [c for n in PILOT_GENS for c in (f"work_avoided_gen1_{n}", f"rule_right_gen1_{n}")]
    return x.sort_values(["algorithm", "rows", "dataset", "fraction"], kind="stable")[cols].reset_index(drop=True)


COST_EXAMPLES = (("GE", "bank_marketing", 0.1, "BankTen"), ("GP", "wine_quality", 0.01, "WineOne"))


def cost_numbers(metrics: pd.DataFrame, props: pd.DataFrame, store: Path = results.STORE) -> dict:
    """Macros \\resCost...: fit of the cost model, the rule, the pilots and two worked examples."""
    from scipy.stats import spearmanr
    d = cost_design(metrics, props, store)
    if d.empty:
        return {}
    out = {"CostN": str(int((d.algorithm == "GE").sum()))}
    for a in ("GE", "GP"):
        y = d[d.algorithm == a]
        out[f"{a}CostRho"] = f"{spearmanr(y.S_T, y.S_T_predicted).statistic:.2f}"
        out[f"{a}CostErr"] = f"{y.error_pct.abs().median():.0f}"
        out[f"{a}CostRight"] = str(int(y.rule_right.sum()))
        for n in PILOT_GENS:
            r = y[f"rule_right_gen1_{n}"]
            out[f"{a}CostPilot{_num_word(n)}"] = str(int(r.sum()))
            wrong = y[~r]
            out[f"{a}CostPilot{_num_word(n)}Cautious"] = str(int((wrong.S_T > 1).sum()))
    curve = cost_pilot_curve(d, store)
    for a in ("GE", "GP"):
        c, whole = curve[curve.algorithm == a], int(d[d.algorithm == a].rule_right.sum())
        out[f"{a}CostPilotNear"] = str(int(c[c.right >= whole - 1].generations.min()))
        out[f"{a}CostGens"] = str(int(c.generations.max()))
    for a, ds, f, key in COST_EXAMPLES:
        r = d[(d.algorithm == a) & (d.dataset == ds) & np.isclose(d.fraction, f)].iloc[0]
        out.update({f"Cost{key}C": f"{r.fp_cost:.2f}", f"Cost{key}W": f"{r.work_avoided:.2f}",
                    f"Cost{key}Pred": f"{r.S_T_predicted:.2f}", f"Cost{key}ST": f"{r.S_T:.2f}"})
    return out


def _num_word(n: int) -> str:
    return {5: "Five", 10: "Ten"}[n]


def cost_claims(metrics: pd.DataFrame, props: pd.DataFrame, store: Path = results.STORE) -> list[tuple[str, str, bool]]:
    """What §6.2 says about the cost model."""
    from scipy.stats import spearmanr
    d = cost_design(metrics, props, store)
    if d.empty:
        return []
    ge, gp = d[d.algorithm == "GE"], d[d.algorithm == "GP"]
    wine = d[(d.algorithm == "GP") & (d.dataset == "wine_quality") & np.isclose(d.fraction, 0.01)].iloc[0]
    gp5 = gp[~gp.rule_right_gen1_5]
    return [
        ("cost/rule", "§6.2: the rule w > c gave the observed side of S_T = 1 in all but at most one configuration per "
         "algorithm", bool(ge.rule_right.sum() >= len(ge) - 1 and gp.rule_right.sum() >= len(gp) - 1)),
        ("cost/rank", "§6.2: predicted and measured S_T rank the configurations alike (Spearman >= 0.85, both)",
         all(spearmanr(y.S_T, y.S_T_predicted).statistic >= 0.85 for y in (ge, gp))),
        ("cost/err-small", "§6.2: the typical (median absolute) prediction error is at most 15% (both)",
         bool(ge.error_pct.abs().median() <= 15 and gp.error_pct.abs().median() <= 15)),
        ("cost/gp-over", "§6.2: in GP the model tends to overestimate S_T (median signed error > 0)",
         bool(gp.error_pct.median() > 0)),
        ("cost/wine-example", "§6.2: GP wine_quality 1%: most work avoided (w > 0.8) but the fingerprint costs a full "
         "evaluation or more (c >= 1) and FEC was slower", bool(wine.work_avoided > 0.8 and wine.fp_cost >= 1
                                                                 and wine.S_T < 1)),
        ("cost/pilot-ge", "§6.2: in GE a 5-generation pilot gave the right decision as often as the whole run",
         bool(ge.rule_right_gen1_5.sum() >= ge.rule_right.sum())),
        ("cost/pilot-gp-cautious", "§6.2: in GP a 5-generation pilot was wrong more often, and all but at most one of "
         "its mistakes rejected FEC where it was faster", bool(len(gp5) > 5 and (gp5.S_T > 1).sum() >= len(gp5) - 1)),
        ("cost/pilot-gp-longer", "§6.2: in GP a 10-generation pilot was right more often than a 5-generation one",
         bool(gp.rule_right_gen1_10.sum() > gp.rule_right_gen1_5.sum())),
        ("cost/pilot-ge-early-gp-late", "§6.2 / Figure S5: GE came within one decision of the whole run after at most "
         "5 generations, GP only after at least 15",
         bool(int(cost_numbers(metrics, props, store)["GECostPilotNear"]) <= 5
              and int(cost_numbers(metrics, props, store)["GPCostPilotNear"]) >= 15)),
    ]


MEMORY_SEEDS = set(PAPER_SEEDS)


def _fitness_only_bytes(n: int, _memo: dict = {}) -> int:
    """Bytes a cache holding only key -> fitness would need for n entries, counted as instrument.cache_bytes counts
    (the dict plus its 32-character hex keys and float values), without the program-text table."""
    if n not in _memo:
        d = {f"{i:032x}": float(i) for i in range(int(n))}
        _memo[n] = sys.getsizeof(d) + sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in d.items())
    return _memo[n]


def memory_runs(keep, props: pd.DataFrame, store: Path = results.STORE) -> pd.DataFrame:
    """One row per instrumented diagnostic run of the paper datasets and seeds,
    random rows. cache_mb: the cache's own two tables at the end of the run (key -> fitness, key -> program text; the
    text only feeds the same-program hit count, not the lookup); fit_only_mb: the same entries without the text table;
    peak_mb: peak resident memory of the whole process (includes the diagnostic recording, so an upper bound)."""
    path = _f(store, "instrumented_runs.csv")
    if not path.is_file():
        return pd.DataFrame()
    r = pd.read_csv(path)
    r = r[r.dataset.isin(set(keep)) & (r.sampler == MAIN_SAMPLER) & r.seed.isin(MEMORY_SEEDS)
          & r.cache_bytes_final.notna() & r.peak_rss_bytes.notna()].copy()
    if r.empty:
        return r
    cls = props.drop_duplicates("dataset").set_index("dataset").classes
    r["multiclass"] = r.dataset.map(cls) > 2
    r["evaluations"] = r.hits + r.misses
    r["cache_mb"] = r.cache_bytes_final / 2**20
    r["fit_only_mb"] = r.cache_size.map(_fitness_only_bytes) / 2**20
    r["peak_mb"] = r.peak_rss_bytes / 2**20
    r["share_of_peak"] = r.cache_bytes_final / r.peak_rss_bytes
    return r


def memory_table(keep, props: pd.DataFrame, store: Path = results.STORE) -> pd.DataFrame:
    """Table S18: per algorithm, dataset and fraction, over the seeds: evaluations, cache entries, cache MB (as run),
    cache MB without the program text, peak process MB and the cache's share of the peak (median and max)."""
    r = memory_runs(keep, props, store)
    if r.empty:
        return r
    rows = props.drop_duplicates("dataset").set_index("dataset").rows
    t = r.groupby(["algorithm", "dataset", "fraction"]).agg(
        runs=("seed", "size"), evaluations_median=("evaluations", "median"),
        entries_median=("cache_size", "median"), entries_max=("cache_size", "max"),
        cache_mb_median=("cache_mb", "median"), cache_mb_max=("cache_mb", "max"),
        fitness_only_mb_median=("fit_only_mb", "median"), fitness_only_mb_max=("fit_only_mb", "max"),
        peak_mb_median=("peak_mb", "median"), peak_mb_max=("peak_mb", "max"),
        cache_share_of_peak_median=("share_of_peak", "median"), cache_share_of_peak_max=("share_of_peak", "max"),
    ).reset_index()
    t.insert(2, "rows", t.dataset.map(rows))
    ages = hit_split(set(keep))
    if len(ages):
        ages = ages.assign(fraction=lambda d: d.fraction.round(2))
        for n in AGE_WINDOWS:
            ages[f"hits_kept_last{n}"] = ages[f"age_le{n}"] / ages.hits
        t = t.assign(fraction=lambda d: d.fraction.round(2)).merge(
            ages[["algorithm", "dataset", "fraction", *[f"hits_kept_last{n}" for n in AGE_WINDOWS]]],
            on=["algorithm", "dataset", "fraction"], how="left")
    return t


def memory_numbers(keep, props: pd.DataFrame, store: Path = results.STORE) -> dict:
    """Macros \\resMem...: the cache-memory subsection."""
    r = memory_runs(keep, props, store)
    if r.empty:
        return {}
    out = {"MemDatasets": str(r.dataset.nunique()), "MemRuns": f"{len(r):,}".replace(",", "{,}"),
           "MemEntriesMax": f"{int(r.cache_size.max()):,}".replace(",", "{,}"),
           "MemEvaluationsMax": f"{int(r.evaluations.max()):,}".replace(",", "{,}"),
           "MemMaxMB": f"{r.cache_mb.max():.0f}", "MemFitOnlyMaxMB": f"{r.fit_only_mb.max():.1f}"}
    for a in ("GE", "GP"):
        x = r[r.algorithm == a]
        out[f"{a}MemEntries"] = f"{int(round(x.cache_size.median(), -2)):,}".replace(",", "{,}")
        out[f"{a}MemEntryShare"] = f"{100 * (x.cache_size / x.evaluations).median():.0f}"
        out[f"{a}MemBinMB"] = f"{x[~x.multiclass].cache_mb.median():.1f}"
        out[f"{a}MemMultiMB"] = f"{x[x.multiclass].cache_mb.median():.1f}"
        out[f"{a}MemMaxMB"] = f"{x.cache_mb.max():.0f}"
        out[f"{a}MemPeakMinMB"], out[f"{a}MemPeakMaxMB"] = f"{x.peak_mb.min():.0f}", f"{x.peak_mb.max():.0f}"
        out[f"{a}MemShareMax"] = f"{100 * x.share_of_peak.max():.0f}"
        out[f"{a}MemShareMed"] = f"{100 * x.share_of_peak.median():.1f}"
    ages = hit_split(set(keep))
    for a in ("GE", "GP"):
        x = ages[ages.algorithm == a]
        for n in AGE_WINDOWS:
            out[f"{a}MemKeptLast{_num_word(n)}"] = f"{100 * x[f'age_le{n}'].sum() / x.hits.sum():.0f}"
    return out


def memory_claims(keep, props: pd.DataFrame, store: Path = results.STORE) -> list[tuple[str, str, bool]]:
    """What the memory subsection says."""
    r = memory_runs(keep, props, store)
    if r.empty:
        return []
    ge, gp = r[r.algorithm == "GE"], r[r.algorithm == "GP"]
    top = r.loc[r.cache_mb.idxmax()]
    return [
        ("mem/all-runs", "§5.3: every paper dataset x algorithm x fraction has its 30 seeds",
         bool((r.groupby(["algorithm", "dataset", "fraction"]).size() == 30).all() and r.dataset.nunique() == len(set(keep)))),
        ("mem/entries-below-evals", "§5.3: no cache held more entries than its run had evaluations",
         bool((r.cache_size <= r.evaluations).all())),
        ("mem/entries-gp-more", "§5.3: GP stored more entries than GE (medians)",
         bool(gp.cache_size.median() > ge.cache_size.median())),
        ("mem/largest-gp-multi", "§5.3: the largest cache was a GP run on a multiclass dataset",
         bool(top.algorithm == "GP" and top.multiclass)),
        ("mem/text-most", "§5.3: in every run most of the cache space held the program text (fitness-only < half)",
         bool((r.fit_only_mb < 0.5 * r.cache_mb).all())),
        ("mem/share-small", "§5.3: the cache was at most 10% of the peak memory of any run (GE at most 5%)",
         bool(r.share_of_peak.max() <= 0.10 and ge.share_of_peak.max() <= 0.05)),
        ("mem/recent-gp-more", "§5.3: keeping only the last 10 generations keeps more of the hits in GP than in GE",
         bool(_kept(keep, "GP", 10) > _kept(keep, "GE", 10))),
        ("mem/recent-most", "§5.3: keeping only the last 10 generations keeps most (> 50%) of the hits in both",
         bool(_kept(keep, "GE", 10) > 0.5 and _kept(keep, "GP", 10) > 0.5)),
    ]


def _kept(keep, algo: str, n: int) -> float:
    """Share of all hits whose cache entry was at most n generations old (a cache keeping only those entries would
    still return them; a lower bound, as an evicted program would be stored again and could hit later)."""
    a = hit_split(set(keep))
    a = a[a.algorithm == algo]
    return float(a[f"age_le{n}"].sum() / a.hits.sum())


def fingerprint_rows(store: Path = results.STORE) -> pd.DataFrame:
    """Fingerprint rows per algorithm, dataset and fraction (random rows), as recorded by the FEC timing runs."""
    r = pd.read_csv(_f(store, "runs.csv"), usecols=["algorithm", "setup", "dataset", "mode", "sampler", "fraction",
                                                 "sample_size"])
    r = r[(r.setup == SETUP) & (r["mode"] == "fec") & (r.sampler == MAIN_SAMPLER)].assign(fraction=lambda d: d.fraction.round(2))
    return r.groupby(["algorithm", "dataset", "fraction"]).sample_size.first().rename("fp_rows").reset_index()


_SEL_CACHE: dict = {}


def selection_generations(keep, store: Path = results.STORE) -> pd.DataFrame:
    """Per generation of the instrumented diagnostic runs (random rows, paper seeds): share of
    the valid programs whose fitness used differs from the true fitness, the rank correlation of used and true fitness,
    the share of tournaments whose winner is truly worse than another aspirant, and how much worse (pp)."""
    path = _f(store, "instrumented_generations.csv")
    if not path.is_file():
        return pd.DataFrame()
    if path not in _SEL_CACHE:
        g = pd.read_csv(path, usecols=["algorithm", "dataset", "sampler", "fraction", "seed", "gen", "ins_n_valid",
                                       "ins_n_wrong", "ins_spearman", "ins_tournaments", "ins_tourn_changed",
                                       "ins_tourn_gap_mean", "ins_elite_wrong"])
        g = g[(g.sampler == MAIN_SAMPLER) & g.seed.isin(set(PAPER_SEEDS))].assign(fraction=lambda d: d.fraction.round(2))
        _SEL_CACHE[path] = g.assign(wrong_share=g.ins_n_wrong / g.ins_n_valid, gap_pp=100 * g.ins_tourn_gap_mean)
    g = _SEL_CACHE[path]
    return g[g.dataset.isin(set(keep))]


def selection_table(keep, store: Path = results.STORE) -> pd.DataFrame:
    """Table S19: per algorithm, dataset and fraction (30 seeds): fingerprint rows; share of programs with a wrong
    fitness (mean over generations, median over seeds); rank correlation used vs true (median, lowest generation);
    share of tournaments with a truly worse winner (whole run; median and largest seed); how much worse (pp, mean over
    those tournaments, median over seeds); generations whose best program carried a wrong fitness."""
    g = selection_generations(keep, store)
    if g.empty:
        return g
    keys = ["algorithm", "dataset", "fraction"]
    run = g.groupby(keys + ["seed"]).agg(wrong=("wrong_share", "mean"), rho=("ins_spearman", "mean"),
                                         rho_min=("ins_spearman", "min"), t=("ins_tournaments", "sum"),
                                         c=("ins_tourn_changed", "sum"), gap=("gap_pp", "mean"),
                                         elite=("ins_elite_wrong", "sum")).reset_index()
    run["changed"] = run.c / run.t
    t = run.groupby(keys).agg(runs=("seed", "size"), wrong_share_median=("wrong", "median"),
                              rank_corr_median=("rho", "median"), rank_corr_lowest=("rho_min", "min"),
                              changed_tournaments_median=("changed", "median"),
                              changed_tournaments_max=("changed", "max"), worse_by_pp_median=("gap", "median"),
                              elite_wrong_generations=("elite", "sum")).reset_index()
    return t.merge(fingerprint_rows(store), on=keys, how="left")[keys[:2] + ["fraction", "fp_rows", *t.columns[3:]]]


def selection_numbers(keep, store: Path = results.STORE) -> dict:
    """Macros \\resSel...: §5.5."""
    from scipy.stats import spearmanr
    t = selection_table(keep, store)
    if t.empty:
        return {}
    g = selection_generations(keep, store)
    out = {"SelGenerations": f"{len(g):,}".replace(",", "{,}"),
           "SelEliteWrong": str(int(g.ins_elite_wrong.sum()))}
    for a in ("GE", "GP"):
        x = t[t.algorithm == a]
        rho, p = spearmanr(x.fp_rows, x.changed_tournaments_median)
        out[f"{a}SelRhoRows"], out[f"{a}SelRhoRowsP"] = f"{rho:.2f}", _p(p)
        for f, w in FRACTIONS:
            y = x[np.isclose(x.fraction, f)]
            out[f"{a}SelChanged{w}"] = _pct_small(y.changed_tournaments_median.median())
            out[f"{a}SelChangedMax{w}"] = f"{100 * y.changed_tournaments_median.max():.0f}"
            out[f"{a}SelRank{w}"] = f"{y.rank_corr_median.median():.3f}"
            out[f"{a}SelGap{w}"] = f"{y.worse_by_pp_median.median():.2f}"
        big = x[x.fp_rows >= GUIDE_ROWS]
        out[f"{a}SelChangedBigMax"] = f"{100 * big.changed_tournaments_median.max():.1f}"
        top = x.loc[x.changed_tournaments_median.idxmax()]
        out[f"{a}SelTopDS"], out[f"{a}SelTopRows"] = _tt(top.dataset), str(int(top.fp_rows))
        out[f"{a}SelTop"] = f"{100 * top.changed_tournaments_median:.0f}"
    out["SelBigRows"] = f"{GUIDE_ROWS:,}".replace(",", "{,}")
    return out


def _pct_small(v: float) -> str:
    v = 100 * v
    return f"{v:.2f}" if v < 0.1 else f"{v:.1f}"


def _sel_trend(g: pd.DataFrame) -> pd.DataFrame:
    """Share of tournaments with a truly worse winner per generation, mean over datasets and seeds."""
    return g[g.gen > 0].groupby(["algorithm", "fraction", "gen"]).apply(
        lambda d: d.ins_tourn_changed.sum() / d.ins_tournaments.sum(), include_groups=False).rename("changed").reset_index()


def selection_claims(keep, store: Path = results.STORE) -> list[tuple[str, str, bool]]:
    """What §5.5 says about selection."""
    from scipy.stats import spearmanr
    t = selection_table(keep, store)
    if t.empty:
        return []
    g = selection_generations(keep, store)
    tr = _sel_trend(g)
    late = tr[tr.gen >= tr.gen.max() - 4].groupby(["algorithm", "fraction"]).changed.mean()
    early = tr[tr.gen <= 5].groupby(["algorithm", "fraction"]).changed.mean()
    ge, gp = t[t.algorithm == "GE"], t[t.algorithm == "GP"]
    out = [
        ("sel/all-runs", "§5.5: every configuration has its 30 seeds", bool((t.runs == 30).all())),
        ("sel/elite-never", "§5.5: the best program of a generation never carried a wrong fitness",
         bool(g.ins_elite_wrong.sum() == 0)),
        ("sel/rank-high", "§5.5: at 5% and 10% the median rank correlation of used and true fitness is at least 0.99 "
         "in both algorithms", bool(all(t[(t.algorithm == a) & ~np.isclose(t.fraction, 0.01)].rank_corr_median.median()
                                        >= 0.99 for a in ("GE", "GP")))),
        ("sel/gp-more", "§5.5: GP had more tournaments with a truly worse winner than GE at every fraction",
         all(gp[np.isclose(gp.fraction, f)].changed_tournaments_median.median()
             > ge[np.isclose(ge.fraction, f)].changed_tournaments_median.median() for f, _ in FRACTIONS)),
        ("sel/falls-fraction", "§5.5: the median share of changed tournaments falls from 1% to 5% to 10% (both)",
         all(np.all(np.diff([t[(t.algorithm == a) & np.isclose(t.fraction, f)].changed_tournaments_median.median()
                             for f, _ in FRACTIONS]) < 0) for a in ("GE", "GP"))),
        ("sel/falls-rows", "§5.5: the share of changed tournaments falls with the fingerprint rows (rho <= -0.7, p < 0.001)",
         all(spearmanr(x.fp_rows, x.changed_tournaments_median)[0] <= -0.7
             and spearmanr(x.fp_rows, x.changed_tournaments_median)[1] < 0.001 for x in (ge, gp))),
        ("sel/gap-small", "§5.5: at 5% and 10% a truly worse winner was worse by a median of less than 0.1 pp (both)",
         bool(t[~np.isclose(t.fraction, 0.01)].groupby(["algorithm", "fraction"]).worse_by_pp_median.median().max() < 0.1)),
        ("sel/top-smallest", "§5.5: the configuration with the most changed tournaments has a fingerprint below 100 rows "
         "(both)", all(x.loc[x.changed_tournaments_median.idxmax()].fp_rows < 100 for x in (ge, gp))),
        ("sel/grows", "§5.5 / Figure fig:selection: changed tournaments were more frequent in the last five generations than in "
         "the first five, for every algorithm and fraction", bool((late > early).all())),
    ]
    return out


def selection_figure(keep, out_dir: Path, store: Path = results.STORE) -> Path:
    """fig_selection.pdf (§5.5): over the generations, mean over datasets and seeds, per fingerprint fraction: the share
    of tournaments with a truly worse winner, how much worse it was (pp) and the rank correlation of used and true
    fitness; GE left, GP right."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs.use()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig_selection.pdf"
    meta = {"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None}
    g = selection_generations(keep, store)
    if g.empty:
        fs.use()
        fig = plt.figure(figsize=(6.3, 1))
        fig.text(0.5, 0.5, "no data", ha="center", va="center", color=INK_SECONDARY)
        fs.save(fig, path)
        plt.close(fig)
        return path
    tr = _sel_trend(g)
    gap = g[g.gen > 0].groupby(["algorithm", "fraction", "gen"]).gap_pp.mean().rename("gap").reset_index()
    rk = g.groupby(["algorithm", "fraction", "gen"]).ins_spearman.mean().rename("rho").reset_index()
    style = FRACTION_STYLE
    fs.use()
    fig, axes = plt.subplots(3, 2, figsize=(6.3, 6.6), sharex=True)
    panels = ((tr, "changed", 100, "tournaments with a truly\nworse winner (%)"),
              (gap, "gap", 1, "how much worse the\nwinner truly was (pp)"),
              (rk, "rho", 1, "rank correlation,\nused vs true fitness"))
    for j, algo in enumerate(("GE", "GP")):
        for i, (d, col, scale, label) in enumerate(panels):
            ax = axes[i, j]
            ends = {}
            for f, (c, ls) in style.items():
                y = d[(d.algorithm == algo) & np.isclose(d.fraction, f)].sort_values("gen", kind="stable")
                ax.plot(y.gen, scale * y[col], color=c, linestyle=ls, linewidth=1.6)
                ends[f] = (y.gen.iloc[-1], scale * y[col].iloc[-1])
            _end_labels(ax, ends)
            ax.grid(color=GRID, linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_xlim(0, 33)
            if j == 0:
                ax.set_ylabel(label)
        fs.title(axes[0, j], algo)
        axes[2, j].set_xlabel("generation")
    fig.tight_layout()
    fs.save(fig, path)
    plt.close(fig)
    return path


_TRAJ_CACHE: dict = {}
TRAJ_STATS = ("avg_nodes", "unique_phenotypes", "avg_pheno_len")   # program statistics: no fitness value in them


def trajectory_pairs(keep, store: Path = results.STORE) -> pd.DataFrame:
    """Every FEC timing run (random rows, paper seeds) beside its baseline of the
    same seed, per generation: same_pop = the program statistics TRAJ_STATS are equal (the populations have not
    separated); d_train / d_test = FEC minus baseline best training / test error (pp). The best training error of an
    FEC run is always its true one (the elite never carries a wrong value, Table S19)."""
    path = _f(store, "generations.csv")
    if not path.is_file():
        return pd.DataFrame()
    if path not in _TRAJ_CACHE:
        cols = ["algorithm", "setup", "dataset", "mode", "sampler", "fraction", "seed", "gen", "min", "fitness_test",
                *TRAJ_STATS]
        g = pd.read_csv(path, usecols=cols)
        g = g[(g.setup == SETUP) & g.seed.isin(set(PAPER_SEEDS))]
        b = g[g["mode"] == "baseline"].drop(columns=["mode", "sampler", "fraction", "setup"])
        f = g[(g["mode"] == "fec") & (g.sampler == MAIN_SAMPLER)].drop(columns=["mode", "setup", "sampler"])
        m = f.assign(fraction=lambda d: d.fraction.round(2)).merge(b, on=["algorithm", "dataset", "seed", "gen"],
                                                                    suffixes=("", "_b"))
        m["same_pop"] = np.logical_and.reduce([m[c] == m[f"{c}_b"] for c in TRAJ_STATS])
        m["d_train"], m["d_test"] = 100 * (m["min"] - m["min_b"]), 100 * (m.fitness_test - m.fitness_test_b)
        _TRAJ_CACHE[path] = m[["algorithm", "dataset", "fraction", "seed", "gen", "same_pop", "d_train", "d_test"]]
    m = _TRAJ_CACHE[path]
    return m[m.dataset.isin(set(keep))]


def trajectory_runs(keep, store: Path = results.STORE) -> pd.DataFrame:
    """Per FEC run: first generation whose population differs from the baseline's (NaN: never), and the differences
    at the last generation."""
    m = trajectory_pairs(keep, store)
    if m.empty:
        return m
    k = ["algorithm", "dataset", "fraction", "seed"]
    last = m[m.gen == m.gen.max()].set_index(k)[["d_train", "d_test"]]
    first = m[~m.same_pop].groupby(k).gen.min().rename("first_div")
    return last.join(first).reset_index()


def trajectory_table(keep, store: Path = results.STORE) -> pd.DataFrame:
    """Table S20: per algorithm, dataset and fraction (30 seeds): share of FEC runs whose population separated from
    the baseline's; median generation of separation; at the last generation, mean absolute and mean signed difference
    (FEC minus baseline) of the best training error and the mean signed difference of its test error (pp)."""
    r = trajectory_runs(keep, store)
    if r.empty:
        return r
    return r.groupby(["algorithm", "dataset", "fraction"]).agg(
        runs=("seed", "size"), diverged_share=("first_div", lambda v: v.notna().mean()),
        first_divergence_gen_median=("first_div", "median"),
        train_diff_abs_pp_end=("d_train", lambda v: v.abs().mean()), train_diff_pp_end=("d_train", "mean"),
        test_diff_pp_end=("d_test", "mean")).reset_index()


def _traj_curve(m: pd.DataFrame) -> pd.DataFrame:
    return m.groupby(["algorithm", "fraction", "gen"]).agg(
        diverged=("same_pop", lambda v: 1 - v.mean()), abs_train=("d_train", lambda v: v.abs().mean()),
        train=("d_train", "mean"), test=("d_test", "mean")).reset_index()


def trajectory_numbers(keep, store: Path = results.STORE) -> dict:
    """Macros \\resDiv...: §5.5."""
    from scipy.stats import wilcoxon
    r = trajectory_runs(keep, store)
    if r.empty:
        return {}
    m = trajectory_pairs(keep, store)
    c = _traj_curve(m)
    out = {}
    for a in ("GE", "GP"):
        for f, w in FRACTIONS:
            x = r[(r.algorithm == a) & np.isclose(r.fraction, f)]
            cc = c[(c.algorithm == a) & np.isclose(c.fraction, f)].set_index("gen")
            out[f"{a}Div{w}"] = f"{100 * x.first_div.notna().mean():.0f}"
            out[f"{a}DivGen{w}"] = f"{x.first_div.median():.0f}"
            out[f"{a}DivAbs{w}"] = f"{x.d_train.abs().mean():.1f}"
            out[f"{a}DivSigned{w}"] = f"{x.d_train.mean():+.2f}"
            out[f"{a}DivTest{w}"] = f"{x.d_test.mean():+.2f}"
            out[f"{a}DivSignedP{w}"] = _p(wilcoxon(x.d_train).pvalue)
            out[f"{a}DivTestP{w}"] = _p(wilcoxon(x.d_test).pvalue)
            out[f"{a}DivSignedGenTen{w}"] = f"{cc.train.loc[10]:+.2f}"
    later = r[r.first_div.notna()]
    out["DivNever"] = str(int(r.first_div.isna().sum()))
    out["DivRuns"] = f"{len(r):,}".replace(",", "{,}")
    rest = r[~np.isclose(r.fraction, 0.01)].groupby(["algorithm", "fraction"]).d_train.mean()
    out["DivSignedRestMax"] = f"{rest.abs().max():.2f}"
    grp = r.groupby(["algorithm", "fraction"])
    gens, gaps = grp.first_div.median(), grp.d_train.apply(lambda v: v.abs().mean())
    out["DivGenMin"], out["DivGenMax"] = f"{gens.min():.0f}", f"{gens.max():.0f}"
    out["DivAbsMin"], out["DivAbsMax"] = f"{gaps.min():.1f}", f"{gaps.max():.1f}"
    return out


def trajectory_claims(keep, store: Path = results.STORE) -> list[tuple[str, str, bool]]:
    """What §5.5 says about divergence and accumulation."""
    r = trajectory_runs(keep, store)
    if r.empty:
        return []
    c = _traj_curve(trajectory_pairs(keep, store))
    k = ["algorithm", "dataset", "fraction", "seed"]
    fake = pd.read_csv(_f(store, "instrumented_generations.csv"), usecols=[*k, "gen", "fec_fake_hits", "sampler"])
    fake = fake[(fake.sampler == MAIN_SAMPLER) & fake.dataset.isin(set(keep)) & fake.seed.isin(set(PAPER_SEEDS))]
    ff = fake[fake.fec_fake_hits > 0].assign(fraction=lambda d: d.fraction.round(2)).groupby(k).gen.min().rename("ff")
    j = r.set_index(k).join(ff)
    gp1 = c[(c.algorithm == "GP") & np.isclose(c.fraction, 0.01)].set_index("gen").train
    rest = r[~np.isclose(r.fraction, 0.01)].groupby(["algorithm", "fraction"])
    return [
        ("div/all-runs", "§5.5: every configuration has 30 paired runs",
         bool((r.groupby(["algorithm", "dataset", "fraction"]).size() == 30).all())),
        ("div/after-fake", "§5.5: no FEC run separated from its baseline before its first fake hit",
         bool(((j.first_div >= j.ff + 1) | j.first_div.isna()).all() and not (j.first_div.notna() & j.ff.isna()).any())),
        ("div/most", "§5.5: most FEC runs separated from their baseline (> 80% in every configuration group)",
         bool((r.groupby(["algorithm", "fraction"]).first_div.apply(lambda v: v.notna().mean()) > 0.8).all())),
        ("div/early", "§5.5: separation came early (median generation <= 6 in every algorithm and fraction)",
         bool((r.groupby(["algorithm", "fraction"]).first_div.median() <= 6).all())),
        ("div/abs-at-ten", "§5.5: at 10% the best training errors still differed by at least 0.5 pp on average",
         bool((r[np.isclose(r.fraction, 0.1)].groupby("algorithm").d_train.apply(lambda v: v.abs().mean()) >= 0.5).all())),
        ("div/no-drift-rest", "§5.5: at 5% and 10% the mean signed difference of best training error at the end is "
         "within 0.1 pp (both)", bool(rest.d_train.mean().abs().max() <= 0.1)),
        ("div/gp1-grows", "§5.5 / Figure fig:divergence: in GP at 1% the mean signed difference grows from generation 5 to 10 to 20 to 30",
         bool(gp1.loc[5] < gp1.loc[10] < gp1.loc[20] < gp1.loc[30])),
        ("div/gp1-worse", "§5.5: GP at 1% ended significantly worse in training and test error (Wilcoxon p < 0.001)",
         _div_sig(r, "GP")),
        ("div/test-rest-ns", "§5.5: at 5% and 10% the test error of FEC and baseline did not differ significantly "
         "(Wilcoxon p >= 0.05, both algorithms)", _div_test_ns(r)),
        ("div/ge1-smaller", "§5.5: GE at 1% ended worse than its baseline, but by less than GP at 1%",
         bool(0 < r[(r.algorithm == "GE") & np.isclose(r.fraction, 0.01)].d_train.mean()
              < r[(r.algorithm == "GP") & np.isclose(r.fraction, 0.01)].d_train.mean())),
    ]


def _div_test_ns(r: pd.DataFrame) -> bool:
    from scipy.stats import wilcoxon
    return all(wilcoxon(x.d_test).pvalue >= 0.05 for (_, f), x in r.groupby(["algorithm", "fraction"])
               if not np.isclose(f, 0.01))


def _div_sig(r: pd.DataFrame, algo: str) -> bool:
    from scipy.stats import wilcoxon
    x = r[(r.algorithm == algo) & np.isclose(r.fraction, 0.01)]
    return bool(x.d_train.mean() > 0 and x.d_test.mean() > 0 and wilcoxon(x.d_train).pvalue < 0.001
                and wilcoxon(x.d_test).pvalue < 0.001)


def trajectory_figure(keep, out_dir: Path, store: Path = results.STORE) -> Path:
    """fig_divergence.pdf (§5.5): over the generations, per fingerprint fraction: share of FEC runs whose population has
    separated from the baseline's; mean absolute and mean signed difference (FEC minus baseline) of the best
    training error (pp); GE left, GP right."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs.use()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig_divergence.pdf"
    meta = {"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None}
    m = trajectory_pairs(keep, store)
    if m.empty:
        fs.use()
        fig = plt.figure(figsize=(6.3, 1))
        fig.text(0.5, 0.5, "no data", ha="center", va="center", color=INK_SECONDARY)
        fs.save(fig, path)
        plt.close(fig)
        return path
    c = _traj_curve(m)
    panels = (("diverged", 100, "FEC runs separated\nfrom the baseline (%)"),
              ("abs_train", 1, "best training error,\n|FEC - baseline| (pp)"),
              ("train", 1, "best training error,\nFEC - baseline (pp)"))
    fs.use()
    fig, axes = plt.subplots(3, 2, figsize=(6.3, 6.6), sharex=True)
    for j, algo in enumerate(("GE", "GP")):
        for i, (col, scale, label) in enumerate(panels):
            ax = axes[i, j]
            if col == "train":
                ax.axhline(0, color=INK_SECONDARY, linewidth=0.8, zorder=1)
            ends = {}
            for f, (cl, ls) in FRACTION_STYLE.items():
                y = c[(c.algorithm == algo) & np.isclose(c.fraction, f)].sort_values("gen", kind="stable")
                ax.plot(y.gen, scale * y[col], color=cl, linestyle=ls, linewidth=1.6, zorder=3)
                ends[f] = (y.gen.iloc[-1], scale * y[col].iloc[-1])
            _end_labels(ax, ends)
            ax.grid(color=GRID, linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_xlim(0, 33)
            if j == 0:
                ax.set_ylabel(label)
        fs.title(axes[0, j], algo)
        axes[2, j].set_xlabel("generation")
    for i in (1, 2):                                            # same y scale for GE and GP: comparable at a glance
        lo = min(axes[i, 0].get_ylim()[0], axes[i, 1].get_ylim()[0])
        hi = max(axes[i, 0].get_ylim()[1], axes[i, 1].get_ylim()[1])
        for ax in axes[i]:
            ax.set_ylim(lo, hi)
    fig.tight_layout()
    fs.save(fig, path)
    plt.close(fig)
    return path


FRACTION_STYLE = {0.01: ("#86b6ef", ":"), 0.05: ("#2a78d6", "--"), 0.1: ("#0d366b", "-")}   # fig_selection, fig_divergence


def _end_labels(ax, ends: dict) -> None:
    """Fraction labels at the line ends, at least 8 points apart vertically (spread around their mean position)."""
    lo_, hi_ = ax.get_ylim()
    height = ax.get_window_extent().height * 72 / ax.figure.dpi          # axis height in points
    order = sorted(ends, key=lambda f: ends[f][1])
    want = [(ends[f][1] - lo_) / (hi_ - lo_) * height for f in order]
    put = list(want)
    for i in range(1, len(put)):                                         # push up where too close ...
        put[i] = max(put[i], put[i - 1] + 8)
    excess = (sum(put) - sum(want)) / len(put)                           # ... then centre the block on the lines
    put = [v - excess for v in put]
    for i in range(1, len(put)):
        put[i] = max(put[i], put[i - 1] + 8)
    for f, w, v in zip(order, want, put):
        ax.annotate(f"{round(100 * f)}%", ends[f], xytext=(3, v - w), textcoords="offset points", fontsize=7,
                    color=INK_SECONDARY, va="center", bbox=dict(fc="white", ec="none", pad=0.3))


def exact_runs(keep, store: Path = results.STORE) -> pd.DataFrame:
    """Per instrumented diagnostic run (random rows, paper seeds), the shadow
    exact cache (program text; recorded only, never supplies a fitness) beside FEC on the same evaluations: hit rates
    and the entries each would store (exact: distinct programs looked up; FEC: misses)."""
    path = _f(store, "instrumented_generations.csv")
    if not path.is_file():
        return pd.DataFrame()
    g = pd.read_csv(path, usecols=["algorithm", "dataset", "sampler", "fraction", "seed", "fec_hits", "fec_misses",
                                   "ins_exact_hits", "ins_exact_lookups"])
    g = g[(g.sampler == MAIN_SAMPLER) & g.dataset.isin(set(keep)) & g.seed.isin(set(PAPER_SEEDS))]
    r = g.assign(fraction=lambda d: d.fraction.round(2)).groupby(["algorithm", "dataset", "fraction", "seed"]).sum(
        numeric_only=True).reset_index()
    return r.assign(exact=r.ins_exact_hits / r.ins_exact_lookups, fec=r.fec_hits / (r.fec_hits + r.fec_misses),
                    entry_ratio=(r.ins_exact_lookups - r.ins_exact_hits) / r.fec_misses)


def exact_numbers(keep, store: Path = results.STORE) -> dict:
    """Macros \\resExact...: the shadow exact cache at REPEAT_FRACTION, medians over datasets."""
    r = exact_runs(keep, store)
    if r.empty:
        return {}
    d = r[np.isclose(r.fraction, REPEAT_FRACTION)].groupby(["algorithm", "dataset"])[["exact", "fec", "entry_ratio"]].median()
    out = {"ExactEntryRatio": f"{d.entry_ratio.median():.1f}"}
    for a in ("GE", "GP"):
        x = d.loc[a]
        out[f"{a}Exact"], out[f"{a}ExactFec"] = f"{100 * x.exact.median():.0f}", f"{100 * x.fec.median():.0f}"
    return out


def exact_claims(keep, store: Path = results.STORE) -> list[tuple[str, str, bool]]:
    r = exact_runs(keep, store)
    if r.empty:
        return []
    c = r.groupby(["algorithm", "dataset", "fraction"])[["exact", "fec", "entry_ratio"]].median()
    return [
        ("exact/below-fec", "§5.3: the shadow exact cache hit less often than FEC in every configuration",
         bool((c.exact < c.fec).all())),
        ("exact/more-entries", "§5.3: an exact cache would store more entries than FEC in every configuration",
         bool((c.entry_ratio > 1).all())),
        ("exact/all-runs", "§5.3: every configuration has its 30 instrumented runs",
         bool((r.groupby(["algorithm", "dataset", "fraction"]).size() == 30).all())),
    ]


def dynamics_figures(corpus, metrics, props, out_tables, min_rows=0, **_) -> list[Path]:
    """§5.5: fig_selection.pdf and fig_divergence.pdf, main text."""
    keep = set(main_text_datasets(corpus, props, min_rows))
    return [selection_figure(keep, out_tables), trajectory_figure(keep, out_tables)]


def cost_figure(design: pd.DataFrame, out_dir: Path) -> Path:
    """FigureS5_cost_model.pdf: predicted against measured S_T (log-log), one point per configuration, coloured by
    the rows of the dataset (as Figure S4), the line of equality and the lines S_T = 1; GE and GP."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    fs.use()
    out_dir.mkdir(parents=True, exist_ok=True)
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
    path = out_dir / "FigureS5_cost_model.pdf"
    meta = {"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None}
    if design.empty:
        fs.use()
        fig = plt.figure(figsize=(6.3, 1))
        fig.text(0.5, 0.5, "no data", ha="center", va="center", color=INK_SECONDARY)
        fs.save(fig, path)
        plt.close(fig)
        return path
    cmap = LinearSegmentedColormap.from_list("rows", SIZE_RAMP)
    norm = LogNorm(design.rows.min(), design.rows.max())
    lo = 0.9 * min(design.S_T.min(), design.S_T_predicted.min())
    hi = 1.1 * max(design.S_T.max(), design.S_T_predicted.max())
    fs.use()
    fig = plt.figure(figsize=(6.3, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=(1, 0.8))
    axes = [fig.add_subplot(gs[0, 0])]
    axes.append(fig.add_subplot(gs[0, 1], sharex=axes[0], sharey=axes[0]))
    pax = fig.add_subplot(gs[1, :])
    for ax, algo in zip(axes, ("GE", "GP")):
        g = design[design.algorithm == algo]
        ax.plot([lo, hi], [lo, hi], color=INK_SECONDARY, linewidth=0.8, zorder=1)
        ax.axhline(1, color=INK_SECONDARY, linewidth=0.6, linestyle="--", zorder=1)
        ax.axvline(1, color=INK_SECONDARY, linewidth=0.6, linestyle="--", zorder=1)
        ax.scatter(g.S_T_predicted, g.S_T, s=18, c=[cmap(norm(v)) for v in g.rows], edgecolor="white",
                   linewidth=0.5, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ticks = [v for v in (0.8, 1, 1.5, 2, 3, 4) if lo <= v <= hi]
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator(ticks))
            axis.set_major_formatter(FixedFormatter([f"{v:g}" for v in ticks]))
            axis.set_minor_locator(NullLocator())
        ax.grid(color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fs.title(ax, algo, f"rule w > c right in {int(g.rule_right.sum())} of {len(g)}")
        ax.set_xlabel("predicted $S_T = 1/(1 + c - w)$ (log scale)")
    axes[0].set_ylabel("measured $S_T$ (log scale)")
    curve = cost_pilot_curve(design)
    for algo, col, ls in (("GE", SERIES_COLOUR, "-"), ("GP", SECOND_COLOUR, "--")):
        c = curve[curve.algorithm == algo]
        pax.plot(c.generations, c.right, color=col, linestyle=ls, linewidth=1.6, zorder=3, marker="o", markersize=3,
                 markeredgecolor="white", markeredgewidth=0.4, label=algo)
    pax.legend(loc="lower right")
    n = int(curve["of"].max())
    pax.set_ylim(min(curve.right.min() - 2, n - 20), n + 1)
    pax.set_xlim(0.5, curve.generations.max() + 0.5)
    pax.axhline(n, color=INK_SECONDARY, linewidth=0.6, linestyle=":", zorder=1)
    pax.grid(color=GRID, linewidth=0.6, zorder=0)
    pax.set_axisbelow(True)
    for side in ("top", "right"):
        pax.spines[side].set_visible(False)
    pax.set_xlabel("pilot length: generations 1 to n used to estimate w")
    pax.set_ylabel(f"rule right (of {n})")
    fs.title(pax, "Short pilot", "rule w > c with w from generations 1 to n only (c from the whole run)")
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    cax = fig.add_axes((0.875, 0.5, 0.015, 0.4))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("rows of the dataset (log scale)", color=INK)
    cb.outline.set_edgecolor(INK_SECONDARY)
    fs.save(fig, path)
    plt.close(fig)
    return path


def numbers(corpus, metrics, props, out_tables, min_rows=0, all_metrics=None, **_) -> list[Path]:
    """numbers.tex: one LaTeX macro per number quoted in the results text, so the prose follows the store.
    Every macro is \\res<Algorithm><Name> or \\res<Name>; claims in the text are the ones computed here."""
    from scipy.stats import spearmanr
    from analysis import wilcoxon_paired
    keep = main_text_datasets(corpus, props, min_rows)
    ann = annotated(metrics[metrics.dataset.isin(keep)]).merge(props[["dataset", "rows"]], on="dataset", how="left")
    every = all_metrics if all_metrics is not None else metrics
    ann_all = annotated(every[every.dataset.isin(keep)]).merge(props[["dataset", "rows"]], on="dataset", how="left")
    out = {}
    both = set(ann[ann.algorithm == "GE"].dataset) & set(ann[ann.algorithm == "GP"].dataset)
    out["NdatasetsBoth"] = str(len(both))
    pairs = ann[["dataset", "algorithm"]].drop_duplicates()
    out["RunsTotal"] = _int(len(pairs) * (30 + 2 * 3 * 30))     # baseline + timing + diagnostic, random rows only
    tt = gegp_train_test()                                      # §6.6.1: no overfitting on the two case tables
    if tt:
        out["GegpTrainTestGap"] = f"{tt['gap_max']:.1f}"
    for algo, v_ in size_partials(ann, props).items():         # §5: size against the cache factors
        rhos = [r for r, _p_ in v_["partial"]]
        out[f"{algo}SizePartialMin"], out[f"{algo}SizePartialMax"] = f"{min(rhos):.2f}", f"{max(rhos):.2f}"
        out[f"{algo}SizePartialPMax"] = _p(max(p_ for _r, p_ in v_["partial"]))
        r_, p_ = v_["work_given_rows"]
        out[f"{algo}WorkGivenRows"], out[f"{algo}WorkGivenRowsP"] = f"{r_:.2f}", _p(p_)
        out[f"{algo}RtwoRows"], out[f"{algo}RtwoRowsWork"] = f"{100 * v_['r2_rows']:.0f}", f"{100 * v_['r2_rows_work']:.0f}"
    if (_f(results.STORE, "runs.csv")).is_file():
        u = undecided_limit(keep=set(ann.dataset))
        if u["undecided"]:                                          # a store without these runs skips the macros
            out["UndecidedN"], out["UndecidedReach"] = str(u["undecided"]), str(u["reach"])
            out["UndecidedSdMin"], out["UndecidedSdMax"] = f"{u['sd_min']:.1f}", f"{u['sd_max']:.1f}"
    w = with_wall(ann)                                          # whole-run wall time, reported beside S_T
    base = pd.read_csv(_f(results.STORE, "runs.csv"), usecols=["algorithm", "setup", "dataset", "mode", "eval_time_sec",
                                                            "wall_time_sec"])
    base = base[(base.setup == SETUP) & (base["mode"] == "baseline") & base.dataset.isin(set(ann.dataset))]
    for algo in ("GE", "GP"):
        g = w[w.algorithm == algo]
        if len(g):
            out[f"{algo}WallFaster"] = f"{100 * (g.wall_speed == 'faster').mean():.0f}"
            out[f"{algo}WallMedian"] = f"{g.S_wall.median():.2f}"
            b = base[base.algorithm == algo]
            out[f"{algo}EvalShare"] = f"{100 * (b.eval_time_sec / b.wall_time_sec).median():.0f}"
    hk = hit_kinds(keep)                                              # kinds of hit (pooled)
    for (a_, fr_), x in hk.assign(n=hk.hits).groupby(["algorithm", "fraction"]):
        word = {0.01: "One", 0.05: "Five", 0.1: "Ten"}[round(fr_, 2)]
        tot = x.n.sum()
        for k, _ in HIT_KINDS:
            out[f"{a_}Hit{k.capitalize()}{word}"] = f"{100 * (x[k] * x.n).sum() / tot:.0f}"
    sv = seconds_saved()                                              # §5.1/§5.2: ratio and seconds, examples
    if len(sv):
        rows_of = props.set_index("dataset")["rows"]
        small, large = min(keep, key=lambda d: rows_of[d]), max(keep, key=lambda d: rows_of[d])
        mm = metrics[metrics.dataset.isin(keep)].assign(fraction=lambda d: d.fraction.round(2)) \
            .merge(sv, on=["algorithm", "dataset", "fraction"])
        out["SavedSmallName"], out["SavedLargeName"] = _tt(small), _tt(large)
        for a_ in ("GE", "GP"):
            g_ = mm[mm.algorithm == a_]
            if g_.empty:
                continue
            for tag, d in (("Small", small), ("Large", large)):
                x = g_[(g_.dataset == d) & (g_.fraction == 0.01)]
                if len(x):
                    out[f"{a_}Saved{tag}ST"], out[f"{a_}Saved{tag}"] = f"{x.S_T.iloc[0]:.2f}", _dur(x.eval_saved_s.iloc[0])
                    out[f"{a_}Saved{tag}Abs"] = _dur(abs(x.eval_saved_s.iloc[0]))
            x = g_.loc[g_.S_T.idxmax()]
            out[f"{a_}SavedMaxST"] = _dur(x.eval_saved_s)
    bs = baseline_seconds()                                           # §5: baseline evaluation time per run (s)
    bs = bs[bs.dataset.isin(keep)] if len(bs) else bs
    for a_ in ("GE", "GP"):
        if a_ in bs and bs[a_].notna().any():
            out[f"BaseSec{a_}Min"], out[f"BaseSec{a_}Max"] = _secs(bs[a_].min()), _secs(bs[a_].max())
            out[f"BaseHours{a_}Max"] = f"{bs[a_].max() / 3600:.1f}"
    gb = props[props.dataset.isin(corpus.dataset)].get("difficulty_gb", pd.Series(dtype=float)).dropna()  # complexity
    out["GBMin"], out["GBMax"] = (f"{gb.min():.2f}", f"{gb.max():.2f}") if len(gb) else ("--", "--")
    gpr = props[props.dataset.isin(keep)]                       # §4.3: grammar diagnostics, full table in S6
    if "range" in gpr and len(gpr):
        for k, word in (("pm1", "One"), ("pm2", "Two"), ("pm3", "Three")):
            out[f"GrammarRange{word}"] = str(int((gpr["range"] == k).sum()))
        # "at least" in the text: round down, so the bound holds (0.958 -> 0.95)
        out["GrammarCoverageMin"] = f"{np.floor(100 * gpr.coverage_min.min()) / 100:.2f}"
        out["GrammarDegenerateMax"] = f"{100 * gpr.degenerate_rate.max():.0f}"
        out["GrammarDegenerateMin"] = f"{100 * gpr.degenerate_rate.min():.0f}"
        out["GrammarDegenerateMedian"] = f"{100 * gpr.degenerate_rate.median():.0f}"
        fr = gpr.fairness_ratio.dropna()
        out["GrammarFairN"] = str(len(fr))
        out["GrammarFairMin"], out["GrammarFairMax"] = (f"{fr.min():.2f}", f"{fr.max():.2f}") if len(fr) else ("--", "--")
        ex = gpr[gpr.dataset == "electricity_cat"]               # §4.2 worked example
        if len(ex):
            out["GrammarExCoverage"] = f"{np.floor(100 * ex.coverage_min.iloc[0]) / 100:.2f}"
            out["GrammarExDegenerate"] = f"{100 * ex.degenerate_rate.iloc[0]:.0f}"
    if (_f(results.STORE, "generations.csv")).is_file():          # Table 2 note: GE mutation intensity
        cols = ["algorithm", "setup", "dataset", "mode", "avg_used_codons"]
        g_ = pd.concat(c[(c.setup == SETUP) & (c.algorithm == "GE") & (c["mode"] == "baseline") & c.dataset.isin(keep)]
                       for c in pd.read_csv(_f(results.STORE, "generations.csv"), usecols=cols, chunksize=1_000_000))
        if len(g_):
            cod = g_.groupby("dataset").avg_used_codons.mean()
            out["GECodonsMedian"] = f"{cod.median():.0f}"
            out["GEMutFlipsMedian"], out["GEMutFlipsMax"] = f"{0.05 * cod.median():.1f}", f"{0.05 * cod.max():.1f}"
            out["GEMutAnyMedian"] = f"{100 * (1 - 0.95 ** cod.median()):.0f}"   # P(at least one codon flipped)
    pre = preprocessing_rows(sorted(keep))                     # §4.2: what the recipes did
    if len(pre):
        out["PreMissingRows"] = str(int(pre.missing_value_rows_removed.sum()))
        out["PreBalancedN"] = str(int((pre.balancing != "none (as published)").sum()))
    out.update(case_numbers(metrics, props))
    out.update(mc_numbers(metrics, props))
    out.update(work_numbers(ann, props))
    if "classes" in props:
        out.update(repeats_numbers(set(keep), props))
        out.update(hit_kind_numbers(set(keep), props))
        out.update(split_numbers(set(keep)))                          # fitness- vs behaviour-equivalent
        out.update(severity_numbers(set(keep), props))
    out.update(size_pair_numbers(ann, props))
    out.update(gegp_equivalent_numbers(ann, props))
    out.update(size_fig_numbers())
    out.update(conclusion_numbers(ann))
    out.update(correlation_numbers(metrics[metrics.dataset.isin(keep)], props))
    out.update(size_numbers(metrics[metrics.dataset.isin(keep)], props))
    out.update(guide_numbers(metrics[metrics.dataset.isin(keep)], props))
    out.update(cost_numbers(metrics[metrics.dataset.isin(keep)], props))
    out.update(memory_numbers(keep, props))
    out.update(selection_numbers(keep))
    out.update(trajectory_numbers(keep))
    out.update(exact_numbers(keep))
    out.update(fraction_tradeoff_numbers(ann))
    out.update(pair_numbers(metrics[metrics.dataset.isin(keep)]))
    for algo in ("GE", "GP"):
        g = ann[ann.algorithm == algo]
        if not len(g):
            continue
        p = f"{algo}"
        out[p + "Ndatasets"] = str(g.dataset.nunique())
        out[p + "Ncells"] = str(len(g))
        out[p + "Faster"] = str((g.speed == "faster").sum())
        out[p + "Slower"] = str((g.speed == "slower").sum())
        out[p + "SlowerDatasets"] = _names(sorted(set(g[g.speed == "slower"].dataset)))
        # The same counts without correction and with Benjamini-Hochberg, beside Holm's
        for tag, col in (("Raw", "speed_raw"), ("BH", "speed_bh")):
            out[p + "Faster" + tag] = str((g[col] == "faster").sum())
            out[p + "Slower" + tag] = str((g[col] == "slower").sum())
            out[p + "SlowerDatasets" + tag] = _names(sorted(set(g[g[col] == "slower"].dataset)))
        lost = g[(g.speed_raw != "") & (g.speed == "")]
        out[p + "HolmLost"] = str(len(lost))
        cfg = [f"{_tt(r.dataset)} at {100 * r.fraction:.0f}\\%"
               for r in lost.sort_values(["dataset", "fraction"], kind="stable").itertuples()]
        out[p + "HolmLostConfigs"] = ("none" if not cfg else cfg[0] if len(cfg) == 1
                                      else ", ".join(cfg[:-1]) + " and " + cfg[-1])
        out[p + "HolmMaxP"] = f"{g[g.speed != ''].p_eval_holm.max():.3f}" if (g.speed != "").any() else "--"
        out.update(holm_size_numbers(g, p))
        both = ann[(ann.speed_raw != "") & (ann.speed == "")]               # both algorithms: one sentence in §3
        out["HolmLostPMin"] = f"{both.p_eval_time.min():.3f}" if len(both) else "--"
        out["HolmLostSTMin"] = f"{both.S_T.min():.2f}" if len(both) else "--"
        out["HolmLostSTMax"] = f"{both.S_T.max():.2f}" if len(both) else "--"
        hi, lo = g.loc[g.S_T.idxmax()], g.loc[g.S_T.idxmin()]
        out[p + "MaxST"], out[p + "MaxSTdataset"], out[p + "MaxSTconfig"] = f"{hi.S_T:.2f}", _tt(hi.dataset), _cfg(hi)
        out[p + "MinST"], out[p + "MinSTdataset"], out[p + "MinSTconfig"] = f"{lo.S_T:.2f}", _tt(lo.dataset), _cfg(lo)
        thr = all_faster_threshold(g)
        out[p + "ThresholdDataset"] = _tt(thr[0]) if thr else "none"
        out[p + "ThresholdRows"] = _int(thr[1]) if thr else "--"
        out[p + "AllFasterDatasets"] = str(int(g.groupby("dataset").speed.apply(lambda s: (s == "faster").all()).sum()))
        for f, word in ((0.01, "One"), (0.05, "Five"), (0.10, "Ten")):
            gf = g[g.fraction.round(2) == f]
            out[p + "MedianST" + word] = f"{gf.S_T.median():.2f}"
            out[p + "EquivShare" + word] = f"{100 * (gf.verdict == 'equivalent').mean():.0f}"
        per = g.groupby("dataset").agg(S_T=("S_T", "mean"), rows=("rows", "first"))
        rho, pval = spearmanr(per.S_T, per.rows)
        out[p + "RhoRows"], out[p + "RhoRowsP"] = f"{rho:.2f}", (f"={pval:.3f}" if pval >= 0.001 else "<0.001")
        per = g.groupby("dataset").agg(dAcc=("dAcc", "mean"), rows=("rows", "first"))
        rho, pval = spearmanr(per.dAcc, per.rows)
        out[p + "RhoDaccRows"], out[p + "RhoDaccRowsP"] = f"{rho:.2f}", (f"={pval:.2f}" if pval >= 0.01 else f"={pval:.3f}")
        v = g.verdict.value_counts()
        for k in ("equivalent", "worse", "better", "undecided"):
            out[p + "Verdict" + k.capitalize()] = str(int(v.get(k, 0)))
        worse = g[g.verdict == "worse"]
        out[p + "WorseCells"] = "; ".join(f"{_tt(r.dataset)} ({_cfg(r)}, {_pp(r.dAcc)} pp)" for r in worse.itertuples()) \
            or "none"
        lowest = g.loc[g.dAcc.idxmin()]
        out[p + "MinDacc"], out[p + "MinDaccDataset"], out[p + "MinDaccConfig"] = _pp(lowest.dAcc), _tt(lowest.dataset), \
            _cfg(lowest)
        out[p + "MedianHit"] = f"{g.hit_rate.median():.2f}"
        gf = g.merge(props.reindex(columns=["dataset", "features"]), on="dataset", how="left")
        for y, x, name in (("fake_hit_ratio", "rows", "FakeRows"),
                           ("hit_rate", "rows", "HitRows"), ("dAcc", "fake_hit_ratio", "DaccFake")):
            rho, pval, _n = dataset_rho(gf, y, x)
            out[p + "Rho" + name], out[p + "Rho" + name + "P"] = f"{rho:.2f}", _p(pval)
        rho, pval, _n = dataset_rho(gf.assign(true_hit=gf.hit_rate * (1 - gf.fake_hit_ratio)), "true_hit", "rows")
        out[p + "RhoTrueHitRows"], out[p + "RhoTrueHitRowsP"] = f"{rho:.2f}", _p(pval)
        out[p + "FeaturesMinP"] = f"{min(dataset_rho(gf, y, 'features')[1] for y in CACHE_MEASURES):.2f}"
        ov = eval_overhead(keep=set(g.dataset))                        # §5: what a hit saves, relative
        if len(ov):
            go = g.assign(fraction=g.fraction.round(2)).merge(ov[ov.algorithm == algo],
                                                              on=["algorithm", "dataset", "fraction"])
            go = go.assign(rows=go.dataset.map(props.set_index("dataset")["rows"]))
            rho_, p_, _n = dataset_rho(go, "overhead", "rows")
            out[p + "RhoOverheadRows"], out[p + "RhoOverheadRowsP"] = f"{rho_:.2f}", _p(p_)
            rho_, p_, _n = dataset_rho(go, "S_T", "overhead")
            out[p + "RhoOverheadST"], out[p + "RhoOverheadSTP"] = f"{rho_:.2f}", _p(p_)
            per_ov = go.groupby("dataset").agg(overhead=("overhead", "mean"), rows=("rows", "first")).sort_values("rows", kind="stable")
            out[p + "OverheadSmall"] = f"{per_ov.overhead.iloc[:3].mean():.2f}"
            out[p + "OverheadLarge"] = f"{per_ov.overhead.iloc[-3:].mean():.2f}"
            out[p + "OverheadTwoSmallMin"] = f"{per_ov.overhead.iloc[:2].min():.2f}"
            out[p + "OverheadTwoSmallMax"] = f"{per_ov.overhead.iloc[:2].max():.2f}"
        for (a_, fr_), (r_, p_, _n) in fake_acc_by_fraction(g).items():   # §5.4 and Figure S2
            word = {0.01: "One", 0.05: "Five", 0.1: "Ten"}[round(fr_, 2)]
            out[p + f"FakeAccRho{word}"], out[p + f"FakeAccRho{word}P"] = f"{r_:.2f}", _p(p_)
        fa = fake_acc_stats(g, algo)                                   # §5.3: fake hits and accuracy
        if fa:
            out[p + "FakeAccLarger"], out[p + "FakeAccN"] = str(fa["larger"]), str(fa["n"])
            out[p + "FakeAccWP"] = _p(fa["wilcoxon_p"])
            out[p + "FakeAccWithinRho"], out[p + "FakeAccWithinP"] = f"{fa['within_rho']:.2f}", _p(fa["within_p"])
            out[p + "FakeAccBetweenRho"], out[p + "FakeAccBetweenP"] = f"{fa['between_rho']:.2f}", _p(fa["between_p"])
        g_all = ann_all[ann_all.algorithm == algo]
        for s_, word in (("random", "Random"),):                         # the paper reports random rows only
            h = g_all[g_all.sampler == s_].assign(b=lambda d: (d.speed == "faster") & (d.verdict == "equivalent"))
            out[p + "FakeHit" + word] = f"{h.fake_hit_ratio.median():.2f}"
            out[p + "AnyBoth" + word] = str(int(h.groupby("dataset").b.any().sum()))
            out[p + "Faster" + word] = f"{100 * (h.speed == 'faster').mean():.0f}"
            out[p + "Comparable" + word] = f"{100 * (h.verdict == 'equivalent').mean():.0f}"
            out[p + "Both" + word] = f"{100 * h.b.mean():.0f}"
            out[p + "MedianST" + word] = f"{h.S_T.median():.2f}"
            out[p + "Worse" + word] = str(int((h.verdict == "worse").sum()))
        rel = relative_rope_detail(g, props)                 # Table S14 (supervisor review): relative ROPE
        for r, word in zip(RELATIVE_ROPES, ("Five", "Ten", "Twenty")):
            out[p + "ComparableRel" + word] = f"{100 * (rel[f'verdict_rel{round(100 * r)}'] == 'equivalent').mean():.0f}"
        for word, (rope, thr) in SENSITIVITY.items():      # practical equivalence under another ROPE / threshold
            eq = equivalent_under(g, rope, thr, props)
            out[p + "Comparable" + word] = f"{100 * eq.mean():.0f}" if eq.notna().any() else "--"
            out[p + "Both" + word] = f"{100 * ((g.speed == 'faster') & eq.fillna(False)).mean():.0f}" \
                if eq.notna().any() else "--"
        oc = outcome_counts(g).join(props.set_index("dataset")["rows"], how="left").sort_values("rows", kind="stable")
        for grp in ("all", "some", "none"):
            sel = oc[oc.group == grp]
            out[p + "Group" + grp.capitalize()] = str(len(sel))
            out[p + "Group" + grp.capitalize() + "Names"] = _names(list(sel.index))
        out[p + "NoSpeedupNames"] = _names(list(oc[oc.faster == 0].index))
        # §5: the sentence explains the "none" group, so only its datasets that were not always faster
        slow = oc[(oc.group == "none") & (oc.faster < oc.configs)]
        out[p + "NotAlwaysFasterNames"] = _names(list(slow.index))
        out[p + "NotAlwaysFaster"] = str(len(slow))
        out[p + "FewSpeedupNames"] = _names(list(oc[(oc.faster > 0) & (oc.faster < oc.configs)].index))
        out[p + "AllFastNoVerdictNames"] = _names(list(oc[(oc.faster == oc.configs) & (oc.comparable == 0)
                                                          & (oc.worse == 0)].index))
        out[p + "WorseDatasetsNames"] = _names(list(oc[oc.worse > 0].index))
        out[p + "BothTotal"] = str(int(oc.both.sum()))
        anyb = oc.both > 0
        out[p + "AnyBoth"], out[p + "AnyBothShare"] = str(int(anyb.sum())), f"{100 * anyb.mean():.0f}"
    out.update(size_group_numbers(ann))
    out.update(gegp_numbers(ann))
    body = header("results/store/metrics.csv, datasets.csv") + "".join(
        f"\\newcommand{{\\res{k}}}{{{v}}}\n" for k, v in out.items())
    return [write(out_tables, "numbers.tex", body)]


# ---------------------------------------------------------------- figure: S_T against table size

SERIES_COLOUR = "#2a78d6"            # one series per panel (dataviz reference palette, categorical slot 1)
SECOND_COLOUR = "#eb6834"            # second series (slot 2; with a dashed line, so not colour alone)
THIRD_COLOUR = "#1baf7a"             # slot 3 (aqua); validated with slots 1-2 (dataviz validator, light: all pass)
INK, INK_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def size_figure(corpus, metrics, props, out_tables, min_rows=0, **_) -> list[Path]:
    """fig_st_rows.pdf and fig_dacc_rows.pdf: per dataset (main sampler), the mean S_T and the mean dAcc (pp, with
    the +-1 pp ROPE) over the fractions against the number of rows; GE left, GP right on a shared y axis, Spearman
    rho in each panel title; no fitted line (the claim is monotonic, not linear)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    keep = main_text_datasets(corpus, props, min_rows)
    rows_of = props.set_index("dataset")["rows"]
    m = metrics[metrics.dataset.isin(keep)].assign(dAcc_pp=lambda d: 100 * d.dAcc)
    grp = m.groupby(["algorithm", "dataset"])
    per = (grp[["S_T", "dAcc_pp", "S_wall"]].mean()
           .join(grp[["S_T", "dAcc_pp"]].min().add_suffix("_lo")).join(grp[["S_T", "dAcc_pp"]].max().add_suffix("_hi"))
           .reset_index().assign(rows=lambda d: d.dataset.map(rows_of)).dropna(subset=["rows"]))
    per["x"] = _spread_ties(per)
    fs.use()
    out_tables.mkdir(parents=True, exist_ok=True)
    paths = []
    for col, ref, label, stem in (("S_T", 1.0, "speedup (mean over fractions)", "fig_st_rows"),
                                  ("dAcc_pp", 0.0, "$\\Delta$Acc, pp (mean over fractions)", "fig_dacc_rows")):
        fs.use()
        fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5), sharey=True)
        for ax, algo in zip(axes, ("GE", "GP")):
            d = per[per.algorithm == algo]
            ax.axhline(ref, color=INK_SECONDARY, linestyle=(0, (4, 3)), linewidth=0.8, zorder=1)
            if col == "dAcc_pp":                  # the region of practical equivalence, +-1 pp
                for r in (-ROPE_PP, ROPE_PP):
                    ax.axhline(r, color=INK_SECONDARY, linestyle=(0, (1, 2)), linewidth=0.8, zorder=1)
            ax.vlines(d.x, d[col + "_lo"], d[col + "_hi"], color=SERIES_COLOUR, alpha=0.45, linewidth=1.0,
                      zorder=2)                     # range over the three fractions
            ax.scatter(d.x, d[col], s=22, color=SERIES_COLOUR, edgecolor="white", linewidth=0.6, zorder=3,
                       label="$S_T$" if col == "S_T" else None)
            if col == "S_T":                          # whole-run speedup beside it, hollow
                ax.scatter(d.x, d.S_wall, s=22, facecolor="none", edgecolor=SERIES_COLOUR, linewidth=0.9, zorder=3,
                           label="$S_\\mathrm{wall}$")
            ax.set_xscale("log")
            ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_xlabel("rows (log scale)")
            if len(d) > 2:
                rho, pval = spearmanr(d.rows, d[col])
                ptxt = f"p = {pval:.3f}" if pval >= 0.001 else "p < 0.001"
                fs.title(ax, algo, f"Spearman $\\rho$ = {rho:.2f}, {ptxt}, n = {len(d)}")
            else:
                fs.title(ax, algo, f"n = {len(d)}")
        axes[0].set_ylabel(label)
        if len(per) and col == "S_T":
            lo = min(0.8, per[["S_T_lo", "S_wall"]].min().min() - 0.1)
            axes[0].set_ylim(lo, per["S_T_hi"].max() + 0.15)
            axes[0].legend(loc="upper left", handletextpad=0.3, borderaxespad=0.2)
        elif len(per):
            top = max(ROPE_PP, per[["dAcc_pp_lo", "dAcc_pp_hi"]].abs().max().max()) + 0.3
            axes[0].set_ylim(-top, top)
        fig.tight_layout()
        path = out_tables / f"{stem}.pdf"
        fs.save(fig, path)
        plt.close(fig)
        paths.append(path)
    paths.append(cache_figure(metrics[metrics.dataset.isin(keep)], rows_of, out_tables,
                              eval_overhead(keep=set(keep))))
    paths.append(fake_acc_figure(metrics[metrics.dataset.isin(keep)], rows_of, out_tables))
    return paths


def _spread_ties(per: pd.DataFrame) -> pd.Series:
    """x positions on the log rows axis: datasets of one algorithm with the same number of rows are moved apart
    by a few per cent so that both points stay visible (e.g. electricity and electricity_cat)."""
    x = per.rows.astype(float).copy()
    for (_a, _r), idx in per.groupby(["algorithm", "rows"]).groups.items():
        if len(idx) > 1:
            for k, i in enumerate(sorted(idx, key=lambda j: per.dataset[j])):
                x[i] *= 1 + 0.06 * (k - (len(idx) - 1) / 2)
    return x


def _per_dataset(metrics: pd.DataFrame, rows_of: pd.Series, cols: list[str]) -> pd.DataFrame:
    """Per (algorithm, dataset): mean, min and max over the fractions of each column, with rows and x position."""
    grp = metrics.groupby(["algorithm", "dataset"])
    per = (grp[cols].mean().join(grp[cols].min().add_suffix("_lo")).join(grp[cols].max().add_suffix("_hi"))
           .reset_index().assign(rows=lambda d: d.dataset.map(rows_of)).dropna(subset=["rows"]))
    per["x"] = _spread_ties(per)
    return per


def _rows_panel(ax, d: pd.DataFrame, col: str, algo: str, xlabel: bool) -> None:
    """One dataset-level panel against rows (log): mean over fractions, range bar, Spearman rho in the title."""
    ax.vlines(d.x, d[col + "_lo"], d[col + "_hi"], color=SERIES_COLOUR, alpha=0.45, linewidth=1.0, zorder=2)
    ax.scatter(d.x, d[col], s=22, color=SERIES_COLOUR, edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xscale("log")
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if xlabel:
        ax.set_xlabel("rows (log scale)")
    rho, pval, n = dataset_rho(d, col, "rows")
    ptxt = f"p = {pval:.3f}" if pval >= 0.001 else "p < 0.001"
    fs.title(ax, algo, f"Spearman $\\rho$ = {rho:.2f}, {ptxt}, n = {n}" if n > 2 else f"n = {n}")


def cache_figure(metrics: pd.DataFrame, rows_of: pd.Series, out_tables: Path,
                 overhead: pd.DataFrame | None = None) -> Path:
    """fig_cache_rows.pdf (where the speedup comes from): per dataset (main sampler), the mean hit rate and the
    measured cost of the fingerprint step per candidate relative to a full evaluation (eval_overhead) against the
    number of rows, and S_T against that cost; GE left, GP right, one shared y axis per row, Spearman rho in each
    panel title."""
    import matplotlib.pyplot as plt
    if overhead is not None and len(overhead):
        metrics = metrics.assign(fraction=metrics.fraction.round(2)).merge(
            overhead, on=["algorithm", "dataset", "fraction"], how="left")
    else:
        metrics = metrics.assign(overhead=np.nan)
    per = _per_dataset(metrics, rows_of, ["hit_rate", "overhead", "S_T"])
    fs.use()
    fig, axes = plt.subplots(3, 2, figsize=(6.3, 6.8), sharey="row")
    for r, (col, label) in enumerate((("hit_rate", "hit rate"),
                                      ("overhead", "fingerprint step per candidate\n/ full evaluation"))):
        for c, algo in enumerate(("GE", "GP")):
            _rows_panel(axes[r, c], per[per.algorithm == algo], col, algo, xlabel=r == 1)
        axes[r, 0].set_ylabel(label)
    for c, algo in enumerate(("GE", "GP")):              # S_T against the cost of the fingerprint step
        ax, d = axes[2, c], per[per.algorithm == algo].dropna(subset=["overhead"])
        ax.axhline(1.0, color=INK_SECONDARY, linestyle=(0, (4, 3)), linewidth=0.8, zorder=1)
        ax.scatter(d.overhead, d.S_T, s=22, color=SERIES_COLOUR, edgecolor="white", linewidth=0.6, zorder=3)
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("fingerprint step per candidate / full evaluation")
        ax.set_xlim(0, max(1.0, float(per.overhead.max()) + 0.05))
        rho, pval, n = dataset_rho(d, "S_T", "overhead")
        ptxt = f"p = {pval:.3f}" if pval >= 0.001 else "p < 0.001"
        fs.title(ax, algo, f"Spearman $\\rho$ = {rho:.2f}, {ptxt}, n = {n}" if n > 2 else f"n = {n}")
    axes[2, 0].set_ylabel("speedup $S_T$")
    axes[0, 0].set_ylim(0, 1)
    if per.overhead_hi.notna().any():
        axes[1, 0].set_ylim(0, max(1.0, float(per.overhead_hi.max()) + 0.05))
    fig.tight_layout()
    path = out_tables / "fig_cache_rows.pdf"
    fs.save(fig, path)
    plt.close(fig)
    return path


def eval_overhead(store: Path = results.STORE, keep=None) -> pd.DataFrame:
    """Per (algorithm, dataset, fraction), main sampler: the measured cost of FEC's fingerprint step per candidate
    (evaluation on the fingerprint rows, key and lookup) relative to the full evaluation of a missed candidate, both
    timed per phase in the same run (c_meas; results/store/phase_timing.csv, runs that reproduce their
    stored twin only), mean over the seeds. Empty without phase-timed runs."""
    path = _f(store, "phase_timing.csv")
    if not path.is_file():
        return pd.DataFrame(columns=["algorithm", "dataset", "fraction", "overhead"])
    t = pd.read_csv(path, usecols=["algorithm", "dataset", "sampler", "fraction", "c_meas", "twin_ok"])
    if not t.twin_ok.all():
        raise TableError("phase_timing.csv holds a run that does not reproduce its stored twin")
    t = t[t.sampler == MAIN_SAMPLER]
    if keep is not None:
        t = t[t.dataset.isin(keep)]
    return t.assign(fraction=t.fraction.round(2)).groupby(["algorithm", "dataset", "fraction"]).c_meas.mean() \
        .rename("overhead").reset_index()


def fake_acc_stats(metrics: pd.DataFrame, algo: str) -> dict:
    """Fake hits against the size of the accuracy difference |dAcc| (pp), one algorithm (main sampler). Within a
    dataset: in how many datasets the fraction with the most fake hits has a larger |dAcc| than the fraction with
    the fewest (Wilcoxon signed-rank over datasets), and the Spearman correlation after removing each dataset's
    mean. Between datasets: Spearman correlation of the dataset means."""
    from scipy.stats import spearmanr, wilcoxon
    g = metrics[metrics.algorithm == algo].assign(absd=lambda d: 100 * d.dAcc.abs()).dropna(
        subset=["fake_hit_ratio", "absd"])
    if g.dataset.nunique() < 5:
        return {}
    hi = g.loc[g.groupby("dataset").fake_hit_ratio.idxmax()].set_index("dataset").absd
    lo = g.loc[g.groupby("dataset").fake_hit_ratio.idxmin()].set_index("dataset").absd
    centre = lambda c: g[c] - g.groupby("dataset")[c].transform("mean")
    w_rho, w_p = spearmanr(centre("fake_hit_ratio"), centre("absd"))
    dm = g.groupby("dataset")[["fake_hit_ratio", "absd"]].mean()
    b_rho, b_p = spearmanr(dm.fake_hit_ratio, dm.absd)
    return {"larger": int((hi > lo).sum()), "n": len(hi), "wilcoxon_p": float(wilcoxon(hi, lo).pvalue),
            "within_rho": float(w_rho), "within_p": float(w_p), "between_rho": float(b_rho), "between_p": float(b_p)}


def fake_acc_by_fraction(metrics: pd.DataFrame) -> dict:
    """Per (algorithm, fraction): Spearman rho and p over datasets between the fake-hit ratio and |dAcc|."""
    from scipy.stats import spearmanr
    out = {}
    for (algo, fr), g in metrics.assign(fr=metrics.fraction.round(2)).groupby(["algorithm", "fr"]):
        g = g.dropna(subset=["fake_hit_ratio", "dAcc"])
        if len(g) > 2:
            rho, pval = spearmanr(g.fake_hit_ratio, g.dAcc.abs())
            out[(algo, fr)] = (float(rho), float(pval), len(g))
    return out


def fake_acc_by_fraction_figure(metrics: pd.DataFrame, out_dir: Path) -> Path:
    """FigureS2_fake_acc_by_fraction.pdf: |dAcc| (pp) against the fake-hit ratio, one point per dataset; rows GE and
    GP, columns 1%, 5% and 10% (main sampler); Spearman rho over datasets in each panel title."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs.use()
    m = metrics.assign(absd=lambda d: 100 * d.dAcc.abs(), fr=lambda d: d.fraction.round(2))
    stats = fake_acc_by_fraction(metrics)
    fs.use()
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.4), sharey=True, sharex="row")
    for r, algo in enumerate(("GE", "GP")):
        for c, fr in enumerate((0.01, 0.05, 0.10)):
            ax, g = axes[r, c], m[(m.algorithm == algo) & (m.fr == fr)]
            ax.scatter(g.fake_hit_ratio, g.absd, s=20, color=SERIES_COLOUR, edgecolor="white", linewidth=0.6, zorder=3)
            ax.grid(color=GRID, linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            rho, pval, n = stats.get((algo, fr), (float("nan"), float("nan"), 0))
            ptxt = f"p = {pval:.3f}" if pval >= 0.001 else "p < 0.001"
            fs.title(ax, f"{algo}, {FRACTION_WORD[fr].replace(chr(92), '')}", f"$\\rho$ = {rho:.2f}, {ptxt}")
            if r == 1:
                ax.set_xlabel("fake-hit ratio")
        axes[r, 0].set_ylabel("|$\\Delta$Acc|, pp")
    axes[0, 0].set_ylim(bottom=0)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "FigureS2_fake_acc_by_fraction.pdf"
    fs.save(fig, path)
    plt.close(fig)
    return path


FLOOR_PP = 0.01                      # smallest |dAcc| drawn on the log axes of fig_fake_acc (pp)


def fake_acc_figure(metrics: pd.DataFrame, rows_of: pd.Series, out_tables: Path) -> Path:
    """fig_fake_acc.pdf: per dataset, the mean fake-hit ratio over the fractions against the number of rows (main
    sampler); GE left, GP right, Spearman rho in each panel title."""
    import matplotlib.pyplot as plt
    per = _per_dataset(metrics, rows_of, ["fake_hit_ratio"])
    fs.use()
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5), sharey=True)
    for ax, algo in zip(axes, ("GE", "GP")):
        _rows_panel(ax, per[per.algorithm == algo], "fake_hit_ratio", algo, xlabel=True)
    axes[0].set_ylabel("fake-hit ratio")
    axes[0].set_ylim(0, 1)
    fig.tight_layout()
    path = out_tables / "fig_fake_acc.pdf"
    fs.save(fig, path)
    plt.close(fig)
    return path


def fake_acc_paired_figure(metrics: pd.DataFrame, out_dir: Path) -> Path:
    """FigureS3_fake_acc_paired.pdf: per dataset, |dAcc| (pp) at the fraction with the fewest fake hits (x) against the
    fraction with the most (y), log axes, with the diagonal; filled above it (more fake hits, larger |dAcc|), hollow
    below. GE left, GP right; count above and Wilcoxon p in each panel title."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs.use()
    m = metrics.assign(absd=lambda d: 100 * d.dAcc.abs())
    fs.use()
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), sharey=True)
    lim = 0.0
    for ax, algo in zip(axes, ("GE", "GP")):
        g = m[m.algorithm == algo].dropna(subset=["fake_hit_ratio", "absd"])
        hi = g.loc[g.groupby("dataset").fake_hit_ratio.idxmax()].set_index("dataset").absd
        lo = g.loc[g.groupby("dataset").fake_hit_ratio.idxmin()].set_index("dataset").absd
        lo = lo.reindex(hi.index)
        hi, lo = hi.clip(lower=FLOOR_PP), lo.clip(lower=FLOOR_PP)  # log axes: an exact zero is drawn at the floor
        lim = max(lim, float(hi.max()), float(lo.max()))
        above = hi > lo
        ax.scatter(lo[above], hi[above], s=22, color=SERIES_COLOUR, edgecolor="white", linewidth=0.6, zorder=3,
                   label="larger with more fake hits")
        ax.scatter(lo[~above], hi[~above], s=22, facecolor="none", edgecolor=SERIES_COLOUR, linewidth=0.9,
                   zorder=3, label="smaller with more fake hits")
        ax.grid(color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("|$\\Delta$Acc|, pp, at 10% (fewest fake hits)")
        st = fake_acc_stats(metrics, algo)
        if st:
            ptxt = f"p = {st['wilcoxon_p']:.3f}" if st["wilcoxon_p"] >= 0.001 else "p < 0.001"
            fs.title(ax, algo, f"{st['larger']} of {st['n']} above the line, Wilcoxon {ptxt}")
    for ax in axes:
        top = lim * 1.5
        ax.plot([FLOOR_PP, top], [FLOOR_PP, top], color=INK_SECONDARY, linestyle=(0, (4, 3)), linewidth=0.8, zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(FLOOR_PP * 0.8, top)
        ax.set_ylim(FLOOR_PP * 0.8, top)
    axes[0].set_ylabel("|$\\Delta$Acc|, pp, at 1% (most)")   # filled/hollow explained in the caption
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "FigureS3_fake_acc_paired.pdf"
    fs.save(fig, path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- claims in the text that depend on the data

def claims(corpus, metrics, props, out_tables, min_rows=0, all_metrics=None, **_) -> list[Path]:
    """claims_check.txt: every sentence of §5 whose wording (not just its numbers) depends on the results, tested on
    the current store. A claim that no longer holds is printed as a warning naming the sentence to rewrite; the
    numbers themselves are macros and update on their own."""
    from scipy.stats import spearmanr
    keep = main_text_datasets(corpus, props, min_rows)
    rows_of = props.set_index("dataset")["rows"]
    ann = annotated(metrics[metrics.dataset.isin(keep)]).assign(rows=lambda d: d.dataset.map(rows_of))
    every = all_metrics if all_metrics is not None else metrics
    ann_all = annotated(every[every.dataset.isin(keep)])
    out = []

    def claim(key: str, sentence: str, holds: bool) -> None:
        out.append((key, sentence, bool(holds)))

    for algo in ("GE", "GP"):
        g = ann[ann.algorithm == algo]
        if not len(g):
            continue
        med = [g[g.fraction.round(2) == f].S_T.median() for f in (0.01, 0.05, 0.10)]
        eq = [(g[g.fraction.round(2) == f].verdict == "equivalent").mean() for f in (0.01, 0.05, 0.10)]
        claim(f"{algo}/fraction-speed", f"§5 {algo}: 'Smaller fingerprints gave larger speedups'",
              med[0] >= med[1] >= med[2])
        claim(f"{algo}/fraction-equivalence", f"§5 {algo}: 'Larger fingerprints, in turn, led to more verdicts of "
              "practical equivalence'", eq[0] <= eq[1] <= eq[2])
        per = g.groupby("dataset").agg(S_T=("S_T", "mean"), rows=("rows", "first"))
        rho, pval = spearmanr(per.S_T, per.rows) if len(per) > 2 else (float("nan"), float("nan"))
        claim(f"{algo}/size", f"§5 {algo}: 'Speed was most strongly associated with the size of the table' (Spearman > 0, p <= 0.05)",
              rho > 0 and pval <= ALPHA)
        per = g.groupby("dataset").agg(dAcc=("dAcc", "mean"), rows=("rows", "first"))
        rho, pval = spearmanr(per.dAcc, per.rows) if len(per) > 2 else (float("nan"), float("nan"))
        claim(f"{algo}/dacc-size", f"§5 {algo}: 'the mean dAcc of a dataset showed no significant correlation with its "
              "number of rows' (Spearman p > 0.05)", pval > ALPHA)
        gf = g.merge(props.reindex(columns=["dataset", "features"]), on="dataset", how="left")
        rho, pval, _n = dataset_rho(gf, "fake_hit_ratio", "rows")
        claim(f"{algo}/fake-size", f"§5 {algo}: 'The fake-hit ratio fell with size' (Spearman < 0, p <= 0.05)",
              rho < 0 and pval <= ALPHA)
        fa = fake_acc_stats(g, algo)
        if fa:
            fmax = g.loc[g.groupby("dataset").fake_hit_ratio.idxmax()].fraction.round(2)
            fmin = g.loc[g.groupby("dataset").fake_hit_ratio.idxmin()].fraction.round(2)
            claim(f"{algo}/fake-1pct-most", f"§5.3 {algo}: 'the 1% fingerprint had the most fake hits and the 10% "
                  "fingerprint the fewest on every table'", (fmax == 0.01).all() and (fmin == 0.10).all())
            claim(f"{algo}/fake-acc-within", f"§5.3 {algo}: 'in most datasets the fraction with the most fake hits had "
                  "the larger accuracy difference' (more than half, Wilcoxon p <= 0.05)",
                  fa["larger"] > fa["n"] / 2 and fa["wilcoxon_p"] <= ALPHA)
            claim(f"{algo}/fake-acc-between", f"§5.3 {algo}: 'between datasets, the level of fake hits did not predict "
                  "the size of the accuracy difference' (p > 0.05)", fa["between_p"] > ALPHA)
        ov = eval_overhead(keep=set(g.dataset))
        if len(ov):
            go = gf.assign(fraction=gf.fraction.round(2)).merge(ov[ov.algorithm == algo],
                                                                on=["algorithm", "dataset", "fraction"])
            r1, p1, _n = dataset_rho(go, "overhead", "rows")
            r2, p2, _n = dataset_rho(go, "S_T", "overhead")
            claim(f"{algo}/overhead-size", f"§5 {algo}: 'relative to a full evaluation, the cost of the fingerprint step "
                  "per candidate fell with size' (Spearman < 0, p <= 0.05)", r1 < 0 and p1 <= ALPHA)
            claim(f"{algo}/overhead-st", f"§5 {algo}: 'the lower this relative cost, the larger S_T' (Spearman < 0, "
                  "p <= 0.05)", r2 < 0 and p2 <= ALPHA)
            strength = "strong" if r2 <= -0.7 else "moderate" if r2 <= -0.4 else "weak"
            claim(f"{algo}/overhead-st-strength", f"§5.3: the association of S_T with the relative cost is "
                  f"{'strong in GE' if algo == 'GE' else 'moderate in GP'} (rho = {r2:.2f})",
                  strength == ("strong" if algo == "GE" else "moderate"))
            if algo == "GP":
                per_c = go.groupby("dataset").agg(c=("overhead", "mean"), rows=("rows", "first")).sort_values("rows", kind="stable")
                claim("GP/step-as-full-smallest", "§5.3: 'on the two smallest tables the fingerprint step of GP cost "
                      "about as much as a full evaluation' (mean relative cost >= 0.9)", bool((per_c.c.iloc[:2] >= 0.9).all()))
            if algo == "GE":
                lowest = go.groupby("dataset").overhead.mean().idxmin()
                claim("GE/step-cheapest-accel", "§6.5: 'the fingerprint step was cheaper on accelerometer than on any "
                      "other table' (GE)", lowest == "accelerometer")
        claim(f"{algo}/features", f"§5 {algo}: 'the number of features was not significantly associated with any of "
              "these measures' (all p > 0.05)", all(dataset_rho(gf, y, "features")[1] > ALPHA for y in CACHE_MEASURES))
        hit_rho, hit_p, _n = dataset_rho(gf, "hit_rate", "rows")
        if algo == "GE":
            claim("GE/hit-size-flat", "§5.1: 'the mean hit rate of a dataset showed no significant correlation "
                  "with its number of rows'", hit_p > ALPHA)
        else:
            claim("GP/hit-size-falls", "§5.2: 'the hit rate fell with size'", hit_rho < 0 and hit_p <= ALPHA)
            th = gf.assign(true_hit=gf.hit_rate * (1 - gf.fake_hit_ratio))
            rho, pval, _n = dataset_rho(th, "true_hit", "rows")
            claim("GP/true-hit-rises", "§5.2: 'The hits lost on larger tables were mostly fake ones: the share of "
                  "evaluations answered with the correct fitness ... rose with size' (Spearman > 0, p <= 0.05)",
                  rho > 0 and pval <= ALPHA)
            per_st = g.groupby("dataset").agg(S_T=("S_T", "mean"), rows=("rows", "first")).sort_values("rows", kind="stable")
            claim("GP/largest-level", "§5.2: 'consistent with the levelling of S_T on the two largest tables' (both "
                  "below the largest mean S_T of the smaller tables)",
                  len(per_st) > 2 and (per_st.S_T.iloc[-2:] < per_st.S_T.iloc[:-2].max()).all())
        claim(f"{algo}/threshold", f"§5 {algo}: 'Every dataset with at least ... rows ... was significantly faster at "
              "all three fractions' (some table from which all are faster exists)", all_faster_threshold(g) is not None)
        oc = outcome_counts(g).join(rows_of, how="left")
        small_slow = oc[oc.faster < oc.configs]
        if algo == "GE":                    # GP's sentence names its tables without calling them small (Holm)
            claim(f"{algo}/smallest-not-faster", f"§5 {algo}: 'For [names], which have relatively few training "
                  "instances, ...' (all of them in the smaller half of the tables)",
                  (small_slow.rows <= oc.rows.median()).all())
        if algo == "GP":
            miss = holm_above_raw_threshold(g).query("speed != 'faster'")
            claim("GP/holm-mid-large-missed", "§5 GP: 'Unlike GE, some configurations on mid-sized and large datasets, "
                  "[configs], were not significantly faster, although FEC was faster on average in each' (some exist, "
                  "none among the three smallest, all S_T > 1)",
                  len(miss) > 0 and (miss.S_T > 1).all()
                  and not set(miss.dataset) & set(g.groupby("dataset").rows.first().nsmallest(N_ENDS).index))
    ge_, gp_ = ann[ann.algorithm == "GE"], ann[ann.algorithm == "GP"]   # §5.2.1: how GP differs from GE
    if len(ge_) and len(gp_):
        med = lambda g, f: g[g.fraction.round(2) == f].S_T.median()
        claim("gegp/median-st", "§5.2.1: 'its typical speedup was slightly higher' (GP median S_T above GE's at "
              "every fraction)", all(med(gp_, f) > med(ge_, f) for f in (0.01, 0.05, 0.10)))
        claim("gegp/max-st", "§5.2.1: 'but its largest was lower' (GP max S_T below GE's)", gp_.S_T.max() < ge_.S_T.max())
        rho_g = lambda g: spearmanr(*g.groupby("dataset").agg(S=("S_T", "mean"), r=("rows", "first")).T.values)[0]
        claim("gegp/size-weaker", "§5.2.1: 'the size of the table explained less of the speedup' (GP rho below GE's)",
              rho_g(gp_) < rho_g(ge_))
        eqf = lambda g, f=None: ((g if f is None else g[g.fraction.round(2) == f]).verdict == "equivalent").mean()
        claim("gegp/comparable-less", "§5.2.1: 'GP was comparable in accuracy less often' (overall and at 1%)",
              eqf(gp_) < eqf(ge_) and eqf(gp_, 0.01) < eqf(ge_, 0.01))
        claim("gegp/fake-more", "§5.2.1: 'its larger share of fake hits' (GP median fake-hit ratio above GE's)",
              gp_.fake_hit_ratio.median() > ge_.fake_hit_ratio.median())
    for a_ in ("GE", "GP"):                                    # §5.4 opening (Figure 8)
        g_ = ann[ann.algorithm == a_].assign(fr=lambda d: d.fraction.round(2))
        if len(g_):
            md = g_.groupby("fr").dAcc.median()
            fk = g_.groupby("fr").fake_hit_ratio.median()
            claim(f"{a_}/fig8-fake-acc", f"§5.4 {a_}: 'from 10% to 1% the median fake-hit ratio rose and the median dAcc "
                  "fell (to below zero)'", fk[0.01] > fk[0.1] and md[0.01] < 0 and md[0.01] < md[0.1])
    for a_ in ("GE", "GP"):                                    # §6.2: what counts as comparable depends on the ROPE
        g_ = ann[ann.algorithm == a_]
        if len(g_):
            eq2 = equivalent_under(g_, 0.02, THRESHOLD, props)
            claim(f"{a_}/rope-two-three-quarters", f"§6.2 {a_}: 'with up to two points about three quarters of the "
                  "configurations count as comparable, against under half with one point' (70-80% vs < 50%)",
                  0.70 <= eq2.mean() <= 0.80 and (g_.verdict == "equivalent").mean() < 0.5)
    hk_ = hit_kinds(keep)
    pooled = lambda x, k: (x[k] * x.hits).sum() / x.hits.sum()
    for a_ in ("GE", "GP"):
        x = hk_[hk_.algorithm == a_]
        if x.empty:
            continue
        fr_ = {f: x[x.fraction == f] for f in (0.01, 0.05, 0.1)}
        claim(f"{a_}/hits-mostly-different", f"§5 {a_}: 'most hits came from a different program than the stored one' "
              "(same-program share < 50% at every fraction)", all(pooled(v, "structural") < 0.5 for v in fr_.values()))
        claim(f"{a_}/behavioural-rises-nonequiv-falls", f"§5 {a_}: 'from 1% to 10% the share of fitness-"
              "equivalent hits rose and that of non-equivalent hits fell'",
              pooled(fr_[0.01], "behavioural") < pooled(fr_[0.1], "behavioural")
              and pooled(fr_[0.01], "nonequiv") > pooled(fr_[0.1], "nonequiv"))
        if a_ == "GE":
            claim("GE/behavioural-beats-nonequiv", "§5.1: 'fitness-equivalent hits outnumbered non-equivalent "
                  "ones at every fraction'", all(pooled(v, "behavioural") > pooled(v, "nonequiv") for v in fr_.values()))
        else:
            claim("GP/nonequiv-beats-behavioural-1pct", "§5.2: 'at 1% non-equivalent hits outnumbered fitness-"
                  "equivalent ones, at 5% and 10% the reverse'", pooled(fr_[0.01], "nonequiv") > pooled(fr_[0.01], "behavioural")
                  and all(pooled(fr_[f], "behavioural") > pooled(fr_[f], "nonequiv") for f in (0.05, 0.1)))
    sv_ = seconds_saved()                                       # §5.1/§5.2 (review): largest table saves most time
    if len(sv_):
        big = max(keep, key=lambda d: rows_of[d])
        for a_ in ("GE", "GP"):
            x = sv_[(sv_.algorithm == a_) & sv_.dataset.isin(keep) & (sv_.fraction.round(2) == 0.01)]
            if len(x):
                small = min(keep, key=lambda d: rows_of[d])
                xs = x[x.dataset == small]
                if a_ == "GP" and len(xs):
                    claim("GP/small-slower-1pct", f"§5.2: 'FEC did not speed up GP on the smallest table ({small}): each "
                          "run lost time at 1%'", xs.eval_saved_s.iloc[0] < 0)
                claim(f"{a_}/saved-largest", f"§5 {a_}: at 1% the largest table ({big}) saved the most time per run, "
                      "although not with the largest ratio", x.loc[x.eval_saved_s.idxmax()].dataset == big)
    pf = fake_acc_by_fraction(ann)
    if len(pf) == 6:
        claim("fake-acc/per-fraction", "§5.3: 'within each fraction, the fake-hit ratio of a dataset was not "
              "significantly associated with its accuracy difference, except for GP at 1%' (Figure S2)",
              all(p_ > ALPHA for (a_, f_), (_r, p_, _n) in pf.items() if (a_, f_) != ("GP", 0.01))
              and pf[("GP", 0.01)][1] <= ALPHA)
    fa_ge, fa_gp = fake_acc_stats(ann, "GE"), fake_acc_stats(ann, "GP")
    if fa_ge and fa_gp:
        claim("fake-acc/gp-stronger", "§5.3: 'clearly for GP, more weakly for GE' (GP within-dataset rho >= 0.5 and "
              "p <= 0.05; GE rho smaller)", fa_gp["within_rho"] >= 0.5 and fa_gp["within_p"] <= ALPHA
              and fa_ge["within_rho"] < fa_gp["within_rho"])
    main_ann = ann[ann.sampler == MAIN_SAMPLER]                 # The undecided share
    claim("verdict/undecided-rest", "§5.1: 'the remaining <n> GE and <n> GP configurations of 51 were undecided' "
          "(no configuration worse or better, so equivalent + undecided = all) and the reply's 'more than half'",
          all(set(main_ann[main_ann.algorithm == a_].verdict) <= {"equivalent", "undecided"}
              and (main_ann[main_ann.algorithm == a_].verdict == "undecided").mean() > 0.5 for a_ in ("GE", "GP")))
    out += size_group_claims(ann)
    out += case_claims(metrics[metrics.dataset.isin(keep)], props)
    out += mc_claims(metrics[metrics.dataset.isin(keep)], props)
    out += pooling_claims()
    out += size_pair_claims(annotated(metrics[metrics.dataset.isin(keep)]), props)
    out += size_fig_claims()
    out += correlation_claims(metrics[metrics.dataset.isin(keep)], props)
    out += size_claims(metrics[metrics.dataset.isin(keep)], props)
    out += guide_claims(metrics[metrics.dataset.isin(keep)], props)
    out += cost_claims(metrics[metrics.dataset.isin(keep)], props)
    out += memory_claims(keep, props)
    out += selection_claims(keep)
    out += trajectory_claims(keep)
    out += exact_claims(keep)
    out += gegp_equivalent_claims(annotated(metrics[metrics.dataset.isin(keep)]), props)
    tt = gegp_train_test()
    if tt:
        claim("gegp/no-overfit", "§6.6.1: 'neither algorithm overfitted on these two tables' (train-test gap of the best "
              "program in the last baseline generation below one percentage point)", tt["gap_max"] < 1.0)
    out += work_claims(annotated(metrics[metrics.dataset.isin(keep)]).merge(props[['dataset', 'rows']], on='dataset', how='left'), props)
    out += conclusion_claims(ann)
    if "classes" in props:
        out += repeats_claims(set(keep), props)
        out += hit_kind_claims(set(keep), props)
        out += severity_claims(set(keep), props, ann)
    from scipy.stats import spearmanr
    st_rows = [spearmanr(*zip(*ann[ann.algorithm == al].groupby("dataset").agg(S=("S_T", "mean"), r=("rows", "first"))
                              .values)) for al in ("GE", "GP")]
    ph = pd.read_csv(_f(results.STORE, "phase_timing.csv")) if (_f(results.STORE, "phase_timing.csv")).is_file() else None
    if ph is not None:
        ph = ph[ph.dataset.isin(keep)].assign(share=lambda d: (d.t_fp_eval + d.t_fp_key + d.t_lookup) / d.eval_time_sec)
        sh = [spearmanr(*zip(*[(rows_of[d], v) for d, v in x.groupby("dataset").share.mean().items()]))
              for _, x in ph.groupby("algorithm")]
        out.append(("abstract/fp-share-size", "Abstract: 'fingerprint computation accounted for a smaller proportion of "
                    "total evaluation time' on larger datasets (Spearman of the fingerprint share of FEC evaluation time "
                    "with rows < 0, p <= 0.05, both algorithms)", all(r.statistic < 0 and r.pvalue <= ALPHA for r in sh)))
    out.append(("abstract/speedup-size", "Abstract: 'speedup grew with dataset size' (Spearman of mean S_T with rows > 0, "
                "p <= 0.05, both algorithms)", all(r.statistic > 0 and r.pvalue <= ALPHA for r in st_rows)))
    rel = relative_rope_detail(ann, props)                     # §5 / Table S14 (supervisor review)
    tag = f"verdict_rel{round(100 * RELATIVE_MAIN)}"
    claim("rope/relative-more", "§5: '+-1 pp is conservative: with a ROPE relative to the baseline's error more "
          "configurations are equivalent' (GE and GP each: >= as many equivalent at 10 % relative as at +-1 pp, "
          "and no configuration moves from equivalent to undecided at 10 %)",
          all((x[tag] == "equivalent").sum() >= (x.verdict_rope1pp == "equivalent").sum()
              for _, x in rel.groupby("algorithm"))
          and not ((rel.verdict_rope1pp == "equivalent") & (rel[tag] != "equivalent")).any())
    claim("rope/relative-all", "§5: the relative-ROPE ranges (5 % to 20 % of the error) are reported as 'from x% to y%' "
          "next to +-1 pp; each relative ROPE gives more equivalent configurations than +-1 pp (GE and GP each), so "
          "the 5 % and 20 % values are the ends of the range",
          all((x[f"verdict_rel{round(100 * r)}"] == "equivalent").sum() > (x.verdict_rope1pp == "equivalent").sum()
              for r in RELATIVE_ROPES for _, x in rel.groupby("algorithm")))
    sp = size_partials(ann, props)                              # §5: size against the cache factors
    if {"GE", "GP"} <= set(sp):
        claim("size/partial", "§5: 'the association with size persisted when the hit rate, the share of work avoided, "
              "the hit/miss program size or the number of classes was held constant' (all partial rho > 0, p <= 0.05)",
              all(r > 0 and p_ <= ALPHA for a_ in ("GE", "GP") for r, p_ in sp[a_]["partial"]))
        claim("size/work-ge", "§5: 'for GE, the share of work avoided added to it' (partial rho > 0, p <= 0.05; GP: p > 0.05)",
              sp["GE"]["work_given_rows"][0] > 0 and sp["GE"]["work_given_rows"][1] <= ALPHA
              and sp["GP"]["work_given_rows"][1] > ALPHA)
    w = with_wall(ann)                                          # §5 intro: wall-time gain smaller, most of all for GE
    drop = {a_: w[w.algorithm == a_].S_T.median() - w[w.algorithm == a_].S_wall.median() for a_ in ("GE", "GP")
            if (w.algorithm == a_).any()}
    if len(drop) == 2:
        claim("wall/smaller", "§5 intro: 'the gain in whole-run wall time is smaller than in evaluation time, most of "
              "all for GE' (median S_wall below median S_T for both; larger gap for GE)",
              drop["GE"] > 0 and drop["GP"] >= 0 and drop["GE"] > drop["GP"])
    out += fraction_tradeoff_claims(ann)
    out += pair_claims(metrics[metrics.dataset.isin(keep)])
    out += gegp_claims(ann)
    broken = [c for c in out if not c[2]]
    for key, sentence, _ in broken:
        print(f"CLAIM NO LONGER HOLDS [{key}]: {sentence} -> rewrite this sentence")
    body = "".join(f"{'ok  ' if ok else 'FAIL'} {key}: {sentence}\n" for key, sentence, ok in out)
    return [write(out_tables, "claims_check.txt", body)]


# ---------------------------------------------------------------- supplementary

SIZE_LIMIT = 5 * 1024 * 1024          # bytes; larger supplementary files are git-ignored
INDEX_HEAD = "| id | file | title | description | source | size | committed |"
GENERATED = {
    "FileS1_data_dictionary.md": ("File S1", "Data dictionary of the supplementary tables",
                                  "Every column of every supplementary table (CSV) with its meaning and unit."),
    "TableS13_fake_severity.csv": ("Table S13", "How large the fitness errors of fake hits were",
                                   "Per algorithm, dataset and fingerprint fraction (random rows, diagnostic runs, 30 "
                                   "seeds; generations with at least one fake hit): number of fake hits; mean absolute "
                                   "and mean signed difference between the cached and the full-training-set error "
                                   "rate per fake hit (pp; negative = the cache showed a lower error); median, 90th "
                                   "percentile and maximum over the generations of the largest difference in a "
                                   "generation (pp); share of generations whose largest difference exceeded 1 pp and "
                                   "5 pp. From per-generation records (fec.py keeps means and the maximum per "
                                   "generation, not each fake hit)."),
    "TableS12_hit_kinds.csv": ("Table S12", "Cache hits by equivalence group per dataset",
                               "Per algorithm, dataset and fingerprint fraction (random rows, diagnostic runs), mean and sd "
                               "over the 30 seeds of the shares of all hits that were (a) structurally equivalent - "
                               "the same program as the one stored, (b) fitness-equivalent - a different program "
                               "with the same fitness on the complete training set - and (c) not equivalent - a different "
                               "program with a different training fitness; the mean number of hits per run; and, where the "
                               "instrumented runs recorded predictions, the share of (b) that predicted every training "
                               "row as the stored program did."),
    "FigureS3_fake_acc_paired.pdf": ("Figure S3", "Accuracy difference at the fractions with the most and fewest "
                                     "fake hits", "Per dataset, the absolute accuracy difference (Delta Acc, pp) at "
                                     "the 10% fraction (fewest fake hits) against the 1% fraction (most fake hits), "
                                     "log axes, with the diagonal; GE and GP (random fingerprint rows)."),
    "FigureS2_fake_acc_by_fraction.pdf": ("Figure S2", "Fake hits and accuracy at each fingerprint fraction",
                                          "Absolute accuracy difference (Delta Acc, pp) against the fake-hit "
                                          "ratio, one point per dataset, one panel per algorithm and fraction "
                                          "(random fingerprint rows), with the Spearman correlation over datasets."),
    "TableS1_metrics.csv": ("Table S1", "Metrics of every dataset and FEC configuration",
                            "All metric columns for every dataset with results, both algorithms, three fingerprint "
                            "fractions, random fingerprint rows."),
    "TableS2_results_GE.csv": ("Table S2", "Full GE results per configuration",
                               "S_T, Wilcoxon p, Delta Acc, Bayesian probabilities and verdict, hit rate, "
                               "fake-hit ratio and S_wall for every dataset and configuration."),
    "TableS3_results_GP.csv": ("Table S3", "Full GP results per configuration", "As Table S2, for GP."),
    "TableS5_preprocessing.csv": ("Table S5", "Preprocessing per dataset",
                                  "Per dataset, from its recorded recipe: source and size at the source, rows removed "
                                  "for missing values, columns dropped (with reason), one-hot encoding, type changes, "
                                  "label encoding, balancing (rows before and after), duplicate rows kept, size used "
                                  "and the checksum of the table."),
    "TableS6_grammar_properties.csv": ("Table S6", "Grammar properties per dataset",
                                       "Constant range, coverage, degenerate rate, feature-type fairness ratio and "
                                       "minimum initial derivation depth of each dataset's grammar (Section 4.3)."),
    "TableS4_hit_work.csv": ("Table S4", "What the cache hits skipped",
                             "Per algorithm, dataset and fraction, from the diagnostic runs (generations 1-30 "
                             "pooled): hit rate, mean size of a hit program relative to a missed one (operator calls), "
                             "estimated share of the full-evaluation work the hits avoided (every operator call counted as one "
                             "unit per row; fingerprint not counted) and S_T."),
}
for _sid, _rope, _thr in SENSITIVITY_TABLES:
    GENERATED[sensitivity_name(_sid, _rope, _thr)] = (
        _sid, f"Accuracy verdicts with a ROPE of ±{round(100 * _rope)} pp and threshold {_thr:.2f}",
        f"As the accuracy columns of Table 5, with the region of practical equivalence ±{round(100 * _rope)} percentage "
        f"points and the verdict threshold {_thr:.2f}: per algorithm, dataset and fingerprint fraction (random rows), "
        "Delta Acc (mean and sd of the 30 paired differences, pp), the posterior probabilities of worse, within the "
        "region and better (Bayesian correlated t-test), and the verdict.")
GENERATED["TableS14_accuracy_relative_rope.csv"] = (
    "Table S14", "Accuracy verdicts with a ROPE relative to the baseline's error",
    "Per algorithm, dataset and fingerprint fraction (random rows): baseline accuracy, Delta Acc (pp), the verdict "
    "with the predefined ROPE of ±1 pp, and the verdicts with a ROPE of ±5 %, ±10 % and ±20 % of the baseline's "
    "error rate (its width in pp beside each); same Bayesian correlated t-test and threshold 0.95.")
GENERATED["TableS15_accuracy_correlation.csv"] = (
    "Table S15", "Accuracy verdicts under other correlations between the runs",
    "Per algorithm, dataset and fingerprint fraction (random rows): Delta Acc (pp) and, for an assumed correlation "
    "rho of 0, 0.1, 0.2, 0.3, 0.4 and 0.5 between the 30 paired runs, the posterior probabilities of worse and of "
    "within ±1 pp and the verdict; same Bayesian correlated t-test, ROPE ±1 pp and threshold 0.95. The analysis uses "
    "rho = 0.2 (Nadeau and Bengio: test share of the data).")
GENERATED["TableS16_size_vs_fingerprint.csv"] = (
    "Table S16", "Dataset size and fingerprint size modelled together",
    "Per algorithm and measure (S_T, hit rate, fake-hit ratio, error per fake hit, absolute Delta Acc), one least-squares "
    "model over the 51 configurations (17 datasets x 3 fractions, random rows): log10(measure) on log10(rows) and "
    "log10(fingerprint rows); absolute Delta Acc in pp without log. Per term: coefficient, 95% bootstrap interval over "
    "datasets (2,000 resamples), range when one dataset is left out, and for log measures the factor for a tenfold "
    "larger table or fingerprint. Also: a tenfold larger table at the same fraction, and S_T with the share of work "
    "avoided added.")
GENERATED["FigureS4_size_vs_fingerprint.pdf"] = (
    "Figure S4", "Fake hits and speedup against the number of fingerprint rows",
    "Fake-hit ratio (log) and S_T against the number of fingerprint rows (log); GE and GP (random fingerprint rows). "
    "Each line is one dataset and connects its three fingerprints: circle 1%, square 5%, triangle 10%. The colour "
    "gives the size of the dataset (colour bar). Moving along a line changes only the fingerprint; comparing lines at "
    "the same position on the x axis changes only the dataset. Panel titles: the factors of the joint model "
    "(Table S16).")
GENERATED["TableS17_cost_model.csv"] = (
    "Table S17", "A simple cost model for the speedup",
    "Per algorithm, dataset and fingerprint fraction (random rows): measured S_T; the measured cost of the fingerprint "
    "step relative to a full evaluation (c, phase-timing runs); the share of full-evaluation work avoided by hits (w, "
    "diagnostic runs, all generations and the first 5 or 10 only); the predicted S_T = 1/(1 + c - w), with no fitted "
    "parameter, and its error; and whether the rule w > c gives the observed side of S_T = 1.")
GENERATED["FigureS5_cost_model.pdf"] = (
    "Figure S5", "Predicted against measured speedup",
    "Top: measured S_T against the S_T predicted by the cost model 1/(1 + c - w) (Table S17), log scales, one point "
    "per configuration, coloured by the number of rows of the dataset; the line of equality and the lines S_T = 1; GE "
    "and GP (random fingerprint rows). Bottom: for how many configurations the rule w > c gives the observed side of "
    "S_T = 1 when w comes from the first n generations only (a short pilot run) and c from the whole run.")
GENERATED["TableS18_cache_memory.csv"] = (
    "Table S18", "Memory used by the cache",
    "Per algorithm, dataset and fingerprint fraction (random rows, diagnostic runs, 30 seeds): evaluations per run; "
    "cache entries at the end of the run; cache size in MB as run (key to fitness and key to program text; the text "
    "only counts hits from the same program); the size without the program text (key to fitness only); peak memory "
    "of the whole process in MB (includes the diagnostic recording, so an upper bound); and the cache's share of "
    "that peak. Median and maximum over the seeds. hits_kept_last5 / hits_kept_last10: share of the hits whose cache "
    "entry was at most 5 / 10 generations old, i.e. the hits a cache keeping only those entries would still return "
    "(a lower bound; from the hit logs).")
GENERATED["TableS19_selection.csv"] = (
    "Table S19", "How wrong cached fitness values affect selection",
    "Per algorithm, dataset and fingerprint fraction (random rows, diagnostic runs, 30 seeds): fingerprint rows; share "
    "of the programs whose fitness used by selection differs from the true fitness (mean over generations, median over "
    "seeds); rank correlation of used and true fitness (median; lowest generation); share of tournaments whose winner "
    "was truly worse than another program in the same tournament (median and largest over seeds); how much worse it "
    "was in percentage points of error (median); generations whose best program carried a wrong fitness.")
GENERATED["TableS20_divergence.csv"] = (
    "Table S20", "How far FEC runs drift from their baseline",
    "Per algorithm, dataset and fingerprint fraction (random rows, FEC timing runs beside the baseline run of the same "
    "seed, 30 seeds): share of FEC runs whose population separated from the baseline's (average program size, number "
    "of distinct programs or average phenotype length differ); median generation of the separation; at the last "
    "generation, the mean absolute and the mean signed difference, FEC minus baseline, of the best training error and "
    "the mean signed difference of its test error, in percentage points.")
SID = r"\b(Figure|Table|File) S(\d+)\b"


def _size(n: int) -> str:
    return f"{n / 1024:.1f} kB" if n < 1024 * 1024 else f"{n / 1024 / 1024:.1f} MB"


def read_index(path: Path) -> tuple[str, list[dict]]:
    """(text before the table, rows) of INDEX.md."""
    text = path.read_text(encoding="utf-8")
    if INDEX_HEAD not in text:
        raise TableError(f"{path}: table header missing ({INDEX_HEAD})")
    preamble, table = text.split(INDEX_HEAD, 1)
    keys = [k.strip() for k in INDEX_HEAD.strip("|").split("|")]
    rows = []
    for line in table.strip().splitlines()[1:]:             # skip the |---| line
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != len(keys):
            raise TableError(f"{path}: malformed row: {line}")
        rows.append(dict(zip(keys, cells)))
    return preamble, rows


def _sid_key(sid: str):
    kind, n = re.match(r"(Figure|Table|File) S(\d+)", sid).groups()
    return ({"Figure": 0, "Table": 1, "File": 2}[kind], int(n))


_FVR_CACHE: dict = {}


def fps_vs_random(all_metrics: pd.DataFrame, props: pd.DataFrame, store: Path = results.STORE) -> pd.DataFrame:
    """Per (algorithm, dataset, fraction): outcomes with random (R) and farthest-point (F) rows, and the paired
    comparison of the two FEC runs of each seed (test accuracy, evaluation time). Empty without both samplers."""
    a = annotated(all_metrics[all_metrics.sampler.isin(["random", "farthest_point"])])
    if a.sampler.nunique() < 2 or not (_f(store, "runs.csv")).is_file():
        return pd.DataFrame()
    key = (store, tuple(sorted(set(a.dataset))))
    if key in _FVR_CACHE:
        return _FVR_CACHE[key]
    from analysis import wilcoxon_paired
    a = a.assign(outcome=np.select([(a.speed == "faster") & (a.verdict == "equivalent"), a.verdict == "equivalent",
                                    a.verdict == "worse"], ["faster, comparable", "comparable", "worse"],
                                   default=a.speed.map({"faster": "faster", "slower": "slower"}).fillna("neither")))
    r = pd.read_csv(_f(store, "runs.csv"), usecols=["algorithm", "setup", "dataset", "mode", "sampler", "fraction", "seed",
                                                 "test_accuracy", "eval_time_sec"])
    r = r[(r.setup == SETUP) & (r["mode"] == "fec") & r.dataset.isin(set(a.dataset))
          & r.sampler.isin(["random", "farthest_point"])]
    rows = []
    for (algo, ds, f), x in r.groupby(["algorithm", "dataset", "fraction"]):
        w = x.pivot_table(index="seed", columns="sampler", values=["test_accuracy", "eval_time_sec"]).dropna()
        if ("test_accuracy", "farthest_point") not in w or ("test_accuracy", "random") not in w or not len(w):
            continue
        m = a[(a.algorithm == algo) & (a.dataset == ds) & np.isclose(a.fraction, f)].set_index("sampler")
        if not {"random", "farthest_point"} <= set(m.index):
            continue
        rows.append({"algorithm": algo, "dataset": ds, "fraction": f,
                     "outcome_random": m.loc["random", "outcome"], "outcome_farthest_point": m.loc["farthest_point", "outcome"],
                     "equivalent_random": m.loc["random", "verdict"] == "equivalent",
                     "equivalent_farthest_point": m.loc["farthest_point", "verdict"] == "equivalent",
                     "acc_diff_pp": float(100 * (w[("test_accuracy", "farthest_point")] - w[("test_accuracy", "random")]).mean()),
                     "acc_p": float(wilcoxon_paired(w[("test_accuracy", "random")], w[("test_accuracy", "farthest_point")])["p"]),
                     "S_T_random": m.loc["random", "S_T"], "S_T_farthest_point": m.loc["farthest_point", "S_T"],
                     "time_p": float(wilcoxon_paired(w[("eval_time_sec", "random")], w[("eval_time_sec", "farthest_point")])["p"]),
                     "fake_random": m.loc["random", "fake_hit_ratio"], "fake_farthest_point": m.loc["farthest_point", "fake_hit_ratio"]})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.merge(props[["dataset", "rows"]], on="dataset", how="left").sort_values(["algorithm", "rows", "fraction"], kind="stable")
    _FVR_CACHE[key] = out
    return out


SAMPLER_CASE = ("GP", "phoneme_balanced", 0.01)     # §6.4 example: largest seed-paired accuracy gain


def sampler_case_numbers(all_metrics: pd.DataFrame, props: pd.DataFrame, store: Path = results.STORE) -> dict:
    """Macros \res SampCase...: the §6.4 example under random and farthest-point rows."""
    algo, ds, f = SAMPLER_CASE
    d = fps_vs_random(all_metrics, props, store)
    x = annotated(all_metrics[(all_metrics.algorithm == algo) & (all_metrics.dataset == ds)
                              & np.isclose(all_metrics.fraction, f)]).set_index("sampler")
    row = d[(d.algorithm == algo) & (d.dataset == ds) & np.isclose(d.fraction, f)] if len(d) else d
    if not len(row) or not {"random", "farthest_point"} <= set(x.index):
        return {}
    r = pd.read_csv(_f(store, "runs.csv"), usecols=["algorithm", "setup", "dataset", "mode", "sampler", "fraction", "sample_size"])
    fp = r[(r.setup == SETUP) & (r.algorithm == algo) & (r.dataset == ds) & (r["mode"] == "fec")
           & np.isclose(r.fraction, f)].sample_size.mean()
    out = {"SampCaseRows": _rows(props.set_index("dataset").rows.get(ds)), "SampCaseFpRows": _int(fp),
           "SampCaseAccDiff": f"{row.acc_diff_pp.iloc[0]:.1f}",      # positive: claim samp-case/paired-significant
           "SampCaseAccP": _p(row.acc_p.iloc[0])}
    for s, k in (("random", "R"), ("farthest_point", "F"), ("stratified", "S")):
        if s in x.index:
            out[f"SampCaseFake{k}"] = f"{100 * x.loc[s, 'fake_hit_ratio']:.0f}"
            out[f"SampCaseDacc{k}"] = _pp(x.loc[s, "dAcc"])
            out[f"SampCaseWorse{k}"] = f"{x.loc[s, 'bayes_worse']:.2f}"
            out[f"SampCaseST{k}"] = f"{x.loc[s, 'S_T']:.2f}"
    return out


def sampler_case_claims(all_metrics: pd.DataFrame, props: pd.DataFrame, store: Path = results.STORE) -> list:
    algo, ds, f = SAMPLER_CASE
    d = fps_vs_random(all_metrics, props, store)
    x = annotated(all_metrics[(all_metrics.algorithm == algo) & (all_metrics.dataset == ds)
                              & np.isclose(all_metrics.fraction, f)]).set_index("sampler")
    row = d[(d.algorithm == algo) & (d.dataset == ds) & np.isclose(d.fraction, f)] if len(d) else d
    if not len(row) or not {"random", "farthest_point", "stratified"} <= set(x.index):
        return []
    R, F, S = x.loc["random"], x.loc["farthest_point"], x.loc["stratified"]
    return [("samp-case/fewer-fakes", f"§6.4 {ds}: farthest-point rows had fewer fake hits", F.fake_hit_ratio < R.fake_hit_ratio),
            ("samp-case/smaller-loss", f"§6.4 {ds}: the accuracy loss against the baseline shrank (dAcc < 0 for both, "
             "|F| < |R|)", R.dAcc < F.dAcc < 0),
            ("samp-case/paired-significant", f"§6.4 {ds}: the seed-paired difference between the FEC runs was significant",
             row.acc_p.iloc[0] <= ALPHA and row.acc_diff_pp.iloc[0] > 0),
            ("samp-case/st-up", f"§6.4 {ds}: S_T rose from below 1 to above 1", R.S_T < 1 < F.S_T),
            ("samp-case/no-verdict", f"§6.4 {ds}: neither random nor farthest-point reached equivalence; stratified was "
             "judged worse", R.verdict != "equivalent" and F.verdict != "equivalent" and S.verdict == "worse"),
            ("samp-case/largest-gain", f"§6.4: {ds} {algo} {f:.0%} is the largest seed-paired accuracy gain of farthest-point "
             "rows over random", len(d) > 0 and d.acc_diff_pp.idxmax() == row.index[0])]


# ---------------------------------------------------------------- File S1: data dictionary of the supplementary CSVs

_PP = "percentage points (pp)"
_SEEDS = "over the 30 paired seeds"
_BAYES = ("posterior probability from the Bayesian correlated t-test over the paired seeds (Nadeau-Bengio "
          "variance correction)")
COLUMN_DOC = {                        # one meaning per column name, shared by every file that has the column
    "eval_saved_s": "Seconds of fitness evaluation saved per run: mean baseline evaluation time minus mean FEC "
                    "evaluation time (timing runs); negative: FEC slower.",
    "wall_saved_s": "Seconds of whole-run wall time saved per run: mean baseline wall time minus mean FEC wall time "
                    "(timing runs); negative: FEC slower.",
    "verdict_rope1pp": "Bayesian accuracy verdict with the predefined ROPE of ±1 pp (as Table 5).",
    **{k: v for rho in CORRELATIONS for k, v in (
        (f"p_worse_rho{rho:.1f}", f"P(FEC accuracy lower by more than 1 pp), Bayesian correlated t-test with an assumed "
                                  f"correlation of {rho:.1f} between the runs."),
        (f"p_rope_rho{rho:.1f}", f"P(FEC and baseline accuracy within ±1 pp), same test with correlation {rho:.1f}."),
        (f"verdict_rho{rho:.1f}", f"Bayesian accuracy verdict (threshold 0.95) with correlation {rho:.1f}"
                                  + (" (as Table 5)." if rho == 0.2 else ".")))},
    "rope_rel5_pp": "Width of the relative ROPE, ±5 % of the baseline's error rate, in percentage points.",
    "rope_rel10_pp": "Width of the relative ROPE, ±10 % of the baseline's error rate, in percentage points.",
    "rope_rel20_pp": "Width of the relative ROPE, ±20 % of the baseline's error rate, in percentage points.",
    "verdict_rel5": "Bayesian accuracy verdict with the ROPE of ±5 % of the baseline's error rate.",
    "verdict_rel10": "Bayesian accuracy verdict with the ROPE of ±10 % of the baseline's error rate.",
    "verdict_rel20": "Bayesian accuracy verdict with the ROPE of ±20 % of the baseline's error rate.",
    "algorithm": "GE (grammatical evolution) or GP (strongly typed genetic programming).",
    "measure": "Measure modelled: S_T, hit_rate, fake_hit_ratio, error_per_fake_pp (mean |cached - full-training "
               "error| per fake hit, pp, as Table S13) or abs_dAcc_pp (|Delta Acc|, pp).",
    "model": "rows+fp: log10(measure) ~ log10(rows) + log10(fingerprint rows); rows+fp+work: the same plus the share "
             "of the work avoided (Table S4). |Delta Acc| is modelled in pp without log.",
    "term": "log_rows: tenfold larger dataset at the same number of fingerprint rows; log_fp: tenfold larger "
            "fingerprint at the same dataset; rows_same_fraction: tenfold larger dataset at the same fingerprint "
            "fraction (log_rows + log_fp); work_avoided: 0.1 more of the work avoided.",
    "coef": "Least-squares coefficient of the term (log10 units of the measure; pp for abs_dAcc_pp).",
    "ci_low": "Lower end of the 95 % percentile bootstrap interval of coef (datasets resampled, 2,000 times).",
    "ci_high": "Upper end of the 95 % percentile bootstrap interval of coef.",
    "loo_min": "Smallest coef when one dataset is left out (17 refits).",
    "loo_max": "Largest coef when one dataset is left out.",
    "factor": "10^coef: the multiplier of the measure for the change named in `term` (empty for abs_dAcc_pp).",
    "factor_low": "10^ci_low.",
    "factor_high": "10^ci_high.",
    "fp_cost": "Measured cost of the fingerprint step per new program, relative to the full evaluation of a missed "
               "program (c; phase-timing runs, mean over seeds).",
    "S_T_predicted": "Speedup predicted by the cost model, 1/(1 + fp_cost - work_avoided); no fitted parameter.",
    "error_pct": "Prediction error, 100 x (S_T_predicted / S_T - 1), in percent.",
    "rule_right": "True when the rule work_avoided > fp_cost gives the observed side of S_T = 1 (faster or not).",
    "work_avoided_gen1_5": "work_avoided from generations 1 to 5 only, as in a short pilot run.",
    "work_avoided_gen1_10": "work_avoided from generations 1 to 10 only.",
    "rule_right_gen1_5": "rule_right with work_avoided_gen1_5.",
    "rule_right_gen1_10": "rule_right with work_avoided_gen1_10.",
    "r2": "Share of the variance of the (log) measure explained by the model, over the 51 configurations.",
    "configurations": "Configurations in the model (datasets x fractions).",
    "datasets": "Datasets in the model.",
    "dataset": "Dataset name, as in Table 1 of the paper.",
    "setup": "Parameter setup of the runs; `default` = the settings of the paper.",
    "fraction": "Fingerprint fraction: share of the training rows used as fingerprint rows (0.01, 0.05, 0.10); "
                "random fingerprint rows.",
    "rows": "Number of rows of the dataset as used.",
    "evaluations_median": "Table S18: evaluations per run (hits + misses), median over the seeds.",
    "entries_median": "Table S18: cache entries at the end of the run, median over the seeds.",
    "entries_max": "Table S18: cache entries at the end of the run, largest over the seeds.",
    "cache_mb_median": "Table S18: cache size in MB as run (key to fitness and key to program text), median.",
    "cache_mb_max": "Table S18: cache size in MB as run (key to fitness and key to program text), largest.",
    "fitness_only_mb_median": "Table S18: cache size in MB without the program text (key to fitness only), median.",
    "fitness_only_mb_max": "Table S18: cache size in MB without the program text (key to fitness only), largest.",
    "peak_mb_median": "Table S18: peak memory of the whole process in MB (diagnostic run), median.",
    "peak_mb_max": "Table S18: peak memory of the whole process in MB (diagnostic run), largest.",
    "cache_share_of_peak_median": "Table S18: cache size as run divided by the peak memory of the process, median.",
    "cache_share_of_peak_max": "Table S18: cache size as run divided by the peak memory of the process, largest.",
    "hits_kept_last5": "Table S18: share of hits whose cache entry was at most 5 generations old (hit logs).",
    "hits_kept_last10": "Table S18: share of hits whose cache entry was at most 10 generations old (hit logs).",
    "fp_rows": "Number of fingerprint rows (random rows, as recorded by the FEC runs).",
    "wrong_share_median": "Table S19: share of valid programs whose fitness used differs from the true fitness (mean over generations), median over seeds.",
    "rank_corr_median": "Table S19: Spearman correlation of used and true fitness across the population (mean over generations), median over seeds.",
    "rank_corr_lowest": "Table S19: lowest Spearman correlation of used and true fitness in any generation of any seed.",
    "changed_tournaments_median": "Table S19: share of tournaments whose winner was truly worse than another aspirant, median over seeds.",
    "changed_tournaments_max": "Table S19: share of tournaments whose winner was truly worse than another aspirant, largest seed.",
    "worse_by_pp_median": "Table S19: how much worse the winner truly was (pp of error, mean over those tournaments), median over seeds.",
    "elite_wrong_generations": "Table S19: generations, over all seeds, whose best program carried a wrong fitness.",
    "diverged_share": "Table S20: share of FEC runs whose population separated from the baseline run of the same seed.",
    "first_divergence_gen_median": "Table S20: median generation in which the population first differed from the baseline's (separated runs).",
    "train_diff_abs_pp_end": "Table S20: mean absolute difference of the best training error, FEC minus baseline, last generation (pp).",
    "train_diff_pp_end": "Table S20: mean difference of the best training error, FEC minus baseline, last generation (pp; above 0 = FEC worse).",
    "test_diff_pp_end": "Table S20: mean difference of the test error of the best program, FEC minus baseline, last generation (pp).",
    "n_runs": "Number of paired seeds behind the row (baseline, FEC timing and FEC diagnostic run with the same seed).",
    "runs": "Number of FEC diagnostic runs (seeds) behind the row.",
    "S_T": "Evaluation-time speedup: mean baseline fitness-evaluation time / mean FEC fitness-evaluation time " + _SEEDS
           + "; > 1 = FEC faster.",
    "S_T_sd": "Standard deviation " + _SEEDS + " of the per-seed ratio baseline time / FEC time.",
    "S_wall": "Wall-time speedup: mean baseline run time / mean FEC run time (whole run, FEC including the sampling of "
              "the fingerprint rows) " + _SEEDS + "; > 1 = FEC faster.",
    "S_wall_sd": "Standard deviation " + _SEEDS + " of the per-seed wall-time ratio.",
    "p_wall": "p-value of the two-sided paired Wilcoxon signed-rank test of FEC against baseline whole-run wall time "
              "over the paired seeds; uncorrected.",
    "p_wall_holm": "`p_wall` Holm-adjusted over the algorithm's 51 configurations (17 datasets x 3 fractions).",
    "wall_speed": "`faster` / `slower`: Holm-adjusted p <= 0.05 (`p_wall_holm`) with S_wall > 1 / < 1; empty = not "
                  "significant.",
    "dAcc": "Accuracy difference: mean FEC test accuracy - mean baseline test accuracy " + _SEEDS + ", as a fraction "
            "(0.01 = 1 pp); < 0 = FEC less accurate.",
    "dAcc_sd": "Standard deviation " + _SEEDS + " of the paired accuracy differences (fraction).",
    "dAcc_pp": "Accuracy difference as in `dAcc`, in " + _PP + ".",
    "dAcc_sd_pp": "Standard deviation of the paired accuracy differences, in " + _PP + ".",
    "baseline_accuracy": "Mean test accuracy of the baseline runs (1 - test error rate of the run's best program).",
    "baseline_accuracy_sd": "Standard deviation of the baseline test accuracy " + _SEEDS + ".",
    "fec_accuracy": "Mean test accuracy of the FEC timing runs.",
    "fec_accuracy_sd": "Standard deviation of the FEC test accuracy " + _SEEDS + ".",
    "hit_rate": "Cache hit rate: hits / (hits + misses), pooled over the FEC timing runs.",
    "hit_rate_sd": "Standard deviation " + _SEEDS + " of the per-run hit rate.",
    "fake_hit_ratio": "Fake-hit ratio: fake hits / hits in the diagnostic runs (a fake hit returns a cached fitness "
                      "that differs from the program's fitness on all training rows).",
    "fake_hit_ratio_sd": "Standard deviation " + _SEEDS + " of the per-run fake-hit ratio.",
    "duplicate_hit_share": "Share of the hits whose program is identical to the one stored under the key "
                           "(structurally equivalent hits, Section 5.4).",
    "p_eval_time": "p-value of the two-sided paired Wilcoxon signed-rank test of FEC against baseline "
                   "fitness-evaluation time over the paired seeds; uncorrected.",
    "p_accuracy": "p-value of the two-sided paired Wilcoxon signed-rank test of FEC against baseline test accuracy; "
                  "uncorrected.",
    "p_equivalence": "p-value of the paired Wilcoxon two one-sided tests (TOST) for equivalence within +-1 pp; small = "
                     "accuracy shown comparable.",
    "bayes_worse": "P(FEC accuracy lower than the baseline's by more than 1 pp), " + _BAYES + ".",
    "bayes_rope": "P(FEC and baseline accuracy within +-1 pp), " + _BAYES + ".",
    "bayes_better": "P(FEC accuracy higher than the baseline's by more than 1 pp), " + _BAYES + ".",
    "p_eval_holm": "`p_eval_time` Holm-adjusted over the algorithm's 51 configurations (17 datasets x 3 fractions); "
                   "decides `speed`.",
    "p_eval_bh": "`p_eval_time` Benjamini-Hochberg-adjusted over the same 51 configurations (check only).",
    "speed": "`faster` / `slower`: Holm-adjusted p <= 0.05 (`p_eval_holm`) with S_T > 1 / < 1; empty = not "
             "significant.",
    "speed_raw": "The same mark from the uncorrected p (`p_eval_time`), for comparison.",
    "verdict": "Bayesian accuracy verdict: `equivalent` if the probability within the ROPE reaches the threshold, "
               "else `worse` or `better` if that probability does, else `undecided`.",
    "rope_pp": "Half-width of the region of practical equivalence (ROPE), in " + _PP + ".",
    "threshold": "Posterior probability a verdict must reach.",
    "p_worse": "P(FEC accuracy lower than the baseline's by more than the ROPE), " + _BAYES + ".",
    "p_rope": "P(FEC and baseline accuracy within the ROPE), " + _BAYES + ".",
    "p_better": "P(FEC accuracy higher than the baseline's by more than the ROPE), " + _BAYES + ".",
    "hit_size_ratio": "Mean size (operator calls) of a program that hit the cache / mean size of one that missed it; "
                      "diagnostic runs, generations 1-30 pooled.",
    "work_avoided": "Estimated share of the full-evaluation work (operator calls x training rows, every operator "
                    "call counted as one unit per row) that the hits avoided; the fingerprint step is not counted.",
    "source": "Where the dataset comes from (repository).",
    "url": "Address of the dataset at its source.",
    "rows_source": "Rows of the table at the source.",
    "columns_source": "Columns of the table at the source.",
    "missing_value_rows_removed": "Rows removed because of missing values.",
    "columns_dropped": "Columns dropped, with the reason.",
    "one_hot": "Categorical columns one-hot encoded.",
    "type_changes": "Columns whose type was changed (cast or override).",
    "label_encoding": "How the class labels were encoded.",
    "balancing": "Class balancing applied (rows before and after), or `none (as published)`.",
    "duplicate_rows_kept": "Exact duplicate rows in the table used (kept).",
    "rows_used": "Rows of the table used in the experiments.",
    "numeric_features": "Numeric features of the table used.",
    "boolean_features": "Boolean (0/1) features of the table used.",
    "classes": "Number of classes.",
    "checksum": "SHA-256 checksum of the table used.",
    "range": "Constant range of the grammar: pm1 / pm2 / pm3 = constants in +-0.99 / +-1.99 / +-2.99 "
             "(standardised values); none = no numeric feature.",
    "coverage_min": "Share of the standardised values of the least covered numeric feature (all rows) that lie "
                    "inside the chosen range.",
    "degenerate_rate": "Share of (numeric feature, constant) pairs whose comparison splits off fewer than 1% or more "
                       "than 99% of the rows.",
    "fairness_ratio": "P(a boolean feature appears) / P(a numeric feature appears) in initial GE programs, mean over "
                      "30 seeded populations; 1 = both feature types equally likely.",
    "min_init_depth": "Smallest initial derivation depth (3, 5, 7 or 9) at which a seeded GE population of 1000 has "
                      "no invalid individual.",
    "hits_per_run": "Mean number of cache hits per diagnostic run.",
    "structural_mean": "Share of the hits that were structurally equivalent (the same program as the one stored "
                       "under the key), mean over the seeds.",
    "structural_sd": "Standard deviation over the seeds of `structural_mean`.",
    "fitness_equivalent_mean": "Share of the hits that were fitness-equivalent (a different program with the same "
                               "fitness on all training rows), mean over the seeds.",
    "fitness_equivalent_sd": "Standard deviation over the seeds of `fitness_equivalent_mean`.",
    "split_runs": "Number of instrumented diagnostic runs (seeds) that recorded the predictions behind "
                  "`behaviour_equivalent_share`; empty where none was run.",
    "behaviour_equivalent_share": "Share of the fitness-equivalent hits whose program predicted every training row as "
                                  "the stored program did (behaviour-equivalent), pooled over `split_runs` runs; the "
                                  "rest had equal fitness with different predictions.",
    "nonequiv_mean": "Share of the hits that were not equivalent, i.e. fake hits (a different program with a different "
                     "fitness), mean over the seeds.",
    "nonequiv_sd": "Standard deviation over the seeds of `nonequiv_mean`.",
    "fake_hits": "Number of fake hits, over all diagnostic runs (30 seeds).",
    "mean_error_pp": "Mean absolute difference between the cached and the true (all training rows) error rate per "
                     "fake hit, in " + _PP + ".",
    "mean_signed_pp": "Mean of (cached - true error rate) per fake hit, in " + _PP + "; negative = the cache "
                      "reported a lower error than the true one.",
    "worst_median_pp": "Median over the generations of the largest difference in a generation, in " + _PP + ".",
    "worst_p90_pp": "90th percentile over the generations of the largest difference in a generation, in " + _PP + ".",
    "worst_max_pp": "Largest difference observed, in " + _PP + ".",
    "gens_worst_over_1pp": "Share of the generations whose largest difference exceeded 1 pp.",
    "gens_worst_over_5pp": "Share of the generations whose largest difference exceeded 5 pp.",
}
DICTIONARY_NOTE = ("Only generations with at least one fake hit are counted. The diagnostic runs record, per "
                   "generation, the mean and the largest difference over the hits, not each fake hit.")


def data_dictionary(folder: Path) -> str:
    """File S1: every column of every supplementary CSV with its meaning; raises when a column has no entry."""
    missing, parts = [], []
    tables = sorted((p for p in folder.glob("Table*.csv")), key=lambda p: _sid_key(GENERATED.get(p.name, ("Table S99",))[0]))
    for path in tables:
        try:
            cols = pd.read_csv(path, nrows=0).columns
        except pd.errors.EmptyDataError:                       # a table written as "no data" (no store)
            continue
        if list(cols) == ["no data"]:
            continue
        missing += [f"{path.name}: {c}" for c in cols if c not in COLUMN_DOC]
        sid, title, _ = GENERATED.get(path.name, ("", path.stem, ""))
        note = f"\n\n{DICTIONARY_NOTE}" if path.name.startswith("TableS13") else ""
        parts.append(f"## {sid}: {title} (`{path.name}`){note}\n\n| Column | Meaning |\n|---|---|\n"
                     + "".join(f"| `{c}` | {COLUMN_DOC.get(c, '')} |\n" for c in cols))
    if missing:
        raise TableError("columns without an entry in COLUMN_DOC (paper_tables.py):\n  " + "\n  ".join(missing))
    head = ("# File S1: Data dictionary of the supplementary tables\n\n"
            "Every column of every supplementary table (CSV). All results use random fingerprint rows and the "
            "settings of the paper; 30 paired seeds (30-59) per configuration. Accuracy is test accuracy "
            "(1 - test error rate of a run's best program).\n\n")
    return head + "\n".join(parts)


def supplementary(corpus, metrics, props, out_tables, out_supp, manuscript: Path | None = None,
                  folder: Path = SUPPLEMENTARY, size_limit: int = SIZE_LIMIT, all_metrics=None, **_) -> list[Path]:
    """Tables S1-S11 (CSV), INDEX.md checked against the folder and the manuscript, the back-matter list."""
    out_supp.mkdir(parents=True, exist_ok=True)
    metrics = metrics[metrics.sampler == MAIN_SAMPLER]                 # the paper reports random fingerprint rows only
    s1 = metrics.drop(columns=["sampler", "sampling_time_sec", "S_E", "S_E_sd", "S_gen", "S_gen_sd", "n_diagnostic"], errors="ignore").sort_values(["algorithm", "dataset", "fraction"], kind="stable")
    written = [write(out_supp, "TableS1_metrics.csv", s1.to_csv(index=False, float_format="%.6g"))]
    ann = with_wall(annotated(metrics))                                # S_wall's Wilcoxon test, as for S_T
    sv = seconds_saved()                                               # seconds saved per run (moved from Table 5)
    if len(sv):
        ann = ann.assign(_fr=ann.fraction.round(2)).merge(sv.rename(columns={"fraction": "_fr"}),
                                                          on=["algorithm", "dataset", "_fr"], how="left").drop(columns="_fr")
    for algo, name in (("GE", "TableS2_results_GE.csv"), ("GP", "TableS3_results_GP.csv")):
        written.append(write(out_supp, name, ann[ann.algorithm == algo][[c for c in FULL_COLUMNS if c in ann]]
                             .to_csv(index=False, float_format="%.6g")))
    s4 = annotated(metrics).assign(fraction=lambda d: d.fraction.round(2)).merge(
        hit_work(), on=["algorithm", "dataset", "sampler", "fraction"], how="inner") if (_f(results.STORE, "generations.csv")).is_file() else pd.DataFrame()
    s4 = (s4[["algorithm", "dataset", "fraction", "hit_rate", "hit_size_ratio", "work_avoided", "S_T"]]
          .sort_values(["algorithm", "dataset", "fraction"], kind="stable") if len(s4) else s4)
    written.append(write(out_supp, "TableS4_hit_work.csv", s4.to_csv(index=False, float_format="%.4g") if len(s4) else "no data\n"))
    keep = list(ordered_corpus(corpus, props).dataset)
    written.append(write(out_supp, "TableS5_preprocessing.csv", preprocessing_rows(keep).to_csv(index=False)))
    gp_cols = ["dataset", "range", "coverage_min", "degenerate_rate", "fairness_ratio", "min_init_depth"]
    s6 = ordered_corpus(corpus, props)[[c for c in gp_cols if c in ordered_corpus(corpus, props)]]
    written.append(write(out_supp, "TableS6_grammar_properties.csv", s6.to_csv(index=False, float_format="%.4g")))
    in_text = metrics[metrics.dataset.isin(keep)]
    written.append(fake_acc_by_fraction_figure(in_text, out_supp))   # Figure S2
    written.append(fake_acc_paired_figure(in_text, out_supp))        # Figure S3
    written.append(write(out_supp, "TableS12_hit_kinds.csv",
                         hit_kinds_table(keep, props).to_csv(index=False)))
    if (_f(results.STORE, "generations.csv")).is_file():
        written.append(write(out_supp, "TableS13_fake_severity.csv",
                             severity_supp_table(keep, props).to_csv(index=False)))
    for sid, rope, thr in SENSITIVITY_TABLES:                          # Tables S7-S11: other ROPE / threshold
        written.append(write(out_supp, sensitivity_name(sid, rope, thr),
                             sensitivity_detail(in_text, props, rope, thr).to_csv(index=False)))
    written.append(write(out_supp, "TableS14_accuracy_relative_rope.csv",   # supervisor review, 2026-10-25
                         relative_rope_detail(in_text, props).to_csv(index=False)))
    written.append(write(out_supp, "TableS15_accuracy_correlation.csv",
                         correlation_detail(in_text, props).to_csv(index=False)))
    if (_f(results.STORE, "runs.csv")).is_file() and (_f(results.STORE, "generations.csv")).is_file():
        design = size_design(in_text, props)
        written.append(write(out_supp, "TableS16_size_vs_fingerprint.csv", "no data\n" if design.empty else
                             size_model(design).to_csv(index=False, float_format="%.4g")))
        written.append(size_fingerprint_figure(design, out_supp))                                   # Figure S4
    if (_f(results.STORE, "phase_timing.csv")).is_file() and (_f(results.STORE, "generations.csv")).is_file():
        cost = cost_design(in_text, props)
        written.append(write(out_supp, "TableS17_cost_model.csv", "no data\n" if cost.empty else
                             cost.to_csv(index=False, float_format="%.4g")))
        written.append(cost_figure(cost, out_supp))                                                 # Figure S5
    mem = memory_table(set(in_text.dataset), props)
    written.append(write(out_supp, "TableS18_cache_memory.csv", "no data\n" if mem.empty else
                         mem.to_csv(index=False, float_format="%.4g")))
    sel = selection_table(set(in_text.dataset))
    written.append(write(out_supp, "TableS19_selection.csv", "no data\n" if sel.empty else
                         sel.to_csv(index=False, float_format="%.4g")))
    traj = trajectory_table(set(in_text.dataset))
    written.append(write(out_supp, "TableS20_divergence.csv", "no data\n" if traj.empty else
                         traj.to_csv(index=False, float_format="%.4g")))
    written.append(write(out_supp, "FileS1_data_dictionary.md", data_dictionary(out_supp)))   # every CSV above
    preamble, rows = read_index(folder / "INDEX.md")
    by_file = {r["file"]: r for r in rows}
    for fname, (sid, title, desc) in GENERATED.items():         # generated rows: the code is the source of truth
        by_file.setdefault(fname, {"id": sid, "file": fname, "source": "python paper_tables.py", "size": "",
                                   "committed": ""}).update({"id": sid, "title": title, "description": desc})
    rows = list(by_file.values())
    problems = []
    present = {p.name for p in folder.iterdir() if p.is_file() and p.name not in ("INDEX.md", ".gitignore")
               and not p.name.startswith(".~lock.") and p.suffix != ".zip"} \
        if folder.is_dir() else set()
    present |= set(GENERATED)
    for name in sorted(present - {r["file"] for r in rows}):
        problems.append(f"file not in INDEX.md: {name}")
    for r in rows:
        if r["file"] not in present:
            problems.append(f"indexed file missing: {r['file']}")
        if not re.fullmatch(r"(Figure|Table|File) S\d+", r["id"]):
            problems.append(f"bad id {r['id']!r} for {r['file']}")
    for key in ("id", "file"):
        seen = [r[key] for r in rows]
        problems += [f"{key} used twice: {v}" for v in sorted({v for v in seen if seen.count(v) > 1})]
    if manuscript is not None and manuscript.is_file():
        cited = {f"{k} S{n}" for k, n in re.findall(SID, manuscript.read_text(encoding="utf-8"))}
        problems += [f"cited in the manuscript but not indexed: {c}" for c in sorted(cited - {r['id'] for r in rows})]
    if problems:
        raise TableError("supplementary folder and INDEX.md disagree:\n  " + "\n  ".join(problems))
    large = []
    for r in rows:
        path = out_supp / r["file"] if (out_supp / r["file"]).is_file() else folder / r["file"]
        n = path.stat().st_size
        r["size"], r["committed"] = _size(n), "yes" if n <= size_limit else "no"
        if n > size_limit:
            large.append(r["file"])
    rows.sort(key=lambda r: _sid_key(r["id"]))
    table = INDEX_HEAD + "\n|---|---|---|---|---|---|---|\n" + "".join(
        "| " + " | ".join(r[k] for k in ("id", "file", "title", "description", "source", "size", "committed")) + " |\n"
        for r in rows)
    written.append(write(out_supp, "INDEX.md", preamble + table))
    written.append(write(out_supp, ".gitignore", "".join(f"{n}\n" for n in large)))
    items = "; ".join(f"{r['id']}: {r['title']}" for r in rows)
    written.append(write(out_tables, "supplementary.tex",
                         header("paper/supplementary/INDEX.md")
                         + "The following supporting information can be downloaded at: \\linksupplementary{s1}, "
                         + items.replace("_", "\\_").replace("%", "\\%") + ".\n"))
    return written


# ---------------------------------------------------------------- Discussion case studies (§6.2, §6.3)

CASE_SIZE = ("wine_quality", "covertype")           # §6.2: a small and a large table
CASE_GEGP = ("electricity_cat", "Diabetes130US")    # §6.3: GE and GP on two tables of different difficulty
CASE_MC = ("accelerometer", "covertype")          # §6.5: a multiclass and a binary table, GE
MC_LIMITS = ("lhc_jets", "sleep")                  # §6.5: the larger and the smaller five-class table
MC_KEY = {"accelerometer": "Accel", "covertype": "Cover", "lhc_jets": "Lhc", "sleep": "Sleep"}
CASE_KEY = {"wine_quality": "Wine", "covertype": "Cover", "electricity_cat": "Elec", "Diabetes130US": "Diab"}
CASE_GEN_COLUMNS = ["algorithm", "setup", "dataset", "mode", "sampler", "fraction", "seed", "gen", "avg_nodes",
                    "fitness_diversity", "fec_hits", "fec_misses", "fec_same_phenotype_hits", "fec_fake_hits",
                    "fec_hit_operators", "fec_miss_operators"]
LAST_GEN = 30
_CASE_CACHE: dict = {}


def case_generations(store: Path = results.STORE) -> pd.DataFrame:
    """Per (dataset, algorithm, generation) of the case-study datasets, from the diagnostic runs with the main
    sampler (all fractions and seeds pooled): hit rate, fake and same-program shares of the hits, the share of
    the evaluation work (operator calls) that the hits avoided, mean program size and fitness diversity.
    The diagnostic runs follow the same course as the timing runs (same seeds; the check never changes a value)."""
    path = _f(store, "generations.csv")
    if path in _CASE_CACHE:
        return _CASE_CACHE[path]
    if not path.is_file():
        raise TableError(f"store file missing: {path} (run `python results.py all`)")
    names = set(CASE_SIZE + CASE_GEGP + CASE_MC + MC_LIMITS + ("covertype",))
    parts = [c[(c.setup == SETUP) & c.dataset.isin(names) & (c["mode"] == "fec_diagnostic") & (c.sampler == MAIN_SAMPLER)]
             for c in pd.read_csv(path, usecols=CASE_GEN_COLUMNS, chunksize=500_000)]
    g = pd.concat(parts)
    g = g.assign(hit_work=g.fec_hits * g.fec_hit_operators.fillna(0), miss_work=g.fec_misses * g.fec_miss_operators.fillna(0))
    hits = g.fec_hits.where(g.fec_hits > 0)                    # per-run ratios, for the spread between runs
    evals = (g.fec_hits + g.fec_misses).where(lambda e: e > 0)
    g = g.assign(r_hit=g.fec_hits / evals, r_fake=g.fec_fake_hits / hits, r_same=g.fec_same_phenotype_hits / hits,
                 r_work=g.hit_work / (g.hit_work + g.miss_work).where(lambda w: w > 0),
                 r_size=(g.hit_work / hits) / (g.miss_work / g.fec_misses.where(g.fec_misses > 0)))
    out = (g.groupby(["dataset", "algorithm", "gen"])
           .agg(hits=("fec_hits", "sum"), misses=("fec_misses", "sum"), fake=("fec_fake_hits", "sum"),
                same=("fec_same_phenotype_hits", "sum"), hit_work=("hit_work", "sum"), miss_work=("miss_work", "sum"),
                nodes=("avg_nodes", "mean"), diversity=("fitness_diversity", "mean"),
                hit_rate_sd=("r_hit", "std"), fake_share_sd=("r_fake", "std"), same_share_sd=("r_same", "std"),
                work_avoided_sd=("r_work", "std"), hit_size_sd=("r_size", "std"),
                nodes_sd=("avg_nodes", "std"), diversity_sd=("fitness_diversity", "std"))
           .reset_index())
    out = out.assign(hit_rate=out.hits / (out.hits + out.misses), fake_share=out.fake / out.hits,
                     same_share=out.same / out.hits, work_avoided=out.hit_work / (out.hit_work + out.miss_work),
                     hit_size=(out.hit_work / out.hits) / (out.miss_work / out.misses))
    _CASE_CACHE[path] = out
    return out


def case_costs(store: Path = results.STORE) -> pd.DataFrame:
    """Per (dataset, algorithm): ms per full evaluation in the baseline runs, and the fingerprint size m at 5%."""
    r = pd.read_csv(_f(store, "runs.csv"), usecols=["algorithm", "setup", "dataset", "mode", "sampler", "fraction",
                                                 "eval_time_sec", "n_full_evals", "sample_size"])
    r = r[(r.setup == SETUP) & r.dataset.isin(set(CASE_KEY) | set(MC_KEY))]
    base = (r[r["mode"] == "baseline"].assign(ms=lambda d: 1e3 * d.eval_time_sec / d.n_full_evals)
            .groupby(["dataset", "algorithm"]).ms.mean())
    fp = (r[(r["mode"] == "fec") & (r.sampler == MAIN_SAMPLER) & (r.fraction.round(2) == 0.05)]
          .groupby(["dataset", "algorithm"]).sample_size.mean())
    return pd.concat([base.rename("ms"), fp.rename("fp_rows")], axis=1).reset_index()


def _at(c: pd.DataFrame, ds: str, algo: str, col: str, gen: int) -> float:
    v = c[(c.dataset == ds) & (c.algorithm == algo) & (c.gen == gen)][col]
    return float(v.iloc[0]) if len(v) else float("nan")


def _late(c: pd.DataFrame, ds: str, algo: str) -> float:
    """Share of all the work avoided over the run that was avoided in the second half (generations 16-30)."""
    x = c[(c.dataset == ds) & (c.algorithm == algo)]
    return float(x[x.gen > LAST_GEN // 2].hit_work.sum() / x.hit_work.sum()) if len(x) else float("nan")


def _ms(x: float) -> str:
    return "--" if pd.isna(x) else (f"{x:.2f}" if x < 10 else f"{x:.0f}")


def case_ready(metrics: pd.DataFrame, props: pd.DataFrame) -> bool:
    """The case studies need every case dataset with GE and GP results and the difficulty column; a store
    without them (a synthetic test store, or before the runs arrive) skips §6.2/§6.3 instead of failing."""
    have = set(zip(metrics.dataset, metrics.algorithm))
    return "difficulty_gb" in props and all((ds, a) in have for ds in CASE_KEY for a in ("GE", "GP"))


def case_numbers(metrics: pd.DataFrame, props: pd.DataFrame, store: Path = results.STORE) -> dict:
    """Macros of §6.2 and §6.3: \res Case<Key><Algo><Name>, \res Case<Key><Name>, \res RhoDifficulty<Algo>(P)."""
    if not case_ready(metrics, props):
        return {}
    c, cost = case_generations(store), case_costs(store).set_index(["dataset", "algorithm"])
    st = metrics.groupby(["dataset", "algorithm"]).S_T.mean()
    pr = props.set_index("dataset")
    out = {}
    pct = lambda x: "--" if pd.isna(x) else f"{100 * x:.0f}"
    for ds, key in CASE_KEY.items():
        if ds in pr.index:
            out[f"Case{key}Rows"] = _rows(pr.loc[ds, "rows"])
            out[f"Case{key}Difficulty"] = f"{pr.loc[ds, 'difficulty_gb']:.2f}"
        for algo in ("GE", "GP"):
            p = f"Case{key}{algo}"
            out[p + "ST"] = _num(st.get((ds, algo), np.nan), 2)
            out[p + "Ms"] = _ms(cost.ms.get((ds, algo), np.nan))
            out[p + "FpRows"] = _int(cost.fp_rows.get((ds, algo), np.nan)) if not pd.isna(cost.fp_rows.get((ds, algo), np.nan)) else "--"
            for col, name, fmt in (("hit_rate", "Hit", lambda x: _num(x, 2)), ("fake_share", "Fake", pct),
                                   ("same_share", "Same", pct), ("work_avoided", "Work", pct),
                                   ("nodes", "Nodes", lambda x: _num(x, 0)), ("diversity", "Div", lambda x: _num(x, 2))):
                out[p + name + "First"] = fmt(_at(c, ds, algo, col, 1))
                out[p + name + "Last"] = fmt(_at(c, ds, algo, col, LAST_GEN))
            x = c[(c.dataset == ds) & (c.algorithm == algo) & (c.gen >= 1)]
            out[p + "FakeMax"] = pct(x.fake_share.max()) if len(x) else "--"
            out[p + "Late"] = pct(_late(c, ds, algo))
    for algo in ("GE", "GP"):
        rho, pval, _n = dataset_rho(metrics[metrics.algorithm == algo].merge(props[["dataset", "difficulty_gb"]], on="dataset"),
                                    "S_T", "difficulty_gb")
        out[f"RhoDifficulty{algo}"], out[f"RhoDifficulty{algo}P"] = f"{rho:.2f}", _p(pval)
    return out


def case_last_by_fraction(store: Path = results.STORE) -> pd.DataFrame:
    """The pooled case-study curves (Figures 10, 11, 13) split by fraction again: per (dataset, algorithm, fraction),
    from the diagnostic runs (main sampler) of the last generation, the fake-hit ratio, the share of work avoided,
    the hit/miss program size and the mean program size; and the evaluations each fraction contributes."""
    path = _f(store, "generations.csv")
    names = set(CASE_SIZE + CASE_GEGP + CASE_MC)
    parts = [c[(c.setup == SETUP) & c.dataset.isin(names) & (c["mode"] == "fec_diagnostic") & (c.sampler == MAIN_SAMPLER)
               & (c.gen == LAST_GEN)] for c in pd.read_csv(path, usecols=CASE_GEN_COLUMNS, chunksize=500_000)]
    g = pd.concat(parts).assign(fraction=lambda d: d.fraction.round(2))
    g = g.assign(hw=g.fec_hits * g.fec_hit_operators.fillna(0), mw=g.fec_misses * g.fec_miss_operators.fillna(0))
    t = g.groupby(["dataset", "algorithm", "fraction"]).agg(hits=("fec_hits", "sum"), misses=("fec_misses", "sum"),
                                                            fake=("fec_fake_hits", "sum"), hw=("hw", "sum"),
                                                            mw=("mw", "sum"), nodes=("avg_nodes", "mean"))
    return t.assign(fake_share=t.fake / t.hits, work_avoided=t.hw / (t.hw + t.mw),
                    hit_size=(t.hw / t.hits) / (t.mw / t.misses), evals=t.hits + t.misses).reset_index()


def gegp_train_test(store: Path = results.STORE) -> dict:
    """§6.6.1: in the baseline runs of the two GE/GP case tables, the mean training and test error of the best
    program in the last generation, per (dataset, algorithm); and the largest gap in percentage points."""
    path = _f(store, "generations.csv")
    if not path.is_file():
        return {}
    cols = ["algorithm", "setup", "dataset", "mode", "gen", "min", "fitness_test"]
    parts = [c[(c.setup == SETUP) & c.dataset.isin(CASE_GEGP) & (c["mode"] == "baseline") & (c.gen == LAST_GEN)]
             for c in pd.read_csv(path, usecols=cols, chunksize=500_000)]
    t = pd.concat(parts).groupby(["dataset", "algorithm"])[["min", "fitness_test"]].mean()
    return {"gap_max": float(100 * (t.fitness_test - t["min"]).abs().max()), "n": len(t)} if len(t) else {}


def pooling_claims(store: Path = results.STORE) -> list[tuple[str, str, bool]]:
    """Captions of Figures 10, 11, 13: the fractions contribute nearly equal numbers of evaluations, and the
    ordering the pooled curves show holds at every fraction in the last generation."""
    if not (_f(store, "generations.csv")).is_file():
        return []
    t = case_last_by_fraction(store).set_index(["dataset", "algorithm", "fraction"])
    fr = sorted(set(t.index.get_level_values(2)))
    ev = t.evals.groupby(level=[0, 1]).agg(lambda v: v.max() / v.min())
    w, cv = CASE_SIZE
    a, b = CASE_MC
    at = lambda ds, al, col, f: t[col].get((ds, al, f), np.nan)
    return [("pool/equal", "Figs 10, 11, 13: the three fractions contribute nearly equal numbers of evaluations "
             "(largest / smallest < 1.25 in the last generation)", bool((ev < 1.25).all())),
            ("pool/fig10", f"Fig 10: fake-hit ratio higher on {w} than on {cv} at every fraction, GE and GP",
             all(at(w, al, "fake_share", f) > at(cv, al, "fake_share", f) for al in ("GE", "GP") for f in fr)),
            ("pool/fig11", f"Fig 11: hit/miss program size larger on {a} than on {b} at every fraction",
             all(at(a, "GE", "hit_size", f) > at(b, "GE", "hit_size", f) for f in fr)),
            ("pool/fig13", "Fig 13: GP programs larger and GP's hits avoiding more work than GE's at every fraction",
             all(at(ds, "GP", c_, f) > at(ds, "GE", c_, f) for ds in CASE_GEGP for c_ in ("nodes", "work_avoided")
                 for f in fr))]


def case_claims(metrics: pd.DataFrame, props: pd.DataFrame, store: Path = results.STORE) -> list[tuple[str, str, bool]]:
    """The wording of §6.2 and §6.3 that depends on the data."""
    if not case_ready(metrics, props):
        return []
    c = case_generations(store)
    cost = case_costs(store).set_index(["dataset", "algorithm"]).ms
    st = metrics.groupby(["dataset", "algorithm"]).S_T.mean()
    diff = props.set_index("dataset").difficulty_gb
    w, cv = CASE_SIZE
    e, d = CASE_GEGP
    at = lambda ds, a, col, gen: _at(c, ds, a, col, gen)
    fake_max = lambda ds, a: c[(c.dataset == ds) & (c.algorithm == a) & (c.gen >= 1)].fake_share.max()
    out = []
    for a in ("GE", "GP"):
        out.append((f"case/{a}/small-slower", f"§6.2 {a}: S_T lower on {w} than on {cv}", st[(w, a)] < st[(cv, a)]))
        out.append((f"case/{a}/cost", f"§6.2 {a}: a full evaluation costs less on {w} than on {cv}",
                    cost[(w, a)] < cost[(cv, a)]))
        out.append((f"case/{a}/fake-rises", f"§6.2 {a}: 'on {w} the fake-hit ratio grew over the run'",
                    at(w, a, "fake_share", LAST_GEN) > at(w, a, "fake_share", 1)))
        out.append((f"case/{a}/fake-low", f"§6.2 {a}: 'on {cv} it stayed below what {w} reached at the end' (max over "
                    f"the run on {cv} < last generation on {w})", fake_max(cv, a) < at(w, a, "fake_share", LAST_GEN)))
        rho, pval, _n = dataset_rho(metrics[metrics.algorithm == a].merge(props[["dataset", "difficulty_gb"]], on="dataset"),
                                    "S_T", "difficulty_gb")
        out.append((f"difficulty/{a}", f"§6.3 {a}: 'S_T showed no significant correlation with difficulty'", pval > ALPHA))
    hit_rose = all(_at(c, d, al, "hit_rate", LAST_GEN) > _at(c, d, al, "hit_rate", 1) for d in (w, cv) for al in ("GE", "GP"))
    out.append(("case/hit-rises", f"§6.3: the hit rate rose from generation 1 to {LAST_GEN} on {w} and {cv} in both "
                "algorithms", hit_rose))
    out.append(("case/GP/same-falls", f"§6.3 GP: the share of structurally equivalent hits fell on {w} and {cv}",
                all(_at(c, d, "GP", "same_share", LAST_GEN) < _at(c, d, "GP", "same_share", 1) for d in (w, cv))))
    out.append(("case/GE/same", f"§6.2 GE: same-program share of hits fell on {w} and rose on {cv}",
                at(w, "GE", "same_share", LAST_GEN) < at(w, "GE", "same_share", 1)
                and at(cv, "GE", "same_share", LAST_GEN) > at(cv, "GE", "same_share", 1)))
    out.append(("gegp/difficulty-differs", f"§6.3: {e} is easier than {d} (GB balanced accuracy)", diff[e] > diff[d]))
    for ds in CASE_GEGP:
        out.append((f"gegp/{ds}/gp-faster", f"§6.3: GP faster than GE on {ds}", st[(ds, "GP")] > st[(ds, "GE")]))
        out.append((f"gegp/{ds}/gp-grows", f"§6.3: GP programs more than doubled in size on {ds}; GE stayed below 15 nodes",
                    at(ds, "GP", "nodes", LAST_GEN) > 2 * at(ds, "GP", "nodes", 1) and at(ds, "GE", "nodes", LAST_GEN) < 15))
        out.append((f"gegp/{ds}/diversity", f"§6.3: GP fitness diversity fell and GE's rose on {ds}",
                    at(ds, "GP", "diversity", LAST_GEN) < at(ds, "GP", "diversity", 1)
                    and at(ds, "GE", "diversity", LAST_GEN) > at(ds, "GE", "diversity", 1)))
        out.append((f"gegp/{ds}/gp-hits-rise", f"§6.3: GP hit rate rose over the run on {ds}",
                    at(ds, "GP", "hit_rate", LAST_GEN) > at(ds, "GP", "hit_rate", 1)))
        out.append((f"gegp/{ds}/work", f"§6.3: at the last generation GP's hits avoided a larger share of the work than GE's on {ds}",
                    at(ds, "GP", "work_avoided", LAST_GEN) > at(ds, "GE", "work_avoided", LAST_GEN)))
        out.append((f"gegp/{ds}/not-same", f"§6.3: on {ds} most GP hits late in the run came from different programs (same < 25%)",
                    at(ds, "GP", "same_share", LAST_GEN) < 0.25))
    out.append(("case/GP/small-below-one", f"§6.2: on {w} GP with FEC was slower than the baseline (S_T < 1)", st[(w, "GP")] < 1))
    ratio = [st[(ds, "GP")] / st[(ds, "GE")] for ds in CASE_GEGP]
    out.append(("gegp/similar-margin", "§6.3: 'GP was faster than GE by a similar margin on both' (GP/GE S_T ratios within "
                "0.15 of each other)", abs(ratio[0] - ratio[1]) < 0.15))
    for ds in CASE_GEGP:
        out.append((f"gegp/{ds}/ge-small-hits", f"§6.3: on {ds} GE hits avoided less work than their number suggests "
                    "(work share < hit rate in the last generation: hit programs smaller than missed ones)",
                    at(ds, "GE", "work_avoided", LAST_GEN) < at(ds, "GE", "hit_rate", LAST_GEN)))
    out.append(("gegp/cover-diverse", "§6.3: covertype GP kept a higher fitness diversity than both case datasets",
                at("covertype", "GP", "diversity", LAST_GEN) > max(at(ds, "GP", "diversity", LAST_GEN) for ds in CASE_GEGP)))
    return out


def case_figures(corpus, metrics, props, out_tables, **_) -> list[Path]:
    """fig_case_size.pdf (§6.2) and fig_case_gegp.pdf (§6.3): per-generation curves of the case-study datasets."""
    if not case_ready(metrics, props):
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter
    c = case_generations()
    c = c[c.gen >= 1]
    out_tables.mkdir(parents=True, exist_ok=True)
    meta = {"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None}

    def tidy(ax):
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlim(0, LAST_GEN + 1)

    # §6.2: rows GE / GP; columns hit rate, fake-hit ratio, same-program share; one line per table
    style = {CASE_SIZE[0]: dict(color=SERIES_COLOUR, linestyle="-"), CASE_SIZE[1]: dict(color=SECOND_COLOUR, linestyle="--")}
    fs.use()
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 3.9), sharex=True, sharey=True)
    for i, algo in enumerate(("GE", "GP")):
        for j, (col, label) in enumerate((("hit_rate", "hit rate"), ("fake_share", "fake-hit ratio"),
                                          ("same_share", "structurally equivalent hits"))):
            ax = axes[i, j]
            for ds in CASE_SIZE:
                x = c[(c.dataset == ds) & (c.algorithm == algo)]
                if col + "_sd" in x:                              # +- sd over the runs
                    ax.fill_between(x.gen, x[col] - x[col + "_sd"], x[col] + x[col + "_sd"], color=style[ds]["color"],
                                    alpha=0.10, linewidth=0)
                ax.plot(x.gen, x[col], linewidth=1.4, label=ds, **style[ds])
            tidy(ax)
            ax.set_ylim(0, 1)
            if i == 0:
                ax.set_title(label, fontsize=8, color=INK)
            if j == 0:
                ax.set_ylabel(algo)
            if i == 1:
                ax.set_xlabel("generation")
    axes[0, 1].legend(frameon=False, fontsize=7, loc="upper left")     # the empty top of the GE fake-share panel
    fig.tight_layout()
    p1 = out_tables / "fig_case_size.pdf"
    fs.save(fig, p1)
    plt.close(fig)

    # §6.3: rows electricity_cat / Diabetes130US; columns program size (log), fitness diversity, work avoided
    style = {"GE": dict(color=SERIES_COLOUR, linestyle="-"), "GP": dict(color=SECOND_COLOUR, linestyle="--")}
    fs.use()
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 3.9), sharex=True)
    for i, ds in enumerate(CASE_GEGP):
        for j, (col, label) in enumerate((("nodes", "mean program size (nodes)"), ("diversity", "fitness diversity"),
                                          ("work_avoided", "share of work avoided by hits"))):
            ax = axes[i, j]
            for algo in ("GE", "GP"):
                x = c[(c.dataset == ds) & (c.algorithm == algo)]
                if col + "_sd" in x:                              # +- sd over the runs
                    ax.fill_between(x.gen, x[col] - x[col + "_sd"], x[col] + x[col + "_sd"], color=style[algo]["color"],
                                    alpha=0.10, linewidth=0)
                ax.plot(x.gen, x[col], linewidth=1.4, label=algo, **style[algo])
            tidy(ax)
            if col == "nodes":
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                ax.yaxis.set_minor_formatter(NullFormatter())
            else:
                ax.set_ylim(0, 1)
            if i == 0:
                ax.set_title(label, fontsize=8, color=INK)
            if j == 0:
                ax.set_ylabel(ds)
            if i == 1:
                ax.set_xlabel("generation")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    p2 = out_tables / "fig_case_gegp.pdf"
    fs.save(fig, p2)
    plt.close(fig)
    return [p1, p2]


# ---------------------------------------------------------------- §6.5: a multiclass and a binary table

def mc_ready(metrics: pd.DataFrame, props: pd.DataFrame) -> bool:
    """§6.5 needs GE results on its four tables and their class counts; a store without them skips the section."""
    have = set(zip(metrics.dataset, metrics.algorithm))
    return "classes" in props and all((ds, "GE") in have for ds in MC_KEY)


def mc_run_seconds(store: Path = results.STORE) -> pd.DataFrame:
    """Per (dataset, mode, fraction): mean evaluation seconds and fingerprint rows per GE run (baseline: fraction NaN)."""
    r = pd.read_csv(_f(store, "runs.csv"), usecols=["algorithm", "setup", "dataset", "mode", "sampler", "fraction",
                                                 "eval_time_sec", "sample_size"])
    r = r[(r.setup == SETUP) & (r.algorithm == "GE") & r.dataset.isin(MC_KEY)
          & ((r["mode"] == "baseline") | ((r["mode"] == "fec") & (r.sampler == MAIN_SAMPLER)))]
    r = r.assign(fraction=r.fraction.where(r["mode"] != "baseline").round(2))
    return r.groupby(["dataset", "mode", "fraction"], dropna=False)[["eval_time_sec", "sample_size"]].mean().reset_index()


def _mc_pooled(c: pd.DataFrame, ds: str) -> pd.Series:
    """Totals over the generations 1-30 of one table, GE: hits, misses, work, same-program and fake hits."""
    x = c[(c.dataset == ds) & (c.algorithm == "GE") & (c.gen >= 1)]
    return x[["hits", "misses", "same", "fake", "hit_work", "miss_work"]].sum()


def mc_numbers(metrics: pd.DataFrame, props: pd.DataFrame, store: Path = results.STORE) -> dict:
    """Macros of §6.5: \\res Mc<Key><Name>, GE, random fingerprint rows."""
    if not mc_ready(metrics, props):
        return {}
    c, cost = case_generations(store), case_costs(store).set_index(["dataset", "algorithm"])
    sec = mc_run_seconds(store)
    ann = annotated(metrics[metrics.algorithm == "GE"])
    pr = props.set_index("dataset")
    pct = lambda x: "--" if pd.isna(x) else f"{100 * x:.0f}"
    out = {"McRowRatio": f"{pr.loc[CASE_MC[1], 'rows'] / pr.loc[CASE_MC[0], 'rows']:.1f}"}
    for ds, key in MC_KEY.items():
        p, g = f"Mc{key}", ann[ann.dataset == ds].set_index(ann[ann.dataset == ds].fraction.round(2))
        out[p + "Rows"], out[p + "Classes"] = _rows(pr.loc[ds, "rows"]), str(int(pr.loc[ds, "classes"]))
        out[p + "Difficulty"] = f"{pr.loc[ds, 'difficulty_gb']:.2f}"
        out[p + "Ms"] = _ms(cost.ms.get((ds, "GE"), np.nan))
        for f, word in FRACTIONS:
            out[p + "ST" + word] = _num(g.S_T.get(f, np.nan), 2)
        out[p + "STMean"] = _num(g.S_T.mean(), 2)
        out[p + "HitFive"] = _num(g.hit_rate.get(0.05, np.nan), 2)
        out[p + "BaseAcc"] = _num(g.baseline_accuracy.mean(), 2)
        eq = [f"{int(round(100 * f))}\\%" for f, v in g.verdict.items() if v == "equivalent"]
        out[p + "EquivFractions"] = "none" if not eq else eq[0] if len(eq) == 1 else ", ".join(eq[:-1]) + " and " + eq[-1]
        s_ = sec[sec.dataset == ds]
        out[p + "BaseSec"] = _int(round(s_[s_["mode"] == "baseline"].eval_time_sec.mean()))
        fec = s_[s_["mode"] == "fec"].set_index("fraction")
        for f, word in FRACTIONS:
            out[p + "FecSec" + word] = _int(round(fec.eval_time_sec.get(f, np.nan))) if f in fec.index else "--"
            out[p + "Prope" + word] = _num(g.bayes_rope.get(f, np.nan), 2)
            out[p + "PairedSd" + word] = _num(100 * g.dAcc_sd.get(f, np.nan), 1)
        out[p + "FpRowsOne"] = _int(fec.sample_size.get(0.01, np.nan)) if 0.01 in fec.index else "--"
        out[p + "BaseAccSd"] = _num(100 * g.baseline_accuracy_sd.mean(), 1)
        out[p + "Worse"] = str(int((g.verdict == "worse").sum()))
        out[p + "NodesFirst"] = _num(_at(c, ds, "GE", "nodes", 1), 0)
        out[p + "NodesLast"] = _num(_at(c, ds, "GE", "nodes", LAST_GEN), 0)
        t = _mc_pooled(c, ds)
        out[p + "HitSize"] = pct((t.hit_work / t.hits) / (t.miss_work / t.misses))
        out[p + "Work"] = pct(t.hit_work / (t.hit_work + t.miss_work))
        out[p + "Same"] = pct(t.same / t.hits)
        out[p + "Fake"] = f"{100 * t.fake / t.hits:.1f}"
    return out


def mc_claims(metrics: pd.DataFrame, props: pd.DataFrame, store: Path = results.STORE) -> list[tuple[str, str, bool]]:
    """The wording of §6.5 that depends on the data."""
    if not mc_ready(metrics, props):
        return []
    a, b = CASE_MC
    c, cost = case_generations(store), case_costs(store).set_index(["dataset", "algorithm"]).ms
    ge = annotated(metrics[metrics.algorithm == "GE"]).assign(fraction=lambda d: d.fraction.round(2))
    st = ge.set_index(["dataset", "fraction"]).S_T
    hit = ge.set_index(["dataset", "fraction"]).hit_rate
    pr = props.set_index("dataset")
    ta, tb = _mc_pooled(c, a), _mc_pooled(c, b)
    size = lambda t: (t.hit_work / t.hits) / (t.miss_work / t.misses)
    sec = mc_run_seconds(store).set_index(["dataset", "mode", "fraction"]).eval_time_sec
    base = lambda ds: sec[(ds, "baseline")].mean()
    return [("mc/classes", f"§6.5: {a} is multiclass, {b} binary", pr.loc[a, "classes"] > 2 and pr.loc[b, "classes"] == 2),
            ("mc/rows", f"§6.5: {b} has more than three times the rows of {a}", pr.loc[b, "rows"] > 3 * pr.loc[a, "rows"]),
            ("mc/faster", f"§6.5: S_T higher on {a} than on {b} at every fraction",
             all(st[(a, f)] > st[(b, f)] for f, _w in FRACTIONS)),
            ("mc/same-cost", f"§6.5: a full baseline evaluation costs about the same on both (within 15%)",
             abs(cost[(a, "GE")] / cost[(b, "GE")] - 1) < 0.15),
            ("mc/same-base-time", f"§6.5: baseline evaluation time per run about the same on both (within 15%)",
             abs(base(a) / base(b) - 1) < 0.15),
            ("mc/fec-time", f"§6.5: with FEC at 5% the evaluation time per run is lower on {a} than on {b}",
             sec[(a, "fec", 0.05)] < sec[(b, "fec", 0.05)]),
            ("mc/larger", f"§6.5: programs on {a} are more than three times as large as on {b} in the last generation",
             _at(c, a, "GE", "nodes", LAST_GEN) > 3 * _at(c, b, "GE", "nodes", LAST_GEN)),
            ("mc/grow", f"§6.5: programs on {a} nearly doubled in size over the run (> 1.8x), those on {b} stayed "
             "at about five nodes (3-6 in every generation)",
             _at(c, a, "GE", "nodes", LAST_GEN) > 1.8 * _at(c, a, "GE", "nodes", 1)
             and c[(c.dataset == b) & (c.algorithm == "GE") & (c.gen >= 1)].nodes.between(3, 6).all()),
            ("mc/similar-hits", "§6.5: similar hit rates at 5% (within 0.1)", abs(hit[(a, 0.05)] - hit[(b, 0.05)]) < 0.1),
            ("mc/hit-size", f"§6.5: relative to a missed program, a hit program was more than twice as large on {a} as on {b}",
             size(ta) > 2 * size(tb)),
            ("mc/work", f"§6.5: hits avoided a larger share of the work on {a}",
             ta.hit_work / (ta.hit_work + ta.miss_work) > tb.hit_work / (tb.hit_work + tb.miss_work)),
            ("mc/not-same", f"§6.5: few hits on {a} were the same program (< 10%), most on {b} (> 50%)",
             ta.same / ta.hits < 0.10 and tb.same / tb.hits > 0.50),
            ("mc/few-fakes", "§6.5: fake hits rare on both (< 5% of the hits)", ta.fake / ta.hits < 0.05 and tb.fake / tb.hits < 0.05),
            ("mc/nearly-twice", f"§6.5: S_T on {a} nearly twice that on {b} at every fraction (1.6-2.1x)",
             all(1.6 <= st[(a, f)] / st[(b, f)] <= 2.1 for f, _w in FRACTIONS)),
            ("mc/work-words", f"§6.5: hits avoided about two thirds of the work on {a} (0.6-0.75) and less than half on {b}",
             0.6 <= ta.hit_work / (ta.hit_work + ta.miss_work) <= 0.75 and tb.hit_work / (tb.hit_work + tb.miss_work) < 0.5),
            ("mc/size-words", f"§6.5: a hit program was nearly as large as a missed one on {a} (> 0.7) and about a third "
             f"of it on {b} (0.25-0.4)", size(ta) > 0.7 and 0.25 <= size(tb) <= 0.4),
            ("mc/weak-baseline", f"§6.5: GE's baseline accuracy on {a} well below gradient boosting (> 0.1 lower)",
             ge[ge.dataset == a].baseline_accuracy.mean() < pr.loc[a, "difficulty_gb"] - 0.1)] + _mc_limit_claims(ge, pr, c, sec)


def _mc_limit_claims(ge, pr, c, sec) -> list[tuple[str, str, bool]]:
    """The closing paragraph of §6.5: the larger five-class table gains less and keeps no confirmed accuracy."""
    a, (big, small) = CASE_MC[0], MC_LIMITS
    x = ge.set_index(["dataset", "fraction"])
    t = _mc_pooled(c, big)
    v = ge[ge.dataset == big]
    return [("mc/lhc-largest", f"§6.5: {big} is the largest table of the corpus", pr.rows.max() == pr.loc[big, "rows"]),
            ("mc/lhc-multiclass", f"§6.5: {big} and {small} have more than two classes",
             pr.loc[big, "classes"] > 2 and pr.loc[small, "classes"] > 2),
            ("mc/lhc-slower", f"§6.5: S_T lower on {big} than on {a} at every fraction",
             all(x.S_T[(big, f)] < x.S_T[(a, f)] for f, _w in FRACTIONS)),
            ("mc/lhc-fewer-hits", f"§6.5: hit rate on {big} lower than on {a} at 5% (by > 0.1)",
             x.hit_rate[(big, 0.05)] < x.hit_rate[(a, 0.05)] - 0.1),
            ("mc/lhc-full-size", f"§6.5: on {big} hits were of the same kind: a hit program > 80% of a missed one, < 10% same program",
             (t.hit_work / t.hits) / (t.miss_work / t.misses) > 0.8 and t.same / t.hits < 0.10),
            ("mc/lhc-no-equiv", f"§6.5: no equivalence on {big} at any fraction, and none worse",
             (v.verdict != "equivalent").all() and (v.verdict != "worse").all()),
            ("mc/lhc-loose", f"§6.5: on {big} the paired accuracy difference varied by more than two points at every fraction",
             (v.dAcc_sd > 0.02).all()),
            ("mc/lhc-long", f"§6.5: an FEC run on {big} still spent more than half the baseline's evaluation time",
             all(sec[(big, "fec", f)] > 0.5 * sec[(big, "baseline")].mean() for f, _w in FRACTIONS)),
            ("mc/lhc-fraction", f"§6.5: on {big} S_T fell as the fraction (and the fingerprint) grew",
             x.S_T[(big, 0.01)] > x.S_T[(big, 0.05)] > x.S_T[(big, 0.10)]),
            ("mc/sleep-equiv", f"§6.5: {small} reached equivalence at 5% and 10%",
             set(ge[(ge.dataset == small) & (ge.verdict == "equivalent")].fraction) == {0.05, 0.10})]


def mc_figure(corpus, metrics, props, out_tables, **_) -> list[Path]:
    """fig_case_mc.pdf (§6.5): GE on the multiclass and the binary table over the generations, diagnostic runs of
    the three fractions pooled: program size, hit rate, size of a hit program relative to a missed one, and the
    share of the evaluation work the hits avoided."""
    if not mc_ready(metrics, props):
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = case_generations()
    c = c[(c.gen >= 1) & (c.algorithm == "GE")]
    style = {CASE_MC[0]: dict(color=SERIES_COLOUR, linestyle="-"), CASE_MC[1]: dict(color=SECOND_COLOUR, linestyle="--")}
    panels = (("nodes", "program size\n(nodes)"), ("hit_rate", "hit rate\n"),
              ("hit_size", "hit program size /\nmissed program size"), ("work_avoided", "share of work\navoided by hits"))
    fs.use()
    fig, axes = plt.subplots(1, 4, figsize=(6.3, 2.3), sharex=True)
    for ax, (col, label) in zip(axes, panels):
        for ds in CASE_MC:
            x = c[c.dataset == ds]
            if col + "_sd" in x:                              # +- sd over the runs
                ax.fill_between(x.gen, x[col] - x[col + "_sd"], x[col] + x[col + "_sd"], color=style[ds]["color"],
                                alpha=0.10, linewidth=0)
            ax.plot(x.gen, x[col], linewidth=1.4, label=ds, **style[ds])
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlim(0, LAST_GEN + 1)
        if col != "nodes":
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(0, None)
        ax.set_title(label, fontsize=8, color=INK)
        ax.set_xlabel("generation")
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")     # the empty bottom of the hit-rate panel
    fig.tight_layout()
    out_tables.mkdir(parents=True, exist_ok=True)
    path = out_tables / "fig_case_mc.pdf"
    fs.save(fig, path)
    plt.close(fig)
    return [path]


# ---------------------------------------------------------------- §5: size against the cache factors

SIZE_CONTROLS = (("hit_rate", "hit rate"), ("work_avoided", "share of work avoided"),
                 ("hit_size_ratio", "hit/miss program size"), ("classes", "number of classes"))


def _partial_spearman(df: pd.DataFrame, y: str, x: str, controls: list[str]) -> tuple[float, float]:
    """Spearman correlation of y and x after removing the linear effect of the controls from their ranks."""
    from scipy.stats import pearsonr, rankdata, t
    rank = lambda c: rankdata(df[c]).astype(float)
    z = np.column_stack([np.ones(len(df))] + [rank(c) for c in controls])
    res = lambda v: v - z @ np.linalg.lstsq(z, v, rcond=None)[0]
    r = pearsonr(res(rank(x)), res(rank(y)))[0]
    dof = len(df) - 2 - len(controls)                  # one degree of freedom lost per control
    p = 2 * t.sf(abs(r) * np.sqrt(dof / max(1 - r * r, 1e-300)), dof)
    return float(r), float(p)


def size_partials(ann: pd.DataFrame, props: pd.DataFrame, store: Path = results.STORE) -> dict:
    """Per algorithm, one value per dataset (mean over fractions, main sampler): the partial Spearman correlation of
    S_T with the number of rows under each control of SIZE_CONTROLS, the partial correlation of S_T with the share
    of work avoided given the rows, and the share of the S_T ranking explained by rows alone and with work avoided."""
    if not (_f(store, "generations.csv")).is_file() or "classes" not in props:
        return {}
    w = hit_work(store)
    w = w[w.sampler == MAIN_SAMPLER].drop(columns="sampler")
    x = ann.assign(fraction=ann.fraction.round(2)).merge(w, on=["algorithm", "dataset", "fraction"])
    out = {}
    for algo, g in x.groupby("algorithm"):
        per = (g.groupby("dataset")[["S_T", "hit_rate", "work_avoided", "hit_size_ratio"]].mean().reset_index()
               .merge(props[["dataset", "rows", "classes"]], on="dataset"))
        if len(per) < 5:
            continue
        part = [_partial_spearman(per, "S_T", "rows", [c]) for c, _l in SIZE_CONTROLS]
        rk = lambda c: per[c].rank().to_numpy(float)
        def r2(cols):
            z = np.column_stack([np.ones(len(per))] + [rk(c) for c in cols])
            y = rk("S_T")
            e = y - z @ np.linalg.lstsq(z, y, rcond=None)[0]
            return 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out[algo] = {"partial": part, "work_given_rows": _partial_spearman(per, "S_T", "work_avoided", ["rows"]),
                     "r2_rows": r2(["rows"]), "r2_rows_work": r2(["rows", "work_avoided"]), "n": len(per)}
    return out


# ---------------------------------------------------------------- §5: what more seeds could change

def undecided_limit(store: Path = results.STORE, keep=None) -> dict:
    """Configurations (random rows) not equivalent with the observed 30 pairs, and how many would reach
    P(rope) >= THRESHOLD with infinitely many seeds if the observed mean and sd of the paired accuracy differences
    held: under Eq. (7) the posterior scale tends to sd * sqrt(n_test / n_train)."""
    from scipy import stats
    r = pd.read_csv(_f(store, "runs.csv"), usecols=["algorithm", "setup", "dataset", "mode", "sampler", "fraction", "seed",
                                                "test_accuracy"])
    r = r[(r.setup == SETUP) & (r.dataset.isin(keep) if keep is not None else True)]
    base = r[r["mode"] == "baseline"][["algorithm", "dataset", "seed", "test_accuracy"]]
    fec = r[(r["mode"] == "fec") & (r.sampler == MAIN_SAMPLER)].assign(fraction=lambda d: d.fraction.round(2))
    x = fec.merge(base, on=["algorithm", "dataset", "seed"], suffixes=("", "_base"))
    m, ratio = results.EQUIVALENCE_MARGIN, 0.25                       # n_test / n_train for the 80/20 split
    rows = []
    for _k, g in x.groupby(["algorithm", "dataset", "fraction"]):
        d = (g.test_accuracy - g.test_accuracy_base).to_numpy(float)
        n, mu, sd = len(d), d.mean(), d.std(ddof=1)
        now = stats.t(n - 1, loc=mu, scale=sd * np.sqrt(1 / n + ratio))
        lim = stats.norm(loc=mu, scale=sd * np.sqrt(ratio))
        rows.append({"now": now.cdf(m) - now.cdf(-m), "limit": lim.cdf(m) - lim.cdf(-m), "sd": sd})
    t = pd.DataFrame(rows, columns=["now", "limit", "sd"])
    und = t[t.now < THRESHOLD]
    return {"n": len(t), "undecided": len(und), "reach": int((und.limit >= THRESHOLD).sum()),
            "sd_min": 100 * und.sd.min(), "sd_max": 100 * und.sd.max()}


# ---------------------------------------------------------------- §6.2/§6.5 and Table S6: what the hits skip

WORK_EXAMPLES = ("credit", "california")    # §6.2: binary tables with high GE hit rates but small speedups
WORK_SMALL = ("wine_quality", "phoneme_balanced")
_WORK_CACHE: dict = {}


def hit_work(store: Path = results.STORE) -> pd.DataFrame:
    """Per (algorithm, dataset, sampler, fraction), from the diagnostic runs, generations 1-30 pooled: the mean size
    (operator calls) of a hit program relative to a missed one, and the share of the full-evaluation work that the
    hits avoided. The fingerprint, computed for every candidate, is not counted."""
    path = _f(store, "generations.csv")
    if path in _WORK_CACHE:
        return _WORK_CACHE[path]
    cols = ["algorithm", "setup", "dataset", "mode", "sampler", "fraction", "gen", "fec_hits", "fec_misses",
            "fec_hit_operators", "fec_miss_operators"]
    parts = [c[(c.setup == SETUP) & (c["mode"] == "fec_diagnostic") & (c.gen >= 1)]
             for c in pd.read_csv(path, usecols=cols, chunksize=1_000_000)]
    g = pd.concat(parts)
    g = g.assign(fraction=g.fraction.round(2), hit_work=g.fec_hits * g.fec_hit_operators.fillna(0),
                 miss_work=g.fec_misses * g.fec_miss_operators.fillna(0))
    t = g.groupby(["algorithm", "dataset", "sampler", "fraction"])[["fec_hits", "fec_misses", "hit_work", "miss_work"]].sum()
    out = pd.DataFrame({"hit_size_ratio": (t.hit_work / t.fec_hits) / (t.miss_work / t.fec_misses),
                        "work_avoided": t.hit_work / (t.hit_work + t.miss_work)}).reset_index()
    _WORK_CACHE[path] = out
    return out


def _work_main(ann: pd.DataFrame, props: pd.DataFrame, store: Path) -> pd.DataFrame:
    w = hit_work(store)
    w = w[w.sampler == MAIN_SAMPLER].drop(columns="sampler")
    x = ann.assign(fraction=ann.fraction.round(2)).merge(w, on=["algorithm", "dataset", "fraction"], how="inner")
    return x.merge(props[["dataset", "classes"]], on="dataset", how="left") if "classes" not in x else x


# §6.6: matched pairs - similar size and hit rate, different hit/miss program size
SIZE_PAIRS = (("GE", 0.01, "sleep", "electricity"), ("GE", 0.05, "dry_bean", "phoneme_balanced"))
SIZE_PAIR_KEY = {"sleep": "Sleep", "electricity": "Elec", "dry_bean": "Bean", "phoneme_balanced": "Phon"}
# §6.6, GP: where a larger hit/miss size did not give a larger speedup
GP_COUNTER = (("GP", 0.05, "electricity", "electricity_cat"), ("GP", 0.05, "wine_quality", "dry_bean"))
GP_COUNTER_KEY = {"electricity": "Elec", "electricity_cat": "ElecCat", "wine_quality": "Wine", "dry_bean": "Bean"}


def _work_table(ann: pd.DataFrame, props: pd.DataFrame, store: Path) -> pd.DataFrame:
    x = _work_main(ann, props, store)
    return x.merge(props[["dataset", "rows"]], on="dataset", how="left", suffixes=("", "_p")) if "rows" not in x else x


def similar_pairs(ann: pd.DataFrame, props: pd.DataFrame, store: Path = results.STORE) -> pd.DataFrame:
    """All pairs of datasets of one algorithm and fraction with rows within a factor of 1.5 and hit rates within
    0.05: which has the larger hit/miss size ratio, and whether it also has the larger S_T."""
    import itertools
    x = _work_table(ann, props, store)
    out = []
    for (a, f), g in x.groupby(["algorithm", "fraction"]):
        for u, v in itertools.combinations(g.itertuples(), 2):
            if max(u.rows, v.rows) / min(u.rows, v.rows) <= 1.5 and abs(u.hit_rate - v.hit_rate) <= 0.05:
                hi, lo = (u, v) if u.hit_size_ratio > v.hit_size_ratio else (v, u)
                out.append({"algorithm": a, "fraction": f, "binary": hi.classes == 2 and lo.classes == 2,
                            "d_size": hi.hit_size_ratio - lo.hit_size_ratio, "st_follows": hi.S_T > lo.S_T})
    return pd.DataFrame(out)


def size_pair_numbers(ann: pd.DataFrame, props: pd.DataFrame, store: Path = results.STORE) -> dict:
    """Macros \\res SizePair...: the matched pairs of SIZE_PAIRS and the binary-only pair statistics."""
    if not (_f(store, "generations.csv")).is_file() or "classes" not in props:
        return {}
    x = _work_table(ann, props, store).set_index(["algorithm", "fraction", "dataset"])
    out = {}
    for a, f, d1, d2 in SIZE_PAIRS:
        for d in (d1, d2):
            if (a, f, d) not in x.index:
                return {}
            r, k = x.loc[(a, f, d)], "SizePair" + SIZE_PAIR_KEY[d]
            out[k + "Rows"], out[k + "Hit"] = _rows(r.rows), f"{r.hit_rate:.2f}"
            out[k + "Ratio"], out[k + "Work"] = f"{100 * r.hit_size_ratio:.0f}", f"{100 * r.work_avoided:.0f}"
            out[k + "ST"], out[k + "Classes"] = f"{r.S_T:.2f}", str(int(r.classes))
    sp = similar_pairs(ann, props, store)
    b = sp[(sp.algorithm == "GE") & sp.binary] if len(sp) else sp
    if len(b):
        out["SizePairBinN"], out["SizePairBinAgree"] = str(len(b)), str(int(b.st_follows.sum()))
        out["SizePairBinDiffMax"] = f"{100 * b.d_size.max():.0f}"
    g = sp[sp.algorithm == "GP"] if len(sp) else sp
    if len(g):
        out["SizePairGPN"], out["SizePairGPAgree"] = str(len(g)), str(int(g.st_follows.sum()))
    for a, f, d1, d2 in GP_COUNTER:
        for d in (d1, d2):
            if (a, f, d) in x.index:
                r, k = x.loc[(a, f, d)], "SizeGP" + GP_COUNTER_KEY[d]
                out[k + "Hit"], out[k + "Ratio"] = f"{r.hit_rate:.2f}", f"{100 * r.hit_size_ratio:.0f}"
                out[k + "Work"], out[k + "ST"] = f"{100 * r.work_avoided:.0f}", f"{r.S_T:.2f}"
    return out


def size_pair_claims(ann: pd.DataFrame, props: pd.DataFrame, store: Path = results.STORE) -> list[tuple[str, str, bool]]:
    if not (_f(store, "generations.csv")).is_file() or "classes" not in props:
        return []
    x = _work_table(ann, props, store).set_index(["algorithm", "fraction", "dataset"])
    out = []
    for a, f, d1, d2 in SIZE_PAIRS:
        if (a, f, d1) not in x.index or (a, f, d2) not in x.index:
            return []
        u, v = x.loc[(a, f, d1)], x.loc[(a, f, d2)]
        out.append((f"sizepair/{d1}-{d2}", f"§6.6: {d1} vs {d2} ({a}, {f:.0%}): similar size (within 1.5x) and hit rate "
                    "(within 0.05), larger hit/miss size ratio, more work avoided and higher S_T for the first",
                    max(u.rows, v.rows) / min(u.rows, v.rows) <= 1.5 and abs(u.hit_rate - v.hit_rate) <= 0.05
                    and u.hit_size_ratio > v.hit_size_ratio and u.work_avoided > v.work_avoided and u.S_T > v.S_T))
    (a, f, e, ec), (_a2, _f2, w, bn) = GP_COUNTER
    if all((a, f, d) in x.index for d in (e, ec, w, bn)):
        E, C, W, B = (x.loc[(a, f, d)] for d in (e, ec, w, bn))
        out.append(("sizepair/gp-elec", f"§6.6 GP: {e} and {ec} at {f:.0%}: same rows, hit rates within 0.05, larger size "
                    f"ratio and more work avoided on {e}, yet the smaller S_T",
                    E.rows == C.rows and abs(E.hit_rate - C.hit_rate) <= 0.05 and E.hit_size_ratio > C.hit_size_ratio
                    and E.work_avoided > C.work_avoided and E.S_T < C.S_T))
        out.append(("sizepair/gp-small", f"§6.6 GP: {w} has the larger size ratio than {bn} at {f:.0%} but S_T at or below "
                    "one and below that of the other", W.hit_size_ratio > B.hit_size_ratio and W.S_T <= 1.0 and W.S_T < B.S_T))
    sp = similar_pairs(ann, props, store)
    g = sp[sp.algorithm == "GP"] if len(sp) else sp
    if len(g):
        out.append(("sizepair/gp-weak", "§6.6 GP: between similar GP tables S_T follows the size ratio in no more than about "
                    "half of the pairs", g.st_follows.mean() <= 0.55))
    b = sp[(sp.algorithm == "GE") & sp.binary] if len(sp) else sp
    if len(b):
        out.append(("sizepair/binary-weak", "§6.6: between similar binary GE tables the size ratio differs little (< 0.15) "
                    "and S_T follows it in no more than about two thirds of the pairs",
                    b.d_size.max() < 0.15 and b.st_follows.mean() <= 0.7))
    return out


SIZE_FIG = (SIZE_PAIRS[0], GP_COUNTER[0])               # Figure (§6.6): one GE and one GP pair over the generations
_SIZE_FIG_CACHE: dict = {}


def size_pair_generations(store: Path = results.STORE) -> pd.DataFrame:
    """Per (algorithm, dataset, generation) for the pairs of SIZE_FIG at their fraction, diagnostic runs, main sampler:
    hit rate, mean size (operator calls) of hit and of missed programs, their ratio, share of work avoided; pooled
    over the 30 runs, with the sd between runs of each (from per-run values)."""
    path = _f(store, "generations.csv")
    if path in _SIZE_FIG_CACHE:
        return _SIZE_FIG_CACHE[path]
    want = {(a, d, f) for a, f, d1, d2 in SIZE_FIG for d in (d1, d2)}
    cols = ["algorithm", "setup", "dataset", "mode", "sampler", "fraction", "seed", "gen", "fec_hits", "fec_misses",
            "fec_hit_operators", "fec_miss_operators"]
    names = {d for _a, d, _fr in want}
    parts = [c[(c.setup == SETUP) & (c["mode"] == "fec_diagnostic") & (c.sampler == MAIN_SAMPLER) & c.dataset.isin(names)]
             for c in pd.read_csv(path, usecols=cols, chunksize=1_000_000)]
    g = pd.concat(parts).assign(fraction=lambda d: d.fraction.round(2))
    g = g[np.array([(a, d, f) in want for a, d, f in zip(g.algorithm, g.dataset, g.fraction)]) & (g.gen >= 1).to_numpy()]
    hw, mw = g.fec_hits * g.fec_hit_operators, g.fec_misses * g.fec_miss_operators
    ev = (g.fec_hits + g.fec_misses).where(lambda e: e > 0)
    g = g.assign(hw=hw, mw=mw, r_hit=g.fec_hits / ev, r_hsize=g.fec_hit_operators, r_msize=g.fec_miss_operators,
                 r_ratio=g.fec_hit_operators / g.fec_miss_operators, r_work=hw / (hw + mw).where(lambda w: w > 0))
    t = (g.groupby(["algorithm", "dataset", "gen"])
         .agg(h=("fec_hits", "sum"), m=("fec_misses", "sum"), hw=("hw", "sum"), mw=("mw", "sum"),
              hit_rate_sd=("r_hit", "std"), hit_size_sd=("r_hsize", "std"), miss_size_sd=("r_msize", "std"),
              ratio_sd=("r_ratio", "std"), work_sd=("r_work", "std")).reset_index())
    t = t.assign(hit_rate=t.h / (t.h + t.m), hit_size=t.hw / t.h, miss_size=t.mw / t.m)
    t = t.assign(ratio=t.hit_size / t.miss_size, work=t.hw / (t.hw + t.mw))
    _SIZE_FIG_CACHE[path] = t
    return t


def size_pair_figure(corpus, metrics, props, out_tables, **_) -> list[Path]:
    """fig_size_pairs.pdf (§6.6): the GE pair (top) and the GP pair (bottom) of SIZE_FIG over the generations: hit
    rate, mean size of hit (solid) and missed (dashed) programs, their ratio, share of work avoided."""
    if not (_f(results.STORE, "generations.csv")).is_file():
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = size_pair_generations()
    if t.empty:
        return []
    fs.use()
    fig, axes = plt.subplots(2, 4, figsize=(6.3, 3.9), sharex=True)
    colours = {}                                               # one colour per dataset across both rows
    for _a, _fr, d1, d2 in SIZE_FIG:
        for d in (d1, d2):
            colours.setdefault(d, (SERIES_COLOUR, SECOND_COLOUR, THIRD_COLOUR)[len(colours)])
    for r, (a, f, d1, d2) in enumerate(SIZE_FIG):
        for i, d in enumerate((d1, d2)):
            x = t[(t.algorithm == a) & (t.dataset == d)]
            c = colours[d]
            for ax, col, sd in ((axes[r, 0], "hit_rate", "hit_rate_sd"), (axes[r, 2], "ratio", "ratio_sd"),
                                (axes[r, 3], "work", "work_sd")):
                ax.fill_between(x.gen, x[col] - x[sd], x[col] + x[sd], color=c, alpha=0.10, linewidth=0)
                ax.plot(x.gen, x[col], color=c, linewidth=1.4, label=d)
            ax = axes[r, 1]
            for col, sd, ls, what in (("hit_size", "hit_size_sd", "-", "hits"), ("miss_size", "miss_size_sd", "--", "misses")):
                ax.fill_between(x.gen, x[col] - x[sd], x[col] + x[sd], color=c, alpha=0.08, linewidth=0)
                ax.plot(x.gen, x[col], color=c, linestyle=ls, linewidth=1.3)
        axes[r, 0].set_ylabel(f"{a}, {FRACTION_WORD[f].replace(chr(92), '')}")
        axes[r, 2].axhline(1.0, color=INK_SECONDARY, linestyle=(0, (4, 3)), linewidth=0.8, zorder=0)
        for ax in axes[r]:
            ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_xlim(0, LAST_GEN + 1)
        for ax in (axes[r, 0], axes[r, 3]):
            ax.set_ylim(0, 1)
        axes[r, 1].set_ylim(bottom=0)
        axes[r, 2].set_ylim(bottom=0)
        axes[r, 0].legend(frameon=False, fontsize=6.5, loc="lower right")
    from matplotlib.lines import Line2D                         # line styles of the size panels, in one key
    axes[1, 1].legend([Line2D([], [], color=INK, linewidth=1.2), Line2D([], [], color=INK, linewidth=1.2, linestyle="--")],
                      ["hit programs", "missed programs"], frameon=False, fontsize=6.5, loc="upper left")
    for j, title in enumerate(("hit rate", "program size\n(operator calls)", "hit / missed\nprogram size",
                               "share of work\navoided by hits")):
        axes[0, j].set_title(title, fontsize=8, color=INK)
    for ax in axes[1]:
        ax.set_xlabel("generation")
    fig.tight_layout()
    out_tables.mkdir(parents=True, exist_ok=True)
    path = out_tables / "fig_size_pairs.pdf"
    fs.save(fig, path)
    plt.close(fig)
    return [path]


def size_fig_numbers(store: Path = results.STORE) -> dict:
    """Macros \\res SizeFig...: what the figure shows - first and last generation of each pair."""
    if not (_f(store, "generations.csv")).is_file():
        return {}
    t = size_pair_generations(store)
    out = {}
    keys = {("GE", "sleep"): "GESleep", ("GE", "electricity"): "GEElec", ("GP", "electricity"): "GPElec",
            ("GP", "electricity_cat"): "GPElecCat"}
    for (a, d), k in keys.items():
        x = t[(t.algorithm == a) & (t.dataset == d)].set_index("gen")
        if x.empty:
            return {}
        out[f"SizeFig{k}RatioFirst"], out[f"SizeFig{k}RatioLast"] = f"{x.ratio.iloc[0]:.2f}", f"{x.ratio.iloc[-1]:.2f}"
        out[f"SizeFig{k}RatioMin"], out[f"SizeFig{k}RatioMax"] = f"{x.ratio.min():.2f}", f"{x.ratio.max():.2f}"
        out[f"SizeFig{k}HitFirst"], out[f"SizeFig{k}HitLast"] = f"{x.hit_size.iloc[0]:.1f}", f"{x.hit_size.iloc[-1]:.1f}"
        out[f"SizeFig{k}MissFirst"], out[f"SizeFig{k}MissLast"] = f"{x.miss_size.iloc[0]:.1f}", f"{x.miss_size.iloc[-1]:.1f}"
        above = x[x.ratio > 1]
        out[f"SizeFig{k}AboveGen"] = str(int(above.index.min())) if len(above) else "--"
        out[f"SizeFig{k}WorkLast"], out[f"SizeFig{k}HitRateLast"] = f"{100 * x.work.iloc[-1]:.0f}", f"{x.hit_rate.iloc[-1]:.2f}"
    return out


def size_fig_claims(store: Path = results.STORE) -> list[tuple[str, str, bool]]:
    if not (_f(store, "generations.csv")).is_file():
        return []
    t = size_pair_generations(store)
    get = lambda a, d: t[(t.algorithm == a) & (t.dataset == d)].set_index("gen")
    s_, e_, ge_, gc_ = get("GE", "sleep"), get("GE", "electricity"), get("GP", "electricity"), get("GP", "electricity_cat")
    if any(x.empty for x in (s_, e_, ge_, gc_)):
        return []
    return [("sizefig/ge-sleep-flat", "Fig size pairs: on sleep (GE 1%) hit and missed programs are about the same size "
             "throughout (ratio 0.9-1.0 in every generation)", s_.ratio.between(0.9, 1.0).all()),
            ("sizefig/ge-elec-half", "Fig size pairs: on electricity (GE 1%) hits are about half the size of misses "
             "throughout (ratio < 0.6 in every generation) and the missed programs grow over the run",
             (e_.ratio < 0.6).all() and e_.miss_size.iloc[-1] > e_.miss_size.iloc[0]),
            ("sizefig/gp-crosses", "Fig size pairs: with GP (5%) the ratio starts below one and ends above one on both "
             "tables, higher on electricity", ge_.ratio.iloc[0] < 1 and gc_.ratio.iloc[0] < 1 and ge_.ratio.iloc[-1] > 1
             and gc_.ratio.iloc[-1] > 1 and ge_.ratio.iloc[-1] > gc_.ratio.iloc[-1]),
            ("sizefig/ge-work", "Fig size pairs: in the last generation the share of work avoided is close to the hit rate on "
             "sleep (within 0.05) and well below it on electricity (more than 0.15 lower)",
             abs(s_.work.iloc[-1] - s_.hit_rate.iloc[-1]) <= 0.05 and e_.hit_rate.iloc[-1] - e_.work.iloc[-1] > 0.15)]


def size_pair_table(corpus, metrics, props, out_tables, min_rows=0, **_) -> list[Path]:
    """size_pairs.tex (§6.6): the matched pairs side by side."""
    keep = main_text_datasets(corpus, props, min_rows)
    ann = annotated(metrics[metrics.dataset.isin(keep)])
    if not (_f(results.STORE, "generations.csv")).is_file() or "classes" not in props:
        return []
    x = _work_table(ann, props, results.STORE).set_index(["algorithm", "fraction", "dataset"])
    rows_ = []
    for k, (a, f, d1, d2) in enumerate(SIZE_PAIRS):
        for i, d in enumerate((d1, d2)):
            if (a, f, d) not in x.index:
                continue
            r = x.loc[(a, f, d)]
            lead = (("\\addlinespace[3pt]" if k and i == 0 else "") + (f"{a}, {FRACTION_WORD[f]}" if i == 0 else ""))
            rows_.append(" & ".join([lead, _tt(d), _rows(r.rows), str(int(r.classes)), f"{r.hit_rate:.2f}",
                                     f"{100 * r.hit_size_ratio:.0f}\\%", f"{100 * r.work_avoided:.0f}\\%",
                                     f"{r.S_T:.2f}"]))
    t = _table("tab:size-pairs",
               "Pairs of tables of similar size and similar hit rate whose cache hits skip programs of different size "
               "(random fingerprint rows). Hit/miss size: mean size, in operator calls, of a program that hit the "
               "cache relative to one that missed it; work avoided: share of the full-evaluation work the hits "
               "avoided (both from the diagnostic runs, generations 1--30 pooled, fingerprint not counted).",
               "llrrrrrr", "Alg., fraction & Dataset & Rows & Cl. & Hit rate & Hit/miss size & Work avoided & $S_T$",
               rows_, "results/store/metrics.csv")
    return [write(out_tables, "size_pairs.tex", t)]


def work_numbers(ann: pd.DataFrame, props: pd.DataFrame, store: Path = results.STORE) -> dict:
    """Macros \\res Work...: hit/miss size ratio ranges, the correlation of S_T with work avoided and with hit rate on
    the tables from the GE threshold upwards, and the §6.2 examples (GE, random rows)."""
    from scipy.stats import spearmanr
    if not (_f(store, "generations.csv")).is_file() or "classes" not in props:
        return {}
    x = _work_main(ann, props, store)
    ge, gp = x[x.algorithm == "GE"], x[x.algorithm == "GP"]
    if ge.empty:
        return {}
    pct = lambda v: f"{100 * v:.0f}"
    out = {"WorkGEBinMin": pct(ge[ge.classes == 2].hit_size_ratio.min()), "WorkGEBinMax": pct(ge[ge.classes == 2].hit_size_ratio.max()),
           "WorkGEMultiMin": pct(ge[ge.classes > 2].hit_size_ratio.min()), "WorkGEMultiMax": pct(ge[ge.classes > 2].hit_size_ratio.max())}
    if len(gp):
        out["WorkGPMin"], out["WorkGPMax"] = pct(gp.hit_size_ratio.min()), pct(gp.hit_size_ratio.max())
        out["WorkGPAboveOne"] = str(int((gp.hit_size_ratio > 1).sum()))
        out["WorkGPN"] = str(len(gp))
    thr = all_faster_threshold(ann[ann.algorithm == "GE"])
    big = ge[ge.rows >= thr[1]] if thr else ge
    for col, key in (("work_avoided", "Work"), ("hit_rate", "Hit")):
        rho, pv = spearmanr(big.S_T, big[col])
        out[f"WorkRho{key}"], out[f"WorkRho{key}P"] = f"{rho:.2f}", _p(pv)
    out["WorkRhoN"] = str(len(big))
    five = ge[ge.fraction == 0.05].set_index("dataset")
    for ds in WORK_EXAMPLES + WORK_SMALL + ("sleep",):
        if ds in five.index:
            k = "Work" + "".join(w.capitalize() for w in ds.split("_"))
            out[k + "Hit"], out[k + "Ratio"] = _num(five.hit_rate[ds], 2), pct(five.hit_size_ratio[ds])
            out[k + "Avoided"], out[k + "ST"] = pct(five.work_avoided[ds]), _num(five.S_T[ds], 2)
    return out


def work_claims(ann: pd.DataFrame, props: pd.DataFrame, store: Path = results.STORE) -> list[tuple[str, str, bool]]:
    from scipy.stats import spearmanr
    if not (_f(store, "generations.csv")).is_file() or "classes" not in props:
        return []
    x = _work_main(ann, props, store)
    ge, gp = x[x.algorithm == "GE"], x[x.algorithm == "GP"]
    if ge.empty:
        return []
    thr = all_faster_threshold(ann[ann.algorithm == "GE"])
    big = ge[ge.rows >= thr[1]] if thr else ge
    rw, pw = spearmanr(big.S_T, big.work_avoided)
    rh, _ph = spearmanr(big.S_T, big.hit_rate)
    five = ge[ge.fraction == 0.05]
    binary = five[five.classes == 2].sort_values("hit_rate", ascending=False, kind="stable")
    f5 = five.set_index("dataset")
    out = [("work/ge-binary-small", "§6.5: on every binary table a GE hit program was less than half the size of a missed one",
            (ge[ge.classes == 2].hit_size_ratio < 0.5).all()),
           ("work/ge-multi-large", "§6.5: on every multiclass table a GE hit program was more than 70% of a missed one",
            (ge[ge.classes > 2].hit_size_ratio > 0.7).all()),
           ("work/rho", "§6.5: from the GE threshold up, S_T follows work avoided (significant) more closely than hit rate",
            pw <= ALPHA and rw > rh),
           ("work/examples-top", f"§6.2: {WORK_EXAMPLES} have the two highest GE hit rates of the binary tables at 5%",
            set(binary.dataset.iloc[:2]) == set(WORK_EXAMPLES)),
           ("work/examples-slow", f"§6.2: at 5% their hit rate is higher but their S_T lower than on sleep and lhc_jets",
            all(f5.hit_rate[ds] > f5.hit_rate[m] and f5.S_T[ds] < f5.S_T[m] for ds in WORK_EXAMPLES for m in MC_LIMITS)),
           ("work/examples-half", f"§6.2: on {WORK_EXAMPLES} the hits avoided only about half of the evaluation work "
            "at 5% (40-60%)", all(0.4 <= f5.work_avoided[ds] <= 0.6 for ds in WORK_EXAMPLES)),
           ("work/small-overhead", f"§6.5: on {WORK_SMALL} the hits avoided at least 40% of the work at 5% but S_T stayed at "
            "about 1 (0.95-1.05)", all(f5.work_avoided[ds] >= 0.4 and 0.95 <= f5.S_T[ds] <= 1.05 for ds in WORK_SMALL))]
    if len(gp):
        out.append(("work/gp-large", "§6.5: GP hit programs at least about as large as missed ones on most configurations "
                    "(ratio > 1 in more than half, min > 0.85)", (gp.hit_size_ratio > 1).mean() > 0.5 and gp.hit_size_ratio.min() > 0.85))
    return out


# ---------------------------------------------------------------- §6.2: small, medium-sized and large tables

# Small / medium-sized / large tables: cut at 20,000 and 100,000 rows. The text does not
# state the cut points; it names the datasets of each group, and Table 1 gives their rows.
SMALL_BELOW, LARGE_ABOVE = 20_000, 100_000


def size_groups(ann: pd.DataFrame, algo: str) -> dict[str, list[str]]:
    rows = ann[ann.algorithm == algo].groupby("dataset").rows.first().sort_values(kind="stable")
    return {"Small": list(rows.index[rows < SMALL_BELOW]),
            "Medium": list(rows.index[(rows >= SMALL_BELOW) & (rows <= LARGE_ABOVE)]),
            "Large": list(rows.index[rows > LARGE_ABOVE])}


PAIR_KEY = {"Diabetes130US": "Diab", "Run_or_walk_information": "Run", "codrna": "Codrna", "covertype": "Cover"}


def size_group_numbers(ann: pd.DataFrame) -> dict:
    """§6.2 walks through the datasets by size without grouping them in the text; only the largest tables are
    named (the large group) and the exceptions outside the three smallest."""
    out = {}
    thr = {a: all_faster_threshold(ann[ann.algorithm == a]) for a in ("GE", "GP") if (ann.algorithm == a).any()}
    if thr.get("GE") and thr.get("GP"):              # one sentence, worded for equal or different starting tables
        ge, gp = _tt(thr["GE"][0]), _tt(thr["GP"][0])
        out["ThresholdSentence"] = (f"From {ge} upwards, every dataset was faster at all three fractions, with GE and with GP."
                                    if ge == gp else f"Every dataset was faster at all three fractions from {ge} upwards "
                                    f"with GE and from {gp} upwards with GP.")
    rows = ann.groupby("dataset").rows.first()
    for ds, key in PAIR_KEY.items():
        if ds in rows.index:
            out[f"Pair{key}Rows"] = _rows(rows[ds])
    for algo in ("GE", "GP"):
        if not (ann.algorithm == algo).any():
            continue
        g = ann[ann.algorithm == algo]
        out[f"{algo}AnyFaster"] = str(int(g.groupby("dataset").speed.apply(lambda s: (s == "faster").any()).sum()))
        for ds, key in PAIR_KEY.items():
            x = g[g.dataset == ds]
            if len(x):
                both = x[(x.speed == "faster") & (x.verdict == "equivalent")]
                words = [FRACTION_WORD[round(float(f), 2)] for f in sorted(both.fraction)]
                out[f"Pair{key}{algo}BothFractions"] = ("none" if not words else words[0] if len(words) == 1
                                                        else ", ".join(words[:-1]) + " and " + words[-1])
        out[f"SizeLarge{algo}N"] = str(len(size_groups(ann, algo)["Large"]))
        out[f"Except{algo}NoBothNames"] = _names(_exceptions(ann, algo))
    return out


def _exceptions(ann: pd.DataFrame, algo: str) -> list[str]:
    """From the first table that was faster at every fraction upwards (the tables §6.2 calls 'these tables'):
    the datasets with no configuration both faster and comparable."""
    g = ann[ann.algorithm == algo]
    thr = all_faster_threshold(g)
    x = g[g.rows >= thr[1]] if thr else g.iloc[0:0]
    nb = x.assign(b=(x.speed == "faster") & (x.verdict == "equivalent")).groupby("dataset").b.any()
    return sorted(nb.index[~nb])


def _pair_ok(g: pd.DataFrame, good: str, bad: str) -> bool:
    a, b = g[g.dataset == good], g[g.dataset == bad]
    if not len(a) or not len(b):
        return True                                  # not in the store yet: the sentence names only what is there
    return (((a.speed == "faster") & (a.verdict == "equivalent")).all() and (b.speed == "faster").all()
            and (b.verdict != "equivalent").all())


def _larger_fraction_ok(g: pd.DataFrame, names: tuple) -> bool:
    x = g[g.dataset.isin(names)]
    if not len(x):
        return True
    one = x[x.fraction.round(2) == 0.01]
    return (one.verdict != "equivalent").all() and (x[x.fraction.round(2) > 0.01].verdict == "equivalent").any()


def _first_both(g: pd.DataFrame, ds: str) -> float:
    """Smallest fraction at which a dataset was both faster and comparable (inf if none or not in the store)."""
    x = g[(g.dataset == ds) & (g.speed == "faster") & (g.verdict == "equivalent")]
    return float(x.fraction.min()) if len(x) else float("inf")


def _most_both_from_threshold(g: pd.DataFrame) -> bool:
    thr = all_faster_threshold(g, "speed_raw")            # §6.2 names the uncorrected threshold (Holm)
    if thr is None:
        return False
    x = g[g.rows >= thr[1]]
    anyb = x.assign(b=(x.speed == "faster") & (x.verdict == "equivalent")).groupby("dataset").b.any()
    return anyb.mean() > 0.5


def size_group_claims(ann: pd.DataFrame) -> list[tuple[str, str, bool]]:
    out = []
    for algo in ("GE", "GP"):
        g = ann[ann.algorithm == algo]
        if not len(g):
            continue
        small, large = _ends(ann, algo)[0], size_groups(ann, algo)["Large"]
        oc = outcome_counts(g)
        all_both = list(oc.index[oc.both == oc.configs])
        out += [(f"size/{algo}/never-worse", f"§6.2 {algo}: no configuration on any table was judged worse",
                 (g.verdict != "worse").all()),
                (f"size/{algo}/all-both-mid-sized", f"§6.2 {algo}: the tables with both at every fraction are mid-sized "
                 "(neither among the three smallest nor the largest)", len(all_both) > 0
                 and not set(all_both) & (set(small) | set(large))),
                (f"size/{algo}/smallest-slower-ten", f"§6.2 {algo}: on the three smallest tables FEC was slower at 10% "
                 + ("on the named tables" if algo == "GE" else "on none (only GE is named)"),
                 (g[g.dataset.isin(small) & (g.fraction.round(2) == 0.10)].speed == "slower").any() == (algo == "GE")),
                (f"size/{algo}/smallest-rarely-all-fast", f"§6.2 {algo}: on the three smallest tables FEC was faster at "
                 "every fraction on at most one", int(g[g.dataset.isin(small)].groupby("dataset").speed
                                                       .apply(lambda s: (s == "faster").all()).sum()) <= 1),
                (f"size/{algo}/pair-diab-run", f"§6.2 {algo}: Diabetes130US both faster and comparable at every fraction, "
                 "Run_or_walk faster at every fraction but comparable at none",
                 _pair_ok(g, "Diabetes130US", "Run_or_walk_information")),
                (f"size/{algo}/codrna-covertype", f"§6.2 {algo}: on codrna and covertype comparable accuracy needed a larger "
                 "fraction than 1% (none at 1%, some at 5% or 10% on at least one of them)",
                 _larger_fraction_ok(g, ("codrna", "covertype"))),
                (f"size/{algo}/most-from-threshold-both", f"§6.2 {algo}: from the first always-faster table (uncorrected) upwards, most "
                 "tables had a configuration both faster and comparable", _most_both_from_threshold(g)),
                (f"size/{algo}/covertype-before-codrna", f"§6.2 {algo}: covertype reached comparable accuracy at a smaller "
                 "fraction than codrna", _first_both(g, "covertype") < _first_both(g, "codrna")),
                (f"size/{algo}/exceptions-faster", f"§6.2 {algo}: the exceptions outside the three smallest tables were "
                 "faster in some configuration and never worse",
                 all((g[g.dataset == d].speed == "faster").any() and (g[g.dataset == d].verdict != "worse").all()
                     for d in _exceptions(ann, algo)))]
    return out


# ---------------------------------------------------------------- Conclusions

N_ENDS = 3                      # "the smallest" / "the largest" tables: the three at each end, by rows (no row threshold)
FRACTIONS = ((0.01, "One"), (0.05, "Five"), (0.10, "Ten"))


def _ends(ann: pd.DataFrame, algo: str) -> tuple[list[str], list[str]]:
    """The N_ENDS smallest and largest datasets of one algorithm (its own datasets with results)."""
    order = ann[ann.algorithm == algo].groupby("dataset").rows.first().sort_values(kind="stable")
    return list(order.index[:N_ENDS]), list(order.index[-N_ENDS:])


def conclusion_numbers(ann: pd.DataFrame) -> dict:
    """Macros of the Conclusions: speed at 1% everywhere; the smallest and the largest tables per fraction.
    `ann`: annotated main-sampler metrics with a rows column."""
    out = {}
    for algo in ("GE", "GP"):
        g = ann[ann.algorithm == algo]
        if not len(g):
            continue
        o1 = g[g.fraction.round(2) == 0.01]
        out[f"{algo}FastOneAll"], out[f"{algo}FastOneN"] = str(int((o1.speed == "faster").sum())), str(len(o1))
        out[f"{algo}FastOneMedianST"] = f"{o1.S_T.median():.2f}"
        out[f"{algo}NotFastOneNames"] = _names(sorted(o1[o1.speed != "faster"].dataset))
        small, large = _ends(ann, algo)[0], size_groups(ann, algo)["Large"]     # the large group of §6.2
        for name, names in (("Smallest", small), ("Largest", large)):
            x = g[g.dataset.isin(names)]
            out[f"{name}{algo}Names"] = _names(names)
            for f, word in FRACTIONS:
                xf = x[x.fraction.round(2) == f]
                out[f"{name}{algo}Fast{word}"] = str(int((xf.speed == "faster").sum()))
                out[f"{name}{algo}Equiv{word}"] = str(int((xf.verdict == "equivalent").sum()))
                out[f"{name}{algo}ST{word}"] = f"{xf.S_T.median():.2f}"
    return out


def conclusion_claims(ann: pd.DataFrame) -> list[tuple[str, str, bool]]:
    out = []
    for algo in ("GE", "GP"):
        g = ann[ann.algorithm == algo]
        if not len(g):
            continue
        small, large = _ends(ann, algo)[0], size_groups(ann, algo)["Large"]
        one = g[g.fraction.round(2) == 0.01]
        out.append((f"concl/{algo}/one-percent-most-faster", f"Conclusions: with a 1% fingerprint {algo} was "
                    "significantly faster on [n] of the datasets (more than half)", (one.speed == "faster").mean() > 0.5))
        sm, lg = g[g.dataset.isin(small)], g[g.dataset.isin(large)]
        at = lambda x, f: x[x.fraction.round(2) == f]
        out.append((f"concl/{algo}/smallest-never-equivalent", f"Conclusions {algo}: on the smallest tables no fraction "
                    "gave accuracy confirmed as equivalent", (sm.verdict != "equivalent").all()))
        out.append((f"concl/{algo}/smallest-fast-at-one", f"Conclusions {algo}: on the smallest tables 1% was enough for a "
                    "speedup on some, and larger fractions were faster on no more of them",
                    (at(sm, 0.01).speed == "faster").sum() > 0
                    and all((at(sm, f).speed == "faster").sum() <= (at(sm, 0.01).speed == "faster").sum() for f in (0.05, 0.10))))
        out.append((f"concl/{algo}/smallest-lose-speed", f"Conclusions {algo}: on the smallest tables a larger fraction "
                    "lowered the median S_T", at(sm, 0.10).S_T.median() < at(sm, 0.01).S_T.median()))
        eq = lambda f: (at(lg, f).verdict == "equivalent").sum()
        out.append((f"concl/{algo}/largest-needs-fraction", f"Conclusions {algo}: on the largest tables no 1% configuration "
                    "was equivalent, more were at 10%", eq(0.01) == 0 and eq(0.10) > 0))
        out.append((f"concl/{algo}/largest-speed-cost", f"Conclusions {algo}: on the largest tables the larger fraction "
                    "lowered the speedup", at(lg, 0.10).S_T.median() < at(lg, 0.01).S_T.median()))
    return out


# ---------------------------------------------------------------- §6.1: the fingerprint fraction as a trade-off

TRADEOFF_MEASURES = (("S_T", "$S_T$", 1, 1.0), ("dAcc", "$\\Delta$Acc (pp)", 100, 0.0), ("hit_rate", "hit rate", 1, None),
                     ("fake_hit_ratio", "fake-hit ratio", 1, None))


def fraction_tradeoff(g: pd.DataFrame) -> dict:
    """One algorithm, main sampler: per measure the median at each fraction and the paired 1%-vs-10% comparison
    over datasets (count of datasets where 10% is higher, Wilcoxon p)."""
    from analysis import wilcoxon_paired
    out = {}
    w = g.pivot_table(index="dataset", columns="fraction", values=[m for m, *_ in TRADEOFF_MEASURES])
    for col, _label, scale, _ref in TRADEOFF_MEASURES:
        med = {round(float(f), 2): scale * g[np.isclose(g.fraction, f)][col].median() for f in sorted(g.fraction.unique())}
        x = w[col].dropna() if col in w else pd.DataFrame()
        lo, hi = [c for c in x.columns if round(float(c), 2) == 0.01], [c for c in x.columns if round(float(c), 2) == 0.10]
        if not lo or not hi:
            out[col] = {"median": med}
            continue
        d = x[hi[0]] - x[lo[0]]
        out[col] = {"median": med, "higher_at_ten": int((d > 0).sum()), "n": len(d),
                    "p": float(wilcoxon_paired(x[lo[0]], x[hi[0]])["p"])}
    return out


def fraction_tradeoff_numbers(ann: pd.DataFrame) -> dict:
    out = {}
    for algo in ("GE", "GP"):
        v = fraction_tradeoff(ann[ann.algorithm == algo])
        if not v:
            continue
        for col, key, digits in (("S_T", "ST", 2), ("dAcc", "Dacc", 2), ("hit_rate", "Hit", 2), ("fake_hit_ratio", "Fake", 2)):
            for f, word in FRACTIONS:
                x = v[col]["median"].get(f, np.nan)
                out[f"{algo}Frac{key}{word}"] = "--" if pd.isna(x) else (f"{x:+.{digits}f}" if col == "dAcc" else f"{x:.{digits}f}")
            if "p" in v[col]:
                out[f"{algo}Frac{key}P"] = _p(v[col]["p"])
                out[f"{algo}Frac{key}LowerAtTen"] = str(v[col]["n"] - v[col]["higher_at_ten"])
                out[f"{algo}Frac{key}N"] = str(v[col]["n"])
    return out


def fraction_tradeoff_claims(ann: pd.DataFrame) -> list[tuple[str, str, bool]]:
    out = []
    for algo in ("GE", "GP"):
        v = fraction_tradeoff(ann[ann.algorithm == algo])
        if not v or "p" not in v["S_T"]:
            continue
        med = lambda c, f: v[c]["median"].get(f, np.nan)
        out.append((f"frac-tradeoff/{algo}/more-hits", f"§6.1 {algo}: a smaller fingerprint gave more hits (median hit "
                    "rate higher at 1% than at 10%, paired p <= 0.05)",
                    med("hit_rate", 0.01) > med("hit_rate", 0.10) and v["hit_rate"]["p"] <= ALPHA))
        out.append((f"frac-tradeoff/{algo}/more-speed", f"§6.1 {algo}: and a larger speedup (paired p <= 0.05)",
                    med("S_T", 0.01) > med("S_T", 0.10) and v["S_T"]["p"] <= ALPHA))
        out.append((f"frac-tradeoff/{algo}/more-fakes-everywhere", f"§6.1 {algo}: the fake-hit ratio was higher at 1% than "
                    "at 10% on every dataset", v["fake_hit_ratio"]["higher_at_ten"] == 0))
        out.append((f"frac-tradeoff/{algo}/accuracy-other-way", f"§6.1 {algo}: the median accuracy difference was higher "
                    "(better) at 10% than at 1%", med("dAcc", 0.10) > med("dAcc", 0.01)))
    vge, vgp = fraction_tradeoff(ann[ann.algorithm == "GE"]), fraction_tradeoff(ann[ann.algorithm == "GP"])
    if vge and vgp and "p" in vge["dAcc"] and "p" in vgp["dAcc"]:
        out.append(("frac-tradeoff/gp-accuracy", "§6.1: 'the accuracy shift was significant in GP but small and not "
                    "significant in GE' (paired p)", vgp["dAcc"]["p"] <= ALPHA < vge["dAcc"]["p"]))
        out.append(("frac-tradeoff/gp-more-fakes", "§6.1: 'GP's 1% configurations had far more fake hits than GE's' "
                    "(median fake-hit ratio at 1% at least twice GE's)",
                    vgp["fake_hit_ratio"]["median"].get(0.01, 0) >= 2 * vge["fake_hit_ratio"]["median"].get(0.01, 1)))
    return out


def fraction_tradeoff_figure(corpus, metrics, props, out_tables, min_rows=0, **_) -> list[Path]:
    """fig_fraction_tradeoff.pdf: S_T, dAcc (pp), hit rate and fake-hit ratio per fingerprint fraction (main sampler),
    GE (top) and GP (bottom); box with one point per dataset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keep = main_text_datasets(corpus, props, min_rows)
    a = metrics[metrics.dataset.isin(keep)]
    if not len(a):
        return []
    rng = np.random.default_rng(0)                      # fixed jitter: the figure is deterministic
    fractions = sorted(a.fraction.unique())
    labels = [FRACTION_WORD[round(float(f), 2)].replace("\\%", "%") for f in fractions]
    fs.use()
    fig, axes = plt.subplots(2, 4, figsize=(6.3, 3.9))
    for r, algo in enumerate(("GE", "GP")):
        g = a[a.algorithm == algo]
        for c, (col, label, scale, ref) in enumerate(TRADEOFF_MEASURES):
            ax = axes[r, c]
            if ref is not None:
                ax.axhline(ref, color=INK_SECONDARY, linestyle=(0, (4, 3)), linewidth=0.8, zorder=1)
            groups = [scale * g[np.isclose(g.fraction, f)][col].dropna().to_numpy() for f in fractions]
            bp = ax.boxplot(groups, widths=0.5, patch_artist=True, showfliers=False,
                            medianprops=dict(color=INK, linewidth=1.2), whiskerprops=dict(color=INK_SECONDARY, linewidth=0.8),
                            capprops=dict(color=INK_SECONDARY, linewidth=0.8), boxprops=dict(linewidth=0.8, color=INK_SECONDARY))
            for patch in bp["boxes"]:
                patch.set_facecolor(SERIES_COLOUR)
                patch.set_alpha(0.25)
            for k, vals in enumerate(groups, start=1):
                ax.scatter(k + rng.uniform(-0.15, 0.15, len(vals)), vals, s=7, color=SERIES_COLOUR, edgecolor="white",
                           linewidth=0.3, zorder=3)
            ax.set_xticks(range(1, len(labels) + 1), labels)
            ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if col in ("hit_rate", "fake_hit_ratio"):
                ax.set_ylim(0, 1)
            if r == 0:
                ax.set_title(label, fontsize=8, color=INK)
            if c == 0:
                ax.set_ylabel(algo)
    for c in range(2):                                   # S_T and dAcc: same y range for GE and GP
        lo = min(axes[0, c].get_ylim()[0], axes[1, c].get_ylim()[0]); hi = max(axes[0, c].get_ylim()[1], axes[1, c].get_ylim()[1])
        for r in range(2):
            axes[r, c].set_ylim(lo, hi)
    fig.tight_layout()
    out_tables.mkdir(parents=True, exist_ok=True)
    path = out_tables / "fig_fraction_tradeoff.pdf"
    fs.save(fig, path)
    plt.close(fig)
    return [path]


# ---------------------------------------------------------------- §6.3: two pairs of tables of similar size (GE)

PAIRS = (("codrna", "covertype"), ("Diabetes130US", "Run_or_walk_information"))
PAIR_ALGO = "GE"
PAIR_FRACTIONS = (0.01, 0.05)
PAIR_MEASURES = (("hit_rate", "hit rate", False), ("fake_share", "fake-hit ratio", False),
                 ("avg_pheno_len", "mean phenotype length", True), ("unique_phenotypes", "unique phenotypes", True),
                 ("avg_features_used", "mean features used", True))
# structural diversity dropped from the pair figures - it is distinct derivations /
# population, i.e. about (unique phenotypes + invalid individuals) / population, so it repeated the unique-phenotype panel
_PAIR_CACHE: dict = {}


def pair_generations(store: Path = results.STORE) -> pd.DataFrame:
    """Per (dataset, mode, fraction, generation) for the pair datasets, GE, main sampler: hit rate and fake-hit ratio
    (diagnostic runs; the timing runs follow the same course) and the phenotype measures of the diagnostic runs and
    of the baseline (mode 'baseline', fraction NaN)."""
    path = _f(store, "generations.csv")
    if path in _PAIR_CACHE:
        return _PAIR_CACHE[path]
    names = {d for pair in PAIRS for d in pair}
    cols = ["algorithm", "setup", "dataset", "mode", "sampler", "fraction", "seed", "gen", "fec_hits", "fec_misses",
            "fec_fake_hits"] + [m for m, _l, ph in PAIR_MEASURES if ph]
    parts = [c[(c.setup == SETUP) & (c.algorithm == PAIR_ALGO) & c.dataset.isin(names)
               & ((c["mode"] == "baseline") | ((c["mode"] == "fec_diagnostic") & (c.sampler == MAIN_SAMPLER)))]
             for c in pd.read_csv(path, usecols=cols, chunksize=500_000)]
    g = pd.concat(parts)
    g["fraction"] = g.fraction.where(g["mode"] != "baseline")
    evals = g.fec_hits + g.fec_misses                          # per-run ratios, for the spread between runs
    g = g.assign(run_hit=g.fec_hits / evals.where(evals > 0), run_fake=g.fec_fake_hits / g.fec_hits.where(g.fec_hits > 0))
    out = (g.groupby(["dataset", "mode", "fraction", "gen"], dropna=False)
           .agg(hits=("fec_hits", "sum"), misses=("fec_misses", "sum"), fake=("fec_fake_hits", "sum"),
                hit_rate_sd=("run_hit", "std"), fake_share_sd=("run_fake", "std"),
                **{m: (m, "mean") for m, _l, ph in PAIR_MEASURES if ph},
                **{m + "_sd": (m, "std") for m, _l, ph in PAIR_MEASURES if ph}).reset_index())
    out["hit_rate"] = out.hits / (out.hits + out.misses)
    out["fake_share"] = out.fake / out.hits
    _PAIR_CACHE[path] = out
    return out


def _pair_mean(c: pd.DataFrame, ds: str, col: str, fraction=None) -> float:
    """Mean over generations 1-30 of one measure (fraction None: the baseline)."""
    x = c[(c.dataset == ds) & (c.gen >= 1)]
    x = x[x["mode"] == "baseline"] if fraction is None else x[(x["mode"] == "fec_diagnostic") & np.isclose(x.fraction, fraction)]
    return float(x[col].mean()) if len(x) else float("nan")


def pair_ready(metrics: pd.DataFrame) -> bool:
    have = set(metrics[metrics.algorithm == PAIR_ALGO].dataset)
    return all(d in have for pair in PAIRS for d in pair)


def pair_numbers(metrics: pd.DataFrame, store: Path = results.STORE) -> dict:
    """Macros \res Pair<Key>...: per dataset and fraction (One, Five) the metrics and the generation means."""
    if not pair_ready(metrics):
        return {}
    c = pair_generations(store)
    a = annotated(metrics[metrics.algorithm == PAIR_ALGO])
    out = {}
    for pair in PAIRS:
        for ds in pair:
            k = PAIR_KEY[ds]
            row = a[a.dataset == ds]
            out[f"Pair{k}BaseSd"] = _num(100 * row.baseline_accuracy_sd.iloc[0], 1)
            for f, word in zip(PAIR_FRACTIONS, ("One", "Five")):
                r = row[np.isclose(row.fraction, f)].iloc[0]
                out[f"Pair{k}ST{word}"] = f"{r.S_T:.2f}"
                out[f"Pair{k}Hit{word}"] = f"{r.hit_rate:.2f}"
                out[f"Pair{k}Fake{word}"] = f"{100 * r.fake_hit_ratio:.1f}"
                out[f"Pair{k}PairedSd{word}"] = f"{100 * r.dAcc_sd:.2f}"
                out[f"Pair{k}Same{word}"] = f"{100 * r.duplicate_hit_share:.0f}"
                for m, key in (("avg_pheno_len", "Len"), ("unique_phenotypes", "Unique"), ("avg_features_used", "Feat")):
                    v = _pair_mean(c, ds, m, f)
                    out[f"Pair{k}{key}{word}"] = f"{v:.2f}" if m == "avg_features_used" else f"{v:.0f}"
            for m, key in (("avg_pheno_len", "Len"), ("unique_phenotypes", "Unique"), ("avg_features_used", "Feat")):
                v = _pair_mean(c, ds, m)
                out[f"Pair{k}{key}Base"] = f"{v:.2f}" if m == "avg_features_used" else f"{v:.0f}"
    return out


def pair_claims(metrics: pd.DataFrame, store: Path = results.STORE) -> list[tuple[str, str, bool]]:
    if not pair_ready(metrics):
        return []
    c = pair_generations(store)
    a = annotated(metrics[metrics.algorithm == PAIR_ALGO]).set_index(["dataset", "fraction"])
    r = lambda ds, f, col: a.loc[(ds, a.loc[ds].index[np.isclose(a.loc[ds].index, f)][0]), col]
    gm = lambda ds, col, f=None: _pair_mean(c, ds, col, f)
    out = []
    for slow, fast in PAIRS:
        tag = f"{PAIR_KEY[slow]}-{PAIR_KEY[fast]}"
        out.append((f"pair/{tag}/faster-more-hits", f"§6.3: {fast} faster than {slow} at 1% and 5%, with a higher hit rate",
                    all(r(fast, f, "S_T") > r(slow, f, "S_T") and r(fast, f, "hit_rate") > r(slow, f, "hit_rate")
                        for f in PAIR_FRACTIONS)))
        out.append((f"pair/{tag}/simpler-population", f"§6.3: {fast} evolved shorter programs, fewer unique phenotypes and fewer "
                    f"features than {slow} (1% and 5%)",
                    all(gm(fast, m, f) < gm(slow, m, f) for f in PAIR_FRACTIONS
                        for m in ("avg_pheno_len", "unique_phenotypes", "avg_features_used"))))
        out.append((f"pair/{tag}/more-repeats", f"§6.3: more hits were the same program on {fast} than on {slow}",
                    all(r(fast, f, "duplicate_hit_share") > r(slow, f, "duplicate_hit_share") for f in PAIR_FRACTIONS)))
    co, cv = PAIRS[0]
    out.append(("pair/codrna-covertype/not-fake-hits", "§6.3: fake hits do not explain codrna's undecided verdicts (low on "
                "both, higher on covertype at 1% and 5%)", all(r(cv, f, "fake_hit_ratio") > r(co, f, "fake_hit_ratio")
                                                             and r(cv, f, "fake_hit_ratio") < 0.10 for f in PAIR_FRACTIONS)))
    out.append(("pair/codrna-covertype/paired-course", "§6.3: at 5% the paired sd of dAcc was below 0.5 pp on covertype and "
                "about 1 pp or more on codrna, although covertype's baseline varies more between seeds",
                r(cv, 0.05, "dAcc_sd") < 0.005 and r(co, 0.05, "dAcc_sd") >= 0.009
                and r(cv, 0.05, "baseline_accuracy_sd") > r(co, 0.05, "baseline_accuracy_sd")))
    out.append(("pair/codrna-covertype/verdicts", "§6.3: covertype comparable at 5%, codrna not",
                r(cv, 0.05, "verdict") == "equivalent" and r(co, 0.05, "verdict") != "equivalent"))
    di, ru = PAIRS[1]
    out.append(("pair/diab-run/more-fakes", "§6.3: Run_or_walk had more fake hits than Diabetes130US at 1% and 5%",
                all(r(ru, f, "fake_hit_ratio") > r(di, f, "fake_hit_ratio") for f in PAIR_FRACTIONS)))
    out.append(("pair/diab-run/fec-changes-population", "§6.3: at 1% FEC lowered the unique phenotypes of Run_or_walk "
                "below its baseline, while on Diabetes130US they stayed within 1% of the baseline",
                gm(ru, "unique_phenotypes", 0.01) < 0.97 * gm(ru, "unique_phenotypes")
                and abs(gm(di, "unique_phenotypes", 0.01) / gm(di, "unique_phenotypes") - 1) < 0.01))
    out.append(("pair/diab-run/paired-course", "§6.3: the paired sd of dAcc was about 0.1 pp on Diabetes130US and above "
                "2.5 pp on Run_or_walk at 1% and 5%", all(r(di, f, "dAcc_sd") < 0.002 and r(ru, f, "dAcc_sd") > 0.025
                                                       for f in PAIR_FRACTIONS)))
    return out


def pair_figures(corpus, metrics, props, out_tables, **_) -> list[Path]:
    """fig_pair_codrna_covertype.pdf, fig_pair_diabetes_run.pdf: the five generation measures of each pair, GE, random
    rows; colour = dataset, solid 1%, dashed 5%, dotted grey-tinted the baseline (phenotype panels)."""
    if not pair_ready(metrics):
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = pair_generations()
    c = c[c.gen >= 1]
    colour = {0: SERIES_COLOUR, 1: SECOND_COLOUR}
    every = c[c.dataset.isin({d for pair in PAIRS for d in pair})
              & ((c["mode"] == "baseline") | c.fraction.round(2).isin([round(f, 2) for f in PAIR_FRACTIONS]))]
    limits = {}
    for col, _label, _ph in PAIR_MEASURES:
        if col == "hit_rate":
            limits[col] = (0, 1)
        elif col == "fake_share":                        # from 0, to just above the largest value of the two pairs
            limits[col] = (0, max(0.05, float(np.ceil(every[col].max() * 1.1 / 0.05) * 0.05)))
        else:
            lo, hi = every[col].min(), every[col].max()
            pad = 0.05 * (hi - lo)
            limits[col] = (lo - pad, hi + pad)
    paths = []
    for pair, stem in zip(PAIRS, ("fig_pair_codrna_covertype", "fig_pair_diabetes_run")):
        fs.use()
        fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.0), sharex=True)
        for ax, (col, label, phen) in zip(axes.flat, PAIR_MEASURES):
            for i, ds in enumerate(pair):
                mark = {}                                  # one colour per dataset, line style per fraction
                for f, ls in zip(PAIR_FRACTIONS, ("-", "--")):
                    x = c[(c.dataset == ds) & (c["mode"] == "fec_diagnostic") & np.isclose(c.fraction, f)]
                    ax.fill_between(x.gen, x[col] - x[col + "_sd"], x[col] + x[col + "_sd"], color=colour[i],
                                    alpha=0.08, linewidth=0)             # +- sd over the 30 runs
                    ax.plot(x.gen, x[col], color=colour[i], linestyle=ls, linewidth=1.2,
                            label=f"{ds}, {FRACTION_WORD[f].replace(chr(92), '')}", **mark)
                if phen:
                    b = c[(c.dataset == ds) & (c["mode"] == "baseline")]
                    ax.plot(b.gen, b[col], color=colour[i], linestyle=":", linewidth=1.0, alpha=0.7,
                            label=f"{ds}, baseline", **mark)
            ax.set_title(label, fontsize=8, color=INK)
            ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_ylim(*limits[col])                    # one scale per measure, shared by both pair figures
        for ax in axes[1]:
            ax.set_xlabel("generation")
        handles, labels = axes[0, 2].get_legend_handles_labels()
        axes[1, 2].axis("off")                           # the sixth slot holds the legend
        axes[1, 2].legend(handles, labels, loc="center", frameon=False, fontsize=6.5)
        axes[0, 2].tick_params(labelbottom=True)
        axes[0, 2].set_xlabel("generation")
        fig.tight_layout()
        out_tables.mkdir(parents=True, exist_ok=True)
        path = out_tables / f"{stem}.pdf"
        fs.save(fig, path)
        plt.close(fig)
        paths.append(path)
    return paths


def diagram_figures(out_tables, **_) -> list[Path]:
    """Figures 1 and 2: the GE/GP and FEC workflow diagrams (diagrams.py), so every figure is built here."""
    return diagrams.build(out_tables)


BUILDERS: list = [diagram_figures, corpus_tables, outcome_map_tables, summary_table, size_figure, case_figures, mc_figure,
                   fraction_tradeoff_figure, pair_figures, ge_vs_gp_figure,
                   ge_vs_gp_table, size_pair_table, size_pair_figure, hit_kinds_figure, severity_table,
                   dynamics_figures, numbers, claims,
                   supplementary]


STORE_INPUTS = ("metrics.csv", "datasets.csv", "runs.csv", "generations.csv", "phase_timing.csv",
                "instrumented_runs.csv", "instrumented_generations.csv")


def missing_inputs(store: Path = results.STORE) -> list[str]:
    """Store files that are not there; the tables and figures that need them are left out of the build."""
    return [n for n in STORE_INPUTS if not _f(store, n).is_file()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: paper/)")
    ap.add_argument("--check", action="store_true", help="rebuild in a temporary folder and compare with the output folder")
    ap.add_argument("--min-rows", type=int, default=0, help="main-text results tables: leave out smaller datasets")
    a = ap.parse_args(argv)
    tables_dir, supp_dir = (a.out / "tables", a.out / "supplementary") if a.out else (TABLES, SUPPLEMENTARY)
    missing = missing_inputs()
    if missing:
        print("error: the results store is incomplete; missing in results/store/: "
              + ", ".join(f"{n}(.gz)" for n in missing) + "\n"
              "generations.csv.gz is attached to the release; download it with\n"
              "  curl -L -o results/store/generations.csv.gz "
              "https://github.com/aghilhooshmand/FEC-GE-GP/releases/download/v1.1.0/generations.csv.gz",
              file=sys.stderr)
        return 1
    try:
        if a.check:
            with tempfile.TemporaryDirectory() as tmp:
                t, s = Path(tmp) / "tables", Path(tmp) / "supplementary"
                written = build(t, s, a.min_rows)
                differ = [p for p in written
                          if not same_content(p, (tables_dir if p.parent == t else supp_dir) / p.name)]
                for p in differ:
                    print(f"differs: {p.name}")
                print(f"check: {len(written) - len(differ)} of {len(written)} files identical")
                return 1 if differ else 0
        written = build(tables_dir, supp_dir, a.min_rows)
    except TableError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
