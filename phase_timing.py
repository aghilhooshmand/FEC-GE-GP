"""Per-phase timing of FEC: twin check, measured fingerprint cost and the per-run table.

A phase-timed run (ge.py / typed_gp.py run ... --phase-timing) is stored under setup 'phase_timing' and records,
besides the usual evaluation time, the time and count of every phase of an FEC evaluation (measurement.PhaseClock).
Its twin is the stored run with the same algorithm, dataset, seed, sampler and fraction under setup 'default' (or
the same parameter overrides): the two must have evolved identically.

Measured fingerprint cost of a run (c_meas): the time of the fingerprint step per candidate that reached it
(evaluation on the fingerprint rows + key + lookup) divided by the time of one full evaluation of a miss. Both come
from the run's own clock; nothing is priced by assumption. Undefined (NaN, flag 'no_misses') without misses.

Commands:
  python phase_timing.py check [GE|GP]          twin-check every phase-timed run on this machine
  python phase_timing.py summary [GE|GP]        per configuration: c_meas mean and sd over seeds, ms per phase
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import datasets

ROOT = datasets.ROOT
SETUP = "phase_timing"
TWIN_FIELDS = ("best_phenotype", "n_train_evals")
TWIN_MEASURES = ("n_evals", "n_full_evals", "n_sample_evals")
TWIN_GEN_FIELDS = ("fec_hits", "fec_misses", "min", "avg")
FP_PHASES = ("t_fp_eval", "t_fp_key", "t_lookup")


def twin_path(path: Path, record: dict) -> Path:
    """The stored run this phase-timed run must reproduce: same place, setup without the 'phase_timing' prefix."""
    setup = record["setup"]
    twin_setup = "default" if setup == SETUP else setup[len(SETUP) + 1:]
    parts = list(path.parts)
    parts[parts.index(setup)] = twin_setup
    return Path(*parts)


def compare(a: dict, b: dict) -> list[str]:
    """Fields in which two runs differ (empty: they evolved identically)."""
    bad = [k for k in TWIN_FIELDS if a.get(k) != b.get(k)]
    bad += [f"measures.{k}" for k in TWIN_MEASURES if a["measures"].get(k) != b["measures"].get(k)]
    ga, gb = a.get("generations", []), b.get("generations", [])
    if len(ga) != len(gb):
        bad.append("generations.length")
    else:
        bad += sorted({f"generations.{k}" for x, y in zip(ga, gb) for k in TWIN_GEN_FIELDS if x.get(k) != y.get(k)})
    return bad


def check_twin(path: Path) -> dict:
    """Compare a phase-timed run with its stored twin, write the outcome into the run record; a mismatch renames the
    file to .mismatch (never stored as a good run). Returns {"path", "ok", "mismatches"}."""
    path = Path(path)
    record = json.loads(path.read_text())
    tp = twin_path(path, record)
    if not tp.is_file():
        result = {"twin": str(tp.relative_to(ROOT)) if tp.is_relative_to(ROOT) else str(tp), "ok": False,
                  "mismatches": ["twin missing"]}
    else:
        bad = compare(record, json.loads(tp.read_text()))
        result = {"twin": str(tp.relative_to(ROOT)) if tp.is_relative_to(ROOT) else str(tp), "ok": not bad,
                  "mismatches": bad}
    record["twin"] = result
    path.write_text(json.dumps(record, indent=1) + "\n")
    if not result["ok"]:
        path.rename(path.with_suffix(".mismatch"))
    return {"path": str(path), **result}


def run_row(record: dict) -> dict:
    """One row of the phase-timing table: identity, every phase field, the measured fingerprint cost."""
    ph, m, f = record["phases"], record["measures"], record.get("fec") or {}
    row = {"algorithm": record["algorithm"], "dataset": record["dataset"], "sampler": f.get("sampler"),
           "fraction": round(float(f.get("fraction", np.nan)), 2), "seed": record["seed"], "setup": record["setup"],
           "code_commit": record.get("code_commit"), "table_checksum": record.get("table_checksum"),
           "eval_time_sec": m["eval_time_sec"], "wall_time_sec": record.get("wall_time_sec"), **ph,
           "twin_ok": (record.get("twin") or {}).get("ok")}
    reached, misses = ph.get("n_reached_fp", 0), ph.get("n_misses", 0)
    fp = sum(ph.get(k, 0.0) for k in FP_PHASES)
    row["ms_fp_per_candidate"] = 1000 * fp / reached if reached else np.nan
    row["ms_full_per_miss"] = 1000 * ph.get("t_full_eval", 0.0) / misses if misses else np.nan
    row["c_meas"] = row["ms_fp_per_candidate"] / row["ms_full_per_miss"] if misses and reached else np.nan
    row["c_meas_flag"] = "" if misses and reached else "no_misses"
    return row


def load_runs(algorithm: str | None = None, root: Path = ROOT) -> list[Path]:
    algos = [algorithm.lower()] if algorithm else ["ge", "gp"]
    return sorted(p for a in algos for p in (root / "results" / a / SETUP).glob("*/*/seed*.json"))


def table(algorithm: str | None = None, root: Path = ROOT) -> pd.DataFrame:
    rows = [run_row(json.loads(p.read_text())) for p in load_runs(algorithm, root)]
    return pd.DataFrame(rows).sort_values(["algorithm", "dataset", "fraction", "seed"]).reset_index(drop=True) \
        if rows else pd.DataFrame()


def summary(algorithm: str | None = None) -> pd.DataFrame:
    t = table(algorithm)
    if t.empty:
        return t
    g = t.groupby(["algorithm", "dataset", "fraction"])
    return g.agg(runs=("seed", "size"), twins_ok=("twin_ok", "sum"), c_meas=("c_meas", "mean"),
                 c_meas_sd=("c_meas", "std"), ms_fp=("ms_fp_per_candidate", "mean"),
                 ms_full=("ms_full_per_miss", "mean")).reset_index()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in ("check", "summary"):
        print(__doc__)
        return 1
    algo = argv[1] if len(argv) > 1 else None
    if argv[0] == "check":
        res = [check_twin(p) for p in load_runs(algo)]
        bad = [r for r in res if not r["ok"]]
        for r in bad:
            print(f"MISMATCH {r['path']}: {', '.join(r['mismatches'])}")
        print(f"twin check: {len(res) - len(bad)} of {len(res)} phase-timed runs reproduce their stored twin")
        return 1 if bad else 0
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(summary(algo).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
