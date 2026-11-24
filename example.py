"""A small, complete example of the experiment: baseline, FEC timing and FEC diagnostic runs, side by side.

For each algorithm and seed it runs, in this process and with the paper's settings unless overridden:
  baseline     the algorithm without the cache
  FEC timing   the same seed with the cache (random fingerprint rows); this is the run whose time is measured
  diagnostic   the same FEC run again, but every cache hit is also evaluated on all training rows, so a hit whose
               stored fitness differs from the program's true fitness is counted as a fake hit; the stored value
               is still used, so the run must evolve exactly as the timing run (checked here; an error otherwise)
and prints, per seed and over the seeds, the measures of the paper: S_T (baseline / FEC evaluation time), the
accuracy difference, the hit rate and the fake-hit ratio. With few seeds and a small test dataset these numbers
show how the pipeline works; they are not the paper's results (those use 30 seeds on 17 larger datasets).
Nothing is written to results/.

    python example.py                                    # GE and GP, prnn_crabs, seeds 30-31, 10% (about 3 min)
    python example.py --algorithm GP --dataset appendicitis --fraction 0.05 --seeds 30 31 32
    python example.py --population 200 --generations 10  # faster, smaller runs
"""
from __future__ import annotations

import argparse
import sys

import ge
import measurement
import typed_gp

RUNNERS = {"GE": ge.run, "GP": typed_gp.run}
TEST_DATASETS = ("appendicitis", "prnn_crabs", "iris")


def run_seed(algorithm: str, dataset: str, seed: int, params: dict, fraction: float) -> dict:
    """The three runs of one seed, after checking that timing and diagnostic evolved identically."""
    run = RUNNERS[algorithm]
    fec = {"fraction": fraction, "sampler": "random"}
    out = {"baseline": run(dataset, seed, params),
           "timing": run(dataset, seed, params, fec),
           "diagnostic": run(dataset, seed, params, {**fec, "diagnostic": True})}
    measurement.same_evolution(out["timing"], out["diagnostic"])      # raises if they differ
    return out


def report(algorithm: str, dataset: str, fraction: float, runs: list[dict]) -> dict:
    print(f"\n{algorithm} on {dataset}, fingerprint {round(100 * fraction)}% of the training rows (random)")
    print(f"{'seed':>4}  {'base acc':>8}  {'FEC acc':>8}  {'hits':>7}  {'hit rate':>8}  {'fake hits':>9}  "
          f"{'eval s base':>11}  {'eval s FEC':>10}  timing = diagnostic")
    for r in runs:
        b, t, d = r["baseline"], r["timing"], r["diagnostic"]
        print(f"{b['seed']:>4}  {1 - b['best_test_error']:>8.3f}  {1 - t['best_test_error']:>8.3f}  "
              f"{t['fec']['hits']:>7}  {t['fec']['hit_rate']:>8.3f}  {d['fec']['fake_hits']:>9}  "
              f"{b['measures']['eval_time_sec']:>11.3f}  {t['measures']['eval_time_sec']:>10.3f}  yes")
    m = measurement.metrics([r["baseline"] for r in runs], [r["timing"] for r in runs],
                            [r["diagnostic"] for r in runs])
    print(f"over {m['n_runs']} seed(s): S_T = {m['S_T']:.2f}, accuracy difference (FEC - baseline) = "
          f"{100 * m['dAcc']:+.2f} pp, hit rate = {m['hit_rate']:.3f}, fake-hit ratio = {m['fake_hit_ratio']:.3f}")
    return m


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--algorithm", choices=("GE", "GP", "both"), default="both")
    ap.add_argument("--dataset", default="prnn_crabs",
                    help=f"a dataset with a built table; the test tables {', '.join(TEST_DATASETS)} are included")
    ap.add_argument("--fraction", type=float, default=0.10, help="fingerprint size as a fraction of the training rows")
    ap.add_argument("--seeds", type=int, nargs="+", default=[30, 31])
    ap.add_argument("--population", type=int, help="population size (paper: 1000)")
    ap.add_argument("--generations", type=int, help="number of generations (paper: 30)")
    a = ap.parse_args(argv)
    params = {k: v for k, v in (("population", a.population), ("generations", a.generations)) if v is not None}
    for algorithm in (("GE", "GP") if a.algorithm == "both" else (a.algorithm,)):
        runs = [run_seed(algorithm, a.dataset, s, params, a.fraction) for s in a.seeds]
        report(algorithm, a.dataset, a.fraction, runs)
    print("\nEvery FEC timing run evolved exactly as its diagnostic run (same best program, evaluations, hits and "
          "misses, and per-generation statistics).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
