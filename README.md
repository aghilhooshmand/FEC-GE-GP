# Fitness Evaluation Caching for Evolutionary Classification

Code, dataset records and results for the manuscript *Fitness Evaluation Caching for Evolutionary
Classification: A Large-Scale Empirical Study* (S. A. Hooshmand and C. Ryan).

Fitness evaluation caching (FEC) evaluates each candidate program on a small set of fingerprint rows, hashes the
outputs, and reuses a stored fitness when the hash has been seen before. The experiments compare grammatical
evolution (GE, with GRAPE) and strongly typed genetic programming (GP, with DEAP) with and without the cache on
17 classification datasets, with 30 paired seeds per configuration.

## Contents

| Path | What it is |
|---|---|
| `datasets.py` | builds each dataset table from its public source (`data/recipes.yaml`) and checks its sha256 |
| `preprocessing.py` | z-score per run; the grammar constant range per dataset |
| `grammar.py`, `grammars/` | the GE grammar of each dataset (and its manifest, minimum initial depth, fairness) |
| `ge.py`, `typed_gp.py`, `operators.py` | the GE and typed GP runs; the operators both use |
| `fec.py` | fingerprint sampling (farthest-point, random, stratified) and the cache |
| `measurement.py` | the definition of every reported measure (evaluation time, S_T, effort, ...) |
| `batch.py`, `experiment.py`, `experiments/` | the experiment files (datasets, seeds 30-59, samplers, fractions) and the batch runner |
| `results.py`, `analysis.py`, `phase_timing.py` | the results store, the statistical tests, the per-phase timing |
| `complexity.py` | PyCol dataset complexity measures |
| `grape/` | GRAPE (de Lima et al., 2022), unchanged, with its BSD-3-Clause licence |
| `data/` | recipes, provenance records (`corpus.json`), constant ranges, complexity; three small test tables |
| `results/store/` | the results of the paper: `metrics.csv` (one row per dataset and configuration), `runs.csv.gz` (one row per run), `datasets.csv`, `phase_timing.csv.gz` |
| `tests/` | pytest suite (runs on the small test tables) |

Larger records are attached to the release (Releases page): `generations.csv.gz` (one row per run and
generation) and `run_files_ge.tar.xz`, `run_files_gp.tar.xz` (the full JSON record of every run).

## Install

Python 3.12:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pytest                      # the test suite, on the small test tables
```

## Rebuild a dataset and run an experiment

```bash
python datasets.py recreate wine_quality         # download from the source, apply the recipe, check sha256
python datasets.py verify                        # checksums of every built table
python ge.py run wine_quality --seed 30          # one GE baseline run (typed_gp.py run ... for GP)
python ge.py run wine_quality --seed 30 --fraction 0.05 --sampler random    # one FEC run
python batch.py pilot experiments/wine_quality_ge.yaml                      # an experiment's first two seeds
python batch.py run experiments/wine_quality_ge.yaml                        # all seeds, on the idle cores
python results.py all                            # the results store and metrics from the run files
```

Run files are written to `results/<algorithm>/<setup>/<dataset>/<configuration>/seed<k>.json`. Each FEC
configuration is run twice per seed: a timing run (evaluation time, S_T) and a diagnostic run that re-checks
every cache hit with a full evaluation (fake hits); both evolve identically (`python batch.py check`).

## Citation

If you use this code, please cite the manuscript. GRAPE: A. de Lima, S. Carvalho, D. M. Dias, E. Naredo,
J. P. Sullivan and C. Ryan, *GRAPE: Grammatical Algorithms in Python for Evolution*, Signals 3(3), 642-663, 2022.
