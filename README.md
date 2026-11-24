# Fitness Evaluation Caching for Evolutionary Classification

[![tests](https://github.com/aghilhooshmand/FEC-GE-GP/actions/workflows/tests.yml/badge.svg)](https://github.com/aghilhooshmand/FEC-GE-GP/actions/workflows/tests.yml)

Code, dataset records and results for the manuscript *Fitness Evaluation Caching for Evolutionary
Classification: A Large-Scale Empirical Study* (S. A. Hooshmand and C. Ryan).

Fitness evaluation caching (FEC) evaluates each candidate program on a small set of fingerprint rows, hashes the
outputs, and reuses a stored fitness when the hash has been seen before. The experiments compare grammatical
evolution (GE, with GRAPE) and strongly typed genetic programming (GP, with DEAP) with and without the cache on
17 classification datasets, with 30 paired seeds per configuration.

| Version | Manuscript |
|---|---|
| v1.0.0 | as submitted |
| v1.1.0 | revised: adds the instrumented runs (cache memory, selection, divergence from the baseline, exact-match cache), the code of every table and figure (`paper_tables.py`, `diagrams.py`, `figstyle.py`) with the published outputs in `paper/`, and `example.py` |

## Checking the work in four steps

Each step can be run on its own on a laptop (Python 3.12, about 2 GB of memory). The times are rough guides.

```bash
git clone https://github.com/aghilhooshmand/FEC-GE-GP.git && cd FEC-GE-GP
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

### 1. Run the tests (about a minute)

```bash
pytest
```

The suite runs on three small test tables that are in the repository (`appendicitis`, `prnn_crabs`, `iris`), so
it needs no download. It checks, among other things:

- that a **timing run and its diagnostic run evolve identically**, also when fake hits occur, for GE and GP, binary
  and multiclass (`tests/test_fec.py::test_diagnostic_evolves_like_timing`,
  `tests/test_typed_gp.py::test_fec_timing_equals_diagnostic`,
  `tests/test_multiclass.py::test_tiny_runs_and_timing_equals_diagnostic`);
- that an instrumented run and a phase-timed run reproduce their stored twins exactly
  (`tests/test_instrument.py`, `tests/test_phase_timing.py`);
- that a seed fixes a run in any process (`tests/test_typed_gp.py::test_run_does_not_depend_on_the_process`);
- the measures (`tests/test_measurement.py`), the statistics (`tests/test_analysis.py`), the grammars, the dataset
  records and checksums, the batch runner, and the code of the tables and figures (`tests/test_paper_tables.py`,
  `tests/test_example_and_diagrams.py`).

A few tests need a dataset table built from its public source (step 4) or the release file of step 3; they are
skipped, with the reason, when it is absent.

### 2. See baseline, timing and diagnostic runs side by side (about 3 minutes)

```bash
python example.py
```

For GE and GP, on `prnn_crabs`, seeds 30 and 31, with a 10% random fingerprint and the paper's settings
(population 1000, 30 generations), it runs the three kinds of run used in the paper:

| run | what it does | what it gives |
|---|---|---|
| baseline | the algorithm without the cache | reference time and accuracy |
| FEC timing | the same seed with the cache | the measured time: S_T = baseline evaluation time / FEC evaluation time |
| FEC diagnostic | the timing run again, but every hit is also evaluated on all training rows; the cached value is still used | the number of fake hits (hits whose cached fitness differs from the true one) |

It prints one line per seed and the summary over the seeds (S_T, accuracy difference, hit rate, fake-hit ratio),
and it stops with an error if a timing run and its diagnostic run did not evolve identically. It writes nothing
to `results/`. Other settings:

```bash
python example.py --algorithm GP --dataset appendicitis --fraction 0.05 --seeds 30 31 32
python example.py --population 200 --generations 10          # faster
python datasets.py recreate wine_quality && python example.py --dataset wine_quality   # a dataset of the paper
```

The test tables have 106 to 200 rows, so even a 10% fingerprint has at most 16 rows: many programs share a key
without sharing their fitness, fake hits are frequent, and the accuracy can differ from the baseline by several
points. This is the effect the paper measures (fake hits fall as the fingerprint gets more rows); on tables this
small the timing is also dominated by overhead. These runs show how the pipeline works. The paper's results come
from 30 seeds on 17 tables of about 2,500 to 805,000 rows.

### 3. Rebuild every table, figure and number of the paper (about 5 minutes)

The results store of every run of the paper is in `results/store/`. One larger file, the per-generation records,
is attached to the release:

```bash
curl -L -o results/store/generations.csv.gz \
  https://github.com/aghilhooshmand/FEC-GE-GP/releases/download/v1.1.0/generations.csv.gz
python paper_tables.py --check        # rebuild in a temporary folder and compare with paper/
```

It ends with `check: 54 of 54 files identical`: the tables, figures, Supplementary Tables S1-S20 and
Figures S2-S5 rebuilt from the store are the ones in the paper. To look at the rebuilt files instead:

```bash
python paper_tables.py --out rebuilt  # writes rebuilt/tables and rebuilt/supplementary
```

Besides the tables and figures, the build writes two files that make the text checkable:

- `paper/tables/numbers.tex`: every number quoted in the text, as one LaTeX macro each (the manuscript prints
  these macros, so no result is typed by hand);
- `paper/tables/claims_check.txt`: every statement of the text about the data (for example "the fake-hit ratio
  fell with size"), with the test behind it and its result, OK or FAIL.

Where each item of the paper comes from (`paper_tables.py` unless stated):

| Paper | File in `paper/` | Function |
|---|---|---|
| Figure 1 | `tables/fig_ge_gp.pdf` | `diagrams.ge_gp` |
| Figure 2 | `tables/fig_fec_workflow.pdf` | `diagrams.fec_workflow` |
| Figures 4, 5 | `tables/fig_st_rows.pdf`, `tables/fig_dacc_rows.pdf` | `size_figure` |
| Figure 6 | `tables/fig_cache_rows.pdf` | `cache_figure` |
| Figure 7 | `tables/fig_hit_kinds.pdf` | `hit_kinds_figure` |
| Figure 8 | `tables/fig_fake_acc.pdf` | `fake_acc_figure` |
| Figures 9, 10 | `tables/fig_selection.pdf`, `tables/fig_divergence.pdf` | `selection_figure`, `trajectory_figure` |
| Figure 11 | `tables/fig_fraction_tradeoff.pdf` | `fraction_tradeoff_figure` |
| Figure 12 | `tables/fig_case_size.pdf` | `case_figures` |
| Figures 13, 14 | `tables/fig_pair_codrna_covertype.pdf`, `tables/fig_pair_diabetes_run.pdf` | `pair_figures` |
| Figure 15 | `tables/fig_case_mc.pdf` | `mc_figure` |
| Figure 16 | `tables/fig_size_pairs.pdf` | `size_pair_figure` |
| Table 1 | `tables/corpus.tex` | `corpus_tables` |
| Table 4 | `tables/summary_by_config.tex` | `summary_table` |
| Table 5 | `tables/outmap.tex` | `outcome_map_tables` |
| Table 6 | `tables/fake_severity.tex` | `severity_table` |
| Table 7 | `tables/size_pairs.tex` | `size_pair_table` |
| numbers in the text | `tables/numbers.tex` | `numbers` |
| statements in the text | `tables/claims_check.txt` | `claims` |
| Tables S1-S20, Figures S2-S5 | `supplementary/` (`INDEX.md` lists them; `FileS1_data_dictionary.md` defines every column) | `supplementary` |

Tables 2 and 3 (settings and operators) and Figures 3 and S1 (grammar listings, from `grammars/*.bnf`) are text in
the manuscript, not results. `fig_case_gegp.pdf`, `fig_ge_vs_gp.pdf`, `ge_vs_gp.tex` and `grammar_props.tex` are
additional views of the same data that the paper does not show.

**Figures.** Every figure uses one style, set in `figstyle.py` (font, sizes, colours, and a PDF without dates, so
a rebuild gives the same file). The font is DejaVu Serif, which ships with matplotlib. The colours are one
colour-blind-safe palette, and a second series always also differs in line style or marker. To change a
figure, edit its function in the table above and run `python paper_tables.py --out rebuilt`.

### 4. Rerun experiments

The dataset tables are not in the repository; each one is rebuilt from its public source and checked against its
recorded checksum:

```bash
python datasets.py recreate wine_quality       # download, apply the recipe in data/recipes.yaml, check sha256
python datasets.py verify                      # checksums of every built table
```

Single runs (each writes `results/<algorithm>/<setup>/<dataset>/<configuration>/seed<k>.json`):

```bash
python ge.py run wine_quality --seed 30                                          # GE baseline
python ge.py run wine_quality --seed 30 --fraction 0.05 --sampler random          # GE, FEC timing
python ge.py run wine_quality --seed 30 --fraction 0.05 --sampler random --diagnostic   # GE, FEC diagnostic
python typed_gp.py run wine_quality --seed 30 --fraction 0.05 --sampler random    # GP (same options)
```

Whole experiments, as run for the paper (`experiments/*.yaml` give the datasets, seeds 30-59, fingerprint
fractions and settings):

```bash
python batch.py plan  experiments/pilot_small.yaml    # jobs per phase
python batch.py pilot experiments/pilot_small.yaml    # the first two seeds: baseline, timing and diagnostic runs
python batch.py check experiments/pilot_small.yaml    # every timing/diagnostic pair evolved identically
python batch.py run   experiments/wine_quality_ge.yaml        # all seeds of a paper experiment, on the idle cores
```

| Experiment files | Runs |
|---|---|
| `wine_quality_*`, `line1_*`, `line2_*` | the main runs: baseline, FEC timing and FEC diagnostic, 17 datasets, GE and GP |
| `phase_timing_*` | FEC timing runs that also time each phase of the evaluation (fingerprint, key, lookup, full evaluation) |
| `instrumented_*` | FEC diagnostic runs that also record selection, memory, every hit, and an exact-match cache beside FEC |
| `pilot_small*` | the pipeline on the test tables |

A full paper experiment takes days on a many-core server. `python results.py all` rebuilds `results/store/` from
the run files in `results/`; it replaces the store, so keep a copy first. The run files of the paper are attached
to the releases (`run_files_ge.tar.xz`, `run_files_gp.tar.xz`; `instrumented_runs_ge.tar.xz` and
`instrumented_runs_gp.tar.xz` for the instrumented runs and their hit logs). Unpacked into the repository folder,
`python batch.py check <experiment file>` and `python instrument.py check` confirm on them that every timing and
diagnostic pair, and every instrumented run, evolved identically.

## Contents

| Path | What it is |
|---|---|
| `datasets.py` | builds each dataset table from its public source (`data/recipes.yaml`) and checks its sha256 |
| `preprocessing.py` | z-score per run; the grammar constant range per dataset |
| `grammar.py`, `grammars/` | the GE grammar of each dataset (and its manifest, minimum initial depth, fairness) |
| `ge.py`, `typed_gp.py`, `operators.py` | the GE and typed GP runs; the operators both use |
| `fec.py` | fingerprint sampling and the cache |
| `measurement.py` | the definition of every reported measure (evaluation time, S_T, effort, ...) |
| `batch.py`, `experiment.py`, `experiments/` | the experiment files (datasets, seeds 30-59, fractions) and the batch runner |
| `results.py`, `analysis.py` | the results store; the statistical tests (Wilcoxon, Holm, Benjamini-Hochberg, Bayesian) |
| `phase_timing.py`, `instrument.py` | per-phase timing; instrumented diagnostic runs |
| `paper_tables.py`, `diagrams.py`, `figstyle.py` | every table, figure and number of the paper |
| `example.py` | baseline, timing and diagnostic runs side by side |
| `complexity.py` | PyCol dataset complexity measures |
| `grape/` | GRAPE (de Lima et al., 2022), unchanged, with its BSD-3-Clause licence |
| `data/` | recipes, provenance records (`corpus.json`), constant ranges, complexity; three small test tables |
| `results/store/` | the results of the paper: `metrics.csv` (one row per dataset and configuration), `runs.csv.gz` (one row per run), `datasets.csv`, `phase_timing.csv.gz`, `instrumented_runs.csv`, `instrumented_generations.csv.gz` |
| `paper/` | the published tables, figures and supplementary files, as written by `paper_tables.py` |
| `tests/` | pytest suite (runs on the small test tables) |

## Citation

If you use this code, please cite the manuscript. GRAPE: A. de Lima, S. Carvalho, D. M. Dias, E. Naredo,
J. P. Sullivan and C. Ryan, *GRAPE: Grammatical Algorithms in Python for Evolution*, Signals 3(3), 642-663, 2022.
