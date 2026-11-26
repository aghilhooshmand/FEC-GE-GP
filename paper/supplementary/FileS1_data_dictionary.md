# File S1: Data dictionary of the supplementary tables

Every column of every supplementary table (CSV). All results use random fingerprint rows and the settings of the paper; 30 paired seeds (30-59) per configuration. Accuracy is test accuracy (1 - test error rate of a run's best program).

## Table S1: Metrics of every dataset and FEC configuration (`TableS1_metrics.csv`)

| Column | Meaning |
|---|---|
| `algorithm` | GE (grammatical evolution) or GP (strongly typed genetic programming). |
| `setup` | Parameter setup of the runs; `default` = the settings of the paper. |
| `dataset` | Dataset name, as in Table 1 of the paper. |
| `fraction` | Fingerprint fraction: share of the training rows used as fingerprint rows (0.01, 0.05, 0.10); random fingerprint rows. |
| `n_runs` | Number of paired seeds behind the row (baseline, FEC timing and FEC diagnostic run with the same seed). |
| `S_T` | Evaluation-time speedup: mean baseline fitness-evaluation time / mean FEC fitness-evaluation time over the 30 paired seeds; > 1 = FEC faster. |
| `dAcc` | Accuracy difference: mean FEC test accuracy - mean baseline test accuracy over the 30 paired seeds, as a fraction (0.01 = 1 pp); < 0 = FEC less accurate. |
| `S_wall` | Wall-time speedup: mean baseline run time / mean FEC run time (whole run, FEC including the sampling of the fingerprint rows) over the 30 paired seeds; > 1 = FEC faster. |
| `hit_rate` | Cache hit rate: hits / (hits + misses), pooled over the FEC timing runs. |
| `fake_hit_ratio` | Fake-hit ratio: fake hits / hits in the diagnostic runs (a fake hit returns a cached fitness that differs from the program's fitness on all training rows). |
| `duplicate_hit_share` | Share of the hits whose program is identical to the one stored under the key (structurally equivalent hits, Section 5.4). |
| `p_eval_time` | p-value of the two-sided paired Wilcoxon signed-rank test of FEC against baseline fitness-evaluation time over the paired seeds; uncorrected. |
| `p_accuracy` | p-value of the two-sided paired Wilcoxon signed-rank test of FEC against baseline test accuracy; uncorrected. |
| `p_equivalence` | p-value of the paired Wilcoxon two one-sided tests (TOST) for equivalence within +-1 pp; small = accuracy shown comparable. |
| `bayes_worse` | P(FEC accuracy lower than the baseline's by more than 1 pp), posterior probability from the Bayesian correlated t-test over the paired seeds (Nadeau-Bengio variance correction). |
| `bayes_rope` | P(FEC and baseline accuracy within +-1 pp), posterior probability from the Bayesian correlated t-test over the paired seeds (Nadeau-Bengio variance correction). |
| `bayes_better` | P(FEC accuracy higher than the baseline's by more than 1 pp), posterior probability from the Bayesian correlated t-test over the paired seeds (Nadeau-Bengio variance correction). |
| `baseline_accuracy` | Mean test accuracy of the baseline runs (1 - test error rate of the run's best program). |
| `fec_accuracy` | Mean test accuracy of the FEC timing runs. |
| `S_T_sd` | Standard deviation over the 30 paired seeds of the per-seed ratio baseline time / FEC time. |
| `S_wall_sd` | Standard deviation over the 30 paired seeds of the per-seed wall-time ratio. |
| `dAcc_sd` | Standard deviation over the 30 paired seeds of the paired accuracy differences (fraction). |
| `baseline_accuracy_sd` | Standard deviation of the baseline test accuracy over the 30 paired seeds. |
| `fec_accuracy_sd` | Standard deviation of the FEC test accuracy over the 30 paired seeds. |
| `hit_rate_sd` | Standard deviation over the 30 paired seeds of the per-run hit rate. |
| `fake_hit_ratio_sd` | Standard deviation over the 30 paired seeds of the per-run fake-hit ratio. |
| `p_eval_holm` | `p_eval_time` Holm-adjusted over the algorithm's 51 configurations (17 datasets x 3 fractions); decides `speed`. |
| `p_eval_bh` | `p_eval_time` Benjamini-Hochberg-adjusted over the same 51 configurations (check only). |

## Table S2: Full GE results per configuration (`TableS2_results_GE.csv`)

| Column | Meaning |
|---|---|
| `algorithm` | GE (grammatical evolution) or GP (strongly typed genetic programming). |
| `dataset` | Dataset name, as in Table 1 of the paper. |
| `fraction` | Fingerprint fraction: share of the training rows used as fingerprint rows (0.01, 0.05, 0.10); random fingerprint rows. |
| `n_runs` | Number of paired seeds behind the row (baseline, FEC timing and FEC diagnostic run with the same seed). |
| `S_T` | Evaluation-time speedup: mean baseline fitness-evaluation time / mean FEC fitness-evaluation time over the 30 paired seeds; > 1 = FEC faster. |
| `S_T_sd` | Standard deviation over the 30 paired seeds of the per-seed ratio baseline time / FEC time. |
| `p_eval_time` | p-value of the two-sided paired Wilcoxon signed-rank test of FEC against baseline fitness-evaluation time over the paired seeds; uncorrected. |
| `p_eval_holm` | `p_eval_time` Holm-adjusted over the algorithm's 51 configurations (17 datasets x 3 fractions); decides `speed`. |
| `p_eval_bh` | `p_eval_time` Benjamini-Hochberg-adjusted over the same 51 configurations (check only). |
| `speed` | `faster` / `slower`: Holm-adjusted p <= 0.05 (`p_eval_holm`) with S_T > 1 / < 1; empty = not significant. |
| `speed_raw` | The same mark from the uncorrected p (`p_eval_time`), for comparison. |
| `dAcc` | Accuracy difference: mean FEC test accuracy - mean baseline test accuracy over the 30 paired seeds, as a fraction (0.01 = 1 pp); < 0 = FEC less accurate. |
| `bayes_worse` | P(FEC accuracy lower than the baseline's by more than 1 pp), posterior probability from the Bayesian correlated t-test over the paired seeds (Nadeau-Bengio variance correction). |
| `bayes_rope` | P(FEC and baseline accuracy within +-1 pp), posterior probability from the Bayesian correlated t-test over the paired seeds (Nadeau-Bengio variance correction). |
| `bayes_better` | P(FEC accuracy higher than the baseline's by more than 1 pp), posterior probability from the Bayesian correlated t-test over the paired seeds (Nadeau-Bengio variance correction). |
| `verdict` | Bayesian accuracy verdict: `equivalent` if the probability within the ROPE reaches the threshold, else `worse` or `better` if that probability does, else `undecided`. |
| `hit_rate` | Cache hit rate: hits / (hits + misses), pooled over the FEC timing runs. |
| `fake_hit_ratio` | Fake-hit ratio: fake hits / hits in the diagnostic runs (a fake hit returns a cached fitness that differs from the program's fitness on all training rows). |
| `S_wall` | Wall-time speedup: mean baseline run time / mean FEC run time (whole run, FEC including the sampling of the fingerprint rows) over the 30 paired seeds; > 1 = FEC faster. |
| `S_wall_sd` | Standard deviation over the 30 paired seeds of the per-seed wall-time ratio. |
| `p_wall` | p-value of the two-sided paired Wilcoxon signed-rank test of FEC against baseline whole-run wall time over the paired seeds; uncorrected. |
| `p_wall_holm` | `p_wall` Holm-adjusted over the algorithm's 51 configurations (17 datasets x 3 fractions). |
| `wall_speed` | `faster` / `slower`: Holm-adjusted p <= 0.05 (`p_wall_holm`) with S_wall > 1 / < 1; empty = not significant. |
| `eval_saved_s` | Seconds of fitness evaluation saved per run: mean baseline evaluation time minus mean FEC evaluation time (timing runs); negative: FEC slower. |
| `wall_saved_s` | Seconds of whole-run wall time saved per run: mean baseline wall time minus mean FEC wall time (timing runs); negative: FEC slower. |

## Table S3: Full GP results per configuration (`TableS3_results_GP.csv`)

| Column | Meaning |
|---|---|
| `algorithm` | GE (grammatical evolution) or GP (strongly typed genetic programming). |
| `dataset` | Dataset name, as in Table 1 of the paper. |
| `fraction` | Fingerprint fraction: share of the training rows used as fingerprint rows (0.01, 0.05, 0.10); random fingerprint rows. |
| `n_runs` | Number of paired seeds behind the row (baseline, FEC timing and FEC diagnostic run with the same seed). |
| `S_T` | Evaluation-time speedup: mean baseline fitness-evaluation time / mean FEC fitness-evaluation time over the 30 paired seeds; > 1 = FEC faster. |
| `S_T_sd` | Standard deviation over the 30 paired seeds of the per-seed ratio baseline time / FEC time. |
| `p_eval_time` | p-value of the two-sided paired Wilcoxon signed-rank test of FEC against baseline fitness-evaluation time over the paired seeds; uncorrected. |
| `p_eval_holm` | `p_eval_time` Holm-adjusted over the algorithm's 51 configurations (17 datasets x 3 fractions); decides `speed`. |
| `p_eval_bh` | `p_eval_time` Benjamini-Hochberg-adjusted over the same 51 configurations (check only). |
| `speed` | `faster` / `slower`: Holm-adjusted p <= 0.05 (`p_eval_holm`) with S_T > 1 / < 1; empty = not significant. |
| `speed_raw` | The same mark from the uncorrected p (`p_eval_time`), for comparison. |
| `dAcc` | Accuracy difference: mean FEC test accuracy - mean baseline test accuracy over the 30 paired seeds, as a fraction (0.01 = 1 pp); < 0 = FEC less accurate. |
| `bayes_worse` | P(FEC accuracy lower than the baseline's by more than 1 pp), posterior probability from the Bayesian correlated t-test over the paired seeds (Nadeau-Bengio variance correction). |
| `bayes_rope` | P(FEC and baseline accuracy within +-1 pp), posterior probability from the Bayesian correlated t-test over the paired seeds (Nadeau-Bengio variance correction). |
| `bayes_better` | P(FEC accuracy higher than the baseline's by more than 1 pp), posterior probability from the Bayesian correlated t-test over the paired seeds (Nadeau-Bengio variance correction). |
| `verdict` | Bayesian accuracy verdict: `equivalent` if the probability within the ROPE reaches the threshold, else `worse` or `better` if that probability does, else `undecided`. |
| `hit_rate` | Cache hit rate: hits / (hits + misses), pooled over the FEC timing runs. |
| `fake_hit_ratio` | Fake-hit ratio: fake hits / hits in the diagnostic runs (a fake hit returns a cached fitness that differs from the program's fitness on all training rows). |
| `S_wall` | Wall-time speedup: mean baseline run time / mean FEC run time (whole run, FEC including the sampling of the fingerprint rows) over the 30 paired seeds; > 1 = FEC faster. |
| `S_wall_sd` | Standard deviation over the 30 paired seeds of the per-seed wall-time ratio. |
| `p_wall` | p-value of the two-sided paired Wilcoxon signed-rank test of FEC against baseline whole-run wall time over the paired seeds; uncorrected. |
| `p_wall_holm` | `p_wall` Holm-adjusted over the algorithm's 51 configurations (17 datasets x 3 fractions). |
| `wall_speed` | `faster` / `slower`: Holm-adjusted p <= 0.05 (`p_wall_holm`) with S_wall > 1 / < 1; empty = not significant. |
| `eval_saved_s` | Seconds of fitness evaluation saved per run: mean baseline evaluation time minus mean FEC evaluation time (timing runs); negative: FEC slower. |
| `wall_saved_s` | Seconds of whole-run wall time saved per run: mean baseline wall time minus mean FEC wall time (timing runs); negative: FEC slower. |

## Table S4: What the cache hits skipped (`TableS4_hit_work.csv`)

| Column | Meaning |
|---|---|
| `algorithm` | GE (grammatical evolution) or GP (strongly typed genetic programming). |
| `dataset` | Dataset name, as in Table 1 of the paper. |
| `fraction` | Fingerprint fraction: share of the training rows used as fingerprint rows (0.01, 0.05, 0.10); random fingerprint rows. |
| `hit_rate` | Cache hit rate: hits / (hits + misses), pooled over the FEC timing runs. |
| `hit_size_ratio` | Mean size (operator calls) of a program that hit the cache / mean size of one that missed it; diagnostic runs, generations 1-30 pooled. |
| `work_avoided` | Estimated share of the full-evaluation work (operator calls x training rows, every operator call counted as one unit per row) that the hits avoided; the fingerprint step is not counted. |
| `S_T` | Evaluation-time speedup: mean baseline fitness-evaluation time / mean FEC fitness-evaluation time over the 30 paired seeds; > 1 = FEC faster. |

## Table S5: Preprocessing per dataset (`TableS5_preprocessing.csv`)

| Column | Meaning |
|---|---|
| `dataset` | Dataset name, as in Table 1 of the paper. |
| `source` | Where the dataset comes from (repository). |
| `url` | Address of the dataset at its source. |
| `rows_source` | Rows of the table at the source. |
| `columns_source` | Columns of the table at the source. |
| `missing_value_rows_removed` | Rows removed because of missing values. |
| `columns_dropped` | Columns dropped, with the reason. |
| `one_hot` | Categorical columns one-hot encoded. |
| `type_changes` | Columns whose type was changed (cast or override). |
| `label_encoding` | How the class labels were encoded. |
| `balancing` | Class balancing applied (rows before and after), or `none (as published)`. |
| `duplicate_rows_kept` | Exact duplicate rows in the table used (kept). |
| `rows_used` | Rows of the table used in the experiments. |
| `numeric_features` | Numeric features of the table used. |
| `boolean_features` | Boolean (0/1) features of the table used. |
| `classes` | Number of classes. |
| `checksum` | SHA-256 checksum of the table used. |

## Table S6: Grammar properties per dataset (`TableS6_grammar_properties.csv`)

| Column | Meaning |
|---|---|
| `dataset` | Dataset name, as in Table 1 of the paper. |
| `range` | Constant range of the grammar: pm1 / pm2 / pm3 = constants in +-0.99 / +-1.99 / +-2.99 (standardised values); none = no numeric feature. |
| `coverage_min` | Share of the standardised values of the least covered numeric feature (all rows) that lie inside the chosen range. |
| `degenerate_rate` | Share of (numeric feature, constant) pairs whose comparison splits off fewer than 1% or more than 99% of the rows. |
| `fairness_ratio` | P(a boolean feature appears) / P(a numeric feature appears) in initial GE programs, mean over 30 seeded populations; 1 = both feature types equally likely. |
| `min_init_depth` | Smallest initial derivation depth (3, 5, 7 or 9) at which a seeded GE population of 1000 has no invalid individual. |

## Table S7: Accuracy verdicts with a ROPE of ±1 pp and threshold 0.90 (`TableS7_accuracy_rope1pp_p90.csv`)

| Column | Meaning |
|---|---|
| `algorithm` | GE (grammatical evolution) or GP (strongly typed genetic programming). |
| `dataset` | Dataset name, as in Table 1 of the paper. |
| `rows` | Number of rows of the dataset as used. |
| `fraction` | Fingerprint fraction: share of the training rows used as fingerprint rows (0.01, 0.05, 0.10); random fingerprint rows. |
| `dAcc_pp` | Accuracy difference as in `dAcc`, in percentage points (pp). |
| `dAcc_sd_pp` | Standard deviation of the paired accuracy differences, in percentage points (pp). |
| `rope_pp` | Half-width of the region of practical equivalence (ROPE), in percentage points (pp). |
| `threshold` | Posterior probability a verdict must reach. |
| `p_worse` | P(FEC accuracy lower than the baseline's by more than the ROPE), posterior probability from the Bayesian correlated t-test over the paired seeds (Nadeau-Bengio variance correction). |
| `p_rope` | P(FEC and baseline accuracy within the ROPE), posterior probability from the Bayesian correlated t-test over the paired seeds (Nadeau-Bengio variance correction). |
| `p_better` | P(FEC accuracy higher than the baseline's by more than the ROPE), posterior probability from the Bayesian correlated t-test over the paired seeds (Nadeau-Bengio variance correction). |
| `verdict` | Bayesian accuracy verdict: `equivalent` if the probability within the ROPE reaches the threshold, else `worse` or `better` if that probability does, else `undecided`. |

## Table S8: Accuracy verdicts with a ROPE of ±2 pp and threshold 0.95 (`TableS8_accuracy_rope2pp_p95.csv`)

| Column | Meaning |
|---|---|
| `algorithm` | GE (grammatical evolution) or GP (strongly typed genetic programming). |
| `dataset` | Dataset name, as in Table 1 of the paper. |
| `rows` | Number of rows of the dataset as used. |
| `fraction` | Fingerprint fraction: share of the training rows used as fingerprint rows (0.01, 0.05, 0.10); random fingerprint rows. |
| `dAcc_pp` | Accuracy difference as in `dAcc`, in percentage points (pp). |
| `dAcc_sd_pp` | Standard deviation of the paired accuracy differences, in percentage points (pp). |
| `rope_pp` | Half-width of the region of practical equivalence (ROPE), in percentage points (pp). |
| `threshold` | Posterior probability a verdict must reach. |
| `p_worse` | P(FEC accuracy lower than the baseline's by more than the ROPE), posterior probability from the Bayesian correlated t-test over the paired seeds (Nadeau-Bengio variance correction). |
| `p_rope` | P(FEC and baseline accuracy within the ROPE), posterior probability from the Bayesian correlated t-test over the paired seeds (Nadeau-Bengio variance correction). |
| `p_better` | P(FEC accuracy higher than the baseline's by more than the ROPE), posterior probability from the Bayesian correlated t-test over the paired seeds (Nadeau-Bengio variance correction). |
| `verdict` | Bayesian accuracy verdict: `equivalent` if the probability within the ROPE reaches the threshold, else `worse` or `better` if that probability does, else `undecided`. |

## Table S9: Accuracy verdicts with a ROPE of ±2 pp and threshold 0.90 (`TableS9_accuracy_rope2pp_p90.csv`)

| Column | Meaning |
|---|---|
| `algorithm` | GE (grammatical evolution) or GP (strongly typed genetic programming). |
| `dataset` | Dataset name, as in Table 1 of the paper. |
| `rows` | Number of rows of the dataset as used. |
| `fraction` | Fingerprint fraction: share of the training rows used as fingerprint rows (0.01, 0.05, 0.10); random fingerprint rows. |
| `dAcc_pp` | Accuracy difference as in `dAcc`, in percentage points (pp). |
| `dAcc_sd_pp` | Standard deviation of the paired accuracy differences, in percentage points (pp). |
| `rope_pp` | Half-width of the region of practical equivalence (ROPE), in percentage points (pp). |
| `threshold` | Posterior probability a verdict must reach. |
| `p_worse` | P(FEC accuracy lower than the baseline's by more than the ROPE), posterior probability from the Bayesian correlated t-test over the paired seeds (Nadeau-Bengio variance correction). |
| `p_rope` | P(FEC and baseline accuracy within the ROPE), posterior probability from the Bayesian correlated t-test over the paired seeds (Nadeau-Bengio variance correction). |
| `p_better` | P(FEC accuracy higher than the baseline's by more than the ROPE), posterior probability from the Bayesian correlated t-test over the paired seeds (Nadeau-Bengio variance correction). |
| `verdict` | Bayesian accuracy verdict: `equivalent` if the probability within the ROPE reaches the threshold, else `worse` or `better` if that probability does, else `undecided`. |

## Table S10: Accuracy verdicts with a ROPE of ±3 pp and threshold 0.95 (`TableS10_accuracy_rope3pp_p95.csv`)

| Column | Meaning |
|---|---|
| `algorithm` | GE (grammatical evolution) or GP (strongly typed genetic programming). |
| `dataset` | Dataset name, as in Table 1 of the paper. |
| `rows` | Number of rows of the dataset as used. |
| `fraction` | Fingerprint fraction: share of the training rows used as fingerprint rows (0.01, 0.05, 0.10); random fingerprint rows. |
| `dAcc_pp` | Accuracy difference as in `dAcc`, in percentage points (pp). |
| `dAcc_sd_pp` | Standard deviation of the paired accuracy differences, in percentage points (pp). |
| `rope_pp` | Half-width of the region of practical equivalence (ROPE), in percentage points (pp). |
| `threshold` | Posterior probability a verdict must reach. |
| `p_worse` | P(FEC accuracy lower than the baseline's by more than the ROPE), posterior probability from the Bayesian correlated t-test over the paired seeds (Nadeau-Bengio variance correction). |
| `p_rope` | P(FEC and baseline accuracy within the ROPE), posterior probability from the Bayesian correlated t-test over the paired seeds (Nadeau-Bengio variance correction). |
| `p_better` | P(FEC accuracy higher than the baseline's by more than the ROPE), posterior probability from the Bayesian correlated t-test over the paired seeds (Nadeau-Bengio variance correction). |
| `verdict` | Bayesian accuracy verdict: `equivalent` if the probability within the ROPE reaches the threshold, else `worse` or `better` if that probability does, else `undecided`. |

## Table S11: Accuracy verdicts with a ROPE of ±3 pp and threshold 0.90 (`TableS11_accuracy_rope3pp_p90.csv`)

| Column | Meaning |
|---|---|
| `algorithm` | GE (grammatical evolution) or GP (strongly typed genetic programming). |
| `dataset` | Dataset name, as in Table 1 of the paper. |
| `rows` | Number of rows of the dataset as used. |
| `fraction` | Fingerprint fraction: share of the training rows used as fingerprint rows (0.01, 0.05, 0.10); random fingerprint rows. |
| `dAcc_pp` | Accuracy difference as in `dAcc`, in percentage points (pp). |
| `dAcc_sd_pp` | Standard deviation of the paired accuracy differences, in percentage points (pp). |
| `rope_pp` | Half-width of the region of practical equivalence (ROPE), in percentage points (pp). |
| `threshold` | Posterior probability a verdict must reach. |
| `p_worse` | P(FEC accuracy lower than the baseline's by more than the ROPE), posterior probability from the Bayesian correlated t-test over the paired seeds (Nadeau-Bengio variance correction). |
| `p_rope` | P(FEC and baseline accuracy within the ROPE), posterior probability from the Bayesian correlated t-test over the paired seeds (Nadeau-Bengio variance correction). |
| `p_better` | P(FEC accuracy higher than the baseline's by more than the ROPE), posterior probability from the Bayesian correlated t-test over the paired seeds (Nadeau-Bengio variance correction). |
| `verdict` | Bayesian accuracy verdict: `equivalent` if the probability within the ROPE reaches the threshold, else `worse` or `better` if that probability does, else `undecided`. |

## Table S12: Cache hits by equivalence group per dataset (`TableS12_hit_kinds.csv`)

| Column | Meaning |
|---|---|
| `algorithm` | GE (grammatical evolution) or GP (strongly typed genetic programming). |
| `dataset` | Dataset name, as in Table 1 of the paper. |
| `rows` | Number of rows of the dataset as used. |
| `fraction` | Fingerprint fraction: share of the training rows used as fingerprint rows (0.01, 0.05, 0.10); random fingerprint rows. |
| `runs` | Number of FEC diagnostic runs (seeds) behind the row. |
| `hits_per_run` | Mean number of cache hits per diagnostic run. |
| `structural_mean` | Share of the hits that were structurally equivalent (the same program as the one stored under the key), mean over the seeds. |
| `structural_sd` | Standard deviation over the seeds of `structural_mean`. |
| `fitness_equivalent_mean` | Share of the hits that were fitness-equivalent (a different program with the same fitness on all training rows), mean over the seeds. |
| `fitness_equivalent_sd` | Standard deviation over the seeds of `fitness_equivalent_mean`. |
| `nonequiv_mean` | Share of the hits that were not equivalent, i.e. fake hits (a different program with a different fitness), mean over the seeds. |
| `nonequiv_sd` | Standard deviation over the seeds of `nonequiv_mean`. |
| `split_runs` | Number of instrumented diagnostic runs (seeds) that recorded the predictions behind `behaviour_equivalent_share`; empty where none was run. |
| `behaviour_equivalent_share` | Share of the fitness-equivalent hits whose program predicted every training row as the stored program did (behaviour-equivalent), pooled over `split_runs` runs; the rest had equal fitness with different predictions. |

## Table S13: How large the fitness errors of fake hits were (`TableS13_fake_severity.csv`)

Only generations with at least one fake hit are counted. The diagnostic runs record, per generation, the mean and the largest difference over the hits, not each fake hit.

| Column | Meaning |
|---|---|
| `algorithm` | GE (grammatical evolution) or GP (strongly typed genetic programming). |
| `dataset` | Dataset name, as in Table 1 of the paper. |
| `rows` | Number of rows of the dataset as used. |
| `fraction` | Fingerprint fraction: share of the training rows used as fingerprint rows (0.01, 0.05, 0.10); random fingerprint rows. |
| `fake_hits` | Number of fake hits, over all diagnostic runs (30 seeds). |
| `mean_error_pp` | Mean absolute difference between the cached and the true (all training rows) error rate per fake hit, in percentage points (pp). |
| `mean_signed_pp` | Mean of (cached - true error rate) per fake hit, in percentage points (pp); negative = the cache reported a lower error than the true one. |
| `worst_median_pp` | Median over the generations of the largest difference in a generation, in percentage points (pp). |
| `worst_p90_pp` | 90th percentile over the generations of the largest difference in a generation, in percentage points (pp). |
| `worst_max_pp` | Largest difference observed, in percentage points (pp). |
| `gens_worst_over_1pp` | Share of the generations whose largest difference exceeded 1 pp. |
| `gens_worst_over_5pp` | Share of the generations whose largest difference exceeded 5 pp. |

## Table S14: Accuracy verdicts with a ROPE relative to the baseline's error (`TableS14_accuracy_relative_rope.csv`)

| Column | Meaning |
|---|---|
| `algorithm` | GE (grammatical evolution) or GP (strongly typed genetic programming). |
| `dataset` | Dataset name, as in Table 1 of the paper. |
| `rows` | Number of rows of the dataset as used. |
| `fraction` | Fingerprint fraction: share of the training rows used as fingerprint rows (0.01, 0.05, 0.10); random fingerprint rows. |
| `baseline_accuracy` | Mean test accuracy of the baseline runs (1 - test error rate of the run's best program). |
| `dAcc_pp` | Accuracy difference as in `dAcc`, in percentage points (pp). |
| `verdict_rope1pp` | Bayesian accuracy verdict with the predefined ROPE of ±1 pp (as Table 5). |
| `rope_rel5_pp` | Width of the relative ROPE, ±5 % of the baseline's error rate, in percentage points. |
| `verdict_rel5` | Bayesian accuracy verdict with the ROPE of ±5 % of the baseline's error rate. |
| `rope_rel10_pp` | Width of the relative ROPE, ±10 % of the baseline's error rate, in percentage points. |
| `verdict_rel10` | Bayesian accuracy verdict with the ROPE of ±10 % of the baseline's error rate. |
| `rope_rel20_pp` | Width of the relative ROPE, ±20 % of the baseline's error rate, in percentage points. |
| `verdict_rel20` | Bayesian accuracy verdict with the ROPE of ±20 % of the baseline's error rate. |

## Table S15: Accuracy verdicts under other correlations between the runs (`TableS15_accuracy_correlation.csv`)

| Column | Meaning |
|---|---|
| `algorithm` | GE (grammatical evolution) or GP (strongly typed genetic programming). |
| `dataset` | Dataset name, as in Table 1 of the paper. |
| `rows` | Number of rows of the dataset as used. |
| `fraction` | Fingerprint fraction: share of the training rows used as fingerprint rows (0.01, 0.05, 0.10); random fingerprint rows. |
| `dAcc_pp` | Accuracy difference as in `dAcc`, in percentage points (pp). |
| `p_worse_kappa0.0` | P(FEC accuracy lower by more than 1 pp), Bayesian correlated t-test with an assumed correlation of 0.0 between the runs. |
| `p_rope_kappa0.0` | P(FEC and baseline accuracy within ±1 pp), same test with correlation 0.0. |
| `verdict_kappa0.0` | Bayesian accuracy verdict (threshold 0.95) with correlation 0.0. |
| `p_worse_kappa0.1` | P(FEC accuracy lower by more than 1 pp), Bayesian correlated t-test with an assumed correlation of 0.1 between the runs. |
| `p_rope_kappa0.1` | P(FEC and baseline accuracy within ±1 pp), same test with correlation 0.1. |
| `verdict_kappa0.1` | Bayesian accuracy verdict (threshold 0.95) with correlation 0.1. |
| `p_worse_kappa0.2` | P(FEC accuracy lower by more than 1 pp), Bayesian correlated t-test with an assumed correlation of 0.2 between the runs. |
| `p_rope_kappa0.2` | P(FEC and baseline accuracy within ±1 pp), same test with correlation 0.2. |
| `verdict_kappa0.2` | Bayesian accuracy verdict (threshold 0.95) with correlation 0.2 (as Table 5). |
| `p_worse_kappa0.3` | P(FEC accuracy lower by more than 1 pp), Bayesian correlated t-test with an assumed correlation of 0.3 between the runs. |
| `p_rope_kappa0.3` | P(FEC and baseline accuracy within ±1 pp), same test with correlation 0.3. |
| `verdict_kappa0.3` | Bayesian accuracy verdict (threshold 0.95) with correlation 0.3. |
| `p_worse_kappa0.4` | P(FEC accuracy lower by more than 1 pp), Bayesian correlated t-test with an assumed correlation of 0.4 between the runs. |
| `p_rope_kappa0.4` | P(FEC and baseline accuracy within ±1 pp), same test with correlation 0.4. |
| `verdict_kappa0.4` | Bayesian accuracy verdict (threshold 0.95) with correlation 0.4. |
| `p_worse_kappa0.5` | P(FEC accuracy lower by more than 1 pp), Bayesian correlated t-test with an assumed correlation of 0.5 between the runs. |
| `p_rope_kappa0.5` | P(FEC and baseline accuracy within ±1 pp), same test with correlation 0.5. |
| `verdict_kappa0.5` | Bayesian accuracy verdict (threshold 0.95) with correlation 0.5. |

## Table S16: Dataset size and fingerprint size modelled together (`TableS16_size_vs_fingerprint.csv`)

| Column | Meaning |
|---|---|
| `algorithm` | GE (grammatical evolution) or GP (strongly typed genetic programming). |
| `measure` | Measure modelled: S_T, hit_rate, fake_hit_ratio, error_per_fake_pp (mean |cached - full-training error| per fake hit, pp, as Table S13) or abs_dAcc_pp (|Delta Acc|, pp). |
| `model` | rows+fp: log10(measure) ~ log10(rows) + log10(fingerprint rows); rows+fp+work: the same plus the share of the work avoided (Table S4). |Delta Acc| is modelled in pp without log. |
| `term` | log_rows: tenfold larger dataset at the same number of fingerprint rows; log_fp: tenfold larger fingerprint at the same dataset; rows_same_fraction: tenfold larger dataset at the same fingerprint fraction (log_rows + log_fp); work_avoided: 0.1 more of the work avoided. |
| `coef` | Least-squares coefficient of the term (log10 units of the measure; pp for abs_dAcc_pp). |
| `ci_low` | Lower end of the 95 % percentile bootstrap interval of coef (datasets resampled, 2,000 times). |
| `ci_high` | Upper end of the 95 % percentile bootstrap interval of coef. |
| `loo_min` | Smallest coef when one dataset is left out (17 refits). |
| `loo_max` | Largest coef when one dataset is left out. |
| `factor` | 10^coef: the multiplier of the measure for the change named in `term` (empty for abs_dAcc_pp). |
| `factor_low` | 10^ci_low. |
| `factor_high` | 10^ci_high. |
| `r2` | Share of the variance of the (log) measure explained by the model, over the 51 configurations. |
| `configurations` | Configurations in the model (datasets x fractions). |
| `datasets` | Datasets in the model. |

## Table S17: A simple cost model for the speedup (`TableS17_cost_model.csv`)

| Column | Meaning |
|---|---|
| `algorithm` | GE (grammatical evolution) or GP (strongly typed genetic programming). |
| `dataset` | Dataset name, as in Table 1 of the paper. |
| `rows` | Number of rows of the dataset as used. |
| `fraction` | Fingerprint fraction: share of the training rows used as fingerprint rows (0.01, 0.05, 0.10); random fingerprint rows. |
| `S_T` | Evaluation-time speedup: mean baseline fitness-evaluation time / mean FEC fitness-evaluation time over the 30 paired seeds; > 1 = FEC faster. |
| `fp_cost` | Measured cost of the fingerprint step per new program, relative to the full evaluation of a missed program (c; phase-timing runs, mean over seeds). |
| `work_avoided` | Estimated share of the full-evaluation work (operator calls x training rows, every operator call counted as one unit per row) that the hits avoided; the fingerprint step is not counted. |
| `S_T_predicted` | Speedup predicted by the cost model, 1/(1 + fp_cost - work_avoided); no fitted parameter. |
| `error_pct` | Prediction error, 100 x (S_T_predicted / S_T - 1), in percent. |
| `rule_right` | True when the rule work_avoided > fp_cost gives the observed side of S_T = 1 (faster or not). |
| `work_avoided_gen1_5` | work_avoided from generations 1 to 5 only, as in a short pilot run. |
| `rule_right_gen1_5` | rule_right with work_avoided_gen1_5. |
| `work_avoided_gen1_10` | work_avoided from generations 1 to 10 only. |
| `rule_right_gen1_10` | rule_right with work_avoided_gen1_10. |

## Table S18: Memory used by the cache (`TableS18_cache_memory.csv`)

| Column | Meaning |
|---|---|
| `algorithm` | GE (grammatical evolution) or GP (strongly typed genetic programming). |
| `dataset` | Dataset name, as in Table 1 of the paper. |
| `rows` | Number of rows of the dataset as used. |
| `fraction` | Fingerprint fraction: share of the training rows used as fingerprint rows (0.01, 0.05, 0.10); random fingerprint rows. |
| `runs` | Number of FEC diagnostic runs (seeds) behind the row. |
| `evaluations_median` | Table S18: evaluations per run (hits + misses), median over the seeds. |
| `entries_median` | Table S18: cache entries at the end of the run, median over the seeds. |
| `entries_max` | Table S18: cache entries at the end of the run, largest over the seeds. |
| `cache_mb_median` | Table S18: cache size in MB as run (key to fitness and key to program text), median. |
| `cache_mb_max` | Table S18: cache size in MB as run (key to fitness and key to program text), largest. |
| `fitness_only_mb_median` | Table S18: cache size in MB without the program text (key to fitness only), median. |
| `fitness_only_mb_max` | Table S18: cache size in MB without the program text (key to fitness only), largest. |
| `peak_mb_median` | Table S18: peak memory of the whole process in MB (diagnostic run), median. |
| `peak_mb_max` | Table S18: peak memory of the whole process in MB (diagnostic run), largest. |
| `cache_share_of_peak_median` | Table S18: cache size as run divided by the peak memory of the process, median. |
| `cache_share_of_peak_max` | Table S18: cache size as run divided by the peak memory of the process, largest. |
| `hits_kept_last5` | Table S18: share of hits whose cache entry was at most 5 generations old (hit logs). |
| `hits_kept_last10` | Table S18: share of hits whose cache entry was at most 10 generations old (hit logs). |

## Table S19: How wrong cached fitness values affect selection (`TableS19_selection.csv`)

| Column | Meaning |
|---|---|
| `algorithm` | GE (grammatical evolution) or GP (strongly typed genetic programming). |
| `dataset` | Dataset name, as in Table 1 of the paper. |
| `fraction` | Fingerprint fraction: share of the training rows used as fingerprint rows (0.01, 0.05, 0.10); random fingerprint rows. |
| `fp_rows` | Number of fingerprint rows (random rows, as recorded by the FEC runs). |
| `runs` | Number of FEC diagnostic runs (seeds) behind the row. |
| `wrong_share_median` | Table S19: share of valid programs whose fitness used differs from the true fitness (mean over generations), median over seeds. |
| `rank_corr_median` | Table S19: Spearman correlation of used and true fitness across the population (mean over generations), median over seeds. |
| `rank_corr_lowest` | Table S19: lowest Spearman correlation of used and true fitness in any generation of any seed. |
| `changed_tournaments_median` | Table S19: share of tournaments whose winner was truly worse than another aspirant, median over seeds. |
| `changed_tournaments_max` | Table S19: share of tournaments whose winner was truly worse than another aspirant, largest seed. |
| `worse_by_pp_median` | Table S19: how much worse the winner truly was (pp of error, mean over those tournaments), median over seeds. |
| `elite_wrong_generations` | Table S19: generations, over all seeds, whose best program carried a wrong fitness. |

## Table S20: How far FEC runs drift from their baseline (`TableS20_divergence.csv`)

| Column | Meaning |
|---|---|
| `algorithm` | GE (grammatical evolution) or GP (strongly typed genetic programming). |
| `dataset` | Dataset name, as in Table 1 of the paper. |
| `fraction` | Fingerprint fraction: share of the training rows used as fingerprint rows (0.01, 0.05, 0.10); random fingerprint rows. |
| `runs` | Number of FEC diagnostic runs (seeds) behind the row. |
| `diverged_share` | Table S20: share of FEC runs whose population separated from the baseline run of the same seed. |
| `first_divergence_gen_median` | Table S20: median generation in which the population first differed from the baseline's (separated runs). |
| `train_diff_abs_pp_end` | Table S20: mean absolute difference of the best training error, FEC minus baseline, last generation (pp). |
| `train_diff_pp_end` | Table S20: mean difference of the best training error, FEC minus baseline, last generation (pp; above 0 = FEC worse). |
| `test_diff_pp_end` | Table S20: mean difference of the test error of the best program, FEC minus baseline, last generation (pp). |
