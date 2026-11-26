"""paper_tables.py: corpus, store loading, tables and supplementary checks."""
import numpy as np
import pandas as pd
import pytest

import paper_tables as pt
import results


NEEDS_STORE = pytest.mark.skipif(bool(pt.missing_inputs()), reason="needs the full results store: "
                                 "results/store/generations.csv.gz is a release file (see README, step 3)")


def test_paper_corpus_has_17_datasets():
    """The paper uses the 17 datasets run with GE and GP (13 binary, 4 multiclass)."""
    corpus = pt.paper_corpus()
    assert len(corpus) == 17 and corpus.dataset.is_unique
    assert (corpus.task == "binary").sum() == 13
    assert not set(corpus.dataset) & {"Higgs", "poker", "Klaverjas2018", "Click_prediction_small"}
    assert not set(corpus.dataset) & {"appendicitis", "prnn_crabs", "iris", "satimage"}
    assert set(corpus[corpus.task == "multiclass"].dataset) == {"dry_bean", "sleep", "accelerometer", "lhc_jets"}


def _store_copy(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    for name in ("metrics.csv", "datasets.csv"):
        (store / name).write_bytes((results.STORE / name).read_bytes())
    return store


@pytest.mark.skipif(not (results.STORE / "metrics.csv").is_file(), reason="no results store")
def test_load_store_missing_column_is_loud(tmp_path):
    store = _store_copy(tmp_path)
    m = pd.read_csv(store / "metrics.csv").drop(columns=["S_T"])
    m.to_csv(store / "metrics.csv", index=False)
    with pytest.raises(pt.TableError, match="S_T"):
        pt.load_store(store)


@pytest.mark.skipif(not (results.STORE / "metrics.csv").is_file(), reason="no results store")
def test_load_store_mixed_setup_is_loud(tmp_path):
    store = _store_copy(tmp_path)
    m = pd.read_csv(store / "metrics.csv")
    row = m[m.setup == "default"].iloc[[0]].copy()
    row["setup"] = "other"
    pd.concat([m, row]).to_csv(store / "metrics.csv", index=False)
    with pytest.raises(pt.TableError, match="more than one setup"):
        pt.load_store(store)


def test_load_store_missing_file_is_loud(tmp_path):
    with pytest.raises(pt.TableError, match="store file missing"):
        pt.load_store(tmp_path)


@pytest.mark.skipif(not (results.STORE / "datasets.csv").is_file(), reason="no results store")
def test_corpus_tables_match_store(tmp_path):
    corpus = pt.paper_corpus()
    metrics, props = pt.load_store(only=corpus.dataset)
    paths = pt.corpus_tables(corpus=corpus, props=props, out_tables=tmp_path)
    text = paths[0].read_text()
    rows = [line for line in text.splitlines() if line.startswith("\\texttt{")]
    assert len(rows) == 17
    missing = set(corpus.dataset) - set(props.dataset)
    for line in rows:
        name = line.split("}")[0].replace("\\texttt{", "").replace("\\_", "_")
        assert ("\\RES{not yet in the store}" in line) == (name in missing)
        if name not in missing:
            r = props.set_index("dataset").loc[name]
            assert f"{r.difficulty_gb:.3f}" in line and f"{int(r.rows):,}".replace(",", "{,}") in line


def test_corpus_tables_deterministic(tmp_path):
    corpus = pt.paper_corpus()
    _, props = pt.load_store(only=corpus.dataset)
    a = [p.read_bytes() for p in pt.corpus_tables(corpus=corpus, props=props, out_tables=tmp_path / "a")]
    b = [p.read_bytes() for p in pt.corpus_tables(corpus=corpus, props=props, out_tables=tmp_path / "b")]
    assert a == b


# ---------------------------------------------------------------- results and supplementary (synthetic store)

def _metrics(datasets=("a", "b"), algorithms=("GE", "GP")):
    rows = []
    for algo in algorithms:
        for di, d in enumerate(datasets):
            for s in pt.SAMPLER_ORDER:
                for f in (0.01, 0.05, 0.10):
                    fast = (d == "a" and f >= 0.05)            # a: significant from 5 %; b: never
                    rows.append({"algorithm": algo, "setup": "default", "dataset": d, "sampler": s, "fraction": f,
                                 "n_runs": 30, "S_T": 1.30 + di * 0.1 if fast else 1.01, "S_T_sd": 0.1,
                                 "dAcc": -0.001234, "S_E": 2.0, "S_wall": 1.1, "hit_rate": 0.5, "fake_hit_ratio": 0.1,
                                 "sampling_time_sec": 0.123, "p_eval_time": 1e-4 if fast else 0.4,
                                 "bayes_worse": 0.01, "bayes_rope": 0.97 if d == "a" else 0.5, "bayes_better": 0.02,
                                 "baseline_accuracy": 0.8, "fec_accuracy": 0.799})
    return pt.with_holm(pd.DataFrame(rows))


def _corpus(datasets=("a", "b", "c")):
    return pd.DataFrame({"dataset": list(datasets), "experiment": "x", "task": "binary"})


def _props(datasets=("a", "b", "c"), rows=(10000, 2000, 50000)):
    return pd.DataFrame({"dataset": list(datasets), "rows": list(rows), "difficulty_class": "hard"})


def test_verdict_rule():
    assert pt.verdict(0.0, 0.95, 0.05) == "equivalent"
    assert pt.verdict(0.96, 0.04, 0.0) == "worse"
    assert pt.verdict(0.0, 0.04, 0.96) == "better"
    assert pt.verdict(0.3, 0.4, 0.3) == "undecided"
    assert pt.verdict(float("nan"), 0.5, 0.5) == ""


def test_holm_family_is_algorithm_and_sampler():
    """Speed is decided on Holm over one algorithm x sampler; a subset keeps the store's p."""
    m = _metrics().copy()
    m.loc[(m.algorithm == "GE") & (m.dataset == "a") & (m.fraction == 0.05), "p_eval_time"] = 0.02   # alone: < 0.05
    m = pt.with_holm(m.drop(columns=["p_eval_holm", "p_eval_bh"]))
    ann = pt.annotated(m)
    x = ann[(ann.algorithm == "GE") & (ann.dataset == "a") & (ann.fraction == 0.05)]
    assert (x.speed_raw == "faster").all() and (x.speed == "").all()
    # family of 6 (2 datasets x 3 fractions): 0.02 is the second smallest p (after 1e-4), so Holm gives 5 x 0.02
    assert np.allclose(x.p_eval_holm, 0.10)
    sub = pt.annotated(m[(m.algorithm == "GE") & (m.dataset == "a")])
    assert list(sub.speed) == list(ann[(ann.algorithm == "GE") & (ann.dataset == "a")].speed)
    with pytest.raises(pt.TableError):
        pt.annotated(m.drop(columns="p_eval_holm"))


def test_smallest_significant_fraction_and_ns():
    ann = pt.annotated(_metrics())
    chosen = pt.chosen_rows(ann[ann.algorithm == "GE"])
    a = chosen[chosen.dataset == "a"]
    b = chosen[chosen.dataset == "b"]
    assert set(a.fraction) == {0.05} and a.significant.all() and (a.speed == "faster").all()
    assert set(b.fraction) == {0.01} and not b.significant.any()
    assert len(chosen) == 6                                      # 2 datasets x 3 samplers


def test_results_table_marks_running_and_excludes(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "MAIN_TEXT_EXCLUDED", ("b",))
    paths = pt.results_tables(corpus=_corpus(), metrics=_metrics(), props=_props(), out_tables=tmp_path)
    text = paths[0].read_text()
    assert "\\texttt{a}" in text and "\\texttt{b} &" not in text
    assert "\\texttt{c} & \\multicolumn{8}{c}{\\RES{running}}" in text
    assert "0.05" in text and "(n.s.)" not in text and "-0.12" in text and "0.12" in text


@NEEDS_STORE
def test_min_rows_only_main_table(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "MAIN_TEXT_EXCLUDED", ())
    t = pt.results_tables(corpus=_corpus(), metrics=_metrics(), props=_props(), out_tables=tmp_path, min_rows=5000)
    text = t[0].read_text()
    assert "\\texttt{b} &" not in text and "fewer than 5,000 rows" in text
    supp = tmp_path / "supp"
    supp.mkdir()
    (supp / "INDEX.md").write_text(pt.INDEX_HEAD + "\n|---|---|---|---|---|---|---|\n")
    pt.supplementary(corpus=_corpus(), metrics=_metrics(), props=_props(), out_tables=tmp_path, out_supp=supp,
                     folder=supp)
    full = pd.read_csv(supp / "TableS2_results_GE.csv")
    assert set(full.dataset) == {"a", "b"} and len(full) == 6          # random rows only: 2 datasets x 3 fractions


def test_ge_vs_gp_one_pair_per_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "MAIN_TEXT_EXCLUDED", ())
    m = _metrics()
    text = pt.ge_vs_gp_table(corpus=_corpus(), metrics=m[m.sampler == "random"], props=_props(),
                             out_tables=tmp_path)[0].read_text()
    rows = [l for l in text.splitlines() if l.startswith(("$S_T$", "hit rate", "$S_E$", "fake-hit ratio", "$\\Delta$Acc"))]
    assert len(rows) == 4                                          # one row per measure (S_T, hit rate, fake hits, dAcc)
    assert all(l.split(" & ")[4] .endswith("of 2") for l in rows)   # one pair per dataset (mean over fractions)
    assert all(l.split(" & ")[7].strip(" \\") in ("yes", "no") for l in rows)   # same outcome within every fraction


def _supp_folder(tmp_path, rows_extra=""):
    folder = tmp_path / "supp"
    folder.mkdir()
    (folder / "INDEX.md").write_text("# index\n\n" + pt.INDEX_HEAD + "\n|---|---|---|---|---|---|---|\n" + rows_extra)
    return folder


def _run_supp(tmp_path, folder, manuscript=None, size_limit=pt.SIZE_LIMIT):
    return pt.supplementary(corpus=_corpus(), metrics=_metrics(), props=_props(), out_tables=tmp_path / "t",
                            out_supp=folder, folder=folder, manuscript=manuscript, size_limit=size_limit)


@NEEDS_STORE
def test_supplementary_valid_index_writes_list(tmp_path):
    folder = _supp_folder(tmp_path, "| Figure S1 | fig.png | A figure | d | app page 6 | | |\n")
    (folder / "fig.png").write_bytes(b"x" * 10)
    _run_supp(tmp_path, folder)
    text = (tmp_path / "t" / "supplementary.tex").read_text()
    assert "Figure S1: A figure; Figure S2: Fake hits" in text and "Figure S3: Accuracy difference" in text
    assert "Figure S3: Accuracy difference at the fractions with the most and fewest fake hits; Figure S4: Fake hits" in text
    assert "against the number of fingerprint rows; Table S1: Metrics" in text
    _, rows = pt.read_index(folder / "INDEX.md")
    assert [r["id"] for r in rows] == ["Figure S1", "Figure S2", "Figure S3", "Figure S4"] + [f"Table S{i}" for i in range(1, 21)] + ["File S1"]
    assert all(r["size"] and r["committed"] == "yes" for r in rows)


def test_supplementary_problems_are_loud(tmp_path):
    folder = _supp_folder(tmp_path, "| Table S1 | ghost.csv | dup | d | s | | |\n")
    (folder / "stray.pdf").write_bytes(b"x")
    ms = tmp_path / "m.tex"
    ms.write_text("see Figure S9 and Table S2")
    with pytest.raises(pt.TableError) as e:
        _run_supp(tmp_path, folder, manuscript=ms)
    msg = str(e.value)
    assert "stray.pdf" in msg and "ghost.csv" in msg and "id used twice: Table S1" in msg and "Figure S9" in msg


@NEEDS_STORE
def test_supplementary_large_file_ignored(tmp_path):
    folder = _supp_folder(tmp_path)
    _run_supp(tmp_path, folder, size_limit=100)
    ignored = (folder / ".gitignore").read_text().split()
    assert "TableS1_metrics.csv" in ignored


def test_check_ignores_provenance_header(tmp_path):
    old, new = tmp_path / "old.tex", tmp_path / "new.tex"
    old.write_text("% written by paper_tables.py from x; code aaa; do not edit by hand\nbody\n")
    new.write_text("% written by paper_tables.py from x; code bbb; do not edit by hand\nbody\n")
    assert pt.same_content(new, old)
    new.write_text("% written by paper_tables.py from x; code bbb; do not edit by hand\nchanged\n")
    assert not pt.same_content(new, old)


def test_all_faster_threshold():
    rows = []
    for d, r, fast in (("s", 100, False), ("m", 1000, True), ("x", 2000, False), ("l", 5000, True), ("xl", 9000, True)):
        rows += [{"dataset": d, "rows": r, "speed": "faster" if fast else ""}] * 3
    assert pt.all_faster_threshold(pd.DataFrame(rows)) == ("l", 5000.0)    # 'm' is fast but 'x' above it is not
    none = pd.DataFrame([{"dataset": "a", "rows": 1, "speed": ""}])
    assert pt.all_faster_threshold(none) is None


def test_numbers_macros_from_synthetic_store(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "MAIN_TEXT_EXCLUDED", ())
    text = pt.numbers(corpus=_corpus(), metrics=_metrics(), props=_props(), out_tables=tmp_path)[0].read_text()
    assert "\\newcommand{\\resGEFaster}{6}" in text            # dataset a: 3 samplers x 5 %, 10 % significant
    assert "\\newcommand{\\resGEVerdictEquivalent}{9}" in text  # dataset a: P(rope) 0.97 in all 9 configurations
    assert "\\newcommand{\\resNdatasetsBoth}{2}" in text


def test_outcome_counts_groups():
    ann = pt.annotated(_metrics())
    t = pt.outcome_counts(ann[ann.algorithm == "GE"])
    a, b = t.loc["a"], t.loc["b"]
    assert (a.faster, a.comparable, a.both, a.group) == (6, 9, 6, "some")     # faster at 5 % and 10 % only
    assert (b.faster, b.comparable, b.both, b.group) == (0, 0, 0, "none")


def test_claims_flags_a_broken_claim(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pt, "MAIN_TEXT_EXCLUDED", ())
    m = _metrics()
    main = m[m.sampler == "random"]
    path = pt.claims(corpus=_corpus(), metrics=main, all_metrics=m, props=_props(), out_tables=tmp_path)[0]
    text = path.read_text()
    assert "GE/fraction-speed" in text and "GE/samplers-similar" not in text      # random rows only
    # synthetic data: dataset a is faster only at 5 % and 10 %, so the median S_T rises with the fraction
    assert "FAIL GE/fraction-speed" in text
    assert "CLAIM NO LONGER HOLDS [GE/fraction-speed]" in capsys.readouterr().out


def test_size_model_separates_rows_and_fingerprint():
    """With the fingerprint a fixed fraction of each table, the joint model recovers a measure
    that depends only on the fingerprint rows (b_rows = 0) and one that depends only on the table (b_fp = 0)."""
    import numpy as np, pandas as pd
    rows = [2_000 * 1.6 ** i for i in range(12)]
    d = pd.DataFrame([{"algorithm": "GE", "dataset": f"d{i}", "rows": n, "fraction": f, "fp_rows": round(0.8 * n * f),
                       "work_avoided": 0.5} for i, n in enumerate(rows) for f in (0.01, 0.05, 0.10)])
    d["fake_hit_ratio"] = 3.0 * d.fp_rows ** -0.5
    d["S_T"] = 0.2 * d.rows ** 0.2
    d["hit_rate"] = d["error_per_fake_pp"] = d["fake_hit_ratio"]
    d["abs_dAcc_pp"] = 0.1
    pt._SIZE_CACHE.clear()
    t = pt.size_model(d).set_index(["measure", "model", "term"])
    assert abs(t.loc[("fake_hit_ratio", "rows+fp", "log_fp"), "coef"] + 0.5) < 1e-6
    assert abs(t.loc[("fake_hit_ratio", "rows+fp", "log_rows"), "coef"]) < 1e-3
    assert abs(t.loc[("S_T", "rows+fp", "log_rows"), "coef"] - 0.2) < 1e-3
    assert abs(t.loc[("S_T", "rows+fp", "log_fp"), "coef"]) < 1e-3
    assert abs(t.loc[("S_T", "rows+fp", "rows_same_fraction"), "factor"] - 10 ** 0.2) < 1e-2
