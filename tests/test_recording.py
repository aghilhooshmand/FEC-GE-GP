"""014 recording: the per-generation diagnostics leave the evolution unchanged (tests/data reference,
recorded before the change) and are filled in every generation. Tiny runs only."""
import json
from pathlib import Path

import pytest

import ge
import typed_gp

REFERENCE = json.loads((Path(__file__).parent / "data" / "recording_reference.json").read_text())
SMALL = {"population": 30, "generations": 3}
SHARED = ("unique_phenotypes", "novel_phenotypes", "cumulative_phenotypes", "avg_depth", "avg_terminals",
          "avg_operators", "avg_pheno_len", "avg_operators_used", "avg_features_used", "fitness_diversity",
          "best_terminals", "best_operators", "best_features_used", "best_pheno_len")
GE_ONLY = ("avg_genome_length", "avg_used_codons", "structural_diversity")
FEC = ("fec_unique_keys", "fec_hit_operators", "fec_miss_operators")
DIAGNOSTIC = ("fec_fitness_diff", "fec_abs_fitness_diff", "fec_max_abs_fitness_diff")


@pytest.mark.parametrize("ref", REFERENCE, ids=lambda r: f"{r['algorithm']}-{r['dataset']}-{'fec' if r['fec'] else 'base'}")
def test_same_evolution_and_new_columns(ref):
    engine = ge if ref["algorithm"] == "GE" else typed_gp
    r = engine.run(ref["dataset"], 30, SMALL, ref["fec"])
    assert r["best_phenotype"] == ref["best_phenotype"]
    assert r["best_test_error"] == ref["best_test_error"]
    assert r["n_train_evals"] == ref["n_train_evals"]
    for old, new in zip(ref["generations"], r["generations"], strict=True):
        assert {k: new[k] for k in old} == old                      # every recorded value unchanged
        expected = SHARED + (GE_ONLY if ref["algorithm"] == "GE" else ()) + (FEC + DIAGNOSTIC if ref["fec"] else ())
        assert set(expected) <= set(new), set(expected) - set(new)
        if not ref["fec"]:
            assert not set(FEC) & set(new)                           # cache columns only in FEC runs
    last = r["generations"][-1]
    assert 0 < last["unique_phenotypes"] <= SMALL["population"]
    assert last["cumulative_phenotypes"] >= last["unique_phenotypes"]
    assert r["generations"][0]["novel_phenotypes"] == r["generations"][0]["unique_phenotypes"]
    if ref["fec"]:
        assert sum(g["fec_unique_keys"] for g in r["generations"]) >= last["fec_cache_size"]


@pytest.mark.parametrize("engine, fec", [(ge, None), (ge, {"fraction": 0.1, "sampler": "random"}),
                                         (typed_gp, {"fraction": 0.1, "sampler": "random"})],
                         ids=["GE-base", "GE-fec", "GP-fec"])
def test_individuals_file(engine, fec, tmp_path, monkeypatch):
    """Optional: on -> population x (generations + 1) rows beside the run; off -> no file; the
    evolution is the same either way."""
    import pandas as pd
    import datasets
    monkeypatch.setattr(datasets, "ROOT", tmp_path / "on")   # save_result writes under ROOT/results
    off = engine.run("iris", 30, SMALL, fec)
    on = engine.run("iris", 30, SMALL, fec, save_individuals=True)
    assert len(on["individuals"]) == SMALL["population"] * (SMALL["generations"] + 1)
    assert on["best_phenotype"] == off["best_phenotype"] and on["n_train_evals"] == off["n_train_evals"]
    path = ge.save_result(on)
    df = pd.read_csv(path.with_name(on["individuals_file"]))
    assert list(df.columns) == list(ge.INDIVIDUAL_COLUMNS) and len(df) == SMALL["population"] * (SMALL["generations"] + 1)
    expected = {"hit", "miss", "invalid", "kept"} if fec else {"kept"}
    assert set(df["fec"].fillna("kept")) <= expected and df.loc[df["gen"] == 0, "fec"].fillna("").ne("kept").all()
    if fec:
        assert (df["fec"] == "hit").any() and (df["fec"] == "miss").any()
    assert "individuals" not in json.loads(path.read_text())
    monkeypatch.setattr(datasets, "ROOT", tmp_path / "off")
    ge.save_result(off)
    assert not list((tmp_path / "off").rglob("*.individuals.csv.gz")) and "individuals_file" not in off
