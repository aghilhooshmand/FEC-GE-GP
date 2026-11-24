"""Tests for example.py (baseline, timing and diagnostic runs side by side) and diagrams.py (Figures 1 and 2)."""
import diagrams
import example


def test_example_runs_all_three_modes(capsys):
    assert example.main(["--population", "30", "--generations", "3", "--seeds", "30", "31"]) == 0
    out = capsys.readouterr().out
    for algo in ("GE", "GP"):
        assert f"{algo} on prnn_crabs" in out
    assert out.count("over 2 seed(s): S_T = ") == 2
    assert "evolved exactly as its diagnostic run" in out


def test_example_seed_has_three_runs():
    r = example.run_seed("GE", "prnn_crabs", 30, {"population": 20, "generations": 2}, 0.1)
    assert r["baseline"]["mode"] == "baseline" and not r["baseline"].get("fec")
    assert r["timing"]["fec"]["hits"] == r["diagnostic"]["fec"]["hits"]
    assert "fake_hits" in r["diagnostic"]["fec"]


def test_diagrams_build(tmp_path):
    diagrams.check_mapping()                                     # the GE panel is a real derivation
    paths = diagrams.build(tmp_path)
    assert sorted(p.name for p in paths) == ["fig_fec_workflow.pdf", "fig_ge_gp.pdf"]
    first = [p.read_bytes() for p in paths]
    assert all(b.startswith(b"%PDF") for b in first)
    assert [p.read_bytes() for p in diagrams.build(tmp_path)] == first   # a rebuild gives the same files
