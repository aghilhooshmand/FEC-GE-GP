"""Tests for experiment.py (the experiment file and the local steps)."""
import pytest

import experiment as ex
from datasets import DatasetError


def write(tmp_path, text):
    path = tmp_path / "e.yaml"
    path.write_text(text)
    return path


def test_load_defaults_and_validation(tmp_path):
    e = ex.load(write(tmp_path, "datasets: [appendicitis]\n"))
    assert e["seeds"] == list(range(30, 60)) and e["fractions"] == [0.01, 0.05, 0.10]
    assert e["samplers"] == ["farthest_point", "random", "stratified"] and e["setup"] == "default"
    with pytest.raises(DatasetError, match="unknown experiment key"):
        ex.load(write(tmp_path, "datasets: [appendicitis]\nfractions: [0.1]\n"))
    with pytest.raises(DatasetError, match="recipe names"):
        ex.load(write(tmp_path, "datasets: [nope]\n"))
    with pytest.raises(DatasetError, match="GE parameter"):
        ex.load(write(tmp_path, "datasets: [appendicitis]\nparams: {popsize: 3}\n"))
    with pytest.raises(DatasetError, match="samplers"):
        ex.load(write(tmp_path, "datasets: [appendicitis]\nfec: {samplers: [kmeans]}\n"))


def test_params_give_their_own_setup(tmp_path):
    e = ex.load(write(tmp_path, "datasets: [appendicitis]\nparams: {population: 200}\nseeds: [30, 35]\n"))
    assert e["setup"] != "default" and e["seeds"] == [30, 35]


def test_example_experiment_loads():
    e = ex.load(ex.ROOT / "experiments" / "pilot_small.yaml")
    assert e["datasets"] == ["appendicitis", "prnn_crabs"] and e["seeds"] == [30, 31]


def test_save_individuals_reaches_every_run(tmp_path):
    """off by default; true puts --save-individuals on every run command (not on fps jobs)."""
    import batch
    assert ex.load(write(tmp_path, "datasets: [appendicitis]\n"))["save_individuals"] is False
    e = ex.load(write(tmp_path, "datasets: [appendicitis]\nseeds: [30]\nsave_individuals: true\n"))
    jobs = batch.plan(e)
    assert all(("--save-individuals" in j.command()) == (j.phase != "fps") for j in jobs)
    with pytest.raises(DatasetError, match="save_individuals"):
        ex.load(write(tmp_path, "datasets: [appendicitis]\nsave_individuals: yes please\n"))
