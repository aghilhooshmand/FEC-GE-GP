"""Experiment pipeline: one file per experiment, laptop -> compute server -> laptop.

An experiment is experiments/<name>.yaml:
    name: line1_ge_main
    algorithm: GE                          # GE or GP (typed GP, typed_gp.py)
    datasets: [MagicTelescope, covertype]
    seeds: 30-59                           # "a-b" or a list
    fec: {samplers: [farthest_point, random, stratified], fractions: [0.01, 0.05, 0.10]}
    params: {}                             # GE parameter overrides (the "setup"); usually empty
    hive: {workers: 39, max_busy: 0.10}
    save_individuals: false                # true: every individual of every generation (large; small datasets)

Commands (python experiment.py <command> experiments/<name>.yaml):
  prepare        laptop: make its datasets ready (table, constant range, grammar, init depth);
                 refuses uncommitted code or records
  pilot [--hive] the first two seeds, on the laptop or on the compute server (batch.py pilot)
  launch         push the commit to the compute server, prepare there, start the full batch in the background
  status         progress on the compute server
  fetch          copy the experiment's results from the compute server and check each run
The batch engine is batch.py; results are results/ge/<setup>/<dataset>/<config>/seed<k>.json and
every experiment has a manifest, results/experiments/<name>.json.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

import datasets
import fec
import ge
import grammar
import preprocessing
from datasets import DatasetError

ROOT = datasets.ROOT
HIVE_HOST = os.environ.get("FEC_REMOTE_HOST", "")  # user@host of the compute server
HIVE_PATH = os.environ.get("FEC_REMOTE_PATH", "projects/FEC")   # relative to the remote home directory
HIVE_REMOTE = os.environ.get("FEC_REMOTE_GIT", "hive")         # git remote: <host>:projects/FEC.git (bare)
KEYS = {"name", "algorithm", "datasets", "seeds", "fec", "params", "hive", "save_individuals", "modes", "phase_timing"}
MODES = ("baseline", "fec", "fec_diagnostic")
COMMITTED = ("*.py", "requirements.txt", "data/recipes.yaml", "data/corpus.json", "data/constant_ranges.csv",
             "data/complexity.csv", "grammars", "experiments")


# ---------------------------------------------------------------- the experiment file

def _seeds(value) -> list[int]:
    if isinstance(value, str) and "-" in value:
        lo, hi = value.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in (value if isinstance(value, list) else [value])]


def load(path: Path) -> dict:
    """The experiment as a dict with defaults filled in; every value checked."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    unknown = set(raw) - KEYS
    if unknown:
        raise DatasetError(path.stem, f"unknown experiment key(s) {sorted(unknown)}; allowed {sorted(KEYS)}")
    algorithm = raw.get("algorithm", "GE")
    if algorithm not in ("GE", "GP"):
        raise DatasetError(path.stem, "algorithm must be GE or GP")
    recipes = datasets.load_recipes()
    names = raw.get("datasets") or []
    missing = [n for n in names if n not in recipes]
    if not names or missing:
        raise DatasetError(path.stem, f"datasets must be recipe names; unknown: {missing}")
    f = raw.get("fec") or {}
    samplers = f.get("samplers", list(fec.SAMPLERS))
    fractions = [float(x) for x in f.get("fractions", [0.01, 0.05, 0.10])]
    if set(samplers) - set(fec.SAMPLERS) or not all(0 < x <= 1 for x in fractions):
        raise DatasetError(path.stem, f"samplers must be in {fec.SAMPLERS}, fractions in (0, 1]")
    params = raw.get("params") or {}
    import typed_gp
    known = typed_gp.PARAMS if algorithm == "GP" else {k: v for k, v in ge.PARAMS.items() if k != "min_init_depth"}
    bad = [k for k in params if k not in known]
    if bad:
        raise DatasetError(path.stem, f"unknown or fixed GE parameter(s) {bad}")
    hive = {"workers": None, "max_busy": 0.10, **(raw.get("hive") or {})}
    if not isinstance(raw.get("save_individuals", False), bool):
        raise DatasetError(path.stem, "save_individuals must be true or false")
    modes = list(raw.get("modes", MODES))
    if not modes or set(modes) - set(MODES):
        raise DatasetError(path.stem, f"modes must be a non-empty subset of {MODES}")
    phase = raw.get("phase_timing", False)
    if not isinstance(phase, bool) or (phase and modes != ["fec"]):
        raise DatasetError(path.stem, "phase_timing: true needs modes: [fec] (FEC timing runs only)")
    return {"name": raw.get("name", path.stem), "path": path, "algorithm": algorithm, "datasets": names,
            "seeds": _seeds(raw.get("seeds", "30-59")), "samplers": samplers, "fractions": fractions,
            "params": params, "setup": ge.phase_setup(params) if phase else ge.setup_id(params), "hive": hive,
            "save_individuals": raw.get("save_individuals", False), "modes": modes, "phase_timing": phase}


# ---------------------------------------------------------------- local steps

def git(*args: str, check: bool = True) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=check).stdout.strip()


def uncommitted() -> list[str]:
    return [line[3:] for line in git("status", "--porcelain", "--", *COMMITTED).splitlines()]


def prepare(exp: dict, remote: bool = False) -> None:
    """Make every dataset ready. On the server (remote) only tables are created, then checked against the
    committed records: a remote table must be byte-identical to the laptop's."""
    for name in exp["datasets"]:
        table = datasets.DATA_DIR / f"{name}.csv"
        record = datasets.read_corpus()["datasets"].get(name)
        if not table.is_file() or record is None or datasets.sha256_file(table) != record["checksum"]:
            datasets.recreate(name, datasets.load_recipes()[name])
        if remote:
            continue
        try:
            preprocessing.constant_range(name)
        except DatasetError:
            preprocessing.write_row(preprocessing.measure(name))
        if grammar.check_one(name):
            grammar.build(name)
        try:
            ge.recorded_depth(name)
        except DatasetError:
            ge.record_depth(name)
    if remote:
        git("checkout", "--", "data/corpus.json")             # records come from the laptop, unchanged
        if datasets.verify(exp["datasets"], datasets.load_recipes()):
            raise DatasetError(exp["name"], "Remote tables differ from the laptop's records")
        return
    dirty = uncommitted()
    if dirty:
        raise DatasetError(exp["name"], f"commit these first (the server runs the committed state): {dirty}")
    print(f"{exp['name']}: {len(exp['datasets'])} dataset(s) ready, commit {git('rev-parse', '--short', 'HEAD')}")


# ---------------------------------------------------------------- compute server

def ssh(command: str, timeout: int = 600) -> str:
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HIVE_HOST,
                          f"cd {HIVE_PATH} && {command}"], capture_output=True, text=True, timeout=timeout)
    if out.returncode:
        raise DatasetError("hive", f"`{command}` failed: {out.stderr.strip() or out.stdout.strip()}")
    return out.stdout


def manifest_path(exp: dict) -> Path:
    return ROOT / "results" / "experiments" / f"{exp['name']}.json"


def update_manifest(exp: dict, **fields) -> None:
    path = manifest_path(exp)
    m = json.loads(path.read_text()) if path.is_file() else {"name": exp["name"], "events": []}
    m.update(experiment_file=exp["path"].read_text(), setup=exp["setup"], datasets=exp["datasets"])
    m["events"].append({"time": datetime.datetime.now().isoformat(timespec="seconds"), **fields})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m, indent=1) + "\n")


def launch(exp: dict, pilot: bool = False) -> None:
    prepare(exp)
    head = git("rev-parse", "HEAD")
    git("push", "-q", HIVE_REMOTE, "main")
    ssh("git pull -q --ff-only")
    remote_head = ssh("git rev-parse HEAD").strip()
    if remote_head != head:
        raise DatasetError("hive", f"the server is at {remote_head[:10]}, laptop at {head[:10]}")
    rel = exp["path"].resolve().relative_to(ROOT)
    ssh(f".venv/bin/python experiment.py remote-prepare {rel}", timeout=3600)
    command = "pilot" if pilot else "run"
    log = f"logs/{exp['name']}_{command}_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
    ssh(f"mkdir -p logs && (nohup .venv/bin/python batch.py {command} {rel} > {log} 2>&1 < /dev/null &)")
    update_manifest(exp, step=f"launch {command}", host=HIVE_HOST, commit=head, log=log)
    print(f"{exp['name']}: {command} started on the compute server at commit {head[:10]}; log {HIVE_PATH}/{log}")


def status(exp: dict) -> None:
    rel = exp["path"].resolve().relative_to(ROOT)
    running = ssh(f"pgrep -f 'batch.py (run|pilot) {rel}' >/dev/null && echo running || echo 'not running'").strip()
    print(f"{exp['name']} on the compute server: batch {running}")
    print(ssh(f".venv/bin/python batch.py status {rel}"), end="")
    print(ssh(f"ls -t logs/{exp['name']}_*.log 2>/dev/null | head -1 | xargs -r grep -c FAIL || true").strip()
          + " failure line(s) in the latest log")


def fetch(exp: dict) -> None:
    """Copy the experiment's results from the compute server and check every run against the laptop's records."""
    corpus = datasets.read_corpus()["datasets"]
    accepted = rejected = 0
    for name in exp["datasets"]:
        algo = exp["algorithm"].lower()
        local = ROOT / "results" / algo / exp["setup"] / name
        local.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", f"{HIVE_HOST}:{HIVE_PATH}/results/{algo}/{exp['setup']}/{name}/", f"{local}/"],
                       check=True, timeout=3600)
        for path in local.rglob("seed*.json"):
            r = json.loads(path.read_text())
            if r["table_checksum"] != corpus[name]["checksum"] or r["setup"] != exp["setup"]:
                print(f"REJECT {path.relative_to(ROOT)}: table or setup differs from the laptop's")
                path.rename(path.with_suffix(".rejected"))
                rejected += 1
            elif exp.get("phase_timing"):                   # twin check again on the laptop
                import phase_timing
                if phase_timing.check_twin(path)["ok"]:
                    accepted += 1
                else:
                    print(f"MISMATCH {path.relative_to(ROOT)}: differs from its stored twin")
                    rejected += 1
            else:
                accepted += 1
    update_manifest(exp, step="fetch", accepted=accepted, rejected=rejected)
    print(f"{exp['name']}: {accepted} run(s) on the laptop, {rejected} rejected")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["prepare", "pilot", "launch", "status", "fetch", "remote-prepare"])
    p.add_argument("experiment", type=Path)
    p.add_argument("--hive", action="store_true", help="pilot: run it on the compute server")
    args = p.parse_args(argv)
    try:
        exp = load(args.experiment)
        if args.command == "prepare":
            prepare(exp)
        elif args.command == "remote-prepare":
            prepare(exp, remote=True)
        elif args.command == "pilot":
            if args.hive:
                launch(exp, pilot=True)
            else:
                prepare(exp)
                import batch
                return batch.main(["pilot", str(exp["path"])])
        elif args.command == "launch":
            launch(exp)
        elif args.command == "status":
            status(exp)
        else:
            fetch(exp)
        return 0
    except (DatasetError, subprocess.SubprocessError) as exc:
        print(f"FAIL {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
