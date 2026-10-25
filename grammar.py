"""Grammar stage (GE only): one grammar per dataset, binary target.

Constant features (one value, flagged by 001) stay in the table but not in the grammar: they can't
split anything (the reference implementation's convention). A rule may have at most 256 alternatives (GRAPE):
a wider feature rule is split into equal sub-rules.

Template: the reference implementation's typed two-hop grammar (generate_dataset_grammars._typed_body_lines) without
the direct feature-vs-feature shortcut, with the signed <constant> rule of the dataset's 003
range. No comment lines: GRAPE's parser has no comments (a stray "::=" in one is read as a rule).

Reads   data/<name>.csv + data/corpus.json, data/constant_ranges.csv
Writes  grammars/<name>.bnf, grammars/manifest.csv (one row per dataset; both committed)

Commands (python grammar.py <command> <names...> | --all):
  build              write the grammar and its manifest row
  check              stale grammars (table, range or file changed) and recipes without one
  compare-reference  byte comparison with the reference implementation grammars_preprocessed/<name>.bnf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

import datasets
import preprocessing
from datasets import DatasetError

REFERENCE_DIR = datasets.ROOT.parent / "reference" / "grammars"
COLUMNS = ("name", "table_checksum", "range", "n_numeric", "n_boolean", "excluded_constant", "grammar_checksum",
           "code_commit")
MAX_ALTERNATIVES = 256      # GRAPE's codon mapping breaks above 256 alternatives in one rule


def grammar_dir() -> Path:
    return datasets.DATA_DIR.parent / "grammars"


def grammar_features(record: dict) -> tuple[list[int], list[int]]:
    """(numeric, boolean) column positions the grammar uses: every non-constant feature by type.
    GP's typed primitive set takes the same lists (typed_gp.py)."""
    feats = record["features"]
    usable = [i for i, f in enumerate(feats) if "constant" not in f["flags"]]
    return ([i for i in usable if feats[i]["type"] == "numeric"], [i for i in usable if feats[i]["type"] == "boolean"])


MAX_CLASSES = 20     # grammars and typed GP: 2..20 classes
MULTICLASS_TOP = ("<score> ::= where(<bool>,<value>,<value>)", "<value> ::= <score_constant>|<score>",
                  "<score_constant> ::= <c>.<c>")     # <c>: the constants' digit rule
DIGITS = "<c> ::= 0|1|2|3|4|5|6|7|8|9"


def grammar_text(numeric_idx: list[int], boolean_idx: list[int], k: int | None, n_classes: int = 2) -> str:
    """Binary (n_classes 2): the <bool>-rooted grammar. Multiclass (3..20): argmax over one <score>
    per class, <score> an if-then-else (where) with nesting, then the binary grammar's lines unchanged."""
    if n_classes > 2:
        if n_classes > MAX_CLASSES:
            raise DatasetError("grammar", f"{n_classes} classes; at most {MAX_CLASSES}")
        start = f"<start> ::= argmax{n_classes}({','.join(['<score>'] * n_classes)})"
        below = grammar_text(numeric_idx, boolean_idx, k)
        if DIGITS not in below.splitlines():            # no numeric features: no constants, so no <c> rule yet
            below += DIGITS + "\n"
        return "\n".join([start, *MULTICLASS_TOP]) + "\n" + below
    return _binary_text(numeric_idx, boolean_idx, k)


def _binary_text(numeric_idx: list[int], boolean_idx: list[int], k: int | None) -> str:
    """The grammar for these feature positions; k = constant range (1, 2, 3) or None (no numeric)."""
    if not numeric_idx and not boolean_idx:
        raise DatasetError("grammar", "no features")
    bool_alts = ["and_(<bool>,<bool>)", "or_(<bool>,<bool>)", "not_(<bool>)"]
    if numeric_idx:
        bool_alts[0:0] = ["less_than_or_equal(<num>,<num>)", "greater_than_or_equal(<num>,<num>)"]
    if boolean_idx:
        bool_alts.append("<bool_feature>")
    lines = [f"<bool> ::= {'|'.join(bool_alts)}"]
    if numeric_idx:
        lines += ["<num> ::= add(<num>,<num>) | sub(<num>,<num>) | mul(<num>,<num>) | pdiv(<num>,<num>) | <terminal>",
                  "<terminal> ::= <num_feature> | <constant>",
                  *feature_rules("<num_feature>", numeric_idx)]
    if boolean_idx:
        lines += feature_rules("<bool_feature>", boolean_idx)
    if numeric_idx:
        if k == 1:
            lines.append("<constant> ::= -0.<c><c>|+0.<c><c>")
        else:
            lines += ["<constant> ::= -<h>.<c><c>|+<h>.<c><c>", "<h> ::= " + "|".join(str(i) for i in range(k))]
        lines.append("<c> ::= 0|1|2|3|4|5|6|7|8|9")
    return "\n".join(lines) + "\n"


def feature_rules(rule: str, idx: list[int]) -> list[str]:
    """`rule ::= x[i]|...`; above MAX_ALTERNATIVES features, `rule ::= <rule_0>|<rule_1>|...` over sub-rules of
    near-equal size, so every feature is still picked with (almost) the same probability."""
    if len(idx) <= MAX_ALTERNATIVES:
        return [f"{rule} ::= {'|'.join(f'x[{i}]' for i in idx)}"]
    n = -(-len(idx) // MAX_ALTERNATIVES)
    parts = [idx[j * len(idx) // n:(j + 1) * len(idx) // n] for j in range(n)]
    subs = [f"{rule[:-1]}_{j}>" for j in range(n)]
    return [f"{rule} ::= {'|'.join(subs)}"] + [f"{s} ::= {'|'.join(f'x[{i}]' for i in p)}" for s, p in zip(subs, parts)]


def build(name: str) -> dict:
    record = datasets.dataset_record(name)
    if not 2 <= record["n_classes"] <= MAX_CLASSES:
        raise DatasetError(name, f"{record['n_classes']} classes; grammars cover 2..{MAX_CLASSES}")
    table = datasets.DATA_DIR / f"{name}.csv"
    if not table.is_file() or datasets.sha256_file(table) != record["checksum"]:
        raise DatasetError(name, f"table {table.name} missing or changed; run `python datasets.py recreate {name}`")
    features = record["features"]
    constant = [i for i, f in enumerate(features) if "constant" in f["flags"]]   # dead alternatives: left out
    numeric, boolean = grammar_features(record)
    rng = preprocessing.constant_range(name)                     # raises if absent or stale
    k = None if rng is None else round(rng[1] + 0.01)            # 0.99 -> 1, 1.99 -> 2, 2.99 -> 3
    if (rng is None) != (not numeric):
        raise DatasetError(name, "constant range and numeric features disagree")
    path = grammar_dir() / f"{name}.bnf"
    path.parent.mkdir(exist_ok=True)
    path.write_text(grammar_text(numeric, boolean, k, record["n_classes"]), encoding="utf-8", newline="\n")
    row = {"name": name, "table_checksum": record["checksum"], "range": "none" if k is None else f"pm{k}",
           "n_numeric": len(numeric), "n_boolean": len(boolean),
           "excluded_constant": ";".join(features[i]["name"] for i in constant),
           "grammar_checksum": datasets.sha256_file(path),
           "code_commit": datasets.git_commit()}
    datasets.upsert_csv_row(grammar_dir() / "manifest.csv", COLUMNS, row)
    return row


def check_one(name: str) -> list[str]:
    """Why the dataset's grammar must be (re)built: missing, edited, or made from another table/range."""
    manifest = grammar_dir() / "manifest.csv"
    df = pd.read_csv(manifest, dtype=str, keep_default_na=False) if manifest.is_file() else pd.DataFrame(columns=COLUMNS)
    hit = df[df["name"] == name]
    path = grammar_dir() / f"{name}.bnf"
    if hit.empty or not path.is_file():
        return ["no grammar"]
    r, problems = hit.iloc[0], []
    if datasets.sha256_file(path) != r["grammar_checksum"]:
        problems.append("grammar file edited")
    if datasets.dataset_record(name)["checksum"] != r["table_checksum"]:
        problems.append("table changed")
    ranges = preprocessing.load_ranges().set_index("name")
    if name not in ranges.index or ranges.loc[name, "winner"] != r["range"]:
        problems.append("constant range changed")
    return problems


def check() -> int:
    manifest = grammar_dir() / "manifest.csv"
    df = pd.read_csv(manifest, dtype=str, keep_default_na=False) if manifest.is_file() else pd.DataFrame(columns=COLUMNS)
    ranges = preprocessing.load_ranges().set_index("name")
    problems = []
    for _, r in df.iterrows():
        path = grammar_dir() / f"{r['name']}.bnf"
        if not path.is_file() or datasets.sha256_file(path) != r["grammar_checksum"]:
            problems.append(f"{r['name']}: grammar file missing or edited")
        try:
            if datasets.dataset_record(r["name"])["checksum"] != r["table_checksum"]:
                problems.append(f"{r['name']}: stale, table changed")
        except DatasetError:
            problems.append(f"{r['name']}: no dataset record")
        if r["name"] not in ranges.index or ranges.loc[r["name"], "winner"] != r["range"]:
            problems.append(f"{r['name']}: stale, constant range changed")
    problems += [f"{n}: recipe without a grammar" for n in datasets.load_recipes() if n not in set(df["name"])]
    for p in problems:
        print(f"FAIL {p}")
    print(f"check: {len(problems)} problem(s)")
    return len(problems)


def compare_reference(name: str) -> bool:
    ref = REFERENCE_DIR / f"{name}.bnf"
    if not ref.is_file():
        print(f"{name:<24} compare   no reference")
        return True
    same = (grammar_dir() / f"{name}.bnf").read_bytes() == ref.read_bytes()
    print(f"{name:<24} compare   {'ok (byte-identical)' if same else 'FAIL differs from ' + str(ref)}")
    return same


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["build", "check", "compare-reference"])
    p.add_argument("names", nargs="*")
    p.add_argument("--all", action="store_true")
    args = p.parse_args(argv)
    if args.command == "check":
        return 1 if check() else 0
    names = list(datasets.load_recipes()) if args.all else args.names
    if not names:
        p.error("give dataset names or --all")
    failed = 0
    for name in names:
        try:
            if args.command == "build":
                row = build(name)
                print(f"{name:<24} build     {row['range']}  {row['n_numeric']} numeric, {row['n_boolean']} boolean")
            elif not compare_reference(name):
                failed += 1
        except DatasetError as exc:
            print(f"FAIL {exc}")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
