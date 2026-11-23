#!/usr/bin/env python3
"""
check_babylm_splits.py
----------------------
Do the BabyLM test sets on this machine include the V+up position fix (commit 15ac008,
2026-07-30)?

Before that fix, resolve_vup_positions() took the LAST "up" token in a sentence without
checking that it belonged to the row's own verb, so a "picked up" row could read the
embedding of a later "build-up". The 350M was regenerated after the fix; this checks
whether the 125M and 1.3B were too. All three share one tokenizer, so a fixed 125M/1.3B
test set should match the 350M's row for row.

Standard library only, so it runs in any Python 3 environment:

    python3 Analyses/check_babylm_splits.py
"""
import csv
import os
import sys
import time

csv.field_size_limit(2**31 - 1)   # sys.maxsize overflows a C long on Windows
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "babylm")
MODELS = ["opt-125m", "opt-350m", "opt-1.3b"]


def load(model):
    """{(verb_up, sentence): token_position} for one model's test set."""
    path = os.path.join(ROOT, model, "Data_up", "test.csv")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows, {(r["verb_up"], r["sentence"]): r["token_position"] for r in rows}


def stamp(path):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path))) \
        if os.path.exists(path) else "MISSING"


print(f"Data directory: {os.path.abspath(ROOT)}\n")
sets = {}
for m in MODELS:
    rows, keyed = load(m)
    sets[m] = keyed
    print(f"{m:9s} test rows: {len(rows):,}   test.csv {stamp(os.path.join(ROOT, m, 'Data_up', 'test.csv'))}")
print("\n(pre-fix files have 78,647 rows; the post-fix 350M has 81,518)\n")

ref = sets["opt-350m"]
for m in ["opt-125m", "opt-1.3b"]:
    s = sets[m]
    shared = s.keys() & ref.keys()
    moved = sum(s[k] != ref[k] for k in shared)
    print(f"{m} vs opt-350m: only in {m}: {len(s.keys() - ref.keys()):,}   "
          f"only in 350m: {len(ref.keys() - s.keys()):,}   "
          f"shared rows reading a different token: {moved:,}")

print("\nResults files (should be dated after the test.csv they were built from):")
for m in MODELS:
    for sub in ["Data_up", "Data_upsubword"]:
        p = os.path.join(ROOT, m, sub, "all_layers_results.csv")
        print(f"  {m:9s} {sub:15s} {stamp(p)}")

print("\nVerdict: if both comparisons show ~0 rows reading a different token, the server's "
      "125M/1.3B results already include the fix. Hundreds of rows means they predate it.")
