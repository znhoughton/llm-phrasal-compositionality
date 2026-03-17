"""
OLMo Corpus Statistics — HuggingFace C4 Streaming
===================================================
Computes verb surface-form frequencies and Forward Transitional Probability
(FTP) by streaming C4 from HuggingFace instead of reading local arrow files.

    FTP("pick up") = count("pick up") / count("pick")

Both counts come from the same streamed text so the ratio is self-consistent.
Target verbs are derived from the existing test CSVs (so FTP is only computed
for V+up types that actually appear in the test data).

Output saved to:
    ../Data/olmo_corpus_stats.pkl
    (vup_freq, verb_freq, ftp)

    vup_freq  : Counter  {"pick up": 12345, ...}   (streamed bigram counts)
    verb_freq : Counter  {"pick": 123456, ...}      (streamed unigram counts)
    ftp       : dict     {"pick up": 0.10, ...}

Usage:
    python get_olmo_corpus_stats_hf.py [--max-docs N] [--out-pkl PATH]

Requires:
    pip install datasets tqdm
"""

import argparse
import collections
import logging
import pickle
import re
import sys

from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[a-z']+")

DEFAULT_MAX_DOCS = 1_000_000   # ~600M words; enough for reliable FTP estimates


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute verb frequencies and FTP by streaming C4 from HuggingFace."
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=DEFAULT_MAX_DOCS,
        help=f"Number of C4 documents to stream. Default: {DEFAULT_MAX_DOCS:,}",
    )
    parser.add_argument(
        "--data-dir-up",
        default="../Data/olmo-3-7b/Data_up",
        help="Path to OLMo Data_up directory (to read target verb_up types). "
             "Default: ../Data/olmo-3-7b/Data_up",
    )
    parser.add_argument(
        "--out-pkl",
        default="../Data/olmo_corpus_stats.pkl",
        help="Output path. Default: ../Data/olmo_corpus_stats.pkl",
    )
    return parser.parse_args()


def get_target_vup_types(data_dir_up):
    """Read unique verb_up types from all_layers_results.csv (layer 0 only for speed)."""
    import os
    import csv

    csv_path = os.path.join(data_dir_up, "all_layers_results.csv")
    if not os.path.exists(csv_path):
        log.error("Cannot find %s — check --data-dir-up", csv_path)
        sys.exit(1)

    vup_types = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("layer", "").strip() == "0":
                vup_types.add(row["verb_up"].strip().lower())

    log.info("Loaded %d unique V+up types from layer 0 of all_layers_results.csv", len(vup_types))
    return vup_types


def stream_and_count(max_docs, target_verbs, target_vup_types):
    """Stream C4 and count verb unigrams and V+up bigrams."""
    verb_freq = collections.Counter()
    vup_freq  = collections.Counter()

    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)

    with tqdm(total=max_docs, desc="Streaming C4 docs", unit="doc") as pbar:
        for i, example in enumerate(ds):
            if i >= max_docs:
                break

            tokens = TOKEN_RE.findall(example["text"].lower())
            n = len(tokens)

            for j, tok in enumerate(tokens):
                if tok in target_verbs:
                    verb_freq[tok] += 1
                    # Check if next token is "up" (standalone word)
                    if j + 1 < n and tokens[j + 1] == "up":
                        vup = f"{tok} up"
                        if vup in target_vup_types:
                            vup_freq[vup] += 1

            pbar.update(1)

    log.info("Done. Distinct verbs counted: %d, V+up bigrams: %d",
             len(verb_freq), len(vup_freq))
    return verb_freq, vup_freq


def compute_ftp(vup_freq, verb_freq):
    ftp = {}
    n_missing = 0
    for vup_type, cnt in vup_freq.items():
        verb = vup_type.split()[0]
        denom = verb_freq.get(verb, 0)
        if denom > 0:
            ftp[vup_type] = cnt / denom
        else:
            n_missing += 1

    log.info("FTP computed for %d V+up types (%d skipped — verb not seen in stream)",
             len(ftp), n_missing)
    log.info("Top 10 V+up types by stream bigram count:")
    for vup_type, cnt in vup_freq.most_common(10):
        ftp_val = ftp.get(vup_type, float("nan"))
        verb = vup_type.split()[0]
        log.info("  %-25s  bigram=%8d  verb=%8d  FTP=%.4f",
                 vup_type, cnt, verb_freq.get(verb, 0), ftp_val)

    return ftp


def main():
    args = parse_args()

    log.info("Loading target V+up types from %s ...", args.data_dir_up)
    target_vup_types = get_target_vup_types(args.data_dir_up)
    target_verbs = {vup.split()[0] for vup in target_vup_types}
    log.info("Target verbs to track: %d", len(target_verbs))

    log.info("Streaming up to %d documents from allenai/c4 ...", args.max_docs)
    verb_freq, vup_freq = stream_and_count(args.max_docs, target_verbs, target_vup_types)

    ftp = compute_ftp(vup_freq, verb_freq)

    with open(args.out_pkl, "wb") as f:
        pickle.dump((vup_freq, verb_freq, ftp), f)
    log.info("Saved to %s", args.out_pkl)


if __name__ == "__main__":
    main()
