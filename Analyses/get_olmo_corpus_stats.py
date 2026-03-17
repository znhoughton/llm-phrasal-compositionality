"""
OLMo Corpus Statistics (C4)
============================
Computes verb surface-form frequencies and Forward Transitional Probability
(FTP) for the OLMo analysis, using the local C4 corpus.

    FTP("pick up") = count("pick up") / count("pick")

V+up frequencies are loaded from the existing corpus_results.pkl (already
computed by create_dataset.py). This script only adds the verb unigram
counts needed for the FTP denominator, using a fast single pass (no spaCy).

Output saved to:
    ../Data/olmo_corpus_stats.pkl
    (vup_freq, verb_freq, ftp)

    vup_freq  : Counter  {"pick up": 5000, ...}  (from corpus_results.pkl)
    verb_freq : Counter  {"pick": 50000, ...}     (surface form, all contexts)
    ftp       : dict     {"pick up": 0.10, ...}

Usage:
    python get_olmo_corpus_stats.py [--data-dir DIR] [--vup-pkl PATH]

Requires:
    pip install pyarrow
"""

import argparse
import collections
import glob
import logging
import os
import pickle
import re

import pyarrow as pa
import pyarrow.ipc as ipc
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MIN_FREQ = 10

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[a-z']+", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute verb frequencies and FTP from local C4 corpus."
    )
    parser.add_argument(
        "--data-dir",
        default="../c4_10B_2B_local/train",
        help="Path to local C4 arrow files. Default: ../c4_10B_2B_local/train",
    )
    parser.add_argument(
        "--vup-pkl",
        default="../Data/corpus_results.pkl",
        help="Path to corpus_results.pkl (from create_dataset.py). "
             "Default: ../Data/corpus_results.pkl",
    )
    parser.add_argument(
        "--out-pkl",
        default="../Data/olmo_corpus_stats.pkl",
        help="Output path. Default: ../Data/olmo_corpus_stats.pkl",
    )
    return parser.parse_args()


def get_arrow_files(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "**/*.arrow"), recursive=True))
    return [f for f in files if "data-" in os.path.basename(f)]


def read_arrow(filepath):
    try:
        with pa.memory_map(filepath, "r") as source:
            return ipc.open_stream(source).read_all()
    except pa.lib.ArrowInvalid:
        with pa.memory_map(filepath, "r") as source:
            return ipc.open_file(source).read_all()


def count_verb_freq_from_arrow(arrow_files, target_verbs):
    verb_freq = collections.Counter()

    log.info("Counting verb surface-form frequencies across %d arrow files ...", len(arrow_files))
    for filepath in tqdm(arrow_files, desc="Arrow files", unit="file"):
        table = read_arrow(filepath)
        for text in table.column("text").to_pylist():
            if not text:
                continue
            for token in TOKEN_RE.findall(text.lower()):
                if token in target_verbs:
                    verb_freq[token] += 1

    log.info("Verb freq pass done. %d distinct verbs counted.", len(verb_freq))
    return verb_freq


def main():
    args = parse_args()

    log.info("Loading V+up frequencies from %s ...", args.vup_pkl)
    with open(args.vup_pkl, "rb") as f:
        _, vup_freq, _, _ = pickle.load(f)
    log.info("  V+up types loaded: %d", len(vup_freq))

    # Only count verbs we need for FTP
    target_verbs = {
        vup_type.split()[0]
        for vup_type, cnt in vup_freq.items()
        if cnt >= MIN_FREQ
    }
    log.info("Target verbs to count: %d", len(target_verbs))

    arrow_files = get_arrow_files(args.data_dir)
    log.info("Found %d arrow files in %s", len(arrow_files), args.data_dir)

    verb_freq = count_verb_freq_from_arrow(arrow_files, target_verbs)

    # Compute FTP
    ftp = {}
    n_missing = 0
    for vup_type, cnt in vup_freq.items():
        if cnt < MIN_FREQ:
            continue
        verb = vup_type.split()[0]
        denom = verb_freq.get(verb, 0)
        if denom > 0:
            ftp[vup_type] = cnt / denom
        else:
            n_missing += 1

    log.info("FTP computed for %d V+up types (%d skipped — verb not in corpus)",
             len(ftp), n_missing)
    log.info("Top 10 V+up types by frequency:")
    for vup_type, cnt in vup_freq.most_common(10):
        ftp_val = ftp.get(vup_type, float("nan"))
        log.info("  %-25s  freq=%8d  FTP=%.4f", vup_type, cnt, ftp_val)

    with open(args.out_pkl, "wb") as f:
        pickle.dump((vup_freq, verb_freq, ftp), f)
    log.info("Saved to %s", args.out_pkl)


if __name__ == "__main__":
    main()
