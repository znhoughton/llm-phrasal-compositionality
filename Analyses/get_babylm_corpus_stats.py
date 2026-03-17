"""
BabyLM Corpus Statistics
=========================
Computes V+up frequencies, verb surface-form frequencies, and Forward
Transitional Probability (FTP) from the BabyLM training corpus:

    FTP("pick up") = count("pick up") / count("pick")

Two passes over the corpus:
  Pass 1 (spaCy, up-containing docs only): count V+up types using POS tags
  Pass 2 (fast, all docs):                 count surface-form frequencies
                                           for the target verbs only

Output saved to:
    ../Data/babylm_corpus_stats.pkl
    (vup_freq, verb_freq, ftp)

    vup_freq  : Counter  {"pick up": 5000, ...}
    verb_freq : Counter  {"pick": 50000, ...}   (surface form, all contexts)
    ftp       : dict     {"pick up": 0.10, ...}

Usage:
    python get_babylm_corpus_stats.py

Requires:
    pip install datasets spacy
    python -m spacy download en_core_web_sm
"""

import collections
import logging
import pickle

import spacy
from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATASET_ID  = "znhoughton/babylm-150m-v3"
DATASET_SPLIT = "train"
OUT_PKL     = "../Data/babylm_corpus_stats.pkl"
MIN_FREQ    = 10   # minimum V+up occurrences to compute FTP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PASS 1: count V+up using spaCy (up-containing docs only)
# ---------------------------------------------------------------------------

def is_verb_up_context(doc, up_idx):
    if up_idx == 0:
        return None
    prev = doc[up_idx - 1]
    if prev.pos_ == "VERB":
        return f"{prev.text.lower()} up"
    return None


def count_vup(ds, nlp):
    vup_freq = collections.Counter()
    up_freq  = 0

    log.info("Pass 1: counting V+up with spaCy (up-containing docs only) ...")
    texts = [
        item["text"][:4000]
        for item in ds
        if item["text"] and " up" in item["text"].lower()
    ]
    log.info("  %d of %d docs contain 'up'", len(texts), len(ds))

    for doc in tqdm(
        nlp.pipe(texts, batch_size=256),
        total=len(texts),
        desc="spaCy pass",
        unit="doc",
    ):
        for sent in doc.sents:
            for tok in sent:
                if tok.text.lower() != "up":
                    continue
                label = is_verb_up_context(sent.as_doc(), tok.i - sent.start)
                if label:
                    vup_freq[label] += 1
                else:
                    up_freq += 1

    log.info("  V+up types found: %d | total V+up tokens: %d | standalone up: %d",
             len(vup_freq), sum(vup_freq.values()), up_freq)
    return vup_freq, up_freq


# ---------------------------------------------------------------------------
# PASS 2: count verb surface-form frequencies (fast, all docs)
# ---------------------------------------------------------------------------

def count_verb_freq(ds, target_verbs):
    """
    Count surface-form occurrences of each target verb across all docs.
    Uses simple whitespace tokenization (no POS tagging) — fast.
    target_verbs: set of lowercase verb strings to count (e.g. {"pick", "set"})
    """
    import re
    token_re = re.compile(r"[a-z']+", re.IGNORECASE)
    verb_freq = collections.Counter()

    log.info("Pass 2: counting verb surface-form frequencies (all %d docs) ...", len(ds))
    for item in tqdm(ds, desc="Verb freq pass", unit="doc"):
        text = item["text"]
        if not text:
            continue
        for token in token_re.findall(text.lower()):
            if token in target_verbs:
                verb_freq[token] += 1

    log.info("  Verbs counted: %d distinct surface forms", len(verb_freq))
    return verb_freq


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    log.info("Loading %s (split=%s) ...", DATASET_ID, DATASET_SPLIT)
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    log.info("Loaded %d documents.", len(ds))

    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    nlp.add_pipe("sentencizer")

    # Pass 1 — V+up counts via spaCy
    vup_freq, up_freq = count_vup(ds, nlp)

    # Pass 2 — verb surface-form counts (only the verbs we need for FTP)
    target_verbs = {vup_type.split()[0] for vup_type in vup_freq if vup_freq[vup_type] >= MIN_FREQ}
    log.info("Target verbs for frequency counting: %d", len(target_verbs))
    verb_freq = count_verb_freq(ds, target_verbs)

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

    log.info("FTP computed for %d V+up types (%d skipped — verb not found in corpus)",
             len(ftp), n_missing)

    log.info("Top 10 V+up types by frequency:")
    for vup_type, cnt in vup_freq.most_common(10):
        ftp_val = ftp.get(vup_type, float("nan"))
        log.info("  %-25s  freq=%8d  FTP=%.4f", vup_type, cnt, ftp_val)

    with open(OUT_PKL, "wb") as f:
        pickle.dump((vup_freq, verb_freq, ftp), f)
    log.info("Saved to %s", OUT_PKL)


if __name__ == "__main__":
    main()
