"""
Dataset Creation Script
========================
Creates two datasets from the local C4 corpus:

Dataset 1 (original V+up analysis):
  - vup_sentences:  dict {verb_up -> [sentences]}  e.g. "pick up" -> [...]
  - vup_freq:       Counter of V+up type frequencies over the full corpus
  - up_sentences:   list of sentences containing standalone "up" (not V+up)
  - up_freq:        total count of standalone "up" occurrences

Dataset 2 (new "up within words" analysis):
  - up_word_sentences:  dict {word -> [sentences]}  e.g. "cup" -> [...]
  - up_word_freq:       Counter of word frequencies over the full corpus
  - (up_sentences and up_freq are shared from Dataset 1)

Both datasets are saved as pickle files:
  - corpus_results.pkl         (Dataset 1, same format as before)
  - corpus_results_upwords.pkl (Dataset 2)

Usage:
    python create_datasets.py

Requires:
    pip install pyarrow spacy tqdm
    python -m spacy download en_core_web_sm
"""

import os
import re
import glob
import pickle
import logging
import collections
import multiprocessing as mp

import pyarrow as pa
import pyarrow.ipc as ipc
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR              = "../c4_10B_2B_local/train"
OUT_VUP_PKL           = "../Data/corpus_results.pkl"
OUT_UPWORD_PKL        = "../Data/corpus_results_upwords.pkl"

MAX_SENTENCES_PER_VUP    = 50
MAX_SENTENCES_PER_UPWORD = 50
MIN_FREQ_VUP             = 10
MIN_FREQ_UPWORD          = 10
N_STANDALONE_UP          = 2000   # 1000 train + 1000 val

# Words to EXCLUDE from "up within words" — these are the standalone particle
# "up" or common false positives we don't want
UPWORD_EXCLUDE = {"up", "ups"}

# Regex: matches any word that contains "up" as a substring but is not
# itself just "up" or "ups". We capture the full word.
UPWORD_RE = re.compile(r'\b([a-z]*up[a-z]+|[a-z]+up[a-z]*)\b', re.IGNORECASE)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def get_all_arrow_files(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "**/*.arrow"), recursive=True))
    return [f for f in files if "data-" in os.path.basename(f)]


def read_arrow(filepath):
    try:
        with pa.memory_map(filepath, "r") as source:
            return ipc.open_stream(source).read_all()
    except pa.lib.ArrowInvalid:
        with pa.memory_map(filepath, "r") as source:
            return ipc.open_file(source).read_all()


def is_verb_up_context(doc, up_idx):
    if up_idx == 0:
        return None
    prev = doc[up_idx - 1]
    if prev.pos_ == "VERB":
        return f"{prev.text.lower()} up"
    return None


# ---------------------------------------------------------------------------
# WORKER: process one arrow file for BOTH datasets simultaneously
# ---------------------------------------------------------------------------

def process_arrow_file(filepath):
    import spacy, collections, re
    import pyarrow as pa
    import pyarrow.ipc as ipc

    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    nlp.add_pipe("sentencizer")

    upword_re = re.compile(r'\b([a-z]*up[a-z]+|[a-z]+up[a-z]*)\b', re.IGNORECASE)
    upword_exclude = {"up", "ups"}

    # Dataset 1 accumulators
    local_vup_freq  = collections.Counter()
    local_vup_sents = collections.defaultdict(list)
    local_up_freq   = 0
    local_up_sents  = []

    # Dataset 2 accumulators
    local_upword_freq  = collections.Counter()
    local_upword_sents = collections.defaultdict(list)

    try:
        with pa.memory_map(filepath, "r") as source:
            table = ipc.open_stream(source).read_all()
    except pa.lib.ArrowInvalid:
        with pa.memory_map(filepath, "r") as source:
            table = ipc.open_file(source).read_all()

    for text in table.column("text").to_pylist():
        if not text:
            continue

        text_lower = text.lower()
        has_up        = " up " in text_lower or text_lower.startswith("up ") or text_lower.endswith(" up")
        has_upword    = bool(upword_re.search(text_lower))

        if not has_up and not has_upword:
            continue

        doc = nlp(text[:4000])

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if len(sent_text) < 10:
                continue

            sent_lower = sent_text.lower()

            # --- Dataset 1: standalone "up" and V+up ---
            if " up" in sent_lower:
                for tok in sent:
                    if tok.text.lower() != "up":
                        continue
                    vup_label = is_verb_up_context(sent.as_doc(), tok.i - sent.start)
                    if vup_label:
                        local_vup_freq[vup_label] += 1
                        if len(local_vup_sents[vup_label]) < MAX_SENTENCES_PER_VUP:
                            local_vup_sents[vup_label].append(sent_text)
                    else:
                        local_up_freq += 1
                        if len(local_up_sents) < N_STANDALONE_UP:
                            local_up_sents.append(sent_text)

            # --- Dataset 2: words containing "up" as substring ---
            upword_matches = upword_re.findall(sent_lower)
            for word in upword_matches:
                word = word.lower()
                if word in upword_exclude:
                    continue
                local_upword_freq[word] += 1
                if len(local_upword_sents[word]) < MAX_SENTENCES_PER_UPWORD:
                    local_upword_sents[word].append(sent_text)

    return (
        dict(local_vup_freq),  dict(local_vup_sents),  local_up_freq,  local_up_sents,
        dict(local_upword_freq), dict(local_upword_sents),
    )


# ---------------------------------------------------------------------------
# MAIN COLLECTION
# ---------------------------------------------------------------------------

def collect_all(data_dir, n_workers=None):
    arrow_files = get_all_arrow_files(data_dir)
    log.info("Found %d arrow files. Starting parallel collection...", len(arrow_files))

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    log.info("Using %d workers", n_workers)

    # Dataset 1
    vup_freq      = collections.Counter()
    vup_sentences = collections.defaultdict(list)
    up_freq       = 0
    up_sentences  = []

    # Dataset 2
    upword_freq      = collections.Counter()
    upword_sentences = collections.defaultdict(list)

    with mp.Pool(n_workers) as pool:
        for (
            local_vup_freq, local_vup_sents, local_up_freq, local_up_sents,
            local_upword_freq, local_upword_sents,
        ) in tqdm(
            pool.imap_unordered(process_arrow_file, arrow_files),
            total=len(arrow_files),
            desc="Corpus collection",
            unit="file",
        ):
            # Merge Dataset 1
            vup_freq += collections.Counter(local_vup_freq)
            up_freq  += local_up_freq

            for vup_type, sents in local_vup_sents.items():
                if len(vup_sentences[vup_type]) < MAX_SENTENCES_PER_VUP:
                    slots = MAX_SENTENCES_PER_VUP - len(vup_sentences[vup_type])
                    vup_sentences[vup_type].extend(sents[:slots])

            if len(up_sentences) < N_STANDALONE_UP:
                slots = N_STANDALONE_UP - len(up_sentences)
                up_sentences.extend(local_up_sents[:slots])

            # Merge Dataset 2
            upword_freq += collections.Counter(local_upword_freq)

            for word, sents in local_upword_sents.items():
                if len(upword_sentences[word]) < MAX_SENTENCES_PER_UPWORD:
                    slots = MAX_SENTENCES_PER_UPWORD - len(upword_sentences[word])
                    upword_sentences[word].extend(sents[:slots])

    # Filter to qualifying types
    vup_sentences = {
        k: v for k, v in vup_sentences.items()
        if vup_freq[k] > MIN_FREQ_VUP and len(v) > 0
    }
    upword_sentences = {
        k: v for k, v in upword_sentences.items()
        if upword_freq[k] > MIN_FREQ_UPWORD and len(v) > 0
    }

    log.info(
        "Dataset 1 — V+up types (freq>%d): %d | standalone up sentences: %d | up_freq: %d",
        MIN_FREQ_VUP, len(vup_sentences), len(up_sentences), up_freq,
    )
    log.info(
        "Dataset 2 — up-word types (freq>%d): %d | total upword tokens: %d",
        MIN_FREQ_UPWORD, len(upword_sentences), sum(upword_freq.values()),
    )

    return (
        dict(vup_sentences), vup_freq, up_sentences, up_freq,
        dict(upword_sentences), upword_freq,
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    (
        vup_sentences, vup_freq, up_sentences, up_freq,
        upword_sentences, upword_freq,
    ) = collect_all(DATA_DIR)

    # Save Dataset 1
    with open(OUT_VUP_PKL, "wb") as f:
        pickle.dump((vup_sentences, vup_freq, up_sentences, up_freq), f)
    log.info("Saved Dataset 1 to %s", OUT_VUP_PKL)

    # Save Dataset 2 (also includes up_sentences/up_freq since they're shared)
    with open(OUT_UPWORD_PKL, "wb") as f:
        pickle.dump((upword_sentences, upword_freq, up_sentences, up_freq), f)
    log.info("Saved Dataset 2 to %s", OUT_UPWORD_PKL)

    # Print some examples
    print("\n=== Top 10 V+up types ===")
    for vup_type, freq in vup_freq.most_common(10):
        print(f"  {vup_type:25s} {freq:>8,}")

    print("\n=== Top 20 up-within-word types ===")
    for word, freq in upword_freq.most_common(20):
        print(f"  {word:25s} {freq:>8,}")

    print(f"\n=== Standalone 'up' sentences collected: {len(up_sentences)} ===")


if __name__ == "__main__":
    main()