"""
Whisper Corpus Data Sufficiency Check
======================================
Scans a speech corpus for instances of:
  1. Standalone "up" (preposition/particle — not part of a V+up phrase)
  2. V+up phrases (e.g. "pick up", "set up")
  3. "up" within words (e.g. "cup", "puppet", "support")

Reports counts and assesses whether we have enough data to run the
layer-by-layer Whisper classifier analysis.

Supported corpus sources (select via --source):
  librispeech  — loads via HuggingFace datasets (no download needed)
  local        — reads plain-text transcript files from a directory

Usage examples:
    # Check LibriSpeech train-clean-100 (default, no setup required)
    python check_whisper_corpus.py --source librispeech

    # Check a local directory of .txt transcript files (one sentence per line)
    python check_whisper_corpus.py --source local --transcript-dir /path/to/transcripts

    # Check a specific LibriSpeech split
    python check_whisper_corpus.py --source librispeech --librispeech-split train.360

Requirements:
    pip install datasets spacy
    python -m spacy download en_core_web_sm
"""

import argparse
import collections
import logging
import os
import re

import spacy
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MIN_SENTENCE_LEN = 10
MAX_SENTENCES_PER_VUP = 50      # cap for collection (not for counting)
MIN_FREQ_VUP      = 5           # minimum V+up occurrences to include in report
N_TARGET_STANDALONE = 2000      # target number of standalone-up sentences
N_TARGET_VUP_TYPES  = 20        # target number of distinct V+up types with >= MIN_FREQ_VUP

UPWORD_RE = re.compile(r'\b([a-z]*up[a-z]+|[a-z]+up[a-z]*)\b', re.IGNORECASE)
UPWORD_EXCLUDE = {"up", "ups"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def is_verb_up_context(doc, up_idx):
    if up_idx == 0:
        return None
    prev = doc[up_idx - 1]
    if prev.pos_ == "VERB":
        return f"{prev.text.lower()} up"
    return None


def process_sentences(sentences, nlp):
    """
    Process a list of sentences through spaCy and count:
      - V+up type frequencies
      - V+up sentences (up to cap)
      - standalone "up" sentences
      - up-within-word frequencies

    Returns a dict of results.
    """
    vup_freq      = collections.Counter()
    vup_sentences = collections.defaultdict(list)
    up_sentences  = []
    up_freq       = 0
    upword_freq   = collections.Counter()
    upword_sentences = collections.defaultdict(list)

    for doc in tqdm(
        nlp.pipe(sentences, batch_size=512),
        total=len(sentences),
        desc="Parsing",
        unit="sent",
    ):
        sent_text = doc.text.strip()
        if len(sent_text) < MIN_SENTENCE_LEN:
            continue

        sent_lower = sent_text.lower()

        # V+up and standalone "up"
        if " up" in sent_lower or sent_lower.startswith("up "):
            for tok in doc:
                if tok.text.lower() != "up":
                    continue
                vup_label = is_verb_up_context(doc, tok.i)
                if vup_label:
                    vup_freq[vup_label] += 1
                    if len(vup_sentences[vup_label]) < MAX_SENTENCES_PER_VUP:
                        vup_sentences[vup_label].append(sent_text)
                else:
                    up_freq += 1
                    if len(up_sentences) < N_TARGET_STANDALONE:
                        up_sentences.append(sent_text)

        # "up" within words
        for match in UPWORD_RE.findall(sent_lower):
            word = match.lower()
            if word in UPWORD_EXCLUDE:
                continue
            upword_freq[word] += 1
            if len(upword_sentences[word]) < MAX_SENTENCES_PER_VUP:
                upword_sentences[word].append(sent_text)

    return {
        "vup_freq":       vup_freq,
        "vup_sentences":  dict(vup_sentences),
        "up_freq":        up_freq,
        "up_sentences":   up_sentences,
        "upword_freq":    upword_freq,
        "upword_sentences": dict(upword_sentences),
    }


# ---------------------------------------------------------------------------
# CORPUS LOADERS
# ---------------------------------------------------------------------------

def load_librispeech(split):
    from datasets import load_dataset
    log.info("Loading LibriSpeech split '%s' from HuggingFace ...", split)
    # LibriSpeech is under 'openslr/librispeech_asr'; try the standard path
    def _extract_text(ds):
        # Select only text to avoid triggering audio decoding
        if hasattr(ds, "select_columns"):
            return ds.select_columns(["text"])["text"]
        return [item["text"] for item in ds]

    try:
        ds = load_dataset("openslr/librispeech_asr", "clean", split=split, trust_remote_code=True)
        sentences = _extract_text(ds)
    except Exception:
        # Fallback: some HuggingFace versions use a different loader
        ds = load_dataset("librispeech_asr", "clean", split=split, trust_remote_code=True)
        sentences = _extract_text(ds)
    log.info("Loaded %d utterances.", len(sentences))
    return sentences


def load_local_transcripts(transcript_dir):
    log.info("Loading transcripts from %s ...", transcript_dir)
    sentences = []
    for fname in os.listdir(transcript_dir):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(transcript_dir, fname), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sentences.append(line)
    log.info("Loaded %d lines from %d files.", len(sentences), len(os.listdir(transcript_dir)))
    return sentences


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------

def print_report(results, source_label):
    vup_freq    = results["vup_freq"]
    up_freq     = results["up_freq"]
    up_sents    = results["up_sentences"]
    upword_freq = results["upword_freq"]

    qualifying_vup = {k: v for k, v in vup_freq.items() if v >= MIN_FREQ_VUP}

    print("\n" + "=" * 65)
    print(f" DATA SUFFICIENCY REPORT — {source_label}")
    print("=" * 65)

    # ── Standalone "up" ────────────────────────────────────────────────
    print("\nSTANDALONE \"up\"")
    print(f"  Total occurrences : {up_freq:,}")
    print(f"  Sentences collected: {len(up_sents):,}  (target: {N_TARGET_STANDALONE:,})")
    ok_up = len(up_sents) >= N_TARGET_STANDALONE
    print(f"  Status: {'OK - SUFFICIENT' if ok_up else 'FAIL - INSUFFICIENT — need more data'}")

    # ── V+up phrases ────────────────────────────────────────────────────
    print(f"\nV+UP PHRASES  (min freq >= {MIN_FREQ_VUP})")
    print(f"  Distinct types (all)          : {len(vup_freq):,}")
    print(f"  Distinct types (freq>={MIN_FREQ_VUP:>2d})    : {len(qualifying_vup):,}  (target: ~{N_TARGET_VUP_TYPES:,}+)")
    ok_vup = len(qualifying_vup) >= N_TARGET_VUP_TYPES
    print(f"  Status: {'OK - SUFFICIENT' if ok_vup else 'FAIL - INSUFFICIENT — few V+up types'}")
    print(f"\n  Top 20 V+up types:")
    for vup, cnt in vup_freq.most_common(20):
        print(f"    {vup:25s} {cnt:>6,}")

    # ── "up" within words ────────────────────────────────────────────────
    qualifying_upword = {k: v for k, v in upword_freq.items() if v >= MIN_FREQ_VUP}
    print(f"\n\"UP\" WITHIN WORDS  (min freq >= {MIN_FREQ_VUP})")
    print(f"  Distinct word types (all)      : {len(upword_freq):,}")
    print(f"  Distinct word types (freq>={MIN_FREQ_VUP:>2d}) : {len(qualifying_upword):,}")
    ok_upword = len(qualifying_upword) >= 20
    print(f"  Status: {'OK - SUFFICIENT' if ok_upword else 'FAIL - INSUFFICIENT'}")
    print(f"\n  Top 20 up-within-word types:")
    for word, cnt in upword_freq.most_common(20):
        print(f"    {word:25s} {cnt:>6,}")

    # ── Overall verdict ──────────────────────────────────────────────────
    print("\n" + "-" * 65)
    if ok_up and ok_vup and ok_upword:
        print(" VERDICT: OK - Corpus appears sufficient for the Whisper analysis.")
    else:
        problems = []
        if not ok_up:
            problems.append(f"standalone 'up' sentences (have {len(up_sents)}, need {N_TARGET_STANDALONE})")
        if not ok_vup:
            problems.append(f"distinct V+up types (have {len(qualifying_vup)}, want ~{N_TARGET_VUP_TYPES}+)")
        if not ok_upword:
            problems.append(f"up-within-word types (have {len(qualifying_upword)}, want ~20+)")
        print(" VERDICT: FAIL - Corpus may be insufficient:")
        for p in problems:
            print(f"   - {p}")
        print(" Consider using a larger or more conversational corpus (e.g. Switchboard, CORAAL).")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Check a speech corpus for 'up'-related data sufficiency."
    )
    parser.add_argument(
        "--source",
        choices=["librispeech", "local"],
        default="librispeech",
        help="Corpus source. Default: librispeech",
    )
    parser.add_argument(
        "--librispeech-split",
        default="train.clean.100",
        help="LibriSpeech split to use (train.clean.100, train.clean.360, validation.clean, test.clean). "
             "Default: train.clean.100  (~100h, ~28k utterances)",
    )
    parser.add_argument(
        "--transcript-dir",
        default=None,
        help="Directory of .txt transcript files (one sentence per line). "
             "Required when --source=local.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.source == "librispeech":
        sentences = load_librispeech(args.librispeech_split)
        label = f"LibriSpeech {args.librispeech_split}"
    elif args.source == "local":
        if not args.transcript_dir:
            raise ValueError("--transcript-dir is required when --source=local")
        sentences = load_local_transcripts(args.transcript_dir)
        label = f"Local transcripts ({args.transcript_dir})"

    log.info("Loading spaCy model ...")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    nlp.add_pipe("sentencizer")

    results = process_sentences(sentences, nlp)
    print_report(results, label)


if __name__ == "__main__":
    main()
