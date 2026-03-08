"""
Whisper Dataset Builder — LibriSpeech
======================================
Scans a LibriSpeech split for utterances containing "up", aligns them with
WhisperX to get word-level timestamps, classifies each "up" as V+up or
standalone using spaCy POS tags, and saves a dataset CSV.

For each utterance a randomly-chosen non-"up" word is also recorded as the
negative example for train/val.

Output:
    OUT_DIR/dataset.csv
    OUT_DIR/audio/<utt_id>.wav     (16 kHz .wav for each utterance with "up")

Columns in dataset.csv:
    utt_id         : unique utterance id
    audio_path     : path to the saved .wav
    sampling_rate  : 16000
    up_start       : start time of "up" in the audio (seconds)
    up_end         : end time of "up" in the audio (seconds)
    neg_start      : start time of the negative (non-"up") word
    neg_end        : end time of the negative word
    neg_word       : text of the negative word
    label          : "vup" or "standalone_up"
    verb_up        : V+up type (e.g. "pick up"), or "" for standalone
    transcript     : full utterance text

Usage:
    python build_whisper_dataset.py [--split SPLIT] [--out-dir DIR] [--device cuda]

    # Check LibriSpeech train-clean-100 (fast, ~100h)
    python build_whisper_dataset.py --split train.clean.100

    # Full train-clean-360 for more V+up coverage
    python build_whisper_dataset.py --split train.clean.360

Requires:
    pip install whisperx soundfile spacy datasets
    python -m spacy download en_core_web_sm
"""

import argparse
import collections
import logging
import os
import random

import numpy as np
import pandas as pd
import soundfile as sf
import spacy
from datasets import load_dataset
from tqdm import tqdm

import whisperx

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MIN_FREQ_VUP = 5      # minimum V+up occurrences to count as qualifying
RANDOM_SEED  = 964

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# PARSE ARGS
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Whisper dataset from LibriSpeech."
    )
    parser.add_argument(
        "--split", default="train.clean.100",
        help="LibriSpeech split. Options: train.clean.100, train.clean.360, "
             "validation.clean, test.clean. Default: train.clean.100",
    )
    parser.add_argument(
        "--out-dir", default="../../Data/whisper",
        help="Output directory. Default: ../../Data/whisper",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device for WhisperX alignment model. Default: cuda",
    )
    parser.add_argument(
        "--compute-type", default="float16",
        help="Compute type for WhisperX. Default: float16",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# SPACY: classify "up" tokens
# ---------------------------------------------------------------------------

def classify_ups(text, nlp):
    """
    Return list of (token_index, label, verb_up_type) for each "up" token.
      label: "vup" or "standalone_up"
      verb_up_type: "pick up" etc., or "" for standalone
    """
    doc = nlp(text)
    results = []
    for tok in doc:
        if tok.text.lower() != "up":
            continue
        if tok.i > 0 and doc[tok.i - 1].pos_ == "VERB":
            verb_up = f"{doc[tok.i - 1].text.lower()} up"
            results.append((tok.i, "vup", verb_up))
        else:
            results.append((tok.i, "standalone_up", ""))
    return results


# ---------------------------------------------------------------------------
# WHISPERX: word-level alignment
# ---------------------------------------------------------------------------

def align_utterance(audio_array, transcript, align_model, metadata, device):
    """
    Run WhisperX forced alignment on a single utterance.
    Returns list of word segments [{"word", "start", "end"}, ...] or None.
    """
    try:
        duration = len(audio_array) / 16000.0
        segments = [{"text": transcript, "start": 0.0, "end": duration}]
        result = whisperx.align(
            segments, align_model, metadata, audio_array, device,
            return_char_alignments=False,
        )
        return result.get("word_segments", [])
    except Exception as e:
        log.debug("Alignment failed: %s", e)
        return None


def match_up_timestamps(word_segments, spacy_ups):
    """
    Match the i-th "up" in word_segments to the i-th "up" in spacy_ups
    (sequential pairing by occurrence order).
    Returns list of (word_segment, label, verb_up_type).
    """
    aligned_ups = [
        ws for ws in word_segments
        if ws.get("word", "").lower().strip(".,!?;:\"'") == "up"
        and "start" in ws and "end" in ws
    ]
    matched = []
    for i, (_, label, verb_up) in enumerate(spacy_ups):
        if i < len(aligned_ups):
            matched.append((aligned_ups[i], label, verb_up))
    return matched


def sample_negative(word_segments):
    """Pick a random non-"up" word with valid timestamps (>= 20ms duration)."""
    candidates = [
        ws for ws in word_segments
        if ws.get("word", "").lower().strip(".,!?;:\"'") != "up"
        and "start" in ws and "end" in ws
        and ws["end"] - ws["start"] >= 0.02
    ]
    return random.choice(candidates) if candidates else None


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    audio_dir = os.path.join(args.out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    # Build list of split names to try: the user-supplied name, plus a variant
    # that strips/adds the "clean." prefix (older librispeech_asr caches use
    # "train.100"/"train.360" while newer openslr/librispeech_asr uses
    # "train.clean.100"/"train.clean.360").
    split = args.split
    if split.startswith("train.clean."):
        split_alt = split.replace("train.clean.", "train.")
    elif split.startswith("train.") and not split.startswith("train.clean."):
        split_alt = split.replace("train.", "train.clean.", 1)
    else:
        split_alt = split

    log.info("Loading LibriSpeech split '%s' from HuggingFace ...", split)
    ds = None
    for repo, sp in [
        ("openslr/librispeech_asr", split),
        ("librispeech_asr",         split),
        ("openslr/librispeech_asr", split_alt),
        ("librispeech_asr",         split_alt),
    ]:
        try:
            ds = load_dataset(repo, "clean", split=sp, trust_remote_code=True)
            log.info("Loaded from %s, split=%s", repo, sp)
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError(
            f"Could not load LibriSpeech split '{split}' (also tried '{split_alt}'). "
            "Check your --split argument and datasets library version."
        )
    log.info("Loaded %d utterances.", len(ds))

    log.info("Loading spaCy model ...")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    nlp.add_pipe("sentencizer")

    log.info("Loading WhisperX alignment model (device=%s) ...", args.device)
    align_model, metadata = whisperx.load_align_model(
        language_code="en", device=args.device,
    )

    rows       = []
    vup_counts = collections.Counter()
    n_standalone = 0
    n_skipped    = 0

    for item in tqdm(ds, desc="Utterances", unit="utt"):
        text = item["text"]
        if not text or "up" not in text.lower():
            continue

        audio_array = np.array(item["audio"]["array"], dtype=np.float32)
        sr = item["audio"]["sampling_rate"]
        if sr != 16000:
            log.warning("Unexpected sample rate %d — skipping", sr)
            n_skipped += 1
            continue

        # spaCy parse
        ups = classify_ups(text, nlp)
        if not ups:
            continue

        # WhisperX alignment
        word_segs = align_utterance(audio_array, text, align_model, metadata, args.device)
        if not word_segs:
            n_skipped += 1
            continue

        matched = match_up_timestamps(word_segs, ups)
        if not matched:
            n_skipped += 1
            continue

        neg = sample_negative(word_segs)
        if neg is None:
            n_skipped += 1
            continue

        # Build utterance id from dataset fields
        utt_id = str(item.get("id", f"utt_{len(rows):06d}")).replace("/", "_").replace(" ", "_")
        audio_path = os.path.join(audio_dir, f"{utt_id}.wav")

        # Save audio (once per utterance — overwrite if same id appears twice, harmless)
        if not os.path.exists(audio_path):
            sf.write(audio_path, audio_array, sr)

        for ws, label, verb_up in matched:
            rows.append({
                "utt_id":        utt_id,
                "audio_path":    audio_path,
                "sampling_rate": sr,
                "up_start":      round(ws["start"], 4),
                "up_end":        round(ws["end"],   4),
                "neg_start":     round(neg["start"], 4),
                "neg_end":       round(neg["end"],   4),
                "neg_word":      neg["word"],
                "label":         label,
                "verb_up":       verb_up,
                "transcript":    text,
            })
            if label == "vup":
                vup_counts[verb_up] += 1
            else:
                n_standalone += 1

    log.info(
        "Finished: %d rows | %d standalone | %d V+up types | %d utterances skipped",
        len(rows), n_standalone, len(vup_counts), n_skipped,
    )

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, "dataset.csv")
    df.to_csv(out_csv, index=False)
    log.info("Saved dataset to %s", out_csv)

    qualifying = {k: v for k, v in vup_counts.items() if v >= MIN_FREQ_VUP}
    log.info("\n--- Data sufficiency check ---")
    log.info("  Standalone 'up' occurrences : %6d  (target: >= 2000)", n_standalone)
    log.info("  Qualifying V+up types       : %6d  (target: >= 20, min freq=%d)",
             len(qualifying), MIN_FREQ_VUP)
    log.info("  Top 20 V+up types:")
    for vup, cnt in vup_counts.most_common(20):
        log.info("    %-25s  %d", vup, cnt)

    if n_standalone >= 2000 and len(qualifying) >= 20:
        log.info("  STATUS: SUFFICIENT for Whisper analysis.")
    else:
        log.info("  STATUS: MAY BE INSUFFICIENT — consider train.clean.360.")


if __name__ == "__main__":
    main()
