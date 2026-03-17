"""
GigaSpeech Dataset Builder
==========================
Reads up-audio-metadata.csv (GigaSpeech), extracts audio segments,
runs WhisperX forced alignment to get word-level timestamps for "up",
classifies each "up" token as "vup" (V+up) or "word_up" (non-V+up) using
spaCy POS tags on the full segment transcript (cleaned_text), and saves a
dataset CSV.  Classification mirrors build_whisper_dataset.py: if the token
immediately preceding "up" in the spaCy parse is a VERB → "vup", else →
"word_up".

For train/val: use "word_up" rows (mirrors OLMo/BabyLM standalone_up design)
For test:      use "vup" rows (V+up types with >= MIN_FREQ_VUP occurrences)

Output:
    OUT_DIR/dataset.csv
    OUT_DIR/audio/<sid>.wav   (16 kHz mono .wav for each extracted segment)

Columns in dataset.csv (same schema as build_whisper_dataset.py):
    utt_id         : segment id (from 'sid' column)
    audio_path     : path to the extracted .wav
    sampling_rate  : 16000
    up_start       : start time of "up" within the extracted segment (seconds)
    up_end         : end time of "up" within the extracted segment (seconds)
    neg_start      : start time of the negative word
    neg_end        : end time of the negative word
    neg_word       : text of the negative word
    label          : "vup" or "word_up"
    verb_up        : V+up type (e.g. "pick up"), or "" for word_up
    transcript     : cleaned_text for the segment

Usage:
    python build_gigaspeech_dataset.py [--metadata CSV] [--out-dir DIR] [--device cuda]

    # Quick test on first 500 rows
    python build_gigaspeech_dataset.py --max-rows 500

Requires:
    pip install whisperx soundfile spacy pandas tqdm
    python -m spacy download en_core_web_sm
    ffmpeg available on PATH (for .opus / audio segment extraction)
"""

import argparse
import collections
import logging
import os
import random
import subprocess

import numpy as np
import pandas as pd
import soundfile as sf
import spacy
from tqdm import tqdm

import whisperx

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MIN_FREQ_VUP = 5
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
        description="Build Whisper dataset from GigaSpeech up-audio-metadata.csv."
    )
    parser.add_argument(
        "--metadata", default="../../Data/up-audio-metadata.csv",
        help="Path to up-audio-metadata.csv. Default: ../../Data/up-audio-metadata.csv",
    )
    parser.add_argument(
        "--out-dir", default="../../Data/whisper_gigaspeech",
        help="Output directory. Default: ../../Data/whisper_gigaspeech",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device for WhisperX alignment model. Default: cuda",
    )
    parser.add_argument(
        "--compute-type", default="float16",
        help="Compute type for WhisperX. Default: float16",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Process at most this many rows (for testing). Default: no limit",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# CLASSIFICATION: classify "up" tokens in transcript context
# ---------------------------------------------------------------------------

def classify_ups(text, nlp):
    """
    Run spaCy on the full transcript and return a list of
    (token_index, label, verb_up_type) for each "up" token.
      label        : "vup" or "word_up"
      verb_up_type : "pick up" etc., or "" for word_up
    Mirrors the logic in build_whisper_dataset.py (classify_ups).
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
            results.append((tok.i, "word_up", ""))
    return results


# ---------------------------------------------------------------------------
# AUDIO: extract segment from source file via ffmpeg
# ---------------------------------------------------------------------------

def extract_segment(audio_file, begin_time, end_time, out_wav):
    """
    Extract [begin_time, end_time] seconds from audio_file,
    save as 16 kHz mono WAV at out_wav.
    Returns True on success.
    """
    duration = max(end_time - begin_time, 0.05)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(begin_time),
        "-t",  str(duration),
        "-i",  audio_file,
        "-ar", "16000",
        "-ac", "1",
        "-f",  "wav",
        out_wav,
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        return result.returncode == 0 and os.path.exists(out_wav)
    except Exception as e:
        log.debug("ffmpeg failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# WHISPERX: word-level alignment
# ---------------------------------------------------------------------------

def align_utterance(audio_array, transcript, align_model, wx_metadata, device):
    """
    Run WhisperX forced alignment on a single utterance.
    Returns list of word segments [{word, start, end}, ...] or None.
    """
    try:
        duration = len(audio_array) / 16000.0
        segments = [{"text": transcript, "start": 0.0, "end": duration}]
        result = whisperx.align(
            segments, align_model, wx_metadata, audio_array, device,
            return_char_alignments=False,
        )
        return result.get("word_segments", [])
    except Exception as e:
        log.debug("Alignment failed: %s", e)
        return None


def find_up_timestamps(word_segments):
    """Return all word segments whose word is 'up' with valid timestamps."""
    return [
        ws for ws in word_segments
        if ws.get("word", "").lower().strip(".,!?;:\"'") == "up"
        and "start" in ws and "end" in ws
    ]


def sample_negative(word_segments):
    """Pick a random non-'up' word with valid timestamps (>= 20 ms duration)."""
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

    # ---- Load metadata ----
    log.info("Loading metadata from %s ...", args.metadata)
    meta = pd.read_csv(args.metadata)
    log.info("Raw rows: %d", len(meta))

    meta = meta.dropna(subset=["begin_time", "end_time"])
    log.info("After dropping null-timestamp rows: %d", len(meta))

    if "source" in meta.columns:
        before = len(meta)
        meta = meta[meta["source"] == "GigaSpeech"].reset_index(drop=True)
        log.info("After filtering source==GigaSpeech: %d (dropped %d)", len(meta), before - len(meta))

    if args.max_rows is not None:
        meta = meta.head(args.max_rows)
        log.info("Capped to %d rows (--max-rows).", args.max_rows)

    # ---- Load models ----
    log.info("Loading spaCy model ...")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    nlp.add_pipe("sentencizer")

    log.info("Loading WhisperX alignment model (device=%s) ...", args.device)
    align_model, wx_metadata = whisperx.load_align_model(
        language_code="en", device=args.device,
    )

    rows       = []
    vup_counts = collections.Counter()
    n_word_up  = 0
    n_skipped  = 0

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Segments", unit="seg"):
        sid        = str(row["sid"])
        dataset_dir = str(row.get("dataset_dir", "")).rstrip("/\\")
        file_rel   = str(row.get("file", ""))
        audio_file = os.path.join(dataset_dir, file_rel) if dataset_dir else file_rel
        begin_time = float(row["begin_time"])
        end_time   = float(row["end_time"])
        transcript = str(row.get("cleaned_text") or row.get("text", "")).strip()

        if not transcript or "up" not in transcript.lower():
            n_skipped += 1
            continue

        if not os.path.exists(audio_file):
            log.debug("Audio file not found: %s", audio_file)
            n_skipped += 1
            continue

        # Classify "up" tokens using full transcript context
        ups = classify_ups(transcript, nlp)
        if not ups:
            n_skipped += 1
            continue

        # Extract audio segment (skip if already done)
        wav_path = os.path.join(audio_dir, f"{sid}.wav")
        if not os.path.exists(wav_path):
            ok = extract_segment(audio_file, begin_time, end_time, wav_path)
            if not ok:
                log.debug("Segment extraction failed: sid=%s", sid)
                n_skipped += 1
                continue

        # Load extracted wav
        try:
            audio_array, sr = sf.read(wav_path)
            audio_array = audio_array.astype(np.float32)
        except Exception as e:
            log.debug("WAV read failed: %s", e)
            n_skipped += 1
            continue

        if sr != 16000:
            log.warning("Unexpected sample rate %d for sid=%s — skipping", sr, sid)
            n_skipped += 1
            continue

        # WhisperX forced alignment
        word_segs = align_utterance(audio_array, transcript, align_model, wx_metadata, args.device)
        if not word_segs:
            n_skipped += 1
            continue

        # Match the i-th aligned "up" to the i-th spaCy classification
        aligned_ups = find_up_timestamps(word_segs)
        if not aligned_ups:
            n_skipped += 1
            continue

        neg = sample_negative(word_segs)
        if neg is None:
            n_skipped += 1
            continue

        for i, (_, label, verb_up) in enumerate(ups):
            if i >= len(aligned_ups):
                break
            ws = aligned_ups[i]
            rows.append({
                "utt_id":        sid,
                "audio_path":    wav_path,
                "sampling_rate": sr,
                "up_start":      round(ws["start"], 4),
                "up_end":        round(ws["end"],   4),
                "neg_start":     round(neg["start"], 4),
                "neg_end":       round(neg["end"],   4),
                "neg_word":      neg["word"],
                "label":         label,
                "verb_up":       verb_up,
                "transcript":    transcript,
            })
            if label == "vup":
                vup_counts[verb_up] += 1
            else:
                n_word_up += 1

    log.info(
        "Finished: %d rows | %d word_up | %d V+up types | %d skipped",
        len(rows), n_word_up, len(vup_counts), n_skipped,
    )

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, "dataset.csv")
    df.to_csv(out_csv, index=False)
    log.info("Saved dataset to %s", out_csv)

    qualifying = {k: v for k, v in vup_counts.items() if v >= MIN_FREQ_VUP}
    log.info("\n--- Data sufficiency check ---")
    log.info(
        "  word_up occurrences   : %6d  (target: >= 1200 for 1000 train + 200 val)",
        n_word_up,
    )
    log.info(
        "  Qualifying V+up types : %6d  (target: >= 20, min freq=%d)",
        len(qualifying), MIN_FREQ_VUP,
    )
    log.info("  Top 20 V+up types:")
    for vup, cnt in vup_counts.most_common(20):
        log.info("    %-25s  %d", vup, cnt)

    if n_word_up >= 1200 and len(qualifying) >= 20:
        log.info("  STATUS: SUFFICIENT for Whisper analysis.")
    else:
        log.info("  STATUS: MAY BE INSUFFICIENT — check metadata coverage.")


if __name__ == "__main__":
    main()
