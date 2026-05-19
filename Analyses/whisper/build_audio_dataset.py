"""
Audio Dataset Builder
=====================
Reads up-audio-metadata.csv (a mixed-source audio metadata file), extracts
audio segments, runs WhisperX forced alignment to get word-level timestamps
for "up", classifies each "up" token as "vup" (V+up) or "word_up" (non-V+up)
using spaCy POS tags on the full segment transcript (cleaned_text), and saves
a dataset CSV.

Classification: if the token immediately preceding "up" in the spaCy parse is
a VERB → "vup", else → "word_up".

For train/val: use "word_up" rows (mirrors OLMo/BabyLM standalone_up design)
For test:      use "vup" rows (V+up types with >= MIN_FREQ_VUP occurrences)

Output:
    OUT_DIR/dataset.csv
    OUT_DIR/audio/<sid>.wav   (16 kHz mono .wav for each extracted segment)

Columns in dataset.csv:
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

Speed
-----
Three-phase pipeline:

    Phase 1 — filter rows; batch spaCy via nlp.pipe()

    Phase 2 — parallel ffmpeg extraction (--ffmpeg-workers, default 32)

    Phase 3 — parallel WhisperX alignment (--align-workers, default 1):
        Each worker process loads its own model copy and processes its
        assigned segments independently — no audio is shared between workers.
        Each segment is aligned in isolation, exactly as in serial mode.
        With 32 workers and wav2vec2-base (~400 MB each), total VRAM cost
        is ~12.8 GB, well within 80 GB.

Usage:
    python build_audio_dataset.py [--metadata CSV] [--out-dir DIR] [--device cuda]

    # Recommended for H100 with 32 cores:
    python build_audio_dataset.py --align-workers 32

    # Quick test on first 500 rows
    python build_audio_dataset.py --max-rows 500

Requires:
    pip install whisperx soundfile spacy pandas tqdm
    python -m spacy download en_core_web_sm
    ffmpeg available on PATH (for segment extraction from any audio format)
"""

import argparse
import collections
import concurrent.futures
import logging
import math
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
        description="Build Whisper dataset from up-audio-metadata.csv."
    )
    parser.add_argument(
        "--metadata", default="../../Data/up-audio-metadata.csv",
        help="Path to up-audio-metadata.csv. Default: ../../Data/up-audio-metadata.csv",
    )
    parser.add_argument(
        "--out-dir", default="../../Data/whisper_audio",
        help="Output directory. Default: ../../Data/whisper_audio",
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
    parser.add_argument(
        "--ffmpeg-workers", type=int, default=32,
        help="Parallel ffmpeg workers for audio extraction. Default: 32",
    )
    parser.add_argument(
        "--align-workers", type=int, default=1,
        help=(
            "Worker processes for alignment. Each loads its own model copy and "
            "processes its segments in isolation (no audio mixing). "
            "Set to the number of CPU cores for maximum throughput (e.g. 32). "
            "Default: 1"
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# CLASSIFICATION: classify "up" tokens from a pre-parsed spaCy Doc
# ---------------------------------------------------------------------------

def classify_ups_from_doc(doc):
    """
    Given an already-parsed spaCy Doc, return a list of
    (token_index, label, verb_up_type) for each "up" token.
      label        : "vup" or "word_up"
      verb_up_type : "pick up" etc., or "" for word_up

    A token is classified as "vup" if the immediately preceding token is a
    VERB (original check) OR if spaCy's dependency parser labels it as a
    particle (dep_ == 'prt'), which catches cases like "picked it up" where
    the particle does not immediately follow the verb.
    """
    results = []
    for tok in doc:
        if tok.text.lower() != "up":
            continue
        prev_is_verb = tok.i > 0 and doc[tok.i - 1].pos_ == "VERB"
        is_particle  = tok.dep_ == "prt"
        if prev_is_verb or is_particle:
            if prev_is_verb:
                verb = doc[tok.i - 1].text.lower()
            elif tok.head.pos_ == "VERB":
                verb = tok.head.text.lower()
            else:
                verb = ""
            verb_up = f"{verb} up" if verb else "up"
            results.append((tok.i, "vup", verb_up))
        elif tok.dep_ == "prep":
            results.append((tok.i, "word_up", ""))
        # else: adverbs, quantmod, etc. — excluded from training
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
# WHISPERX: word-level alignment (single utterance)
# ---------------------------------------------------------------------------

def align_utterance(audio_array, transcript, align_model, wx_metadata, device):
    """
    Run WhisperX forced alignment on a single utterance in isolation.
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
    """
    Pick a random non-'up' word with valid timestamps (>= 20 ms duration).
    Restricted to purely alphabetic words (word.isalpha()), mirroring the
    token.isalpha() criterion used for negatives in the OLMo/BabyLM pipeline.
    """
    candidates = [
        ws for ws in word_segments
        if ws.get("word", "").lower().strip(".,!?;:\"'") != "up"
        and "start" in ws and "end" in ws
        and ws["end"] - ws["start"] >= 0.02
        and ws.get("word", "").isalpha()
    ]
    return random.choice(candidates) if candidates else None


# ---------------------------------------------------------------------------
# WORKER PROCESS: alignment (used when --align-workers > 1)
#
# Each worker is a fully independent process with its own model copy on the
# GPU.  Segments are never mixed across workers; every call to whisperx.align()
# operates on exactly one utterance's audio, identical to serial mode.
# ---------------------------------------------------------------------------

def _alignment_worker(items_chunk, device):
    """
    Entry point for each spawned worker process.

    Loads the alignment model once, then processes each segment in the chunk
    independently.  Returns list of (orig_idx, word_segments_or_None).
    """
    import whisperx as _wx
    import soundfile as _sf
    import numpy as _np

    align_model, wx_meta = _wx.load_align_model(language_code="en", device=device)

    results = []
    for orig_idx, c in items_chunk:
        try:
            audio, sr = _sf.read(c["wav_path"])
            if sr != 16000:
                results.append((orig_idx, None))
                continue
            audio    = audio.astype(_np.float32)
            duration = len(audio) / 16000.0
            segs     = [{"text": c["transcript"], "start": 0.0, "end": duration}]
            out      = _wx.align(
                segs, align_model, wx_meta, audio, device,
                return_char_alignments=False,
            )
            results.append((orig_idx, out.get("word_segments", [])))
        except Exception:
            results.append((orig_idx, None))

    return results


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

    if args.max_rows is not None:
        meta = meta.head(args.max_rows)
        log.info("Capped to %d rows (--max-rows).", args.max_rows)

    # ====================================================================
    # PHASE 1: pre-filter rows + batch spaCy
    # ====================================================================
    log.info("Phase 1: pre-filtering and batch spaCy classification ...")

    log.info("Loading spaCy model ...")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    # Build candidate list with cheap filters only — no disk I/O yet
    candidates      = []
    n_skipped_early = 0
    for _, row in meta.iterrows():
        sid         = str(row["sid"])
        dataset_dir = str(row.get("dataset_dir", "")).rstrip("/\\")
        file_rel    = str(row.get("file", ""))
        audio_file  = os.path.join(dataset_dir, file_rel) if dataset_dir else file_rel
        begin_time  = float(row["begin_time"])
        end_time    = float(row["end_time"])
        transcript  = str(row.get("cleaned_text") or row.get("text", "")).strip()

        if not transcript or "up" not in transcript.lower():
            n_skipped_early += 1
            continue
        if not os.path.exists(audio_file):
            log.debug("Audio file not found: %s", audio_file)
            n_skipped_early += 1
            continue

        candidates.append({
            "sid":        sid,
            "audio_file": audio_file,
            "begin_time": begin_time,
            "end_time":   end_time,
            "transcript": transcript,
            "wav_path":   os.path.join(audio_dir, f"{sid}.wav"),
        })

    log.info(
        "Candidates after pre-filter: %d  (skipped early: %d)",
        len(candidates), n_skipped_early,
    )

    # Batch spaCy — much faster than one call per row
    transcripts = [c["transcript"] for c in candidates]
    log.info("Running spaCy on %d transcripts (batch_size=256) ...", len(transcripts))
    docs = list(tqdm(
        nlp.pipe(transcripts, batch_size=256),
        total=len(transcripts), desc="spaCy", unit="doc",
    ))

    spacy_valid     = []
    n_skipped_spacy = 0
    for c, doc in zip(candidates, docs):
        ups = classify_ups_from_doc(doc)
        if not ups:
            n_skipped_spacy += 1
            continue
        c["ups"] = ups
        spacy_valid.append(c)

    log.info(
        "After spaCy filter: %d  (no 'up' tokens: %d)",
        len(spacy_valid), n_skipped_spacy,
    )

    # ====================================================================
    # PHASE 2: parallel ffmpeg extraction
    # ====================================================================
    log.info(
        "Phase 2: extracting audio segments (%d workers) ...", args.ffmpeg_workers
    )

    def _extract(c):
        if os.path.exists(c["wav_path"]):
            return True
        return extract_segment(
            c["audio_file"], c["begin_time"], c["end_time"], c["wav_path"]
        )

    extract_ok = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.ffmpeg_workers) as ex:
        future_to_sid = {ex.submit(_extract, c): c["sid"] for c in spacy_valid}
        for fut in tqdm(
            concurrent.futures.as_completed(future_to_sid),
            total=len(future_to_sid), desc="ffmpeg", unit="seg",
        ):
            sid = future_to_sid[fut]
            try:
                extract_ok[sid] = fut.result()
            except Exception as e:
                log.debug("ffmpeg worker error sid=%s: %s", sid, e)
                extract_ok[sid] = False

    align_candidates = [c for c in spacy_valid if extract_ok.get(c["sid"], False)]
    n_skipped_ffmpeg = len(spacy_valid) - len(align_candidates)
    log.info(
        "After ffmpeg phase: %d ready for alignment  (failed extraction: %d)",
        len(align_candidates), n_skipped_ffmpeg,
    )

    # ====================================================================
    # PHASE 3: WhisperX alignment
    #
    # Each segment is aligned independently — whisperx.align() is called
    # with only that segment's audio.  With --align-workers > 1, N worker
    # processes each load their own model copy and work through their share
    # of segments in parallel.
    # ====================================================================
    n_workers = args.align_workers
    log.info(
        "Phase 3: aligning %d segments (%d worker%s) ...",
        len(align_candidates), n_workers, "s" if n_workers > 1 else "",
    )

    alignment_results: dict = {}  # orig_idx → word_segs or None

    if n_workers <= 1:
        # ---- Serial path ----
        log.info("Loading WhisperX alignment model (device=%s) ...", args.device)
        align_model, wx_metadata = whisperx.load_align_model(
            language_code="en", device=args.device,
        )
        for orig_idx, c in tqdm(
            enumerate(align_candidates), total=len(align_candidates),
            desc="Aligning", unit="seg",
        ):
            try:
                audio, sr = sf.read(c["wav_path"])
                if sr != 16000:
                    alignment_results[orig_idx] = None
                    continue
                word_segs = align_utterance(
                    audio.astype(np.float32), c["transcript"],
                    align_model, wx_metadata, args.device,
                )
                alignment_results[orig_idx] = word_segs
            except Exception:
                alignment_results[orig_idx] = None

    else:
        # ---- Parallel path ----
        # Split into N roughly equal chunks; each worker gets one chunk.
        chunk_size = math.ceil(len(align_candidates) / n_workers)
        indexed    = list(enumerate(align_candidates))
        chunks     = [
            indexed[i : i + chunk_size]
            for i in range(0, len(indexed), chunk_size)
        ]
        log.info(
            "Splitting %d segments into %d chunks (~%d each); "
            "each worker loads its own model copy (~400 MB each on GPU).",
            len(align_candidates), len(chunks), chunk_size,
        )

        import torch.multiprocessing as tmp
        ctx = tmp.get_context("spawn")

        with tqdm(total=len(align_candidates), desc="Aligning", unit="seg") as pbar:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=n_workers, mp_context=ctx
            ) as pool:
                futures = {
                    pool.submit(_alignment_worker, chunk, args.device): chunk
                    for chunk in chunks
                }
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        chunk_results = fut.result()
                        for orig_idx, ws in chunk_results:
                            alignment_results[orig_idx] = ws
                        pbar.update(len(futures[fut]))
                    except Exception as e:
                        log.warning("Worker chunk failed: %s", e)
                        for orig_idx, _ in futures[fut]:
                            alignment_results[orig_idx] = None
                        pbar.update(len(futures[fut]))

    # ====================================================================
    # POST-PROCESS: build output rows from alignment results
    # ====================================================================
    rows            = []
    vup_counts      = collections.Counter()
    n_word_up       = 0
    n_skipped_align = 0

    for orig_idx, c in enumerate(align_candidates):
        word_segs = alignment_results.get(orig_idx)
        if not word_segs:
            n_skipped_align += 1
            continue

        aligned_ups = find_up_timestamps(word_segs)
        if not aligned_ups:
            n_skipped_align += 1
            continue

        neg = sample_negative(word_segs)
        if neg is None:
            n_skipped_align += 1
            continue

        for i, (_, label, verb_up) in enumerate(c["ups"]):
            if i >= len(aligned_ups):
                break
            ws = aligned_ups[i]
            rows.append({
                "utt_id":        c["sid"],
                "audio_path":    c["wav_path"],
                "sampling_rate": 16000,
                "up_start":      round(ws["start"], 4),
                "up_end":        round(ws["end"],   4),
                "neg_start":     round(neg["start"], 4),
                "neg_end":       round(neg["end"],   4),
                "neg_word":      neg["word"],
                "label":         label,
                "verb_up":       verb_up,
                "transcript":    c["transcript"],
            })
            if label == "vup":
                vup_counts[verb_up] += 1
            else:
                n_word_up += 1

    n_skipped_total = (
        n_skipped_early + n_skipped_spacy + n_skipped_ffmpeg + n_skipped_align
    )
    log.info(
        "Finished: %d rows | %d word_up | %d V+up types | %d skipped total",
        len(rows), n_word_up, len(vup_counts), n_skipped_total,
    )

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, "dataset.csv")
    df.to_csv(out_csv, index=False)
    log.info("Saved dataset to %s", out_csv)

    qualifying = {k: v for k, v in vup_counts.items() if v >= MIN_FREQ_VUP}
    log.info("\n--- Data sufficiency check ---")
    log.info(
        "  word_up occurrences   : %6d  (target: >= 2000 for 1000 train + 1000 val)",
        n_word_up,
    )
    log.info(
        "  Qualifying V+up types : %6d  (target: >= 20, min freq=%d)",
        len(qualifying), MIN_FREQ_VUP,
    )
    log.info("  Top 20 V+up types:")
    for vup, cnt in vup_counts.most_common(20):
        log.info("    %-25s  %d", vup, cnt)

    if n_word_up >= 2000 and len(qualifying) >= 20:
        log.info("  STATUS: SUFFICIENT for Whisper analysis.")
    else:
        log.info("  STATUS: MAY BE INSUFFICIENT — check metadata coverage.")


if __name__ == "__main__":
    main()
