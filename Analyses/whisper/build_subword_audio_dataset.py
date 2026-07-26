"""
Subword "up" Audio Dataset Builder (Experiment 2 replication for Whisper)
==========================================================================
Sibling script to build_audio_dataset.py, following the same
one-script-per-condition convention as up_independently.py /
subwords_containing_up.py for the text models.

Reads a metadata CSV of segments containing an "up"-within-word match (e.g.
"update", "upon", "backup") -- found via the same UPWORD_RE/UPWORD_EXCLUDE
criteria as Analyses/create_dataset.py's Dataset 2 -- extracts audio
segments, runs WhisperX forced alignment WITH character-level output, and
isolates the specific "up" substring's timestamp within each matched word.

This is the audio analog of resolve_upword_positions() in
create_train_val_test.py, which finds the BPE token containing "up" as a
substring within the tokenizer's encoding of the up-word, rather than the
whole word's representation. There's no discrete "subword token" in
continuous audio, so character-level alignment is used instead to recover
just the "up" portion of the word's pronunciation, rather than pooling over
the entire up-containing word (which is what would happen if we reused the
word-level up_start/up_end approach from build_audio_dataset.py -- that
approach only isolates "up" cleanly for build_audio_dataset.py's own rows
because "up" is always the *entire* word there, not embedded in a larger one).

Expected metadata schema (same columns as up-audio-metadata.csv, but
matched_phrase is the up-containing WORD, e.g. "update", not a V+up phrase):
    sid, file, segment_speaker, begin_time, end_time, source, ds_source,
    dataset_dir, text, cleaned_text, matched_phrase, found

Restricted to ONE row per unique up-containing word type at the metadata
level is NOT done here -- that dedup happens downstream in
run_whisper_classifier.py's build_splits(), mirroring how
create_train_val_test.py dedups all_upword_pairs to one sentence per type
only when building the actual train/val split, not at corpus-collection time.

Output: appends rows to the SAME schema as build_audio_dataset.py's
dataset.csv (label="subword_up", upword_type=<the word>, verb_up=""), saved
separately as OUT_DIR/dataset_subword.csv so the existing dataset.csv from
build_audio_dataset.py is never touched; run_whisper_classifier.py reads
and concatenates both.

Usage:
    python build_subword_audio_dataset.py --metadata <path> --out-dir ../../Data/whisper

Requires: same as build_audio_dataset.py (whisperx, soundfile, pandas, tqdm)
"""

import argparse
import concurrent.futures
import logging
import math
import os
import random

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

import whisperx

from build_audio_dataset import (
    RANDOM_SEED, UPWORD_RE, UPWORD_EXCLUDE,
    extract_segment, align_utterance, sample_negative,
    find_word_segment, find_upword_char_span, extract_char_entries,
    _alignment_worker,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build subword-'up' Whisper dataset (Experiment 2 replication)."
    )
    parser.add_argument(
        "--metadata", required=True,
        help="Path to subword-up metadata CSV (same schema as "
             "up-audio-metadata.csv, but matched_phrase is the up-containing "
             "word, e.g. 'update').",
    )
    parser.add_argument(
        "--out-dir", default="../../Data/whisper",
        help="Output directory (same one build_audio_dataset.py wrote dataset.csv "
             "and audio/ to). Default: ../../Data/whisper",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--ffmpeg-workers", type=int, default=32)
    parser.add_argument("--align-workers", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    audio_dir = os.path.join(args.out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    log.info("Loading metadata from %s ...", args.metadata)
    meta = pd.read_csv(args.metadata)
    log.info("Raw rows: %d", len(meta))
    meta = meta.dropna(subset=["begin_time", "end_time", "matched_phrase"])
    if args.max_rows is not None:
        meta = meta.head(args.max_rows)
        log.info("Capped to %d rows (--max-rows).", args.max_rows)

    # ---- PHASE 1: pre-filter -- target word must actually satisfy
    # UPWORD_RE (belt-and-suspenders in case the metadata's own search used a
    # different rule); no spaCy needed here, the target word is already given.
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
        target_word = str(row["matched_phrase"]).strip().lower()

        if not target_word or not UPWORD_RE.fullmatch(target_word) or target_word in UPWORD_EXCLUDE:
            n_skipped_early += 1
            continue
        if not transcript or target_word not in transcript.lower():
            n_skipped_early += 1
            continue
        if not os.path.exists(audio_file):
            n_skipped_early += 1
            continue

        candidates.append({
            "sid":              sid,
            "audio_file":       audio_file,
            "begin_time":       begin_time,
            "end_time":         end_time,
            "transcript":       transcript,
            "target_word":      target_word,
            "wav_path":         os.path.join(audio_dir, f"{sid}.wav"),
            "need_char_alignment": True,
        })

    log.info(
        "Candidates after pre-filter: %d  (skipped: %d)",
        len(candidates), n_skipped_early,
    )

    # ---- PHASE 2: parallel ffmpeg extraction (identical to build_audio_dataset.py) ----
    log.info("Phase 2: extracting audio segments (%d workers) ...", args.ffmpeg_workers)

    def _extract(c):
        if os.path.exists(c["wav_path"]):
            return True
        return extract_segment(c["audio_file"], c["begin_time"], c["end_time"], c["wav_path"])

    extract_ok = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.ffmpeg_workers) as ex:
        future_to_sid = {ex.submit(_extract, c): c["sid"] for c in candidates}
        for fut in tqdm(concurrent.futures.as_completed(future_to_sid),
                         total=len(future_to_sid), desc="ffmpeg", unit="seg"):
            sid = future_to_sid[fut]
            try:
                extract_ok[sid] = fut.result()
            except Exception:
                extract_ok[sid] = False

    align_candidates = [c for c in candidates if extract_ok.get(c["sid"], False)]
    log.info("After ffmpeg phase: %d ready for alignment (failed: %d)",
              len(align_candidates), len(candidates) - len(align_candidates))

    # ---- PHASE 3: WhisperX alignment WITH character-level output ----
    n_workers = args.align_workers
    log.info("Phase 3: aligning %d segments with char-level output (%d worker%s) ...",
              len(align_candidates), n_workers, "s" if n_workers > 1 else "")

    alignment_results = {}  # orig_idx -> (word_segs, char_entries) or (None, None)

    if n_workers <= 1:
        log.info("Loading WhisperX alignment model (device=%s) ...", args.device)
        align_model, wx_metadata = whisperx.load_align_model(language_code="en", device=args.device)
        for orig_idx, c in tqdm(enumerate(align_candidates), total=len(align_candidates),
                                 desc="Aligning", unit="seg"):
            try:
                audio, sr = sf.read(c["wav_path"])
                if sr != 16000:
                    alignment_results[orig_idx] = (None, None)
                    continue
                word_segs, char_entries = align_utterance(
                    audio.astype(np.float32), c["transcript"],
                    align_model, wx_metadata, args.device,
                    return_char_alignments=True,
                )
                alignment_results[orig_idx] = (word_segs, char_entries)
            except Exception:
                alignment_results[orig_idx] = (None, None)
    else:
        chunk_size = math.ceil(len(align_candidates) / n_workers)
        indexed    = list(enumerate(align_candidates))
        chunks     = [indexed[i:i + chunk_size] for i in range(0, len(indexed), chunk_size)]
        log.info("Splitting %d segments into %d chunks (~%d each).",
                  len(align_candidates), len(chunks), chunk_size)

        import torch.multiprocessing as tmp
        ctx = tmp.get_context("spawn")

        with tqdm(total=len(align_candidates), desc="Aligning", unit="seg") as pbar:
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                futures = {
                    pool.submit(_alignment_worker, chunk, args.device): chunk
                    for chunk in chunks
                }
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        for orig_idx, result in fut.result():
                            alignment_results[orig_idx] = result  # already (word_segs, chars) — need_char_alignment=True on every item
                        pbar.update(len(futures[fut]))
                    except Exception as e:
                        log.warning("Worker chunk failed: %s", e)
                        for orig_idx, _ in futures[fut]:
                            alignment_results[orig_idx] = (None, None)
                        pbar.update(len(futures[fut]))

    # ---- POST-PROCESS: isolate the "up" character span within each target word ----
    rows                 = []
    upword_counts        = {}
    n_skipped_no_align    = 0
    n_skipped_no_word_seg = 0
    n_skipped_char_mismatch = 0
    n_skipped_no_neg      = 0

    for orig_idx, c in enumerate(align_candidates):
        word_segs, char_entries = alignment_results.get(orig_idx, (None, None))
        if not word_segs or not char_entries:
            n_skipped_no_align += 1
            continue

        target_ws = find_word_segment(word_segs, c["target_word"])
        if target_ws is None:
            n_skipped_no_word_seg += 1
            continue

        up_span = find_upword_char_span(char_entries, target_ws, c["target_word"])
        if up_span is None:
            n_skipped_char_mismatch += 1
            continue

        neg = sample_negative(word_segs, exclude_upword=True)
        if neg is None:
            n_skipped_no_neg += 1
            continue

        up_start, up_end = up_span
        rows.append({
            "utt_id":        c["sid"],
            "audio_path":    c["wav_path"],
            "sampling_rate": 16000,
            "up_start":      round(up_start, 4),
            "up_end":        round(up_end,   4),
            "neg_start":     round(neg["start"], 4),
            "neg_end":       round(neg["end"],   4),
            "neg_word":      neg["word"],
            "label":         "subword_up",
            "verb_up":       "",
            "upword_type":   c["target_word"],
            "transcript":    c["transcript"],
        })
        upword_counts[c["target_word"]] = upword_counts.get(c["target_word"], 0) + 1

    n_skipped_total = (
        n_skipped_early + n_skipped_no_align + n_skipped_no_word_seg
        + n_skipped_char_mismatch + n_skipped_no_neg
    )
    log.info(
        "Finished: %d rows | %d unique up-word types | skipped: %d early, "
        "%d no-alignment, %d word-not-found, %d char-span-mismatch, %d no-negative "
        "(total skipped %d)",
        len(rows), len(upword_counts), n_skipped_early, n_skipped_no_align,
        n_skipped_no_word_seg, n_skipped_char_mismatch, n_skipped_no_neg, n_skipped_total,
    )

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, "dataset_subword.csv")
    df.to_csv(out_csv, index=False)
    log.info("Saved subword-up dataset to %s", out_csv)

    log.info("--- Data sufficiency check ---")
    log.info("  Unique up-word types : %6d  (target: >= 2000 for 1000 train + 1000 val)",
              len(upword_counts))
    log.info("  Top 20 up-word types:")
    for word, cnt in sorted(upword_counts.items(), key=lambda kv: -kv[1])[:20]:
        log.info("    %-20s  %d", word, cnt)
    if len(upword_counts) >= 2000:
        log.info("  STATUS: SUFFICIENT for Experiment 2 replication.")
    else:
        log.info("  STATUS: MAY BE INSUFFICIENT -- check metadata coverage / char-alignment skip rate above.")


if __name__ == "__main__":
    main()
