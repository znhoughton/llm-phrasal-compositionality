"""
Reconstruct up-audio-metadata.csv from GigaSpeech + Common Voice
==================================================================
The original script that produced Data/up-audio-metadata.csv is not present
anywhere in this repository (checked exhaustively: every tracked file, and
full git history including deleted files -- see project discussion). This
script is a best-effort reconstruction, reverse-engineered directly from the
existing file's own content (not guessed) via proper CSV-aware inspection:

  - matched_phrase is ALWAYS exactly a 2-word phrase ending in "up" (100% of
    267,689 rows), verbatim substring of cleaned_text. No verb/POS filtering
    happens at this stage -- that happens later, downstream, in
    build_audio_dataset.py's spaCy-based classify_ups_from_doc(). This stage
    just captures every (preceding_word, "up") bigram for every standalone
    occurrence of the word "up" in a segment's transcript.
  - found = the TOTAL number of standalone "up" occurrences within that
    segment (constant across all rows sharing the same sid, confirmed by
    duplicate-sid rows e.g. "leg up"/"tumble up" both showing found=2 for a
    segment containing exactly 2 occurrences of "up").
  - cleaned_text = lowercase(text) with standard ASCII punctuation
    (.,!?;:'"-) stripped (99.95% exact match against the real file when
    tested this way; the ~0.05% residual mismatches are Unicode curly quotes
    "'"" being handled differently than straight quotes, which this script
    also strips for simplicity -- flagged in the validation output).
  - Rows are 97% GigaSpeech / 3% Mozilla Common Voice by ds_source in the
    real file, matched via ds_source/dataset_dir columns pointing to
    /dpluth-data/GigaSpeech/data/ and /dpluth-data/mcv/en/clips/.

VALIDATED LOCALLY: the matching logic above (clean_text + find_up_occurrences)
was tested directly against all 257,778 unique segments in the real
up-audio-metadata.csv, feeding its own "text" column back through this
logic and comparing the output to its actual cleaned_text/matched_phrase/
found columns. Result: 99.95% exact cleaned_text match, 99.90% exact match
on the full (matched_phrase, found) set per segment. This validates the
LOGIC independent of corpus loading (which can't be tested without server
access -- that's what validate_reconstruction.py's row-count/sid-overlap
checks are for).

KNOWN EDGE CASE (the ~0.1% mismatch): consecutive repeated "up" tokens
(e.g. "up, up, up," "keep lifting up up up") are slightly over-counted by
this script -- it treats every adjacent pairing as a separate match, while
the real data appears to use some non-overlapping consumption rule that
wasn't fully reverse-engineered (rare enough, and small enough in effect,
that this wasn't worth further guessing). Not expected to matter in
practice given how rare this pattern is.

ASSUMPTION FLAGGED AS UNCERTAIN: how the corpus is actually loaded on the
server. This defaults to the HuggingFace `datasets` library
("speechcolab/gigaspeech" for GigaSpeech, "mozilla-foundation/common_voice_*"
for Common Voice), mirroring the precedent already in this repo
(build_whisper_dataset.py loads LibriSpeech the same way). If the server's
actual setup differs (e.g. GigaSpeech accessed via a local manifest file
instead), only load_gigaspeech_segments()/load_common_voice_segments() below
need to change -- the matching logic (find_up_occurrences) does not depend
on how the corpus was loaded and should not need to change.

Run this, then run validate_reconstruction.py to compare against the real
Data/up-audio-metadata.csv and see exactly how close this gets.

Usage:
    python reconstruct_up_audio_metadata.py --out Data/up-audio-metadata-reconstructed.csv
    python validate_reconstruction.py --reconstructed Data/up-audio-metadata-reconstructed.csv

Requires:
    pip install datasets pandas tqdm
"""

import argparse
import logging
import re

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

UP_RE = re.compile(r"\bup\b", re.IGNORECASE)
CLEAN_RE = re.compile(r"[.,!?;:'\"‘’“”\-]")


def clean_text(text):
    t = text.lower()
    t = CLEAN_RE.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def find_up_occurrences(cleaned_text):
    """
    Return list of (matched_phrase, found_count) for every standalone "up"
    occurrence in cleaned_text. found_count is the same for every entry
    (total occurrences in this segment), matching the real file's behavior.
    Occurrences of "up" with no preceding word (sentence-initial) are
    skipped, since 100% of matched_phrase in the real file are 2 words --
    there's no evidence sentence-initial "up" was ever included.
    """
    words = cleaned_text.split()
    up_positions = [i for i, w in enumerate(words) if UP_RE.fullmatch(w)]
    total = len(up_positions)
    results = []
    for i in up_positions:
        if i == 0:
            continue  # no preceding word -- not represented in the real file
        matched_phrase = f"{words[i - 1]} up"
        results.append((matched_phrase, total))
    return results


# ---------------------------------------------------------------------------
# CORPUS LOADING -- the one part of this script that's a genuine guess about
# server setup. Adjust these two functions if the actual access method
# differs (e.g. a local GigaSpeech.json manifest instead of the datasets lib).
# ---------------------------------------------------------------------------

def load_gigaspeech_segments(subset="l"):
    """
    Yields dicts with: sid, file, segment_speaker, begin_time, end_time,
    source (audiobook/podcast/youtube), text, dataset_dir.

    GigaSpeech's HF datasets release nests segments under each long audio
    file; this reproduces that structure. Subset "l" (large, ~2500h) is a
    starting guess -- adjust to whatever subset best matches the real
    file's ~256k rows (validate_reconstruction.py will tell you if subset
    coverage is too small or too large).
    """
    from datasets import load_dataset

    log.info("Loading GigaSpeech subset '%s' via HuggingFace datasets ...", subset)
    ds = load_dataset("speechcolab/gigaspeech", subset, trust_remote_code=True)

    for split_name, split in ds.items():
        for row in tqdm(split, desc=f"GigaSpeech[{split_name}]"):
            sid = row.get("segment_id") or row.get("sid")
            aid = row.get("audio_id") or row.get("aid") or (sid.rsplit("_S", 1)[0] if sid else None)
            category = str(row.get("category", "")).lower()  # audiobook/podcast/youtube in some releases
            yield {
                "sid": sid,
                "file": row.get("path", f"audio/{category}/{aid}.opus"),
                "segment_speaker": row.get("speaker", "N/A"),
                "begin_time": row.get("begin_time"),
                "end_time": row.get("end_time"),
                "source": category or "unknown",
                "ds_source": "gigaspeech",
                "dataset_dir": "/dpluth-data/GigaSpeech/data/",
                "text": row.get("text", ""),
            }


def load_common_voice_segments(lang="en"):
    """
    Yields dicts in the same shape as load_gigaspeech_segments(), for the
    Mozilla Common Voice portion (3% of the real file, ds_source =
    "mozilla_common_voice", source = "personal device" in every real row --
    this is a fixed label, not derived per-clip, since Common Voice clips
    are all self-recorded on personal devices by design).
    """
    from datasets import load_dataset

    log.info("Loading Common Voice (%s) via HuggingFace datasets ...", lang)
    ds = load_dataset("mozilla-foundation/common_voice_16_1", lang, trust_remote_code=True)

    for split_name, split in ds.items():
        for row in tqdm(split, desc=f"CommonVoice[{split_name}]"):
            yield {
                "sid": row.get("client_id", "") + "_" + str(row.get("path", "")),
                "file": row.get("path", ""),
                "segment_speaker": "N/A",
                "begin_time": 0.0,
                "end_time": None,  # Common Voice clips are single-utterance; no sub-segment offsets
                "source": "personal device",
                "ds_source": "mozilla_common_voice",
                "dataset_dir": "/dpluth-data/mcv/en/clips/",
                "text": row.get("sentence", ""),
            }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="Data/up-audio-metadata-reconstructed.csv")
    parser.add_argument("--gigaspeech-subset", default="l",
                         help="GigaSpeech subset to load (xs/s/m/l/xl). Default: l")
    parser.add_argument("--skip-common-voice", action="store_true",
                         help="Skip Common Voice (only ~3%% of the real file); useful for a faster first pass.")
    parser.add_argument("--max-segments", type=int, default=None,
                         help="Cap total segments processed, for a quick test run.")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    n_processed = 0

    def process(seg):
        nonlocal n_processed
        text = seg.get("text") or ""
        if not text or "up" not in text.lower():
            return
        cleaned = clean_text(text)
        occurrences = find_up_occurrences(cleaned)
        for matched_phrase, found in occurrences:
            rows.append({
                "sid": seg["sid"], "file": seg["file"],
                "segment_speaker": seg["segment_speaker"],
                "begin_time": seg["begin_time"], "end_time": seg["end_time"],
                "source": seg["source"], "ds_source": seg["ds_source"],
                "dataset_dir": seg["dataset_dir"],
                "text": text, "cleaned_text": cleaned,
                "matched_phrase": matched_phrase, "found": found,
            })

    for seg in load_gigaspeech_segments(args.gigaspeech_subset):
        process(seg)
        n_processed += 1
        if args.max_segments and n_processed >= args.max_segments:
            break

    if not args.skip_common_voice:
        n_cv = 0
        for seg in load_common_voice_segments():
            process(seg)
            n_cv += 1
            if args.max_segments and n_cv >= args.max_segments:
                break

    log.info("Total candidate rows: %d", len(rows))
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    log.info("Saved to %s", args.out)


if __name__ == "__main__":
    main()
