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

ASSUMPTION FLAGGED AS UNCERTAIN: the exact manifest filename/schema on the
server's local corpus mount. up-audio-metadata.csv's own dataset_dir column
confirms the corpus is read from local disk at /dpluth-data/GigaSpeech/data/
and /dpluth-data/mcv/en/clips/ -- NOT via the HuggingFace `datasets` library,
which would pull a different release/version/file-layout than whatever's
actually on that mount (an earlier version of this script wrongly assumed
HF `datasets`, matching the precedent in build_whisper_dataset.py for
LibriSpeech -- that was a mistake specific to this corpus, since the real
file's own paths point at a local mount, not an HF cache). This version
reads GigaSpeech's native JSON manifest and Common Voice's native TSV
manifest directly from --gigaspeech-root/--cv-root. The specific manifest
filename (GIGASPEECH_MANIFEST_CANDIDATES / CV_MANIFEST_CANDIDATES) and field
names within it are still a best guess at the official release format and
haven't been verified against this specific mount -- if load_gigaspeech_
segments()/load_common_voice_segments() can't find or parse the manifest,
that's the part to fix; find_up_occurrences() and clean_text() (the matching
logic) are validated separately and shouldn't need to change.

Run this, then run validate_reconstruction.py to compare against the real
Data/up-audio-metadata.csv and see exactly how close this gets.

Usage (run from Analyses/whisper/, matching build_audio_dataset.py's convention):
    python reconstruct_up_audio_metadata.py
    # if the manifest isn't found automatically:
    python reconstruct_up_audio_metadata.py --gigaspeech-manifest /path/to/GigaSpeech.json \
        --cv-manifest /path/to/validated.tsv
    python validate_reconstruction.py --reconstructed ../../Data/up-audio-metadata-reconstructed.csv

Requires:
    pip install pandas tqdm
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
# CORPUS LOADING -- reads directly from the local corpus mount that produced
# up-audio-metadata.csv (confirmed via its dataset_dir column:
# /dpluth-data/GigaSpeech/data/ and /dpluth-data/mcv/en/clips/), NOT via the
# HuggingFace `datasets` library -- that would pull a different release/
# version/layout than whatever's actually on disk here, with different
# audio paths than what the real file's `file`/`dataset_dir` columns record.
# This is still the one part of this script that's a genuine guess (exact
# manifest filename/schema on this specific mount), since it can't be
# verified without server access. Adjust GIGASPEECH_MANIFEST_CANDIDATES /
# CV_MANIFEST_CANDIDATES below to point at the real file(s) if none of the
# guessed names/fields match what's actually there.
# ---------------------------------------------------------------------------

GIGASPEECH_MANIFEST_CANDIDATES = [
    "GigaSpeech.json",
    "data/GigaSpeech.json",
    "metadata/GigaSpeech.json",
]

CV_MANIFEST_CANDIDATES = [
    "validated.tsv",
    "train.tsv",
]


def _find_manifest(root, candidates):
    import os
    for rel in candidates:
        p = os.path.join(root, rel)
        if os.path.exists(p):
            return p
    return None


def load_gigaspeech_segments(root="/dpluth-data/GigaSpeech", manifest_path=None):
    """
    Yields dicts with: sid, file, segment_speaker, begin_time, end_time,
    source (audiobook/podcast/youtube), text, dataset_dir.

    Reads GigaSpeech's own native JSON manifest directly (the official
    SpeechColab release format: a top-level {"audios": [...]} list, each
    entry with an "aid", a "path" relative to the release root, a
    "category" (audiobook/podcast/youtube/...), and a nested "segments"
    list with per-segment "sid"/"begin_time"/"end_time"/"text_tn" (or
    "text")/"speaker"). Field names are checked defensively (a few
    plausible alternatives per field) since the exact schema on this
    specific mount hasn't been verified against server access.
    """
    import json
    import os

    manifest_path = manifest_path or _find_manifest(root, GIGASPEECH_MANIFEST_CANDIDATES)
    if manifest_path is None:
        raise FileNotFoundError(
            f"No GigaSpeech manifest found under {root} (tried: {GIGASPEECH_MANIFEST_CANDIDATES}). "
            "Pass --gigaspeech-manifest explicitly if it lives somewhere else / under a different name."
        )
    log.info("Loading GigaSpeech manifest from %s ...", manifest_path)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    audios = manifest.get("audios", manifest if isinstance(manifest, list) else [])
    for audio in tqdm(audios, desc="GigaSpeech audios"):
        aid = audio.get("aid") or audio.get("audio_id")
        path = audio.get("path") or audio.get("file")
        category = str(audio.get("category", audio.get("source", ""))).lower()
        for seg in audio.get("segments", []):
            sid = seg.get("sid") or seg.get("segment_id")
            text = seg.get("text_tn") or seg.get("text") or ""
            yield {
                "sid": sid,
                "file": path,
                "segment_speaker": seg.get("speaker", "N/A"),
                "begin_time": seg.get("begin_time"),
                "end_time": seg.get("end_time"),
                "source": category or "unknown",
                "ds_source": "gigaspeech",
                "dataset_dir": os.path.join(root, "data") + "/",
                "text": text,
            }


def load_common_voice_segments(root="/dpluth-data/mcv/en", manifest_path=None):
    """
    Yields dicts in the same shape as load_gigaspeech_segments(), reading
    Common Voice's own native TSV manifest directly (columns: client_id,
    path, sentence, ... -- the standard raw Common Voice distribution
    format). ds_source="mozilla_common_voice", source="personal device"
    for every row (a fixed label, not derived per-clip -- Common Voice
    clips are all self-recorded on personal devices by design, matching
    what's already in the real up-audio-metadata.csv).
    """
    import csv
    import os

    manifest_path = manifest_path or _find_manifest(root, CV_MANIFEST_CANDIDATES)
    if manifest_path is None:
        raise FileNotFoundError(
            f"No Common Voice manifest found under {root} (tried: {CV_MANIFEST_CANDIDATES}). "
            "Pass --cv-manifest explicitly if it lives somewhere else / under a different name."
        )
    log.info("Loading Common Voice manifest from %s ...", manifest_path)

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in tqdm(reader, desc="Common Voice clips"):
            client_id = row.get("client_id", "")
            clip_path = row.get("path", "")
            yield {
                "sid": f"{client_id}_{clip_path}",
                "file": clip_path,
                "segment_speaker": "N/A",
                "begin_time": 0.0,
                "end_time": None,  # single-utterance clips; no sub-segment offsets
                "source": "personal device",
                "ds_source": "mozilla_common_voice",
                "dataset_dir": os.path.join(root, "clips") + "/",
                "text": row.get("sentence", ""),
            }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="../../Data/up-audio-metadata-reconstructed.csv",
                         help="Default assumes running from Analyses/whisper/, matching "
                              "build_audio_dataset.py's own convention.")
    parser.add_argument("--gigaspeech-root", default="/dpluth-data/GigaSpeech",
                         help="Local GigaSpeech release root. Default: /dpluth-data/GigaSpeech")
    parser.add_argument("--gigaspeech-manifest", default=None,
                         help="Explicit path to GigaSpeech's manifest JSON, if it's not found "
                              "automatically under --gigaspeech-root.")
    parser.add_argument("--cv-root", default="/dpluth-data/mcv/en",
                         help="Local Common Voice release root. Default: /dpluth-data/mcv/en")
    parser.add_argument("--cv-manifest", default=None,
                         help="Explicit path to Common Voice's manifest TSV, if it's not found "
                              "automatically under --cv-root.")
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

    for seg in load_gigaspeech_segments(args.gigaspeech_root, args.gigaspeech_manifest):
        process(seg)
        n_processed += 1
        if args.max_segments and n_processed >= args.max_segments:
            break

    if not args.skip_common_voice:
        n_cv = 0
        for seg in load_common_voice_segments(args.cv_root, args.cv_manifest):
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
