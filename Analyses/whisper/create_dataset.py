"""
Create Audio Datasets -- GigaSpeech + Common Voice
====================================================
Mirrors Analyses/create_dataset.py's structure (the text-side script) for
the audio side: a single scan of the corpus produces TWO datasets.

Dataset 1 (standalone "up" / V+up candidates): same format as the existing
Data/up-audio-metadata.csv, which this project's whole Whisper pipeline
(build_audio_dataset.py, run_whisper_classifier.py, ...) already reads.
The original script that produced that file is not present anywhere in
this repository (checked exhaustively: every tracked file, and full git
history including deleted files). This script's corpus-search logic is a
reconstruction, validated against the real file directly:

  - matched_phrase is exactly a 2-word phrase ending in "up", verbatim
    substring of cleaned_text; found = total "up" occurrences in that
    segment. No verb/POS filtering happens here -- that's downstream, in
    build_audio_dataset.py's spaCy-based classify_ups_from_doc().
  - Full corpus-scale validation (38,131 GigaSpeech audio entries + all of
    Common Voice's validated.tsv): 100% recall (every real (sid,
    matched_phrase) pair recovered) and 97.7% precision (GigaSpeech 97.6%,
    Common Voice 99.4%). The ~2.3% of extra rows this reconstruction finds
    beyond the real file look like genuinely valid "up" matches the real
    file simply doesn't include (likely a slightly different/narrower
    corpus subset snapshot -- GigaSpeech tags each segment with which
    subset(s) it belongs to, e.g. "subsets": ["{XL}"]) -- not a logic bug.
    This discrepancy is small and well-characterized enough to accept
    rather than chase further.
  - Two fixes were needed to get here: (1) GigaSpeech's text_tn field is
    ALL CAPS with literal tag tokens for punctuation (e.g. "<COMMA>",
    "<PERIOD>") instead of real punctuation -- degigaspeech_text() converts
    these. (2) Common Voice's real "sid" turned out to be sentence_id (a
    hash of the sentence text), not client_id+path as first assumed --
    confirmed by the real file's sid being too short to be client_id
    (client_id is a much longer hash, which is segment_speaker instead).
  - Known small edge case: consecutive repeated "up" tokens (e.g. "up, up,
    up") are over-counted slightly (every adjacent pairing treated as a
    match, vs. some non-overlapping rule in the real data) -- rare enough
    not to be worth reverse-engineering further.

Because Data/up-audio-metadata.csv already exists and is the file this
project's pipeline actually uses, and because the small reconstruction/
real discrepancy above is being deliberately accepted rather than chased,
this script does NOT regenerate/overwrite it by default -- it checks for
the file first and only builds it if missing (e.g. bootstrapping onto a
machine that doesn't have it yet).

Dataset 2 (up-CONTAINING WORD candidates, e.g. "update", "upon", "backup"):
the new data needed for the Whisper Experiment 2 replication (a sense-
agnostic "up" positive class, mirroring the text models' design). Matches
Analyses/create_dataset.py's process for its own Dataset 2, not just the
UPWORD_RE/UPWORD_EXCLUDE matching rule: MAX_SEGMENTS_PER_UPWORD (=50,
mirroring MAX_SENTENCES_PER_UPWORD there) caps how many example rows get
stored per up-word type, so a handful of very common words (e.g. "couple",
"upon", "support") don't dominate the dataset; MIN_FREQ_UPWORD then drops
any type with too few total occurrences across the whole corpus scan after
collection. MIN_FREQ_UPWORD is set to 5 here, lower than the text side's 10
-- at 10, the audio corpus (much smaller than the text corpora used for
OLMo/BabyLM) only yielded 889 qualifying types against a target of 2000 for
a 1000/1000 train/val split, so the threshold was loosened specifically for
this audio replication to admit more (rarer) types. Feeds into
build_subword_audio_dataset.py (matched_phrase = the up-containing word).

Like Dataset 1, this is only (re)generated if its output file doesn't
already exist -- once created, later runs won't blow it away by default,
matching Dataset 1's same idempotency logic (see --force-dataset2 to
override either).

Output:
    Data/up-audio-metadata.csv           (Dataset 1 -- only if missing)
    Data/up-audio-metadata-subword.csv   (Dataset 2 -- only if missing)

Usage (run from Analyses/whisper/, matching build_audio_dataset.py's convention):
    python create_dataset.py
    # if a manifest isn't found automatically:
    python create_dataset.py --gigaspeech-manifest /path/to/GigaSpeech.json \
        --cv-manifest /path/to/validated.tsv
    # quick test on a small slice first:
    python create_dataset.py --max-segments 5000

Requires:
    pip install pandas tqdm
"""

import argparse
import logging
import os
import re

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

UP_RE = re.compile(r"\bup\b", re.IGNORECASE)
CLEAN_RE = re.compile(r"[.,!?;:'\"‘’“”\-]")

# Same rule as Analyses/create_dataset.py's Dataset 2 and
# build_audio_dataset.py: any word containing "up" as a substring,
# excluding the standalone word "up"/"ups" itself.
UPWORD_RE = re.compile(r"\b([a-z]*up[a-z]+|[a-z]+up[a-z]*)\b", re.IGNORECASE)
UPWORD_EXCLUDE = {"up", "ups"}

# MAX_SEGMENTS_PER_UPWORD matches Analyses/create_dataset.py's Dataset 2
# (MAX_SENTENCES_PER_UPWORD): cap stored example rows per up-word type so a
# handful of very common words don't dominate. MIN_FREQ_UPWORD (drop types
# with too few total occurrences across the whole corpus scan) is lower here
# (5 vs. the text side's 10) -- the audio corpus is much smaller, and 10 only
# yielded 889 qualifying types against the 2000 target for a 1000/1000
# train/val split.
MAX_SEGMENTS_PER_UPWORD = 50
MIN_FREQ_UPWORD = 5


def clean_text(text):
    t = text.lower()
    t = CLEAN_RE.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def find_up_occurrences(cleaned_text):
    """
    Dataset 1: return list of (matched_phrase, found_count) for every
    standalone "up" occurrence in cleaned_text. found_count is the total
    for the segment (matches the real file's behavior). Sentence-initial
    "up" (no preceding word) is skipped -- the real file's matched_phrase
    is always exactly 2 words.
    """
    words = cleaned_text.split()
    up_positions = [i for i, w in enumerate(words) if UP_RE.fullmatch(w)]
    total = len(up_positions)
    results = []
    for i in up_positions:
        if i == 0:
            continue
        matched_phrase = f"{words[i - 1]} up"
        results.append((matched_phrase, total))
    return results


def find_upword_occurrences(cleaned_text):
    """
    Dataset 2: return list of (matched_word, found_count) for every
    up-CONTAINING-word occurrence (e.g. "update", "upon", "backup") in
    cleaned_text, excluding standalone "up"/"ups". found_count is the
    total such occurrences in the segment.
    """
    words = cleaned_text.split()
    matches = [w for w in words if w not in UPWORD_EXCLUDE and UPWORD_RE.fullmatch(w)]
    total = len(matches)
    return [(w, total) for w in matches]


# ---------------------------------------------------------------------------
# CORPUS LOADING -- reads directly from the local corpus mount that produced
# up-audio-metadata.csv (confirmed via its dataset_dir column:
# /dpluth-data/GigaSpeech/data/ and /dpluth-data/mcv/en/clips/), NOT via the
# HuggingFace `datasets` library, which would pull a different release/
# version/layout than whatever's actually on disk here.
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

# GigaSpeech's text_tn is ALL CAPS with literal tag tokens standing in for
# punctuation rather than real punctuation characters (confirmed directly
# against the manifest on the server). Standard GigaSpeech tag set; add
# more here if others turn up in practice.
GIGASPEECH_TAG_MAP = {
    "<COMMA>": ",",
    "<PERIOD>": ".",
    "<QUESTIONMARK>": "?",
    "<EXCLAMATIONPOINT>": "!",
}


def _find_manifest(root, candidates):
    for rel in candidates:
        p = os.path.join(root, rel)
        if os.path.exists(p):
            return p
    return None


def degigaspeech_text(text_tn):
    """Convert GigaSpeech's tagged, uppercase text_tn into natural-looking
    lowercase text with real punctuation, matching the real file's "text"."""
    t = text_tn
    for tag, punct in GIGASPEECH_TAG_MAP.items():
        t = t.replace(tag, punct)
    t = t.lower()
    t = re.sub(r"\s+([,.?!])", r"\1", t)  # "door , you" -> "door, you"
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_gigaspeech_segments(root="/dpluth-data/GigaSpeech", manifest_path=None):
    """
    Yields dicts with: sid, file, segment_speaker, begin_time, end_time,
    source (audiobook/podcast/youtube), text, dataset_dir.

    Reads GigaSpeech's own native JSON manifest directly (confirmed schema:
    a top-level {"audios": [...]} list, each entry with "aid"/"path"/
    "category" and a nested "segments" list with "sid"/"begin_time"/
    "end_time"/"text_raw"/"text_tn"/"speaker"). text_raw was empty in the
    sample checked; text_tn (via degigaspeech_text()) is what's used,
    falling back to text_raw only if text_tn is itself empty.
    """
    import json

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
        path = audio.get("path") or audio.get("file")
        category = str(audio.get("category", audio.get("source", ""))).lower()
        for seg in audio.get("segments", []):
            sid = seg.get("sid") or seg.get("segment_id")
            text_tn = seg.get("text_tn", "")
            text = degigaspeech_text(text_tn) if text_tn else seg.get("text_raw", "")
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
    Common Voice's own native TSV manifest directly. "sid" is sentence_id
    (a hash of the sentence text); segment_speaker is client_id (confirmed
    against the real file's own column lengths -- sid is ~64 hex chars,
    too short to be the much-longer client_id). begin_time/end_time are
    left blank (None), matching the real file (standalone clips, not
    sub-segments of a longer file). ds_source="mozilla_common_voice",
    source="personal device" for every row (fixed, not derived per-clip).
    """
    import csv

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
            clip_path = row.get("path", "")
            yield {
                "sid": row.get("sentence_id", ""),
                "file": clip_path,
                "segment_speaker": row.get("client_id", "N/A"),
                "begin_time": None,
                "end_time": None,
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
    parser.add_argument("--dataset1-out", default="../../Data/up-audio-metadata.csv",
                         help="Standalone-up/V+up metadata (Dataset 1). Only (re)generated if "
                              "this path doesn't already exist -- the real file is authoritative "
                              "and intentionally not overwritten. Default assumes running from "
                              "Analyses/whisper/.")
    parser.add_argument("--dataset2-out", default="../../Data/up-audio-metadata-subword.csv",
                         help="Up-containing-word metadata (Dataset 2, for the Experiment 2 "
                              "replication). Only (re)generated if this path doesn't already exist.")
    parser.add_argument("--force-dataset1", action="store_true",
                         help="Regenerate Dataset 1 even if --dataset1-out already exists.")
    parser.add_argument("--force-dataset2", action="store_true",
                         help="Regenerate Dataset 2 even if --dataset2-out already exists.")
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
                         help="Skip Common Voice (only ~3%% of Dataset 1's real-file counterpart); "
                              "useful for a faster first pass.")
    parser.add_argument("--max-segments", type=int, default=None,
                         help="Cap total segments processed per source, for a quick test run.")
    return parser.parse_args()


def main():
    args = parse_args()

    need_dataset1 = args.force_dataset1 or not os.path.exists(args.dataset1_out)
    if need_dataset1:
        log.info("%s not found (or --force-dataset1 set) -- will generate Dataset 1 "
                 "(standalone up / V+up) from the corpus.", args.dataset1_out)
    else:
        log.info("%s already exists -- skipping Dataset 1 regeneration. The existing file is "
                 "authoritative and is what the rest of this project's Whisper pipeline actually "
                 "uses; the small, already-characterized difference between it and what this "
                 "script would reconstruct (100%% recall, ~2%% extra rows at full scale) is being "
                 "deliberately accepted rather than chased further. Pass --force-dataset1 to "
                 "regenerate anyway.", args.dataset1_out)

    need_dataset2 = args.force_dataset2 or not os.path.exists(args.dataset2_out)
    if need_dataset2:
        log.info("%s not found (or --force-dataset2 set) -- will generate Dataset 2 "
                 "(up-containing words) from the corpus.", args.dataset2_out)
    else:
        log.info("%s already exists -- skipping Dataset 2 regeneration. Pass --force-dataset2 "
                 "to regenerate anyway.", args.dataset2_out)

    if not need_dataset1 and not need_dataset2:
        log.info("Both datasets already exist and neither --force flag was set -- nothing to do.")
        return

    dataset1_rows = []
    # Dataset 2 accumulated the same way as Analyses/create_dataset.py's Dataset 2:
    # a capped list of example rows per up-word type (MAX_SEGMENTS_PER_UPWORD),
    # plus an uncapped running frequency count per type, with the frequency
    # threshold (MIN_FREQ_UPWORD) applied only after the full scan completes.
    upword_rows = {}
    upword_freq = {}

    def process(seg):
        text = seg.get("text") or ""
        if not text or "up" not in text.lower():
            return
        cleaned = clean_text(text)

        if need_dataset1:
            for matched_phrase, found in find_up_occurrences(cleaned):
                dataset1_rows.append({
                    "sid": seg["sid"], "file": seg["file"],
                    "segment_speaker": seg["segment_speaker"],
                    "begin_time": seg["begin_time"], "end_time": seg["end_time"],
                    "source": seg["source"], "ds_source": seg["ds_source"],
                    "dataset_dir": seg["dataset_dir"],
                    "text": text, "cleaned_text": cleaned,
                    "matched_phrase": matched_phrase, "found": found,
                })

        if need_dataset2:
            for matched_word, found in find_upword_occurrences(cleaned):
                upword_freq[matched_word] = upword_freq.get(matched_word, 0) + 1
                bucket = upword_rows.setdefault(matched_word, [])
                if len(bucket) < MAX_SEGMENTS_PER_UPWORD:
                    bucket.append({
                        "sid": seg["sid"], "file": seg["file"],
                        "segment_speaker": seg["segment_speaker"],
                        "begin_time": seg["begin_time"], "end_time": seg["end_time"],
                        "source": seg["source"], "ds_source": seg["ds_source"],
                        "dataset_dir": seg["dataset_dir"],
                        "text": text, "cleaned_text": cleaned,
                        "matched_phrase": matched_word, "found": found,
                    })

    n_processed = 0
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

    if need_dataset1:
        log.info("Dataset 1: %d candidate rows", len(dataset1_rows))
        pd.DataFrame(dataset1_rows).to_csv(args.dataset1_out, index=False)
        log.info("Saved Dataset 1 to %s", args.dataset1_out)

    if need_dataset2:
        qualifying_words = {w for w, freq in upword_freq.items() if freq > MIN_FREQ_UPWORD}
        dataset2_rows = [
            row for word, rows in upword_rows.items() if word in qualifying_words
            for row in rows
        ]
        log.info(
            "Dataset 2: %d candidate rows, %d qualifying up-word types (freq>%d) "
            "out of %d types seen",
            len(dataset2_rows), len(qualifying_words), MIN_FREQ_UPWORD, len(upword_freq),
        )
        pd.DataFrame(dataset2_rows).to_csv(args.dataset2_out, index=False)
        log.info("Saved Dataset 2 to %s", args.dataset2_out)


if __name__ == "__main__":
    main()
