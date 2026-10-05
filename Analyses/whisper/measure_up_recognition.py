#!/usr/bin/env python
"""
How often does Whisper actually recognise the *up* that the ground-truth transcript says is there?

WHY. Reviewer mjt8 asked what accuracy the ASR model had. Items are selected from ground-truth
transcripts, so recognition error cannot *select* items -- but it could still vary with frequency:
*up* in a frequent phrase may be acoustically reduced, so Whisper may be likelier to miss it there.
That would track frequency in the same direction as our effect, so it is worth measuring rather
than arguing away.

WHAT IT DOES. For each segment, Whisper freely transcribes the audio (greedy, no teacher forcing
-- this is the one place in the pipeline where the model is asked to produce text). We then ask
whether a standalone "up" appears in its output, and whether the V+up bigram does. Results are
broken down by frequency decile, which is the comparison that matters: a flat recognition rate
across deciles means recognition cannot be driving the frequency effect.

NOTE. Recognition is scored per segment, not force-aligned to the token position, so a segment
containing another "up" could score as a hit. Segments with more than one "up" are dropped by the
upstream data-quality filter, so this is rare; --strict additionally requires the bigram.

Usage:
    python measure_up_recognition.py --dataset ../../Data/whisper/dataset.csv \
        --freq ../../Data/olmo-3-7b/Data_up/all_layers_results.csv --out up_recognition.csv
    python measure_up_recognition.py --limit 500        # quick check
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor


# Defaults resolve against the repo, not the shell's cwd, so the script runs the same from
# Analyses/whisper/ or from the repo root.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=os.path.join(REPO, "Data", "whisper", "dataset.csv"),
                   help="dataset.csv with audio_path, verb_up, transcript")
    p.add_argument("--freq",
                   default=os.path.join(REPO, "Data", "olmo-3-7b", "Data_up",
                                        "all_layers_results.csv"),
                   help="CSV with verb_up + frequency, to break results out by decile")
    p.add_argument("--model", default="openai/whisper-small")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None, help="only process the first N rows")
    p.add_argument("--strict", action="store_true",
                   help="count a hit only if the full V+up bigram appears, not just 'up'")
    p.add_argument("--out", default="up_recognition.csv")
    return p.parse_args()


def normalise(s):
    """Lowercase, strip punctuation, collapse whitespace -- so 'Up,' matches 'up'."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(s).lower())).strip()


def main():
    args = parse_args()
    df = pd.read_csv(args.dataset)
    # Positives only: rows whose label marks them as an "up" instance.
    if "label" in df.columns:
        df = df[df["label"].astype(str).str.contains("up", case=False, na=False)]
    if args.limit:
        df = df.head(args.limit)
    if not len(df):
        sys.exit("no rows to process")

    # audio_path is stored relative to Analyses/whisper/, so resolve it against the repo rather
    # than the cwd; fall back to the literal value if that does not exist.
    def resolve(p):
        p = str(p)
        if os.path.isabs(p) and os.path.exists(p):
            return p
        cand = os.path.normpath(os.path.join(REPO, "Analyses", "whisper", p))
        return cand if os.path.exists(cand) else p

    df = df.assign(audio_path=df["audio_path"].map(resolve))

    missing = [p for p in df["audio_path"].head(20) if not os.path.exists(p)]
    if missing:
        sys.exit("audio not found, e.g.\n  %s\nThe GigaSpeech wavs are not in the repo. Run this "
                 "where they live, or pass --dataset a CSV whose audio_path column points at "
                 "them." % missing[0])

    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(args.device).eval()

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="transcribing", unit="seg"):
        try:
            audio, sr = sf.read(r["audio_path"])
            if sr != 16000:
                raise ValueError("expected 16kHz, got %s" % sr)
            feats = processor(np.asarray(audio, dtype=np.float32), sampling_rate=16000,
                              return_tensors="pt").input_features.to(args.device)
            with torch.no_grad():
                ids = model.generate(feats, max_new_tokens=200)
            hyp = normalise(processor.batch_decode(ids, skip_special_tokens=True)[0])
        except Exception as e:                       # keep going; count as not-evaluated
            rows.append({"verb_up": r.get("verb_up"), "ok": np.nan, "error": str(e)[:80]})
            continue

        verb = normalise(r.get("verb_up", "")).split()
        bigram_hit = bool(re.search(r"\b%s\s+up\b" % re.escape(verb[0]), hyp)) if verb else False
        up_hit = bool(re.search(r"\bup\b", hyp))
        rows.append({"verb_up": r.get("verb_up"),
                     "up_recognised": up_hit,
                     "bigram_recognised": bigram_hit,
                     "ok": bigram_hit if args.strict else up_hit,
                     "hypothesis": hyp[:200],
                     "error": ""})

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    scored = out.dropna(subset=["ok"])
    print("\nsegments scored: %d (%d failed to process)" % (len(scored), len(out) - len(scored)))
    print("  'up' recognised:        %.1f%%" % (100 * scored["up_recognised"].mean()))
    print("  V+up bigram recognised: %.1f%%" % (100 * scored["bigram_recognised"].mean()))

    # The decile breakdown is the part that answers the reviewer: if recognition is flat across
    # frequency, recognition error cannot be generating a frequency effect.
    if args.freq and os.path.exists(args.freq):
        f = pd.read_csv(args.freq)[["verb_up", "frequency"]].drop_duplicates("verb_up")
        m = scored.merge(f, on="verb_up", how="inner")
        if len(m):
            m["decile"] = pd.qcut(np.log(m["frequency"] + 1), 10, labels=False, duplicates="drop")
            g = m.groupby("decile").agg(n=("ok", "size"), recognised=("ok", "mean"))
            print("\nrecognition rate by frequency decile (low -> high):")
            for d, row in g.iterrows():
                print("  %2d  n=%6d  %.1f%%" % (d, row["n"], 100 * row["recognised"]))
            r = np.corrcoef(np.log(m["frequency"] + 1), m["ok"].astype(float))[0, 1]
            print("\n  r(log-frequency, recognised) = %+.4f" % r)
            print("  A near-zero r means recognition cannot drive the frequency effect.")
        else:
            print("\n(no verb_up overlap with --freq file; skipped decile breakdown)")


if __name__ == "__main__":
    main()
