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
whether a standalone "up" appears in its output, and whether the V+up bigram does. We then report the
correlation between log-frequency and recognition, which is the comparison that matters: if
recognition does not track frequency, it cannot be driving the frequency effect.

NOTE. Recognition is scored per segment, not force-aligned to the token position, so a segment
containing another "up" could score as a hit. Segments with more than one "up" are dropped by the
upstream data-quality filter, so this is rare; --strict additionally requires the bigram.

Usage (paths default to the repo, so this runs from anywhere inside it):
    python Analyses/whisper/measure_up_recognition.py
    python Analyses/whisper/measure_up_recognition.py --limit 500     # quick check first
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
                   help="CSV with verb_up + frequency, for the frequency association")
    p.add_argument("--model", default="openai/whisper-small")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None, help="only process the first N rows")
    p.add_argument("--sample", type=int, default=None,
                   help="randomly subsample N segments (preferred over --limit, which takes them "
                        "in file order). A few thousand pins the recognition rate and the "
                        "correlation tightly enough; the full set is a quarter of a million "
                        "segments decoded one at a time.")
    p.add_argument("--seed", type=int, default=964)
    p.add_argument("--strict", action="store_true",
                   help="count a hit only if the full V+up bigram appears, not just 'up'")
    p.add_argument("--out", default="up_recognition.csv")
    p.add_argument("--from-csv", dest="from_csv", default=None,
                   help="skip transcription and re-summarise an existing --out file. Transcription "
                        "is the expensive part and is already saved per segment, so a run that "
                        "finished under an older version of this script can be summarised with "
                        "this instead of being repeated.")
    return p.parse_args()


def normalise(s):
    """Lowercase, strip punctuation, collapse whitespace -- so 'Up,' matches 'up'."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(s).lower())).strip()


def summarise(out, freq_path):
    """Print recognition rates and the frequency association for a scored table."""
    scored = out.dropna(subset=["ok"])
    print("\nsegments scored: %d (%d failed to process)" % (len(scored), len(out) - len(scored)))
    print("  'up' recognised:        %.1f%%" % (100 * scored["up_recognised"].mean()))
    print("  V+up bigram recognised: %.1f%%" % (100 * scored["bigram_recognised"].mean()))

    # The association between frequency and recognition is what answers the reviewer: if
    # recognition does not track frequency, recognition error cannot generate a frequency effect.
    if freq_path and os.path.exists(freq_path):
        f = pd.read_csv(freq_path)[["verb_up", "frequency"]].drop_duplicates("verb_up")
        m = scored.merge(f, on="verb_up", how="inner")
        if not len(m):
            print("\n(no verb_up overlap with --freq file; skipped the frequency association)")
            return
        lf = np.log(m["frequency"] + 1).to_numpy(float)
        y = m["ok"].to_numpy(float)
        r = np.corrcoef(lf, y)[0, 1]          # Pearson; with y dichotomous this is point-biserial
        n = len(m)
        ci = ""
        if n > 3 and -1 < r < 1:              # Fisher z interval
            se = 1.0 / np.sqrt(n - 3)
            lo, hi = np.tanh(np.arctanh(r) - 1.96 * se), np.tanh(np.arctanh(r) + 1.96 * se)
            ci = "  95%% CI [%+.4f, %+.4f]" % (lo, hi)
        print("\n  r(log-frequency, recognised) = %+.4f%s   (n=%d)" % (r, ci, n))
        print("  An interval spanning zero means recognition does not track frequency, so it")
        print("  cannot be driving the frequency effect on divergence.")


def main():
    args = parse_args()
    if args.from_csv:
        summarise(pd.read_csv(args.from_csv), args.freq)
        return
    df = pd.read_csv(args.dataset)
    # Positives only: rows whose label marks them as an "up" instance.
    if "label" in df.columns:
        df = df[df["label"].astype(str).str.contains("up", case=False, na=False)]
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=args.seed)
    elif args.limit:
        df = df.head(args.limit)
    if not len(df):
        sys.exit("no rows to process")

    # audio_path is stored relative to Analyses/whisper/, so resolve it against the repo rather
    # than the cwd; fall back to the literal value if that does not exist.
    # One stat() per row over 255k rows is minutes of silence on network storage, so probe a
    # single path to decide whether rewriting is needed, then rewrite the column as a string op.
    probe = str(df["audio_path"].iloc[0])
    base = os.path.join(REPO, "Analyses", "whisper")
    if not os.path.exists(probe) and os.path.exists(os.path.normpath(os.path.join(base, probe))):
        print("resolving audio paths against %s" % base)
        sys.stdout.flush()
        df = df.assign(audio_path=base + os.sep + df["audio_path"].astype(str))
        df = df.assign(audio_path=df["audio_path"].map(os.path.normpath))

    missing = [p for p in df["audio_path"].head(20) if not os.path.exists(p)]
    if missing:
        sys.exit("audio not found, e.g.\n  %s\nThe GigaSpeech wavs are not in the repo. Run this "
                 "where they live, or pass --dataset a CSV whose audio_path column points at "
                 "them." % missing[0])

    # State the device up front: torch silently falls back to CPU if this build has no CUDA,
    # and at one generate() call per segment that is the difference between hours and days.
    print("model:    %s" % args.model)
    print("device:   %s%s" % (args.device,
          "" if args.device != "cpu" else "   <-- CPU: this will be very slow, consider --sample"))
    if args.device.startswith("cuda") and torch.cuda.is_available():
        print("gpu:      %s" % torch.cuda.get_device_name(0))
    print("segments: %d" % len(df))
    sys.stdout.flush()

    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(args.device).eval()

    rows = []
    bar = tqdm(df.iterrows(), total=len(df), desc="transcribing", unit="seg",
               file=sys.stdout, dynamic_ncols=True, mininterval=1.0)
    for i, (_, r) in enumerate(bar):
        if i and i % 200 == 0:          # visible even when tqdm's bar is swallowed by a pipe
            print("  ... %d/%d segments" % (i, len(df)))
            sys.stdout.flush()
        try:
            audio, sr = sf.read(r["audio_path"])
            if sr != 16000:
                raise ValueError("expected 16kHz, got %s" % sr)
            feats = processor(np.asarray(audio, dtype=np.float32), sampling_rate=16000,
                              return_tensors="pt").input_features.to(args.device)
            with torch.no_grad():
                ids = model.generate(feats, max_new_tokens=200)
            hyp = normalise(processor.batch_decode(ids, skip_special_tokens=True)[0])
        except Exception as e:                   # keep going; count as not-evaluated
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
    summarise(out, args.freq)


if __name__ == "__main__":
    main()
