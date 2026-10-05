#!/usr/bin/env python
"""
Time every stage of the Whisper recognition pipeline on ONE segment, to find where it stalls.

measure_up_recognition.py has shown no output on an A100 for ten minutes. Rather than guess
again, this runs each step separately and prints a timing as soon as it completes, so whichever
line is slow is the last one printed.

    python -u Analyses/whisper/diagnose_whisper.py

Nothing here is cached or reused; it is purely a probe.
"""
import os
import sys
import time

t0 = time.time()
def mark(label):
    print("[%7.1fs] %s" % (time.time() - t0, label), flush=True)

mark("python started")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                    # noqa: E402
mark("numpy imported")
import pandas as pd                                                   # noqa: E402
mark("pandas imported")
try:
    import soundfile as sf
    mark("soundfile imported")
except ImportError as e:
    mark("soundfile MISSING: %s  <-- pip install soundfile" % e); sys.exit(1)
try:
    import torch
    mark("torch imported (version %s)" % torch.__version__)
except ImportError as e:
    mark("torch MISSING: %s" % e); sys.exit(1)

mark("cuda available: %s" % torch.cuda.is_available())
if torch.cuda.is_available():
    mark("gpu: %s" % torch.cuda.get_device_name(0))
else:
    mark("NO CUDA -- this is the problem; the run would take days on CPU")

from transformers import WhisperForConditionalGeneration, WhisperProcessor   # noqa: E402
mark("transformers imported")

csv = os.path.join(REPO, "Data", "whisper", "dataset.csv")
mark("reading %s (59MB)" % csv)
df = pd.read_csv(csv)
mark("csv read: %d rows" % len(df))

row = df.iloc[0]
p = str(row["audio_path"])
if not os.path.exists(p):
    cand = os.path.normpath(os.path.join(REPO, "Analyses", "whisper", p))
    mark("audio_path as stored does not exist; trying %s" % cand)
    p = cand
mark("audio exists: %s  (%s)" % (os.path.exists(p), p))
if not os.path.exists(p):
    mark("STOP: the wavs are not on this machine. Nothing below can run.")
    sys.exit(1)

audio, sr = sf.read(p)
mark("audio read: %.1fs of samples at %dHz" % (len(audio) / float(sr), sr))

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
mark("processor loaded")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
mark("model weights loaded (on cpu)")
model = model.to(device).eval()
mark("model moved to %s" % device)

feats = processor(np.asarray(audio, dtype=np.float32), sampling_rate=16000,
                  return_tensors="pt").input_features.to(device)
mark("features extracted: %s" % (tuple(feats.shape),))

with torch.no_grad():
    ids = model.generate(feats, max_new_tokens=200)
mark("FIRST generate() done -- this is the per-segment cost")

txt = processor.batch_decode(ids, skip_special_tokens=True)[0]
mark("decoded: %r" % txt[:70])

n = 5
with torch.no_grad():
    t = time.time()
    for _ in range(n):
        model.generate(feats, max_new_tokens=200)
    per = (time.time() - t) / n
mark("steady-state: %.2fs per segment" % per)
print()
print("  at %.2fs/segment:  500 segments = %.1f min   5000 = %.1f min   78682 = %.1f hours"
      % (per, per * 500 / 60, per * 5000 / 60, per * 78682 / 3600), flush=True)
