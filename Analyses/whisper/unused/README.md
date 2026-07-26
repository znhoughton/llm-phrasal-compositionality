# Unused

Scripts kept for reference but not part of the active pipeline.

## `build_whisper_dataset.py`

An earlier prototype that scanned **LibriSpeech** directly (via the HuggingFace `datasets` library) for "up" occurrences. Superseded by the GigaSpeech + Common Voice pipeline (`create_dataset.py` → `build_audio_dataset.py`) before Experiment 3 was run — none of the paper's results come from this script. Confirmed by direct inspection: `Data/up-audio-metadata.csv` (the file the active pipeline reads) contains only GigaSpeech/Common Voice rows, and this script has no code path that reads or writes that file at all.
