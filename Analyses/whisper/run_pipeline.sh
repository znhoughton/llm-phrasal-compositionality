#!/usr/bin/env bash
# run_pipeline.sh — Whisper-small
#
# Runs the full Whisper classifier pipeline:
#   Step 1: build dataset CSV + audio files from up-audio-metadata.csv via WhisperX
#   Step 2: layer-by-layer encoder + decoder classifier
#
# Run from the Analyses/whisper/ directory:
#   bash run_pipeline.sh

set -euo pipefail

DATA_DIR="../../Data/whisper_audio"
METADATA="../../Data/up-audio-metadata.csv"
VUP_PKL="../../Data/corpus_results.pkl"
MODEL="openai/whisper-small"
DEVICE="cuda"

echo "========================================"
echo " Whisper-small pipeline"
echo "========================================"

echo ""
echo "--- Step 1: build dataset from audio metadata ---"
python build_audio_dataset.py \
  --metadata         "$METADATA" \
  --out-dir          "$DATA_DIR"  \
  --device           "$DEVICE"    \
  --ffmpeg-workers   32           \
  --align-workers    32

echo ""
echo "--- Step 2: layer-by-layer encoder + decoder classifier ---"
python run_whisper_classifier.py \
  --data-dir "$DATA_DIR" \
  --model    "$MODEL"    \
  --device   "$DEVICE"   \
  --vup-pkl  "$VUP_PKL"

echo ""
echo "========================================"
echo " Done. Results in $DATA_DIR"
echo "========================================"
