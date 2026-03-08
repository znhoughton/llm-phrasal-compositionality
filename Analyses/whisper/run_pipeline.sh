#!/usr/bin/env bash
# run_pipeline.sh — Whisper-small (LibriSpeech)
#
# Runs the full Whisper classifier pipeline:
#   Step 1: build dataset CSV + audio files from LibriSpeech via WhisperX
#   Step 2: layer-by-layer encoder + decoder classifier
#
# Run from the Analyses/whisper/ directory:
#   bash run_pipeline.sh
#
# Options:
#   --split train.clean.100   (default, ~100h, faster)
#   --split train.clean.360   (more V+up coverage, recommended)

set -euo pipefail

SPLIT="${1:-train.clean.100}"
DATA_DIR="../../Data/whisper"
VUP_PKL="../../Data/corpus_results.pkl"
MODEL="openai/whisper-small"
DEVICE="cuda"

echo "========================================"
echo " Whisper-small pipeline  (split: $SPLIT)"
echo "========================================"

echo ""
echo "--- Step 1: build dataset from LibriSpeech ---"
python build_whisper_dataset.py \
  --split   "$SPLIT"   \
  --out-dir "$DATA_DIR" \
  --device  "$DEVICE"

echo ""
echo "--- Step 2: layer-by-layer encoder + decoder classifier ---"
python run_whisper_classifier.py \
  --data-dir "$DATA_DIR" \
  --model    "$MODEL"    \
  --device   "$DEVICE"   \
  --vup-pkl  "$VUP_PKL"

echo ""
echo "========================================"
echo " Done. Open Analyses/whisper/analysis-script.Rmd to run the R analysis."
echo "========================================"
