#!/usr/bin/env bash
# run_pipeline.sh — OLMo-3 7B
#
# Runs the full classifier pipeline for OLMo-3 7B:
#   Step 1: build train/val/test CSVs (tokenizer-specific)
#   Step 2: layer-by-layer standalone-up classifier
#   Step 3: layer-by-layer up-morpheme classifier
#
# Run from the Analyses/olmo-3-7b/ directory:
#   bash run_pipeline.sh

set -euo pipefail

MODEL="allenai/Olmo-3-1025-7B"
DATA_UP="../../Data/olmo-3-7b/Data_up"
DATA_UPSUB="../../Data/olmo-3-7b/Data_upsubword"
VUP_PKL="../../Data/corpus_results.pkl"
UPWORD_PKL="../../Data/corpus_results_upwords.pkl"
CORPUS_STATS_PKL="../../Data/olmo_corpus_stats.pkl"

cd "$(dirname "$0")/.."   # cd into Analyses/

echo "========================================"
echo " OLMo-3 7B pipeline"
echo "========================================"

echo ""
echo "--- Step 1: build train/val/test CSVs ---"
python create_train_val_test.py \
  --model              "$MODEL"            \
  --data-dir-up        "$DATA_UP"          \
  --data-dir-upsubword "$DATA_UPSUB"       \
  --vup-pkl            "$VUP_PKL"          \
  --upword-pkl         "$UPWORD_PKL"       \
  --corpus-stats-pkl   "$CORPUS_STATS_PKL"

echo ""
echo "--- Step 2: standalone-up classifier ---"
python up_independently.py \
  --model    "$MODEL"   \
  --data-dir "$DATA_UP" \
  --vup-pkl  "$VUP_PKL"

echo ""
echo "--- Step 3: up-morpheme classifier ---"
python subwords_containing_up.py \
  --model    "$MODEL"    \
  --data-dir "$DATA_UPSUB" \
  --vup-pkl  "$VUP_PKL"

echo ""
echo "========================================"
echo " Done. Open Analyses/olmo-3-7b/analysis-script.Rmd to run the R analysis."
echo "========================================"
