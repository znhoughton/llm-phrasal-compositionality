#!/usr/bin/env bash
# run_pipeline.sh — BabyLM OPT models (125m / 350m / 1.3b)
#
# Runs the full classifier pipeline for all three BabyLM OPT models
# sequentially:
#   Step 1: build train/val/test CSVs (tokenizer-specific, per model)
#   Step 2: layer-by-layer standalone-up classifier (per model)
#   Step 3: layer-by-layer up-morpheme classifier (per model)
#
# Run from the Analyses/babylm/ directory:
#   bash run_pipeline.sh
#
# To run a single model only, pass its tag as an argument:
#   bash run_pipeline.sh opt-125m

set -euo pipefail

VUP_PKL="../Data/corpus_results.pkl"
UPWORD_PKL="../Data/corpus_results_upwords.pkl"
CORPUS_STATS_PKL="../Data/babylm_corpus_stats.pkl"

declare -A MODELS=(
  ["opt-125m"]="znhoughton/opt-babylm-125m-64eps-seed964"
  ["opt-350m"]="znhoughton/opt-babylm-350m-64eps-seed964"
  ["opt-1.3b"]="znhoughton/opt-babylm-1.3b-64eps-seed964"
)

# If a tag is passed as argument, run only that model; otherwise run all three
if [[ $# -gt 0 ]]; then
  TAGS=("$1")
else
  TAGS=("opt-125m" "opt-350m" "opt-1.3b")
fi

cd "$(dirname "$0")/.."   # cd into Analyses/

for TAG in "${TAGS[@]}"; do
  MODEL="${MODELS[$TAG]}"
  DATA_UP="../Data/babylm/$TAG/Data_up"
  DATA_UPSUB="../Data/babylm/$TAG/Data_upsubword"

  echo "========================================"
  echo " BabyLM $TAG  ($MODEL)"
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
  echo " $TAG done."
  echo ""
done

echo "========================================"
echo " All models done. Open Analyses/babylm/analysis-script.Rmd to run the R analysis."
echo "========================================"
