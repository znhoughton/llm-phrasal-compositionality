#!/usr/bin/env bash
set -euo pipefail

cd Analyses
python get_olmo_corpus_stats.py
python get_babylm_corpus_stats.py
rm -rf ../model_cache/olmo ../model_cache/babylm ../model_cache/whisper
bash olmo-3-7b/run_pipeline.sh
bash babylm/run_pipeline.sh
(cd whisper && bash run_pipeline.sh)
sed -i "s/N_WORKERS <- [0-9]*L/N_WORKERS <- 1L/" run_models_parallel.R
Rscript run_models_parallel.R
sed -i "s/N_WORKERS <- 1L/N_WORKERS <- 6L/" run_models_parallel.R
cd ..
Rscript writeup/prepare_results.R
quarto render writeup/writeup.qmd --to acl-pdf