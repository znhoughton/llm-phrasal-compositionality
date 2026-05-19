#!/usr/bin/env bash
set -euo pipefail

cd Analyses
sed -i "s/N_WORKERS <- [0-9]*L/N_WORKERS <- 6L/" run_models_parallel.R
Rscript run_models_parallel.R           # fits ~80 brms mode
Rscript -e "rmarkdown::render('analysis-script.Rmd')"  # fits GAM (bam) models needed by prepare_results.R
cd ..
Rscript writeup/prepare_results.R
quarto render writeup/writeup.qmd --to acl-pdf
