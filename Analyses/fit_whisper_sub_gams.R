# fit_whisper_sub_gams.R
#
# SUPERSEDED for reproducing the paper: the by-layer GAMs reported in
# Appendix H (Whisper Subword Replication, Table tbl-exp3-sub-gam-smooth)
# also control for "up" segment duration, matching Experiment 3's own
# duration-controlled design (Equation eq-gam-layers-duration) -- this
# script's model_freq_layer_sub_*/model_predic_layer_sub_* do NOT include
# duration and will NOT reproduce the numbers in the current manuscript.
# Use fit_whisper_sub_duration_gams.R instead, which fits the equivalent
# duration-controlled models (model_freq_layer_duration_sub_*,
# model_predic_layer_duration_sub_*). Kept here for reference only.
#
# Fits (or loads-if-cached) the two by-layer GAMs needed for the Whisper
# subword condition (Experiment 2 audio replication):
#   model_freq_layer_sub_<encoder|decoder>
#   model_predic_layer_sub_<encoder|decoder>
#
# These are also fit inline in analysis-script.Rmd's whisper-sub-bam-freq-layer
# / whisper-sub-bam-ftp-layer chunks, but that requires knitting the entire
# (heavy) Rmd. run_models_parallel.R doesn't cover them either -- it only
# pre-fits brms models, not GAMs (bam() calls live only in the Rmd). This
# script exists so paper/prepare_results.R has these two .rds files ready
# to load the moment Data/whisper_subword/ exists, without waiting on a full
# Rmd knit.
#
# Run this once after run_whisper_classifier.py's subword run finishes
# (i.e. once Data/whisper_subword/{encoder,decoder}/all_layers_results.csv
# exist), from the Analyses/ directory.

suppressPackageStartupMessages({
  library(tidyverse)
  library(mgcv)
})

WH_CACHE_DIR = "../model_cache/whisper"
dir.create(WH_CACHE_DIR, recursive = TRUE, showWarnings = FALSE)

WH_COMPONENTS = c("encoder", "decoder")

wh_rds_path  = function(name) file.path(WH_CACHE_DIR, paste0(name, ".rds"))
wh_cache_bam = function(name, expr_fn) {
  p = wh_rds_path(name)
  if (file.exists(p)) {
    message("  [cached] ", name)
    readRDS(p)
  } else {
    message("  Fitting ", name, " ...")
    m = expr_fn()
    saveRDS(m, p)
    m
  }
}

# ---- Data loading (mirrors analysis-script.Rmd's load-whisper chunk) -------
message("Loading Whisper subword data...")

ftp_lookup = read_csv("../Data/ftp_lookup.csv", show_col_types = FALSE) %>%
  rename(predic = ftp)

load_whisper_component = function(path, comp) {
  read_csv(path, show_col_types = FALSE) %>%
    select(-any_of("predic")) %>%
    mutate(component = comp, log_freq = log(frequency),
           verb_up_chr = as.character(verb_up)) %>%
    left_join(ftp_lookup, by = c("verb_up_chr" = "verb_up")) %>%
    filter(!is.na(predic)) %>%
    mutate(log_predic = log(predic / (1 - predic)), verb_up = factor(verb_up_chr)) %>%
    filter(is.finite(log_predic)) %>%
    select(-verb_up_chr)
}

for (comp in WH_COMPONENTS) {
  path = paste0("../Data/whisper_subword/", comp, "/all_layers_results.csv")
  if (!file.exists(path)) {
    stop("Not found: ", path, "\n",
         "Run build_subword_audio_dataset.py and run_whisper_classifier.py ",
         "(--subword-dataset) first.")
  }
}

encoder_sub = load_whisper_component("../Data/whisper_subword/encoder/all_layers_results.csv", "encoder")
decoder_sub = load_whisper_component("../Data/whisper_subword/decoder/all_layers_results.csv", "decoder")

whisper_all_sub = bind_rows(encoder_sub, decoder_sub) %>%
  mutate(component = factor(component, levels = c("encoder", "decoder")))
whisper_ftp_sub = whisper_all_sub %>% filter(!is.na(log_predic))

# ---- Fit (or load) the two by-layer GAMs, matching analysis-script.Rmd -----
message("Fitting/loading by-layer frequency GAM (subword condition)...")
set.seed(964)
wh_sub_freq_layer_models = lapply(setNames(WH_COMPONENTS, WH_COMPONENTS), function(comp) {
  wh_cache_bam(paste0("model_freq_layer_sub_", comp), function() {
    bam(logit ~ te(log_freq, layer) + s(verb_up, bs = 're'),
        data = whisper_all_sub %>% filter(component == comp), method = 'fREML', discrete = TRUE)
  })
})
lapply(wh_sub_freq_layer_models, summary)

message("Fitting/loading by-layer predictability GAM (subword condition)...")
set.seed(964)
wh_sub_predic_layer_models = lapply(setNames(WH_COMPONENTS, WH_COMPONENTS), function(comp) {
  wh_cache_bam(paste0("model_predic_layer_sub_", comp), function() {
    bam(logit ~ te(log_predic, layer) + s(verb_up, bs = 're'),
        data = whisper_ftp_sub %>% filter(component == comp), method = 'fREML', discrete = TRUE)
  })
})
lapply(wh_sub_predic_layer_models, summary)

message("Done. Cached to ", WH_CACHE_DIR, "/model_{freq,predic}_layer_sub_{encoder,decoder}.rds")
