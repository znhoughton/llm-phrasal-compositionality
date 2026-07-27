# fit_whisper_logduration_gams.R
#
# Exploratory-only robustness check (not part of the reported analysis):
# refits the duration-controlled by-layer GAMs using log(duration) instead
# of raw duration, for BOTH the indep (main Experiment 3) and subword
# (Appendix: Whisper Subword Replication) conditions:
#   model_{freq,predic}_layer_logduration_<encoder|decoder>      (indep)
#   model_{freq,predic}_layer_logduration_sub_<encoder|decoder>  (subword)
#
# See fit_whisper_logduration_joint.R for the motivation (checking
# sensitivity to raw-vs-log duration parameterization given duration's
# right skew). Raw-duration models remain the ones actually reported.
#
# Run from the Analyses/ directory.

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

ftp_lookup = read_csv("../Data/ftp_lookup.csv", show_col_types = FALSE) %>%
  rename(predic = ftp)

load_whisper_component = function(path, comp) {
  read_csv(path, show_col_types = FALSE) %>%
    select(-any_of("predic")) %>%
    group_by(layer, verb_up) %>% mutate(.pos = row_number()) %>% ungroup() %>%
    mutate(component = comp, log_freq = log(frequency),
           verb_up_chr = as.character(verb_up)) %>%
    left_join(ftp_lookup, by = c("verb_up_chr" = "verb_up")) %>%
    filter(!is.na(predic)) %>%
    mutate(log_predic = log(predic / (1 - predic)), verb_up = factor(verb_up_chr)) %>%
    filter(is.finite(log_predic)) %>% select(-verb_up_chr)
}

MIN_FREQ_VUP    = 5L
N_TEST_PER_TYPE = 20L

wh_dataset = read_csv("../Data/whisper/dataset.csv", show_col_types = FALSE)
vup_df     = wh_dataset %>% filter(label == "vup")
qualifying = vup_df %>% count(verb_up, name = "n") %>%
  filter(n >= MIN_FREQ_VUP) %>% pull(verb_up)

recon_duration = vup_df %>%
  filter(verb_up %in% qualifying) %>%
  group_by(verb_up) %>%
  slice_head(n = N_TEST_PER_TYPE) %>%
  mutate(.pos = row_number(), duration = up_end - up_start) %>%
  ungroup() %>%
  select(verb_up, .pos, duration)

attach_duration = function(df, comp_name) {
  ref_layer = min(df$layer); ref = df %>% filter(layer == ref_layer)
  counts = inner_join(recon_duration %>% count(verb_up, name = "n_recon"),
                       ref %>% count(verb_up, name = "n_saved"), by = "verb_up")
  good_types = counts %>% filter(n_recon == n_saved) %>% pull(verb_up)
  out = df %>% left_join(recon_duration %>% filter(verb_up %in% good_types), by = c("verb_up", ".pos")) %>%
    mutate(verb_up = factor(verb_up))  # re-join drops factor-ness; restore it
  n_assigned = out %>% filter(layer == ref_layer) %>% summarise(n = sum(!is.na(duration))) %>% pull(n)
  message(sprintf("  [%s] duration assigned for %d/%d reference-layer rows (%d/%d V+up types retained).",
                   comp_name, n_assigned, nrow(ref), length(good_types), nrow(counts)))
  out
}

fit_condition = function(condition_label, data_dir_fn, freq_name_fn, predic_name_fn) {
  message("== ", condition_label, " ==")
  encoder = load_whisper_component(data_dir_fn("encoder"), "encoder")
  decoder = load_whisper_component(data_dir_fn("decoder"), "decoder")

  message("Reconstructing duration (all layers)...")
  all_dur = bind_rows(attach_duration(encoder, "encoder"), attach_duration(decoder, "decoder")) %>%
    mutate(component = factor(component, levels = WH_COMPONENTS))

  ftp_dur = all_dur %>%
    filter(!is.na(log_predic), !is.na(duration), duration > 0) %>%
    mutate(log_duration = log(duration), c_log_duration = c(scale(log_duration)))

  freq_models = lapply(setNames(WH_COMPONENTS, WH_COMPONENTS), function(comp) {
    wh_cache_bam(freq_name_fn(comp), function() {
      bam(logit ~ te(log_freq, layer) + c_log_duration + s(verb_up, bs = 're'),
          data = ftp_dur %>% filter(component == comp), method = 'fREML', discrete = FALSE)
    })
  })
  lapply(freq_models, summary)

  predic_models = lapply(setNames(WH_COMPONENTS, WH_COMPONENTS), function(comp) {
    wh_cache_bam(predic_name_fn(comp), function() {
      bam(logit ~ te(log_predic, layer) + c_log_duration + s(verb_up, bs = 're'),
          data = ftp_dur %>% filter(component == comp), method = 'fREML', discrete = FALSE)
    })
  })
  lapply(predic_models, summary)
}

set.seed(964)
fit_condition(
  "Indep (main Experiment 3)",
  function(comp) paste0("../Data/whisper/", comp, "/all_layers_results.csv"),
  function(comp) paste0("model_freq_layer_logduration_", comp),
  function(comp) paste0("model_predic_layer_logduration_", comp)
)

set.seed(964)
fit_condition(
  "Subword (Appendix replication)",
  function(comp) paste0("../Data/whisper_subword/", comp, "/all_layers_results.csv"),
  function(comp) paste0("model_freq_layer_logduration_sub_", comp),
  function(comp) paste0("model_predic_layer_logduration_sub_", comp)
)

message("Done.")
