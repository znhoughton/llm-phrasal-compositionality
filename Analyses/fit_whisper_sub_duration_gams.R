# fit_whisper_sub_duration_gams.R
#
# Fits (or loads-if-cached) the two by-layer GAMs, controlling for "up"
# segment duration, for the Whisper subword condition (Appendix: Whisper
# Subword Replication):
#   model_freq_layer_duration_sub_<encoder|decoder>
#   model_predic_layer_duration_sub_<encoder|decoder>
#
# Mirrors the indep condition's model_freq_layer_duration_<comp> /
# model_predic_layer_duration_<comp> (eq-gam-layers-duration), applied to the
# subword condition. Duration is reconstructed per-layer from the same
# Data/whisper/dataset.csv source used for the indep condition and for
# fit_whisper_sub_duration_joint.R, via the same (verb_up, .pos) matching
# logic -- but here applied at every layer (not just the final one), since
# the by-layer GAM needs duration attached across all 12 layers.
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

# ---- Data loading -------------------------------------------------------
message("Loading Whisper subword data...")

ftp_lookup = read_csv("../Data/ftp_lookup.csv", show_col_types = FALSE) %>%
  rename(predic = ftp)

load_whisper_component = function(path, comp) {
  read_csv(path, show_col_types = FALSE) %>%
    select(-any_of("predic")) %>%
    group_by(layer, verb_up) %>%
    mutate(.pos = row_number()) %>%
    ungroup() %>%
    mutate(component = comp, log_freq = log(frequency),
           verb_up_chr = as.character(verb_up)) %>%
    left_join(ftp_lookup, by = c("verb_up_chr" = "verb_up")) %>%
    filter(!is.na(predic)) %>%
    mutate(log_predic = log(predic / (1 - predic)), verb_up = factor(verb_up_chr)) %>%
    filter(is.finite(log_predic)) %>%
    select(-verb_up_chr)
}

encoder_sub = load_whisper_component("../Data/whisper_subword/encoder/all_layers_results.csv", "encoder")
decoder_sub = load_whisper_component("../Data/whisper_subword/decoder/all_layers_results.csv", "decoder")

# ---- Duration reconstruction (identical logic/source to the joint model script) ----
message("Reconstructing 'up' duration for Whisper subword items (all layers)...")

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
  ref_layer = min(df$layer)
  ref       = df %>% filter(layer == ref_layer)
  counts = inner_join(
    recon_duration %>% count(verb_up, name = "n_recon"),
    ref             %>% count(verb_up, name = "n_saved"),
    by = "verb_up"
  )
  good_types = counts %>% filter(n_recon == n_saved) %>% pull(verb_up)
  out = df %>% left_join(
    recon_duration %>% filter(verb_up %in% good_types),
    by = c("verb_up", ".pos")
  )
  n_assigned = out %>% filter(layer == ref_layer) %>% summarise(n = sum(!is.na(duration))) %>% pull(n)
  message(sprintf("  [%s] duration assigned for %d/%d reference-layer rows (%d/%d V+up types retained).",
                   comp_name, n_assigned, nrow(ref), length(good_types), nrow(counts)))
  out
}

whisper_all_sub_dur = bind_rows(
  attach_duration(encoder_sub, "encoder"),
  attach_duration(decoder_sub, "decoder")
) %>% mutate(component = factor(component, levels = WH_COMPONENTS)) %>%
  # left_join against recon_duration (verb_up read in as character from
  # dataset.csv) silently coerces the verb_up factor back to character;
  # re-factor before fitting, since mgcv's s(bs='re') requires a real factor.
  mutate(verb_up = factor(verb_up)) %>%
  mutate(c_duration = c(scale(duration)))

whisper_ftp_sub_dur = whisper_all_sub_dur %>% filter(!is.na(log_predic), !is.na(duration))

# ---- Fit (or load) the two by-layer GAMs, matching the indep duration GAMs --
message("Fitting/loading by-layer frequency GAM (subword + duration condition)...")
set.seed(964)
wh_sub_dur_freq_layer_models = lapply(setNames(WH_COMPONENTS, WH_COMPONENTS), function(comp) {
  wh_cache_bam(paste0("model_freq_layer_duration_sub_", comp), function() {
    # discrete = FALSE: duration has a small number of extreme outliers
    # (up to 1.5s vs. a 0.04-0.14s bulk), which breaks discrete=TRUE's
    # internal bin-boundary computation (non-finite 'from' in seq()).
    bam(logit ~ te(log_freq, layer) + c_duration + s(verb_up, bs = 're'),
        data = whisper_ftp_sub_dur %>% filter(component == comp), method = 'fREML', discrete = FALSE)
  })
})
lapply(wh_sub_dur_freq_layer_models, summary)

message("Fitting/loading by-layer predictability GAM (subword + duration condition)...")
set.seed(964)
wh_sub_dur_predic_layer_models = lapply(setNames(WH_COMPONENTS, WH_COMPONENTS), function(comp) {
  wh_cache_bam(paste0("model_predic_layer_duration_sub_", comp), function() {
    bam(logit ~ te(log_predic, layer) + c_duration + s(verb_up, bs = 're'),
        data = whisper_ftp_sub_dur %>% filter(component == comp), method = 'fREML', discrete = FALSE)
  })
})
lapply(wh_sub_dur_predic_layer_models, summary)

message("Done. Cached to ", WH_CACHE_DIR, "/model_{freq,predic}_layer_duration_sub_{encoder,decoder}.rds")
