# fit_whisper_sub_duration_joint.R
#
# Fits (or loads-if-cached) the joint frequency x predictability model,
# controlling for "up" segment duration, for the Whisper subword condition
# (Appendix: Whisper Subword Replication):
#   model_joint_duration_sub_final_<encoder|decoder>
#
# Mirrors run_models_parallel.R's JOINT_DURATION_FORM / model_joint_duration_final_*
# (the indep condition's duration-controlled robustness check), applied to the
# subword condition's test-set rows. The test set (V+up phrases) is identical
# between the indep and subword conditions -- only the classifier differs --
# so duration is reconstructed from the same Data/whisper/dataset.csv used for
# the indep condition, via the same (verb_up, .pos) position-matching logic as
# run_models_parallel.R's attach_duration().
#
# Run from the Analyses/ directory, after run_whisper_classifier.py
# (--subword-dataset) has produced Data/whisper_subword/{encoder,decoder}/
# all_layers_results.csv.

suppressPackageStartupMessages({
  library(tidyverse)
  library(brms)
})

if (!requireNamespace("cmdstanr", quietly = TRUE))
  stop("cmdstanr not found.")

WH_CACHE_DIR <- "../model_cache/whisper"
dir.create(WH_CACHE_DIR, recursive = TRUE, showWarnings = FALSE)
wh_brms_path <- function(name) file.path(WH_CACHE_DIR, name)

WH_COMPONENTS  <- c("encoder", "decoder")
WH_FINAL_LAYER <- 11L

# ---- Data loading (mirrors run_models_parallel.R) ---------------------------
message("Loading Whisper subword data...")
ftp_lookup <- read_csv("../Data/ftp_lookup.csv", show_col_types = FALSE) %>%
  rename(predic = ftp)

load_whisper_component <- function(path, comp) {
  read_csv(path, show_col_types = FALSE) %>%
    select(-any_of("predic")) %>%
    group_by(layer, verb_up) %>%
    mutate(.pos = row_number()) %>%
    ungroup() %>%
    mutate(component = comp,
           log_freq = log(frequency),
           verb_up_chr = as.character(verb_up)) %>%
    left_join(ftp_lookup, by = c("verb_up_chr" = "verb_up")) %>%
    filter(!is.na(predic)) %>%
    mutate(log_predic = log(predic / (1 - predic)), verb_up = factor(verb_up_chr)) %>%
    filter(is.finite(log_predic)) %>%
    select(-verb_up_chr)
}

encoder_sub <- load_whisper_component("../Data/whisper_subword/encoder/all_layers_results.csv", "encoder")
decoder_sub <- load_whisper_component("../Data/whisper_subword/decoder/all_layers_results.csv", "decoder")

# ---- Duration reconstruction (identical logic/source to run_models_parallel.R) ----
message("Reconstructing 'up' duration for Whisper subword items...")

MIN_FREQ_VUP    <- 5L
N_TEST_PER_TYPE <- 20L

wh_dataset <- read_csv("../Data/whisper/dataset.csv", show_col_types = FALSE)
vup_df     <- wh_dataset %>% filter(label == "vup")
qualifying <- vup_df %>% count(verb_up, name = "n") %>%
  filter(n >= MIN_FREQ_VUP) %>% pull(verb_up)

recon_duration <- vup_df %>%
  filter(verb_up %in% qualifying) %>%
  group_by(verb_up) %>%
  slice_head(n = N_TEST_PER_TYPE) %>%
  mutate(.pos = row_number(), duration = up_end - up_start) %>%
  ungroup() %>%
  select(verb_up, .pos, duration)

attach_duration <- function(df, comp_name) {
  ref_layer <- min(df$layer)
  ref       <- df %>% filter(layer == ref_layer)

  counts <- inner_join(
    recon_duration %>% count(verb_up, name = "n_recon"),
    ref             %>% count(verb_up, name = "n_saved"),
    by = "verb_up"
  )
  good_types <- counts %>% filter(n_recon == n_saved) %>% pull(verb_up)

  out <- df %>% left_join(
    recon_duration %>% filter(verb_up %in% good_types),
    by = c("verb_up", ".pos")
  )

  n_assigned <- out %>% filter(layer == ref_layer) %>% summarise(n = sum(!is.na(duration))) %>% pull(n)
  message(sprintf(
    "  [%s] duration assigned for %d/%d reference-layer rows (%d/%d V+up types retained).",
    comp_name, n_assigned, nrow(ref), length(good_types), nrow(counts)
  ))
  out
}

whisper_all_sub_dur <- bind_rows(
  attach_duration(encoder_sub, "encoder"),
  attach_duration(decoder_sub, "decoder")
) %>% mutate(component = factor(component, levels = WH_COMPONENTS))

whisper_final_sub_duration_ftp <- whisper_all_sub_dur %>%
  filter(layer == WH_FINAL_LAYER, !is.na(log_predic), !is.na(duration)) %>%
  group_by(component) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic)),
         c_duration  = c(scale(duration))) %>%
  ungroup()

# ---- Fit (or load) the 2 joint+duration models -------------------------------
JOINT_DURATION_FORM <- "logit ~ c_log_freq * c_log_predic + c_duration + (1 | verb_up)"

for (comp in WH_COMPONENTS) {
  file <- wh_brms_path(paste0("model_joint_duration_sub_final_", comp))
  if (file.exists(paste0(file, ".rds"))) {
    message("  [cached] ", file)
    next
  }
  message("  Fitting model_joint_duration_sub_final_", comp, " ...")
  brm(
    formula = as.formula(JOINT_DURATION_FORM),
    data    = whisper_final_sub_duration_ftp %>% filter(component == comp),
    prior   = set_prior("normal(0, 1)", class = "b"),
    iter    = 6000L, warmup = 3000L, chains = 4L, cores = 4L, seed = 964L,
    backend = "cmdstanr",
    silent  = 2L,
    refresh = 0L,
    file    = file
  )
}

message("Done. Cached to ", WH_CACHE_DIR, "/model_joint_duration_sub_final_{encoder,decoder}.rds")
