# fit_whisper_logduration_joint.R
#
# Exploratory-only robustness check (not part of the reported analysis):
# refits the duration-controlled joint model using log(duration) instead of
# raw duration, for BOTH the indep (main Experiment 3) and subword
# (Appendix: Whisper Subword Replication) conditions:
#   model_joint_logduration_final_<encoder|decoder>      (indep)
#   model_joint_logduration_sub_final_<encoder|decoder>  (subword)
#
# Motivation: duration is heavily right-skewed (99% of values in 0.04-0.14s,
# but a long tail out to 1.5s), and log-duration is a common alternative
# parameterization for exactly this kind of skew. This script exists purely
# to check whether the reported (raw-duration) results are sensitive to that
# choice -- the raw-duration models remain the ones actually reported.
#
# Run from the Analyses/ directory.

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
  ) %>% mutate(verb_up = factor(verb_up))  # re-join drops factor-ness; restore it
  n_assigned <- out %>% filter(layer == ref_layer) %>% summarise(n = sum(!is.na(duration))) %>% pull(n)
  message(sprintf("  [%s] duration assigned for %d/%d reference-layer rows (%d/%d V+up types retained).",
                   comp_name, n_assigned, nrow(ref), length(good_types), nrow(counts)))
  out
}

JOINT_LOGDURATION_FORM <- "logit ~ c_log_freq * c_log_predic + c_log_duration + (1 | verb_up)"

fit_condition <- function(condition_label, data_dir_fn, model_name_fn) {
  message("== ", condition_label, " ==")
  encoder <- load_whisper_component(data_dir_fn("encoder"), "encoder")
  decoder <- load_whisper_component(data_dir_fn("decoder"), "decoder")

  message("Reconstructing duration...")
  all_dur <- bind_rows(attach_duration(encoder, "encoder"), attach_duration(decoder, "decoder")) %>%
    mutate(component = factor(component, levels = WH_COMPONENTS))

  final_ftp <- all_dur %>%
    filter(layer == WH_FINAL_LAYER, !is.na(log_predic), !is.na(duration), duration > 0) %>%
    mutate(log_duration = log(duration)) %>%
    group_by(component) %>%
    mutate(c_log_freq   = c(scale(log_freq)),
           c_log_predic = c(scale(log_predic)),
           c_log_duration = c(scale(log_duration))) %>%
    ungroup()

  for (comp in WH_COMPONENTS) {
    file <- wh_brms_path(model_name_fn(comp))
    if (file.exists(paste0(file, ".rds"))) {
      message("  [cached] ", file)
      next
    }
    message("  Fitting ", basename(file), " ...")
    brm(
      formula = as.formula(JOINT_LOGDURATION_FORM),
      data    = final_ftp %>% filter(component == comp),
      prior   = set_prior("normal(0, 1)", class = "b"),
      iter    = 6000L, warmup = 3000L, chains = 4L, cores = 4L, seed = 964L,
      backend = "cmdstanr",
      silent  = 2L,
      refresh = 0L,
      file    = file
    )
  }
}

fit_condition(
  "Indep (main Experiment 3)",
  function(comp) paste0("../Data/whisper/", comp, "/all_layers_results.csv"),
  function(comp) paste0("model_joint_logduration_final_", comp)
)

fit_condition(
  "Subword (Appendix replication)",
  function(comp) paste0("../Data/whisper_subword/", comp, "/all_layers_results.csv"),
  function(comp) paste0("model_joint_logduration_sub_final_", comp)
)

message("Done.")
