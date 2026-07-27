# run_models_parallel.R
#
# Pre-fits all ~80 brm() models in parallel using the cmdstanr backend.
# Run this script from the Analyses/ directory BEFORE knitting analysis-script.Rmd.
# After this completes, the Rmd loads every model from cache instantly.
#
# Requirements:
#   install.packages(c("tidyverse", "brms", "future", "furrr"))
#   install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
#   cmdstanr::install_cmdstan()
#
# Hardware: adjust N_WORKERS below to your core count. Default here is 1
# (fully sequential -- one model at a time, using 4 chains/4 cores for that
# model only) for running locally without a cluster. Raise it if you have
# enough cores to fit multiple models at once (e.g. 6 workers x 4 chains =
# 24 cores).

suppressPackageStartupMessages({
  library(tidyverse)
  library(brms)
  library(future)
  library(furrr)
})

if (!requireNamespace("cmdstanr", quietly = TRUE))
  stop("cmdstanr not found. Install with:\n  install.packages('cmdstanr', repos = c('https://mc-stan.org/r-packages/', getOption('repos')))\n  cmdstanr::install_cmdstan()")

N_WORKERS <- 1L  # sequential: one model at a time (4 chains/4 cores for that model)

# ---- Cache directories (mirror analysis-script.Rmd) -------------------------
OLMO_CACHE_DIR <- "../model_cache/olmo"
BLM_CACHE_DIR  <- "../model_cache/babylm"
WH_CACHE_DIR   <- "../model_cache/whisper"

dir.create(OLMO_CACHE_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(BLM_CACHE_DIR,  recursive = TRUE, showWarnings = FALSE)
dir.create(WH_CACHE_DIR,   recursive = TRUE, showWarnings = FALSE)

olmo_brms_path <- function(name) file.path(OLMO_CACHE_DIR, name)
blm_brms_path  <- function(name) file.path(BLM_CACHE_DIR,  name)
wh_brms_path   <- function(name) file.path(WH_CACHE_DIR,   name)

BLM_TAGS            <- c("opt-125m", "opt-350m", "opt-1.3b")
BLM_FINAL_LAYER_MAP <- list("opt-125m" = 11L, "opt-350m" = 23L, "opt-1.3b" = 23L)
blm_slug            <- function(tag) gsub("\\.", "", gsub("opt-", "", tag))
WH_COMPONENTS       <- c("encoder", "decoder")

# ---- Data loading (mirrors analysis-script.Rmd) -----------------------------
message("Loading OLMo data...")
olmo_indep <- read_csv("../Data/olmo-3-7b/Data_up/all_layers_results.csv",
                        show_col_types = FALSE) %>%
  mutate(log_freq = log(frequency),
         log_predic = log(predic / (1 - predic)),
         verb_up = factor(verb_up)) %>%
  filter(!is.na(predic), is.finite(log_predic))

olmo_sub <- read_csv("../Data/olmo-3-7b/Data_upsubword/all_layers_results.csv",
                      show_col_types = FALSE) %>%
  mutate(log_freq = log(frequency),
         log_predic = log(predic / (1 - predic)),
         verb_up = factor(verb_up)) %>%
  filter(!is.na(predic), is.finite(log_predic))

OLMO_FINAL_LAYER <- 31L

olmo_indep_final <- olmo_indep %>% filter(layer == OLMO_FINAL_LAYER)
olmo_sub_final   <- olmo_sub   %>% filter(layer == OLMO_FINAL_LAYER)
olmo_indep_first <- olmo_indep %>% filter(layer == 0L)
olmo_sub_first   <- olmo_sub   %>% filter(layer == 0L)

olmo_indep_final_ftp <- olmo_indep %>%
  filter(layer == OLMO_FINAL_LAYER, !is.na(log_predic)) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic)))
olmo_sub_final_ftp <- olmo_sub %>%
  filter(layer == OLMO_FINAL_LAYER, !is.na(log_predic)) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic)))
olmo_indep_first_ftp <- olmo_indep %>%
  filter(layer == 0L, !is.na(log_predic)) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic)))
olmo_sub_first_ftp <- olmo_sub %>%
  filter(layer == 0L, !is.na(log_predic)) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic)))

message("Loading BabyLM data...")
load_babylm_model <- function(tag) {
  ui <- read_csv(paste0("../Data/babylm/", tag, "/Data_up/all_layers_results.csv"),
                 show_col_types = FALSE) %>%
    rename_with(~ sub("^ftp$", "predic", .)) %>%
    mutate(model = tag,
           log_freq = log(frequency),
           log_predic = log(predic / (1 - predic)),
           verb_up = factor(verb_up)) %>%
    filter(!is.na(predic), is.finite(log_predic))
  us <- read_csv(paste0("../Data/babylm/", tag, "/Data_upsubword/all_layers_results.csv"),
                 show_col_types = FALSE) %>%
    rename_with(~ sub("^ftp$", "predic", .)) %>%
    mutate(model = tag,
           log_freq = log(frequency),
           log_predic = log(predic / (1 - predic)),
           verb_up = factor(verb_up)) %>%
    filter(!is.na(predic), is.finite(log_predic))
  list(ui = ui, us = us)
}

blm_raw   <- setNames(lapply(BLM_TAGS, load_babylm_model), BLM_TAGS)
blm_indep <- map_dfr(blm_raw, "ui") %>%
  mutate(model = factor(model, levels = BLM_TAGS)) %>%
  group_by(model) %>% mutate(layer_norm = layer / max(layer)) %>% ungroup()
blm_sub   <- map_dfr(blm_raw, "us") %>%
  mutate(model = factor(model, levels = BLM_TAGS)) %>%
  group_by(model) %>% mutate(layer_norm = layer / max(layer)) %>% ungroup()

blm_indep_final <- blm_indep %>%
  group_by(model) %>%
  filter(layer == BLM_FINAL_LAYER_MAP[[as.character(model[1])]]) %>% ungroup()
blm_sub_final   <- blm_sub %>%
  group_by(model) %>%
  filter(layer == BLM_FINAL_LAYER_MAP[[as.character(model[1])]]) %>% ungroup()
blm_indep_first <- blm_indep %>% filter(layer == 0L)
blm_sub_first   <- blm_sub   %>% filter(layer == 0L)

blm_indep_final_ftp <- blm_indep_final %>% filter(!is.na(log_predic)) %>%
  group_by(model) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic))) %>% ungroup()
blm_sub_final_ftp   <- blm_sub_final   %>% filter(!is.na(log_predic)) %>%
  group_by(model) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic))) %>% ungroup()
blm_indep_first_ftp <- blm_indep_first %>% filter(!is.na(log_predic)) %>%
  group_by(model) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic))) %>% ungroup()
blm_sub_first_ftp   <- blm_sub_first   %>% filter(!is.na(log_predic)) %>%
  group_by(model) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic))) %>% ungroup()

message("Loading Whisper data...")
ftp_lookup <- read_csv("../Data/ftp_lookup.csv", show_col_types = FALSE) %>%
  rename(predic = ftp)

load_whisper_component <- function(path, comp) {
  read_csv(path, show_col_types = FALSE) %>%
    select(-any_of("predic")) %>%
    # Position within each (layer, verb_up) block, computed on the untouched
    # file order -- used below to reattach "up" duration, which was computed
    # internally during feature extraction but never saved to this CSV.
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

encoder     <- load_whisper_component("../Data/whisper/encoder/all_layers_results.csv", "encoder")
decoder     <- load_whisper_component("../Data/whisper/decoder/all_layers_results.csv", "decoder")
whisper_all <- bind_rows(encoder, decoder) %>%
  mutate(component = factor(component, levels = c("encoder", "decoder")))

WH_FINAL_LAYER <- 11L
whisper_final  <- whisper_all %>% filter(layer == WH_FINAL_LAYER)
whisper_first  <- whisper_all %>% filter(layer == 0L)

whisper_final_ftp <- whisper_final %>% filter(!is.na(log_predic)) %>%
  group_by(component) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic))) %>% ungroup()
whisper_first_ftp <- whisper_first %>% filter(!is.na(log_predic)) %>%
  group_by(component) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic))) %>% ungroup()

# ---- Duration control (reviewer-requested, ARR 2026 May, Y2UE) -------------
# "up" duration (up_end - up_start) isn't saved in all_layers_results.csv --
# it's computed internally during feature extraction (run_whisper_classifier.py)
# but dropped before writing the per-layer CSVs. We reconstruct it from the
# raw dataset.csv using the exact same deterministic selection logic as
# build_splits() in that script: filter to label == "vup", keep types with
# >= MIN_FREQ_VUP occurrences, take the first N_TEST_PER_TYPE rows per type
# in their original (unshuffled) file order.
#
# The reconstructed rows are matched back to all_layers_results.csv by
# position within each (layer, verb_up) block (via .pos above), not by row
# content, since the saved CSVs retain no row ID. Where a V+up type's row
# count doesn't match between the reconstruction and the saved data (some
# row(s) were dropped during extraction -- e.g. a failed audio read, or for
# the decoder, no matching token in the tokenized transcript), we cannot tell
# *which* occurrence(s) were dropped, so that type is excluded from the
# duration analysis rather than guessed at.
message("Reconstructing 'up' duration for Whisper items...")

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
  mismatched <- counts %>% filter(n_recon != n_saved)
  if (nrow(mismatched) > 0) {
    message(sprintf(
      "  [%s] %d/%d V+up types have mismatched row counts (rows dropped during extraction); excluding them from the duration analysis.",
      comp_name, nrow(mismatched), nrow(counts)
    ))
  }
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

whisper_all_dur <- bind_rows(
  attach_duration(encoder, "encoder"),
  attach_duration(decoder, "decoder")
)

whisper_final_duration_ftp <- whisper_all_dur %>%
  filter(layer == WH_FINAL_LAYER, !is.na(log_predic), !is.na(duration)) %>%
  group_by(component) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic)),
         c_duration  = c(scale(duration))) %>%
  ungroup()

# ---- Subword condition (Experiment 2 replication) ---------------------------
# Classifier trained with subword positives ("up" within a larger word, e.g.
# "update") combined with standalone positives (see build_splits() in
# run_whisper_classifier.py); the test set is unchanged (still V+up
# instances), so all_layers_results.csv here has the same schema as the
# non-subword file above -- only the classifier's training data differs.
message("Loading Whisper subword data...")
encoder_sub     <- load_whisper_component("../Data/whisper_subword/encoder/all_layers_results.csv", "encoder")
decoder_sub     <- load_whisper_component("../Data/whisper_subword/decoder/all_layers_results.csv", "decoder")
whisper_all_sub <- bind_rows(encoder_sub, decoder_sub) %>%
  mutate(component = factor(component, levels = c("encoder", "decoder")))

whisper_final_sub <- whisper_all_sub %>% filter(layer == WH_FINAL_LAYER)
whisper_first_sub <- whisper_all_sub %>% filter(layer == 0L)

whisper_final_ftp_sub <- whisper_final_sub %>% filter(!is.na(log_predic)) %>%
  group_by(component) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic))) %>% ungroup()
whisper_first_ftp_sub <- whisper_first_sub %>% filter(!is.na(log_predic)) %>%
  group_by(component) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic))) %>% ungroup()

# ---- Pre-subset per model / component (avoids serializing full datasets) ----
blm_data <- setNames(lapply(BLM_TAGS, function(tag) list(
  indep_final     = blm_indep_final     %>% filter(model == tag),
  indep_first     = blm_indep_first     %>% filter(model == tag),
  sub_final       = blm_sub_final       %>% filter(model == tag),
  sub_first       = blm_sub_first       %>% filter(model == tag),
  indep_final_ftp = blm_indep_final_ftp %>% filter(model == tag),
  indep_first_ftp = blm_indep_first_ftp %>% filter(model == tag),
  sub_final_ftp   = blm_sub_final_ftp   %>% filter(model == tag),
  sub_first_ftp   = blm_sub_first_ftp   %>% filter(model == tag)
)), BLM_TAGS)

wh_data <- setNames(lapply(WH_COMPONENTS, function(comp) list(
  final              = whisper_final              %>% filter(component == comp),
  first              = whisper_first              %>% filter(component == comp),
  final_ftp          = whisper_final_ftp          %>% filter(component == comp),
  first_ftp          = whisper_first_ftp          %>% filter(component == comp),
  final_duration_ftp = whisper_final_duration_ftp %>% filter(component == comp),
  sub_final          = whisper_final_sub          %>% filter(component == comp),
  sub_first          = whisper_first_sub          %>% filter(component == comp),
  sub_final_ftp      = whisper_final_ftp_sub      %>% filter(component == comp),
  sub_first_ftp      = whisper_first_ftp_sub      %>% filter(component == comp)
)), WH_COMPONENTS)

# ---- Build data lookup (exported once per worker, not once per task) --------
DATA_LOOKUP <- c(
  list(
    olmo_indep_final     = olmo_indep_final,
    olmo_indep_first     = olmo_indep_first,
    olmo_sub_final       = olmo_sub_final,
    olmo_sub_first       = olmo_sub_first,
    olmo_indep_final_ftp = olmo_indep_final_ftp,
    olmo_sub_final_ftp   = olmo_sub_final_ftp,
    olmo_indep_first_ftp = olmo_indep_first_ftp,
    olmo_sub_first_ftp   = olmo_sub_first_ftp
  ),
  unlist(lapply(BLM_TAGS, function(tag) {
    sl <- blm_slug(tag)
    d  <- blm_data[[tag]]
    setNames(list(
      d$indep_final, d$indep_first, d$sub_final,    d$sub_first,
      d$indep_final_ftp, d$sub_final_ftp, d$indep_first_ftp, d$sub_first_ftp
    ), paste0("blm_", sl, c("_indep_final", "_indep_first", "_sub_final", "_sub_first",
                             "_indep_final_ftp", "_sub_final_ftp", "_indep_first_ftp", "_sub_first_ftp")))
  }), recursive = FALSE),
  unlist(lapply(WH_COMPONENTS, function(comp) {
    d <- wh_data[[comp]]
    setNames(list(d$final, d$first, d$final_ftp, d$first_ftp, d$final_duration_ftp,
                  d$sub_final, d$sub_first, d$sub_final_ftp, d$sub_first_ftp),
             paste0("wh_", comp, c("_final", "_first", "_final_ftp", "_first_ftp", "_final_duration_ftp",
                                    "_sub_final", "_sub_first", "_sub_final_ftp", "_sub_first_ftp")))
  }), recursive = FALSE)
)

# ---- Build model spec list --------------------------------------------------
FREQ_FORM       <- "logit ~ log_freq + (1 | verb_up)"
PREDIC_FORM     <- "logit ~ log_predic + (1 | verb_up)"
JOINT_FORM      <- "logit ~ c_log_freq * c_log_predic + (1 | verb_up)"
POLY_JOINT_FORM <- paste0("logit ~ c_log_freq + I(c_log_freq^2) + ",
                           "c_log_predic + I(c_log_predic^2) + ",
                           "c_log_freq:c_log_predic + (1 | verb_up)")
# Duration-controlled robustness check for Whisper (ARR 2026 May, Y2UE):
# does the frequency/predictability effect survive once "up"'s own duration
# (a proxy for phonetic reduction) is partialled out?
JOINT_DURATION_FORM <- "logit ~ c_log_freq * c_log_predic + c_duration + (1 | verb_up)"

mk <- function(formula, data_key, file) list(formula = formula, data_key = data_key, file = file)

model_specs <- c(
  # ---- OLMo (16 models) -----------------------------------------------------
  list(
    # frequency
    mk(FREQ_FORM, "olmo_indep_final",     olmo_brms_path("model_freq_up_independently")),
    mk(FREQ_FORM, "olmo_indep_first",     olmo_brms_path("model_freq_up_independently_first_layer")),
    mk(FREQ_FORM, "olmo_sub_final",       olmo_brms_path("model_freq_up_subword")),
    mk(FREQ_FORM, "olmo_sub_first",       olmo_brms_path("model_freq_up_subword_first_layer")),
    # predictability
    mk(PREDIC_FORM, "olmo_indep_final_ftp", olmo_brms_path("model_predic_up_independently")),
    mk(PREDIC_FORM, "olmo_sub_final_ftp",   olmo_brms_path("model_predic_up_subword")),
    # joint (interaction)
    mk(JOINT_FORM, "olmo_indep_final_ftp",  olmo_brms_path("model_freq_predic_up_independently")),
    mk(JOINT_FORM, "olmo_sub_final_ftp",    olmo_brms_path("model_freq_predic_up_subword"))
    # "_first_layer"/"_first" (predictability, joint) and all "poly_joint"
    # specs removed: confirmed unused in prepare_results.R/writeup (see notes
    # there) and never fully fit -- no point fitting them now.
  ),
  # ---- BabyLM (48 models: 16 groups × 3 tags) --------------------------------
  unlist(lapply(BLM_TAGS, function(tag) {
    sl <- blm_slug(tag)
    list(
      # frequency
      mk(FREQ_FORM, paste0("blm_", sl, "_indep_final"), blm_brms_path(paste0("model_freq_indep_final_", sl))),
      mk(FREQ_FORM, paste0("blm_", sl, "_indep_first"), blm_brms_path(paste0("model_freq_indep_first_", sl))),
      mk(FREQ_FORM, paste0("blm_", sl, "_sub_final"),   blm_brms_path(paste0("model_freq_sub_final_",   sl))),
      mk(FREQ_FORM, paste0("blm_", sl, "_sub_first"),   blm_brms_path(paste0("model_freq_sub_first_",   sl))),
      # predictability
      mk(PREDIC_FORM, paste0("blm_", sl, "_indep_final_ftp"), blm_brms_path(paste0("model_predic_indep_final_", sl))),
      mk(PREDIC_FORM, paste0("blm_", sl, "_sub_final_ftp"),   blm_brms_path(paste0("model_predic_sub_final_",   sl))),
      mk(PREDIC_FORM, paste0("blm_", sl, "_indep_first_ftp"), blm_brms_path(paste0("model_predic_indep_first_", sl))),
      mk(PREDIC_FORM, paste0("blm_", sl, "_sub_first_ftp"),   blm_brms_path(paste0("model_predic_sub_first_",   sl))),
      # joint
      mk(JOINT_FORM, paste0("blm_", sl, "_indep_final_ftp"), blm_brms_path(paste0("model_joint_indep_final_", sl))),
      mk(JOINT_FORM, paste0("blm_", sl, "_sub_final_ftp"),   blm_brms_path(paste0("model_joint_sub_final_",   sl))),
      mk(JOINT_FORM, paste0("blm_", sl, "_indep_first_ftp"), blm_brms_path(paste0("model_joint_indep_first_", sl))),
      mk(JOINT_FORM, paste0("blm_", sl, "_sub_first_ftp"),   blm_brms_path(paste0("model_joint_sub_first_",   sl))),
      # polynomial joint
      mk(POLY_JOINT_FORM, paste0("blm_", sl, "_indep_final_ftp"), blm_brms_path(paste0("model_poly_joint_indep_final_", sl))),
      mk(POLY_JOINT_FORM, paste0("blm_", sl, "_sub_final_ftp"),   blm_brms_path(paste0("model_poly_joint_sub_final_",   sl))),
      mk(POLY_JOINT_FORM, paste0("blm_", sl, "_indep_first_ftp"), blm_brms_path(paste0("model_poly_joint_indep_first_", sl))),
      mk(POLY_JOINT_FORM, paste0("blm_", sl, "_sub_first_ftp"),   blm_brms_path(paste0("model_poly_joint_sub_first_",   sl)))
    )
  }), recursive = FALSE),
  # ---- Whisper (26 models: 13 groups × 2 components) -------------------------
  unlist(lapply(WH_COMPONENTS, function(comp) {
    list(
      # frequency
      mk(FREQ_FORM,       paste0("wh_", comp, "_final"),     wh_brms_path(paste0("model_freq_final_",       comp))),
      mk(FREQ_FORM,       paste0("wh_", comp, "_first"),     wh_brms_path(paste0("model_freq_first_",       comp))),
      # predictability
      mk(PREDIC_FORM,     paste0("wh_", comp, "_final_ftp"), wh_brms_path(paste0("model_predic_final_",     comp))),
      mk(PREDIC_FORM,     paste0("wh_", comp, "_first_ftp"), wh_brms_path(paste0("model_predic_first_",     comp))),
      # joint
      mk(JOINT_FORM,      paste0("wh_", comp, "_final_ftp"), wh_brms_path(paste0("model_joint_final_",      comp))),
      mk(JOINT_FORM,      paste0("wh_", comp, "_first_ftp"), wh_brms_path(paste0("model_joint_first_",      comp))),
      # polynomial joint: removed -- confirmed unused in prepare_results.R/writeup
      # (never read downstream) and 2 of the 4 were never fully fit anyway.
      # joint, controlling for "up" duration (phonetic-reduction robustness check)
      mk(JOINT_DURATION_FORM, paste0("wh_", comp, "_final_duration_ftp"), wh_brms_path(paste0("model_joint_duration_final_", comp))),
      # ---- subword condition (Experiment 2 replication) ------------------------
      # frequency
      mk(FREQ_FORM,       paste0("wh_", comp, "_sub_final"),     wh_brms_path(paste0("model_freq_sub_final_",   comp))),
      mk(FREQ_FORM,       paste0("wh_", comp, "_sub_first"),     wh_brms_path(paste0("model_freq_sub_first_",   comp))),
      # predictability
      mk(PREDIC_FORM,     paste0("wh_", comp, "_sub_final_ftp"), wh_brms_path(paste0("model_predic_sub_final_", comp))),
      mk(PREDIC_FORM,     paste0("wh_", comp, "_sub_first_ftp"), wh_brms_path(paste0("model_predic_sub_first_", comp))),
      # joint
      mk(JOINT_FORM,      paste0("wh_", comp, "_sub_final_ftp"), wh_brms_path(paste0("model_joint_sub_final_",  comp))),
      mk(JOINT_FORM,      paste0("wh_", comp, "_sub_first_ftp"), wh_brms_path(paste0("model_joint_sub_first_",  comp)))
    )
  }), recursive = FALSE)
)

# Skip models whose .rds cache already exists
pending <- Filter(function(s) !file.exists(paste0(s$file, ".rds")), model_specs)

message(sprintf(
  "%d / %d models need fitting (rest already cached).",
  length(pending), length(model_specs)
))

if (length(pending) == 0L) {
  message("All models cached. Nothing to do.")
  quit(status = 0L)
}

# ---- Worker function --------------------------------------------------------
run_one_model <- function(spec) {
  library(brms)
  data <- DATA_LOOKUP[[spec$data_key]]
  brm(
    formula = as.formula(spec$formula),
    data    = data,
    prior   = set_prior("normal(0, 1)", class = "b"),
    iter    = 6000L, warmup = 3000L, chains = 4L, cores = 4L, seed = 964L,
    backend = "cmdstanr",
    silent  = 2L,
    refresh = 0L,
    file    = spec$file
  )
  invisible(spec$file)
}

# ---- Run in parallel --------------------------------------------------------
message(sprintf(
  "Fitting %d models: %d workers × 4 chains = %d cores.",
  length(pending), N_WORKERS, N_WORKERS * 4L
))

plan(multisession, workers = N_WORKERS)
future_map(
  pending,
  run_one_model,
  .progress = TRUE,
  .options  = furrr_options(packages = "brms", globals = list(DATA_LOOKUP = DATA_LOOKUP))
)
plan(sequential)

message("Done. All fitted models saved to model_cache/.")
message("You can now knit analysis-script.Rmd — all brm() calls will load from cache.")