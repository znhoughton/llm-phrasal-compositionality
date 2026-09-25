# dedup_replication_check.R
#
# Replication check: refits every model actually reported in the paper
# (per paper/prepare_results.R's readRDS() calls -- 24 text models + 12
# Whisper models, NOT the full ~80-model exploratory cache) on data with
# duplicate test *instances* removed, and compares against the originals.
#
# Motivation: some V+up types' nominal N_TEST_PER_TYPE=20 test instances are
# backed by fewer unique underlying sentences (text) / audio clips (Whisper),
# because the source corpora contain overlapping/duplicate documents that
# create_dataset.py / build_audio_dataset.py never deduplicate against. This
# inflates a verb_up type's apparent instance count without adding new
# information. Since every model already has (1 | verb_up) / s(verb_up,
# bs='re') random intercepts, duplication *within* a type is largely absorbed
# -- but we check rather than assume.
#
# Dedup key:
#   text:    distinct(layer, verb_up, sentence)      -- exact sentence match
#   whisper: distinct(verb_up, utt_id, up_start, up_end) -- exact same audio
#            clip, reconstructed from dataset.csv and matched back to
#            all_layers_results.csv by (verb_up, .pos), same approach
#            run_models_parallel.R uses to reattach "up" duration.
#
# Refit models are saved to a SEPARATE cache (model_cache/dedup-check/...)
# so the paper's actual reported models are never touched. This script then
# prints a side-by-side comparison (sign flips / credibility changes for
# joint models; EDF/F/p for GAM smooths) between original and deduped fits.
#
# Run from Analyses/ directory:
#   Rscript dedup_replication_check.R
#
# The Whisper section is gated behind RUN_WHISPER (default FALSE) because it
# depends on Data/whisper{,_subword}/{encoder,decoder}/all_layers_results.csv
# reflecting the current classifier rerun (neg_word crash fix, commit
# b71a70d) -- flip it to TRUE once that rerun completes and results are
# pulled.

suppressPackageStartupMessages({
  library(tidyverse)
  library(brms)
  library(mgcv)
  library(future)
  library(furrr)
})

if (!requireNamespace("cmdstanr", quietly = TRUE))
  stop("cmdstanr not found. Install with:\n  install.packages('cmdstanr', repos = c('https://mc-stan.org/r-packages/', getOption('repos')))\n  cmdstanr::install_cmdstan()")

RUN_TEXT    <- TRUE
RUN_WHISPER <- FALSE  # flip to TRUE once the current classifier rerun's output is pulled

N_WORKERS <- 1L

# ---- Cache directories -------------------------------------------------------
OLMO_CACHE_DIR   <- "../model_cache/olmo"
BLM_CACHE_DIR    <- "../model_cache/babylm"
WH_CACHE_DIR     <- "../model_cache/whisper"
OLMO_DEDUP_DIR   <- "../model_cache/dedup-check/olmo"
BLM_DEDUP_DIR    <- "../model_cache/dedup-check/babylm"
WH_DEDUP_DIR     <- "../model_cache/dedup-check/whisper"

for (d in c(OLMO_DEDUP_DIR, BLM_DEDUP_DIR, WH_DEDUP_DIR)) dir.create(d, recursive = TRUE, showWarnings = FALSE)

BLM_TAGS            <- c("opt-125m", "opt-350m", "opt-1.3b")
BLM_FINAL_LAYER_MAP <- list("opt-125m" = 11L, "opt-350m" = 23L, "opt-1.3b" = 23L)
blm_slug            <- function(tag) gsub("\\.", "", gsub("opt-", "", tag))
WH_COMPONENTS       <- c("encoder", "decoder")
OLMO_FINAL_LAYER    <- 31L
WH_FINAL_LAYER      <- 11L

JOINT_FORM <- "logit ~ c_log_freq * c_log_predic + (1 | verb_up)"
JOINT_DURATION_FORM <- "logit ~ c_log_freq * c_log_predic + c_duration + (1 | verb_up)"

# Dedup-refit models are saved under the same base name + "_dedup" suffix
# (in addition to living in model_cache/dedup-check/), so the filename alone
# unambiguously flags it as the deduped replication, not the reported fit.
dd <- function(name) paste0(name, "_dedup")

dedup_text <- function(df, label) {
  n_before <- nrow(df)
  out <- df %>% distinct(layer, verb_up, sentence, .keep_all = TRUE)
  message(sprintf("  [%s] %d -> %d rows after dedup (%d dropped, %.1f%%)",
                   label, n_before, nrow(out), n_before - nrow(out),
                   100 * (n_before - nrow(out)) / n_before))
  out
}

# ==============================================================================
# TEXT MODELS (24 reported: 8 joint brms + 16 GAM bam)
# ==============================================================================
if (RUN_TEXT) {

message("Loading + deduping OLMo data...")
olmo_indep <- read_csv("../Data/olmo-3-7b/Data_up/all_layers_results.csv", show_col_types = FALSE) %>%
  mutate(log_freq = log(frequency), log_predic = log(predic / (1 - predic)), verb_up = factor(verb_up)) %>%
  filter(!is.na(predic), is.finite(log_predic)) %>%
  dedup_text("OLMo indep")
olmo_sub <- read_csv("../Data/olmo-3-7b/Data_upsubword/all_layers_results.csv", show_col_types = FALSE) %>%
  mutate(log_freq = log(frequency), log_predic = log(predic / (1 - predic)), verb_up = factor(verb_up)) %>%
  filter(!is.na(predic), is.finite(log_predic)) %>%
  dedup_text("OLMo subword")

olmo_indep_final_ftp <- olmo_indep %>% filter(layer == OLMO_FINAL_LAYER, !is.na(log_predic)) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic)))
olmo_sub_final_ftp <- olmo_sub %>% filter(layer == OLMO_FINAL_LAYER, !is.na(log_predic)) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic)))
olmo_indep_ftp <- olmo_indep %>% filter(!is.na(log_predic))
olmo_sub_ftp   <- olmo_sub   %>% filter(!is.na(log_predic))

message("Loading + deduping BabyLM data...")
load_babylm_dedup <- function(tag) {
  ui <- read_csv(paste0("../Data/babylm/", tag, "/Data_up/all_layers_results.csv"), show_col_types = FALSE) %>%
    rename_with(~ sub("^ftp$", "predic", .)) %>%
    mutate(model = tag, log_freq = log(frequency), log_predic = log(predic / (1 - predic)), verb_up = factor(verb_up)) %>%
    filter(!is.na(predic), is.finite(log_predic)) %>%
    dedup_text(paste0(tag, " indep"))
  us <- read_csv(paste0("../Data/babylm/", tag, "/Data_upsubword/all_layers_results.csv"), show_col_types = FALSE) %>%
    rename_with(~ sub("^ftp$", "predic", .)) %>%
    mutate(model = tag, log_freq = log(frequency), log_predic = log(predic / (1 - predic)), verb_up = factor(verb_up)) %>%
    filter(!is.na(predic), is.finite(log_predic)) %>%
    dedup_text(paste0(tag, " subword"))
  list(ui = ui, us = us)
}
blm_raw   <- setNames(lapply(BLM_TAGS, load_babylm_dedup), BLM_TAGS)
blm_indep <- map_dfr(blm_raw, "ui") %>% mutate(model = factor(model, levels = BLM_TAGS))
blm_sub   <- map_dfr(blm_raw, "us") %>% mutate(model = factor(model, levels = BLM_TAGS))

blm_indep_final_ftp <- blm_indep %>% group_by(model) %>%
  filter(layer == BLM_FINAL_LAYER_MAP[[as.character(model[1])]], !is.na(log_predic)) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic))) %>% ungroup()
blm_sub_final_ftp <- blm_sub %>% group_by(model) %>%
  filter(layer == BLM_FINAL_LAYER_MAP[[as.character(model[1])]], !is.na(log_predic)) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic))) %>% ungroup()
blm_indep_ftp <- blm_indep %>% filter(!is.na(log_predic))
blm_sub_ftp   <- blm_sub   %>% filter(!is.na(log_predic))

# ---- Fit the 8 joint brms models ---------------------------------------------
DATA_LOOKUP <- list(
  olmo_indep_final_ftp = olmo_indep_final_ftp,
  olmo_sub_final_ftp   = olmo_sub_final_ftp
)
for (tag in BLM_TAGS) {
  sl <- blm_slug(tag)
  DATA_LOOKUP[[paste0("blm_", sl, "_indep_final_ftp")]] <- blm_indep_final_ftp %>% filter(model == tag)
  DATA_LOOKUP[[paste0("blm_", sl, "_sub_final_ftp")]]   <- blm_sub_final_ftp   %>% filter(model == tag)
}

mk <- function(formula, data_key, file) list(formula = formula, data_key = data_key, file = file)
joint_specs <- c(
  list(
    mk(JOINT_FORM, "olmo_indep_final_ftp", file.path(OLMO_DEDUP_DIR, dd("model_freq_predic_up_independently"))),
    mk(JOINT_FORM, "olmo_sub_final_ftp",   file.path(OLMO_DEDUP_DIR, dd("model_freq_predic_up_subword")))
  ),
  unlist(lapply(BLM_TAGS, function(tag) {
    sl <- blm_slug(tag)
    list(
      mk(JOINT_FORM, paste0("blm_", sl, "_indep_final_ftp"), file.path(BLM_DEDUP_DIR, dd(paste0("model_joint_indep_final_", sl)))),
      mk(JOINT_FORM, paste0("blm_", sl, "_sub_final_ftp"),   file.path(BLM_DEDUP_DIR, dd(paste0("model_joint_sub_final_",   sl))))
    )
  }), recursive = FALSE)
)

pending <- Filter(function(s) !file.exists(paste0(s$file, ".rds")), joint_specs)
message(sprintf("Text joint models: %d / %d need fitting (rest already cached in dedup-check/).",
                 length(pending), length(joint_specs)))

if (length(pending) > 0L) {
  run_one_model <- function(spec) {
    library(brms)
    data <- DATA_LOOKUP[[spec$data_key]]
    brm(
      formula = as.formula(spec$formula), data = data,
      prior = set_prior("normal(0, 1)", class = "b"),
      iter = 6000L, warmup = 3000L, chains = 4L, cores = 4L, seed = 964L,
      backend = "cmdstanr", silent = 2L, refresh = 0L, file = spec$file
    )
    invisible(spec$file)
  }
  plan(multisession, workers = N_WORKERS)
  future_map(pending, run_one_model, .progress = TRUE,
             .options = furrr_options(packages = "brms", globals = list(DATA_LOOKUP = DATA_LOOKUP)))
  plan(sequential)
}
message("Done fitting text joint models.")

# ---- Fit the 16 by-layer GAM models -------------------------------------------
cache_bam <- function(dir, name, expr_fn) {
  p <- file.path(dir, paste0(name, ".rds"))
  if (file.exists(p)) { message("  [cached] ", name); return(invisible(NULL)) }
  message("  Fitting ", name, " ...")
  saveRDS(expr_fn(), p)
  invisible(NULL)
}

message("Fitting OLMo by-layer GAMs (deduped)...")
set.seed(964)
cache_bam(OLMO_DEDUP_DIR, dd("model_freq_layer_up_independently"), function()
  bam(logit ~ te(log_freq, layer) + s(verb_up, bs = 're'), data = olmo_indep, method = 'fREML', discrete = TRUE))
set.seed(964)
cache_bam(OLMO_DEDUP_DIR, dd("model_freq_layer_up_subword"), function()
  bam(logit ~ te(log_freq, layer) + s(verb_up, bs = 're'), data = olmo_sub, method = 'fREML', discrete = TRUE))
set.seed(964)
cache_bam(OLMO_DEDUP_DIR, dd("model_predic_layer_up_independently"), function()
  bam(logit ~ te(log_predic, layer) + s(verb_up, bs = 're'), data = olmo_indep_ftp, method = 'fREML', discrete = TRUE))
set.seed(964)
cache_bam(OLMO_DEDUP_DIR, dd("model_predic_layer_up_subword"), function()
  bam(logit ~ te(log_predic, layer) + s(verb_up, bs = 're'), data = olmo_sub_ftp, method = 'fREML', discrete = TRUE))

message("Fitting BabyLM by-layer GAMs (deduped)...")
for (tag in BLM_TAGS) {
  sl <- blm_slug(tag)
  set.seed(964)
  cache_bam(BLM_DEDUP_DIR, dd(paste0("model_freq_layer_indep_", sl)), function()
    bam(logit ~ te(log_freq, layer) + s(verb_up, bs = 're'), data = blm_indep %>% filter(model == tag), method = 'fREML', discrete = TRUE))
  set.seed(964)
  cache_bam(BLM_DEDUP_DIR, dd(paste0("model_freq_layer_sub_", sl)), function()
    bam(logit ~ te(log_freq, layer) + s(verb_up, bs = 're'), data = blm_sub %>% filter(model == tag), method = 'fREML', discrete = TRUE))
  set.seed(964)
  cache_bam(BLM_DEDUP_DIR, dd(paste0("model_predic_layer_indep_", sl)), function()
    bam(logit ~ te(log_predic, layer) + s(verb_up, bs = 're'), data = blm_indep_ftp %>% filter(model == tag), method = 'fREML', discrete = TRUE))
  set.seed(964)
  cache_bam(BLM_DEDUP_DIR, dd(paste0("model_predic_layer_sub_", sl)), function()
    bam(logit ~ te(log_predic, layer) + s(verb_up, bs = 're'), data = blm_sub_ftp %>% filter(model == tag), method = 'fREML', discrete = TRUE))
}
message("Done fitting text GAM models.")

} # RUN_TEXT

# ==============================================================================
# WHISPER MODELS (12 reported: 4 joint brms + 8 GAM bam)
# ==============================================================================
if (RUN_WHISPER) {

message("Loading Whisper data + reconstructing test-item identity for dedup...")

MIN_FREQ_VUP    <- 5L
N_TEST_PER_TYPE <- 20L

load_whisper_component <- function(path, comp) {
  read_csv(path, show_col_types = FALSE) %>%
    select(-any_of("predic")) %>%
    group_by(layer, verb_up) %>%
    mutate(.pos = row_number()) %>%
    ungroup() %>%
    mutate(component = comp, log_freq = log(frequency), verb_up_chr = as.character(verb_up))
}

ftp_lookup <- read_csv("../Data/ftp_lookup.csv", show_col_types = FALSE) %>% rename(predic = ftp)

attach_predic <- function(df) {
  df %>% left_join(ftp_lookup, by = c("verb_up_chr" = "verb_up")) %>%
    filter(!is.na(predic)) %>%
    mutate(log_predic = log(predic / (1 - predic)), verb_up = factor(verb_up_chr)) %>%
    filter(is.finite(log_predic)) %>% select(-verb_up_chr)
}

# Dedup key reconstructed from dataset.csv, matched back by (verb_up, .pos) --
# same join strategy run_models_parallel.R uses to reattach "up" duration,
# since all_layers_results.csv itself carries no row identifier.
dedup_whisper_pos <- function(dataset_path) {
  ds <- read_csv(dataset_path, show_col_types = FALSE)
  vup_df <- ds %>% filter(label == "vup")
  qualifying <- vup_df %>% count(verb_up, name = "n") %>% filter(n >= MIN_FREQ_VUP) %>% pull(verb_up)
  vup_df %>%
    filter(verb_up %in% qualifying) %>%
    group_by(verb_up) %>%
    slice_head(n = N_TEST_PER_TYPE) %>%
    mutate(.pos = row_number()) %>%
    ungroup() %>%
    distinct(verb_up, utt_id, up_start, up_end, .keep_all = TRUE) %>%
    transmute(verb_up = as.character(verb_up), .pos, keep = TRUE)
}

dedup_component <- function(df, keep_pos, label) {
  n_before <- nrow(df)
  out <- df %>% inner_join(keep_pos, by = c("verb_up_chr" = "verb_up", ".pos")) %>% select(-keep)
  message(sprintf("  [%s] %d -> %d rows after dedup (%d dropped, %.1f%%)",
                   label, n_before, nrow(out), n_before - nrow(out), 100 * (n_before - nrow(out)) / n_before))
  out
}

# Subword has no dataset.csv of its own -- same underlying V+up test items as
# indep, only the classifier differs -- so both conditions reconstruct from
# the same source (matches fit_whisper_sub_duration_gams.R).
keep_pos_main <- dedup_whisper_pos("../Data/whisper/dataset.csv")
keep_pos_sub  <- keep_pos_main

encoder     <- load_whisper_component("../Data/whisper/encoder/all_layers_results.csv", "encoder") %>% dedup_component(keep_pos_main, "encoder") %>% attach_predic()
decoder     <- load_whisper_component("../Data/whisper/decoder/all_layers_results.csv", "decoder") %>% dedup_component(keep_pos_main, "decoder") %>% attach_predic()
encoder_sub <- load_whisper_component("../Data/whisper_subword/encoder/all_layers_results.csv", "encoder") %>% dedup_component(keep_pos_sub, "encoder subword") %>% attach_predic()
decoder_sub <- load_whisper_component("../Data/whisper_subword/decoder/all_layers_results.csv", "decoder") %>% dedup_component(keep_pos_sub, "decoder subword") %>% attach_predic()

whisper_all     <- bind_rows(encoder, decoder) %>% mutate(component = factor(component, levels = WH_COMPONENTS))
whisper_all_sub <- bind_rows(encoder_sub, decoder_sub) %>% mutate(component = factor(component, levels = WH_COMPONENTS))

# "up" duration reconstruction, mirroring run_models_parallel.R's attach_duration()
recon_duration <- function(dataset_path) {
  ds <- read_csv(dataset_path, show_col_types = FALSE)
  vup_df <- ds %>% filter(label == "vup")
  qualifying <- vup_df %>% count(verb_up, name = "n") %>% filter(n >= MIN_FREQ_VUP) %>% pull(verb_up)
  vup_df %>% filter(verb_up %in% qualifying) %>% group_by(verb_up) %>% slice_head(n = N_TEST_PER_TYPE) %>%
    mutate(.pos = row_number(), duration = up_end - up_start) %>% ungroup() %>%
    select(verb_up, .pos, duration) %>% mutate(verb_up = as.character(verb_up))
}
rd_main <- recon_duration("../Data/whisper/dataset.csv")
rd_sub  <- rd_main

attach_duration <- function(df, rd) df %>% mutate(verb_up_chr = as.character(verb_up)) %>%
  left_join(rd, by = c("verb_up_chr" = "verb_up", ".pos")) %>% select(-verb_up_chr)

whisper_final_duration_ftp <- bind_rows(attach_duration(encoder, rd_main), attach_duration(decoder, rd_main)) %>%
  mutate(component = factor(component, levels = WH_COMPONENTS)) %>%
  filter(layer == WH_FINAL_LAYER, !is.na(log_predic), !is.na(duration)) %>%
  group_by(component) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic)), c_duration = c(scale(duration))) %>% ungroup()

whisper_sub_final_duration_ftp <- bind_rows(attach_duration(encoder_sub, rd_sub), attach_duration(decoder_sub, rd_sub)) %>%
  mutate(component = factor(component, levels = WH_COMPONENTS)) %>%
  filter(layer == WH_FINAL_LAYER, !is.na(log_predic), !is.na(duration)) %>%
  group_by(component) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic)), c_duration = c(scale(duration))) %>% ungroup()

# ---- Fit the 4 joint (duration-controlled) brms models -----------------------
WH_DATA_LOOKUP <- list()
for (comp in WH_COMPONENTS) {
  WH_DATA_LOOKUP[[paste0("wh_", comp, "_final_duration_ftp")]]     <- whisper_final_duration_ftp %>% filter(component == comp)
  WH_DATA_LOOKUP[[paste0("wh_", comp, "_sub_final_duration_ftp")]] <- whisper_sub_final_duration_ftp %>% filter(component == comp)
}

wh_joint_specs <- unlist(lapply(WH_COMPONENTS, function(comp) list(
  mk(JOINT_DURATION_FORM, paste0("wh_", comp, "_final_duration_ftp"),     file.path(WH_DEDUP_DIR, dd(paste0("model_joint_duration_final_", comp)))),
  mk(JOINT_DURATION_FORM, paste0("wh_", comp, "_sub_final_duration_ftp"), file.path(WH_DEDUP_DIR, dd(paste0("model_joint_duration_sub_final_", comp))))
)), recursive = FALSE)

pending_wh <- Filter(function(s) !file.exists(paste0(s$file, ".rds")), wh_joint_specs)
message(sprintf("Whisper joint models: %d / %d need fitting.", length(pending_wh), length(wh_joint_specs)))
if (length(pending_wh) > 0L) {
  run_one_wh <- function(spec) {
    library(brms)
    data <- WH_DATA_LOOKUP[[spec$data_key]]
    brm(formula = as.formula(spec$formula), data = data,
        prior = set_prior("normal(0, 1)", class = "b"),
        iter = 6000L, warmup = 3000L, chains = 4L, cores = 4L, seed = 964L,
        backend = "cmdstanr", silent = 2L, refresh = 0L, file = spec$file)
    invisible(spec$file)
  }
  plan(multisession, workers = N_WORKERS)
  future_map(pending_wh, run_one_wh, .progress = TRUE,
             .options = furrr_options(packages = "brms", globals = list(WH_DATA_LOOKUP = WH_DATA_LOOKUP)))
  plan(sequential)
}

# ---- Fit the 8 by-layer duration GAMs -----------------------------------------
# Formulas/settings mirror fit_whisper_duration_gams.R / fit_whisper_sub_duration_gams.R
# EXACTLY: c_duration is a covariate (not just a filter), scaled POOLED across
# encoder+decoder (not grouped by component, unlike the joint models above --
# this asymmetry exists in the original scripts and is preserved here for a
# faithful replication), discrete=FALSE (duration has extreme outliers that
# break discrete=TRUE's bin-boundary computation), and both the freq and
# predic GAMs are fit on the SAME predic-valid + duration-valid row set.
whisper_all_dur <- bind_rows(attach_duration(encoder, rd_main), attach_duration(decoder, rd_main)) %>%
  mutate(component = factor(component, levels = WH_COMPONENTS), verb_up = factor(verb_up)) %>%
  mutate(c_duration = c(scale(duration)))
whisper_all_sub_dur <- bind_rows(attach_duration(encoder_sub, rd_sub), attach_duration(decoder_sub, rd_sub)) %>%
  mutate(component = factor(component, levels = WH_COMPONENTS), verb_up = factor(verb_up)) %>%
  mutate(c_duration = c(scale(duration)))

whisper_ftp_dur     <- whisper_all_dur     %>% filter(!is.na(log_predic), !is.na(duration))
whisper_ftp_sub_dur <- whisper_all_sub_dur %>% filter(!is.na(log_predic), !is.na(duration))

for (comp in WH_COMPONENTS) {
  set.seed(964)
  cache_bam(WH_DEDUP_DIR, dd(paste0("model_freq_layer_duration_", comp)), function()
    bam(logit ~ te(log_freq, layer) + c_duration + s(verb_up, bs = 're'), data = whisper_ftp_dur %>% filter(component == comp), method = 'fREML', discrete = FALSE))
  set.seed(964)
  cache_bam(WH_DEDUP_DIR, dd(paste0("model_predic_layer_duration_", comp)), function()
    bam(logit ~ te(log_predic, layer) + c_duration + s(verb_up, bs = 're'), data = whisper_ftp_dur %>% filter(component == comp), method = 'fREML', discrete = FALSE))
  set.seed(964)
  cache_bam(WH_DEDUP_DIR, dd(paste0("model_freq_layer_duration_sub_", comp)), function()
    bam(logit ~ te(log_freq, layer) + c_duration + s(verb_up, bs = 're'), data = whisper_ftp_sub_dur %>% filter(component == comp), method = 'fREML', discrete = FALSE))
  set.seed(964)
  cache_bam(WH_DEDUP_DIR, dd(paste0("model_predic_layer_duration_sub_", comp)), function()
    bam(logit ~ te(log_predic, layer) + c_duration + s(verb_up, bs = 're'), data = whisper_ftp_sub_dur %>% filter(component == comp), method = 'fREML', discrete = FALSE))
}
message("Done fitting Whisper models.")

} # RUN_WHISPER

# ==============================================================================
# COMPARISON: original (reported) vs deduped refit
# ==============================================================================
message("\n\n################## COMPARISON: reported vs deduped ##################\n")

compare_joint <- function(orig_path, dedup_path, label) {
  if (!file.exists(orig_path) || !file.exists(dedup_path)) {
    message(sprintf("  [%s] skipped -- missing %s", label, if (!file.exists(orig_path)) orig_path else dedup_path))
    return(invisible(NULL))
  }
  orig_m  <- readRDS(orig_path)
  dedup_m <- readRDS(dedup_path)
  orig_fx  <- fixef(orig_m)
  dedup_fx <- fixef(dedup_m)
  cat(sprintf("\n=== %s ===\n", label))
  for (param in rownames(orig_fx)) {
    o_est <- orig_fx[param, "Estimate"];  o_lo <- orig_fx[param, "Q2.5"];  o_hi <- orig_fx[param, "Q97.5"]
    d_est <- dedup_fx[param, "Estimate"]; d_lo <- dedup_fx[param, "Q2.5"]; d_hi <- dedup_fx[param, "Q97.5"]
    o_sig <- (o_lo > 0 || o_hi < 0); d_sig <- (d_lo > 0 || d_hi < 0)
    sign_flip <- sign(o_est) != sign(d_est) && o_sig && d_sig
    sig_flip  <- o_sig != d_sig
    flag <- if (sign_flip) " <<< SIGN FLIP (both credible)" else if (sig_flip) " <<< CREDIBILITY CHANGED" else ""
    cat(sprintf("  %-25s orig=%7.3f [%6.3f,%6.3f] cred=%s  ->  dedup=%7.3f [%6.3f,%6.3f] cred=%s%s\n",
                param, o_est, o_lo, o_hi, o_sig, d_est, d_lo, d_hi, d_sig, flag))
  }
}

compare_gam <- function(orig_path, dedup_path, label) {
  if (!file.exists(orig_path) || !file.exists(dedup_path)) {
    message(sprintf("  [%s] skipped -- missing %s", label, if (!file.exists(orig_path)) orig_path else dedup_path))
    return(invisible(NULL))
  }
  orig_m  <- readRDS(orig_path)
  dedup_m <- readRDS(dedup_path)
  orig_s  <- summary(orig_m)$s.table
  dedup_s <- summary(dedup_m)$s.table
  orig_te  <- orig_s[grep("^te\\(", rownames(orig_s)), , drop = FALSE][1, ]
  dedup_te <- dedup_s[grep("^te\\(", rownames(dedup_s)), , drop = FALSE][1, ]
  cat(sprintf("  %-30s orig: edf=%.2f F=%.1f p=%.4g  ->  dedup: edf=%.2f F=%.1f p=%.4g\n",
              label, orig_te["edf"], orig_te["F"], orig_te["p-value"],
              dedup_te["edf"], dedup_te["F"], dedup_te["p-value"]))
}

ddp <- function(dir, name) file.path(dir, paste0(dd(name), ".rds"))

cat("############## JOINT MODELS (8 text + 4 whisper) ##############\n")
compare_joint(file.path(OLMO_CACHE_DIR, "model_freq_predic_up_independently.rds"),
              ddp(OLMO_DEDUP_DIR, "model_freq_predic_up_independently"), "OLMo indep (Exp 1)")
compare_joint(file.path(OLMO_CACHE_DIR, "model_freq_predic_up_subword.rds"),
              ddp(OLMO_DEDUP_DIR, "model_freq_predic_up_subword"), "OLMo subword (Exp 2)")
for (sl in c("125m", "350m", "13b")) {
  compare_joint(file.path(BLM_CACHE_DIR, paste0("model_joint_indep_final_", sl, ".rds")),
                ddp(BLM_DEDUP_DIR, paste0("model_joint_indep_final_", sl)), paste0("BabyLM ", sl, " indep (Exp 1)"))
  compare_joint(file.path(BLM_CACHE_DIR, paste0("model_joint_sub_final_", sl, ".rds")),
                ddp(BLM_DEDUP_DIR, paste0("model_joint_sub_final_", sl)), paste0("BabyLM ", sl, " subword (Exp 2)"))
}
for (comp in WH_COMPONENTS) {
  compare_joint(file.path(WH_CACHE_DIR, paste0("model_joint_duration_final_", comp, ".rds")),
                ddp(WH_DEDUP_DIR, paste0("model_joint_duration_final_", comp)), paste0("Whisper ", comp, " (Exp 3)"))
  compare_joint(file.path(WH_CACHE_DIR, paste0("model_joint_duration_sub_final_", comp, ".rds")),
                ddp(WH_DEDUP_DIR, paste0("model_joint_duration_sub_final_", comp)), paste0("Whisper ", comp, " subword (Exp 3 replication)"))
}

cat("\n############## GAM MODELS (16 text + 8 whisper) ##############\n")
compare_gam(file.path(OLMO_CACHE_DIR, "model_freq_layer_up_independently.rds"),
            ddp(OLMO_DEDUP_DIR, "model_freq_layer_up_independently"), "OLMo freq indep")
compare_gam(file.path(OLMO_CACHE_DIR, "model_predic_layer_up_independently.rds"),
            ddp(OLMO_DEDUP_DIR, "model_predic_layer_up_independently"), "OLMo predic indep")
compare_gam(file.path(OLMO_CACHE_DIR, "model_freq_layer_up_subword.rds"),
            ddp(OLMO_DEDUP_DIR, "model_freq_layer_up_subword"), "OLMo freq sub")
compare_gam(file.path(OLMO_CACHE_DIR, "model_predic_layer_up_subword.rds"),
            ddp(OLMO_DEDUP_DIR, "model_predic_layer_up_subword"), "OLMo predic sub")
for (sl in c("125m", "350m", "13b")) {
  compare_gam(file.path(BLM_CACHE_DIR, paste0("model_freq_layer_indep_", sl, ".rds")),
              ddp(BLM_DEDUP_DIR, paste0("model_freq_layer_indep_", sl)), paste0("BLM ", sl, " freq indep"))
  compare_gam(file.path(BLM_CACHE_DIR, paste0("model_predic_layer_indep_", sl, ".rds")),
              ddp(BLM_DEDUP_DIR, paste0("model_predic_layer_indep_", sl)), paste0("BLM ", sl, " predic indep"))
  compare_gam(file.path(BLM_CACHE_DIR, paste0("model_freq_layer_sub_", sl, ".rds")),
              ddp(BLM_DEDUP_DIR, paste0("model_freq_layer_sub_", sl)), paste0("BLM ", sl, " freq sub"))
  compare_gam(file.path(BLM_CACHE_DIR, paste0("model_predic_layer_sub_", sl, ".rds")),
              ddp(BLM_DEDUP_DIR, paste0("model_predic_layer_sub_", sl)), paste0("BLM ", sl, " predic sub"))
}
for (comp in WH_COMPONENTS) {
  compare_gam(file.path(WH_CACHE_DIR, paste0("model_freq_layer_duration_", comp, ".rds")),
              ddp(WH_DEDUP_DIR, paste0("model_freq_layer_duration_", comp)), paste0("Whisper ", comp, " freq"))
  compare_gam(file.path(WH_CACHE_DIR, paste0("model_predic_layer_duration_", comp, ".rds")),
              ddp(WH_DEDUP_DIR, paste0("model_predic_layer_duration_", comp)), paste0("Whisper ", comp, " predic"))
  compare_gam(file.path(WH_CACHE_DIR, paste0("model_freq_layer_duration_sub_", comp, ".rds")),
              ddp(WH_DEDUP_DIR, paste0("model_freq_layer_duration_sub_", comp)), paste0("Whisper ", comp, " freq sub"))
  compare_gam(file.path(WH_CACHE_DIR, paste0("model_predic_layer_duration_sub_", comp, ".rds")),
              ddp(WH_DEDUP_DIR, paste0("model_predic_layer_duration_sub_", comp)), paste0("Whisper ", comp, " predic sub"))
}

message("\nDone.")
