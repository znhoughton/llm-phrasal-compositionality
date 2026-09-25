# whisper_sub_refit_check.R
#
# Subword-condition counterpart to whisper_indep_refit_check.R, now that the
# subword classifier rerun has finished (commit ceecc64). Covers the 6
# subword-condition reported models:
#   model_joint_duration_sub_final_{encoder,decoder}     (2 joint brms)
#   model_freq_layer_duration_sub_{encoder,decoder}      (2 GAM bam)
#   model_predic_layer_duration_sub_{encoder,decoder}    (2 GAM bam)
#
# IMPORTANT: the subword condition shares the SAME underlying V+up test set
# as indep -- only the classifier differs (trained with subword positives
# too). There is no Data/whisper_subword/dataset.csv; duration/dedup-key
# reconstruction uses Data/whisper/dataset.csv, exactly as
# fit_whisper_sub_duration_gams.R already does. (dedup_replication_check.R's
# RUN_WHISPER path had this wrong -- fixed there too, but never executed
# since it was still gated off.)
#
# Same two checks as the indep script:
#   1. pre-fix backup vs. fresh (does the V+up fix change subword results?)
#   2. fresh vs. deduped (does duplicate-audio-clip inflation change them?)
#
# Run from Analyses/ directory.

suppressPackageStartupMessages({
  library(tidyverse)
  library(brms)
  library(mgcv)
})

if (!requireNamespace("cmdstanr", quietly = TRUE))
  stop("cmdstanr not found.")

WH_CACHE_DIR   <- "../model_cache/whisper"
WH_BACKUP_DIR  <- "../model_cache/pre-vup-fix-backup/whisper"
WH_DEDUP_DIR   <- "../model_cache/dedup-check/whisper"
dir.create(WH_DEDUP_DIR, recursive = TRUE, showWarnings = FALSE)

WH_COMPONENTS    <- c("encoder", "decoder")
WH_FINAL_LAYER   <- 11L
MIN_FREQ_VUP     <- 5L
N_TEST_PER_TYPE  <- 20L

JOINT_DURATION_FORM <- "logit ~ c_log_freq * c_log_predic + c_duration + (1 | verb_up)"
dd <- function(name) paste0(name, "_dedup")

# ==============================================================================
# Load fresh subword data (raw + deduped)
# ==============================================================================
message("Loading fresh Whisper subword data...")

ftp_lookup <- read_csv("../Data/ftp_lookup.csv", show_col_types = FALSE) %>% rename(predic = ftp)

load_whisper_component <- function(path, comp) {
  read_csv(path, show_col_types = FALSE) %>%
    select(-any_of("predic")) %>%
    group_by(layer, verb_up) %>%
    mutate(.pos = row_number()) %>%
    ungroup() %>%
    mutate(component = comp, log_freq = log(frequency), verb_up_chr = as.character(verb_up))
}
attach_predic <- function(df) {
  df %>% left_join(ftp_lookup, by = c("verb_up_chr" = "verb_up")) %>%
    filter(!is.na(predic)) %>%
    mutate(log_predic = log(predic / (1 - predic)), verb_up = factor(verb_up_chr)) %>%
    filter(is.finite(log_predic)) %>% select(-verb_up_chr)
}

encoder_raw <- load_whisper_component("../Data/whisper_subword/encoder/all_layers_results.csv", "encoder")
decoder_raw <- load_whisper_component("../Data/whisper_subword/decoder/all_layers_results.csv", "decoder")

# ---- Dedup key: exact same audio clip, reconstructed from the SHARED
# Data/whisper/dataset.csv (subword has no dataset.csv of its own -- same
# V+up test items, only the classifier differs), matched back by (verb_up, .pos).
wh_dataset <- read_csv("../Data/whisper/dataset.csv", show_col_types = FALSE)
vup_df     <- wh_dataset %>% filter(label == "vup")
qualifying <- vup_df %>% count(verb_up, name = "n") %>% filter(n >= MIN_FREQ_VUP) %>% pull(verb_up)

test_items <- vup_df %>%
  filter(verb_up %in% qualifying) %>%
  group_by(verb_up) %>%
  slice_head(n = N_TEST_PER_TYPE) %>%
  mutate(.pos = row_number()) %>%
  ungroup()

n_test_total <- nrow(test_items)
n_dup_rows   <- test_items %>% add_count(verb_up, utt_id, up_start, up_end) %>% filter(n > 1) %>% nrow()
n_types_affected <- test_items %>% add_count(verb_up, utt_id, up_start, up_end) %>% filter(n > 1) %>%
  distinct(verb_up) %>% nrow()
n_types_total <- length(qualifying)
message(sprintf(
  "Duplicate test-instance audit (subword, shared test set): %d/%d test rows (%.1f%%) are exact duplicates; %d/%d types (%.1f%%) affected.",
  n_dup_rows, n_test_total, 100 * n_dup_rows / n_test_total,
  n_types_affected, n_types_total, 100 * n_types_affected / n_types_total
))

keep_pos <- test_items %>%
  distinct(verb_up, utt_id, up_start, up_end, .keep_all = TRUE) %>%
  transmute(verb_up = as.character(verb_up), .pos, keep = TRUE)

dedup_component <- function(df, label) {
  n_before <- nrow(df)
  out <- df %>% inner_join(keep_pos, by = c("verb_up_chr" = "verb_up", ".pos")) %>% select(-keep)
  message(sprintf("  [%s] %d -> %d rows after dedup", label, n_before, nrow(out)))
  out
}
encoder_dedup <- dedup_component(encoder_raw, "encoder")
decoder_dedup <- dedup_component(decoder_raw, "decoder")

encoder <- attach_predic(encoder_raw)
decoder <- attach_predic(decoder_raw)
encoder_dd <- attach_predic(encoder_dedup)
decoder_dd <- attach_predic(decoder_dedup)

recon_duration <- vup_df %>%
  filter(verb_up %in% qualifying) %>%
  group_by(verb_up) %>% slice_head(n = N_TEST_PER_TYPE) %>%
  mutate(.pos = row_number(), duration = up_end - up_start) %>% ungroup() %>%
  select(verb_up, .pos, duration)

attach_duration <- function(df) {
  ref_layer <- min(df$layer)
  ref <- df %>% filter(layer == ref_layer)
  counts <- inner_join(
    recon_duration %>% count(verb_up, name = "n_recon"),
    ref %>% count(verb_up, name = "n_saved"), by = "verb_up"
  )
  good_types <- counts %>% filter(n_recon == n_saved) %>% pull(verb_up)
  df %>% left_join(recon_duration %>% filter(verb_up %in% good_types), by = c("verb_up", ".pos"))
}

# ---- final-layer / joint-model data (c_duration grouped by component, matching run_models_parallel.R) ----
whisper_final_duration_ftp <- bind_rows(attach_duration(encoder), attach_duration(decoder)) %>%
  mutate(component = factor(component, levels = WH_COMPONENTS)) %>%
  filter(layer == WH_FINAL_LAYER, !is.na(log_predic), !is.na(duration)) %>%
  group_by(component) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic)), c_duration = c(scale(duration))) %>% ungroup()

whisper_final_duration_ftp_dd <- bind_rows(attach_duration(encoder_dd), attach_duration(decoder_dd)) %>%
  mutate(component = factor(component, levels = WH_COMPONENTS)) %>%
  filter(layer == WH_FINAL_LAYER, !is.na(log_predic), !is.na(duration)) %>%
  group_by(component) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic)), c_duration = c(scale(duration))) %>% ungroup()

# ---- all-layer / GAM data (c_duration pooled across components, matching fit_whisper_sub_duration_gams.R) ----
whisper_ftp_dur <- bind_rows(attach_duration(encoder), attach_duration(decoder)) %>%
  mutate(component = factor(component, levels = WH_COMPONENTS), verb_up = factor(verb_up)) %>%
  mutate(c_duration = c(scale(duration))) %>%
  filter(!is.na(log_predic), !is.na(duration))

whisper_ftp_dur_dd <- bind_rows(attach_duration(encoder_dd), attach_duration(decoder_dd)) %>%
  mutate(component = factor(component, levels = WH_COMPONENTS), verb_up = factor(verb_up)) %>%
  mutate(c_duration = c(scale(duration))) %>%
  filter(!is.na(log_predic), !is.na(duration))

# ==============================================================================
# Fit: fresh (non-dedup, replaces the stale pre-fix cache) + deduped
# ==============================================================================
fit_joint <- function(data, file) {
  if (file.exists(paste0(file, ".rds"))) { message("  [cached] ", basename(file)); return(invisible(NULL)) }
  message("  Fitting ", basename(file), " ...")
  brm(formula = as.formula(JOINT_DURATION_FORM), data = data,
      prior = set_prior("normal(0, 1)", class = "b"),
      iter = 6000L, warmup = 3000L, chains = 4L, cores = 4L, seed = 964L,
      backend = "cmdstanr", silent = 2L, refresh = 0L, file = file)
  invisible(NULL)
}
fit_gam <- function(formula, data, file) {
  if (file.exists(paste0(file, ".rds"))) { message("  [cached] ", basename(file)); return(invisible(NULL)) }
  message("  Fitting ", basename(file), " ...")
  set.seed(964)
  m <- bam(as.formula(formula), data = data, method = "fREML", discrete = FALSE)
  saveRDS(m, paste0(file, ".rds"))
  invisible(NULL)
}

message("\n=== Fitting fresh (non-dedup) subword models -> model_cache/whisper/ ===")
for (comp in WH_COMPONENTS) {
  fit_joint(whisper_final_duration_ftp %>% filter(component == comp),
            file.path(WH_CACHE_DIR, paste0("model_joint_duration_sub_final_", comp)))
  fit_gam("logit ~ te(log_freq, layer) + c_duration + s(verb_up, bs = 're')",
          whisper_ftp_dur %>% filter(component == comp),
          file.path(WH_CACHE_DIR, paste0("model_freq_layer_duration_sub_", comp)))
  fit_gam("logit ~ te(log_predic, layer) + c_duration + s(verb_up, bs = 're')",
          whisper_ftp_dur %>% filter(component == comp),
          file.path(WH_CACHE_DIR, paste0("model_predic_layer_duration_sub_", comp)))
}

message("\n=== Fitting deduped subword models -> model_cache/dedup-check/whisper/ ===")
for (comp in WH_COMPONENTS) {
  fit_joint(whisper_final_duration_ftp_dd %>% filter(component == comp),
            file.path(WH_DEDUP_DIR, dd(paste0("model_joint_duration_sub_final_", comp))))
  fit_gam("logit ~ te(log_freq, layer) + c_duration + s(verb_up, bs = 're')",
          whisper_ftp_dur_dd %>% filter(component == comp),
          file.path(WH_DEDUP_DIR, dd(paste0("model_freq_layer_duration_sub_", comp))))
  fit_gam("logit ~ te(log_predic, layer) + c_duration + s(verb_up, bs = 're')",
          whisper_ftp_dur_dd %>% filter(component == comp),
          file.path(WH_DEDUP_DIR, dd(paste0("model_predic_layer_duration_sub_", comp))))
}

# ==============================================================================
# Comparisons
# ==============================================================================
compare_joint <- function(a_path, b_path, label) {
  if (!file.exists(a_path) || !file.exists(b_path)) { message("  [", label, "] skipped -- missing file"); return(invisible(NULL)) }
  a_fx <- fixef(readRDS(a_path)); b_fx <- fixef(readRDS(b_path))
  cat(sprintf("\n=== %s ===\n", label))
  for (param in rownames(b_fx)) {
    a_est <- a_fx[param, "Estimate"]; a_lo <- a_fx[param, "Q2.5"]; a_hi <- a_fx[param, "Q97.5"]
    b_est <- b_fx[param, "Estimate"]; b_lo <- b_fx[param, "Q2.5"]; b_hi <- b_fx[param, "Q97.5"]
    a_sig <- (a_lo > 0 || a_hi < 0); b_sig <- (b_lo > 0 || b_hi < 0)
    sign_flip <- sign(a_est) != sign(b_est) && a_sig && b_sig
    sig_flip  <- a_sig != b_sig
    flag <- if (sign_flip) " <<< SIGN FLIP (both credible)" else if (sig_flip) " <<< CREDIBILITY CHANGED" else ""
    cat(sprintf("  %-25s A=%7.3f [%6.3f,%6.3f] cred=%s  ->  B=%7.3f [%6.3f,%6.3f] cred=%s%s\n",
                param, a_est, a_lo, a_hi, a_sig, b_est, b_lo, b_hi, b_sig, flag))
  }
}
compare_gam <- function(a_path, b_path, label) {
  if (!file.exists(a_path) || !file.exists(b_path)) { message("  [", label, "] skipped -- missing file"); return(invisible(NULL)) }
  a_s <- summary(readRDS(a_path))$s.table; b_s <- summary(readRDS(b_path))$s.table
  a_te <- a_s[grep("^te\\(", rownames(a_s)), , drop = FALSE][1, ]
  b_te <- b_s[grep("^te\\(", rownames(b_s)), , drop = FALSE][1, ]
  cat(sprintf("  %-30s A: edf=%.2f F=%.1f p=%.4g  ->  B: edf=%.2f F=%.1f p=%.4g\n",
              label, a_te["edf"], a_te["F"], a_te["p-value"], b_te["edf"], b_te["F"], b_te["p-value"]))
}

cat("\n\n################## CHECK 1: pre-fix backup vs fresh (does the V+up fix change subword results?) ##################\n")
for (comp in WH_COMPONENTS) {
  compare_joint(file.path(WH_BACKUP_DIR, paste0("model_joint_duration_sub_final_", comp, ".rds")),
                file.path(WH_CACHE_DIR, paste0("model_joint_duration_sub_final_", comp, ".rds")),
                paste0("Whisper ", comp, " subword joint (Exp 3 replication)"))
}
for (comp in WH_COMPONENTS) {
  compare_gam(file.path(WH_BACKUP_DIR, paste0("model_freq_layer_duration_sub_", comp, ".rds")),
              file.path(WH_CACHE_DIR, paste0("model_freq_layer_duration_sub_", comp, ".rds")), paste0("Whisper ", comp, " sub freq GAM"))
  compare_gam(file.path(WH_BACKUP_DIR, paste0("model_predic_layer_duration_sub_", comp, ".rds")),
              file.path(WH_CACHE_DIR, paste0("model_predic_layer_duration_sub_", comp, ".rds")), paste0("Whisper ", comp, " sub predic GAM"))
}

cat("\n\n################## CHECK 2: fresh vs deduped (does duplicate-instance inflation change subword results?) ##################\n")
for (comp in WH_COMPONENTS) {
  compare_joint(file.path(WH_CACHE_DIR, paste0("model_joint_duration_sub_final_", comp, ".rds")),
                file.path(WH_DEDUP_DIR, paste0(dd(paste0("model_joint_duration_sub_final_", comp)), ".rds")),
                paste0("Whisper ", comp, " subword joint (Exp 3 replication)"))
}
for (comp in WH_COMPONENTS) {
  compare_gam(file.path(WH_CACHE_DIR, paste0("model_freq_layer_duration_sub_", comp, ".rds")),
              file.path(WH_DEDUP_DIR, paste0(dd(paste0("model_freq_layer_duration_sub_", comp)), ".rds")), paste0("Whisper ", comp, " sub freq GAM"))
  compare_gam(file.path(WH_CACHE_DIR, paste0("model_predic_layer_duration_sub_", comp, ".rds")),
              file.path(WH_DEDUP_DIR, paste0(dd(paste0("model_predic_layer_duration_sub_", comp)), ".rds")), paste0("Whisper ", comp, " sub predic GAM"))
}

message("\nDone.")
