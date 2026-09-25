# whisper_alone_duration_models.R
#
# Fits (or refits, since the subword-condition ones are stale pre-fix
# copies) the single-predictor "frequency alone" / "predictability alone"
# models, controlling for duration, for BOTH Whisper conditions x both
# components -- 8 models total:
#   model_freq_duration_final_{encoder,decoder}       (indep, NEW)
#   model_predic_duration_final_{encoder,decoder}      (indep, NEW)
#   model_freq_duration_sub_final_{encoder,decoder}    (subword, refit)
#   model_predic_duration_sub_final_{encoder,decoder}  (subword, refit)
#
# Purpose: the joint-model refit found the encoder's freq/predic effects
# became unstable (indep: lost credibility; subword: credible sign flip)
# once the V+up fix was applied, and a recomputed zero-order-correlation
# suppression table (Equation eq-suppression-threshold in the writeup)
# predicts this is the same statistical-suppression mechanism already
# documented for the decoder-subword case, now also triggered for the
# encoder. These single-predictor models are the direct empirical check:
# if suppression is the explanation, frequency and predictability should
# each still show a credible, correctly-signed effect ALONE (matching
# their zero-order correlation), even where the JOINT model doesn't.
#
# Formulas match FREQ_DURATION_FORM/PREDIC_DURATION_FORM in
# run_models_parallel.R exactly (log_freq/log_predic left uncentered,
# c_duration scaled -- see that script's comment for why).
#
# Run from Analyses/ directory.

suppressPackageStartupMessages({
  library(tidyverse)
  library(brms)
})
if (!requireNamespace("cmdstanr", quietly = TRUE)) stop("cmdstanr not found.")

WH_CACHE_DIR  <- "../model_cache/whisper"
WH_COMPONENTS <- c("encoder", "decoder")
WH_FINAL_LAYER <- 11L
MIN_FREQ_VUP <- 5L
N_TEST_PER_TYPE <- 20L

FREQ_DURATION_FORM   <- "logit ~ log_freq + c_duration + (1 | verb_up)"
PREDIC_DURATION_FORM <- "logit ~ log_predic + c_duration + (1 | verb_up)"

ftp_lookup <- read_csv("../Data/ftp_lookup.csv", show_col_types = FALSE) %>% rename(predic = ftp)

load_whisper_component <- function(path, comp) {
  read_csv(path, show_col_types = FALSE) %>%
    select(-any_of("predic")) %>%
    group_by(layer, verb_up) %>%
    mutate(.pos = row_number()) %>%
    ungroup() %>%
    mutate(component = comp, log_freq = log(frequency), verb_up_chr = as.character(verb_up)) %>%
    left_join(ftp_lookup, by = c("verb_up_chr" = "verb_up")) %>%
    filter(!is.na(predic)) %>%
    mutate(log_predic = log(predic / (1 - predic)), verb_up = factor(verb_up_chr)) %>%
    filter(is.finite(log_predic)) %>% select(-verb_up_chr)
}

wh_dataset <- read_csv("../Data/whisper/dataset.csv", show_col_types = FALSE)
vup_df     <- wh_dataset %>% filter(label == "vup")
qualifying <- vup_df %>% count(verb_up, name = "n") %>% filter(n >= MIN_FREQ_VUP) %>% pull(verb_up)
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

build_final_ftp <- function(enc_path, dec_path) {
  encoder <- load_whisper_component(enc_path, "encoder")
  decoder <- load_whisper_component(dec_path, "decoder")
  bind_rows(attach_duration(encoder), attach_duration(decoder)) %>%
    mutate(component = factor(component, levels = WH_COMPONENTS)) %>%
    filter(layer == WH_FINAL_LAYER, !is.na(log_predic), !is.na(duration)) %>%
    group_by(component) %>%
    mutate(c_duration = c(scale(duration))) %>%
    ungroup()
}

message("Loading indep-condition data...")
indep_ftp <- build_final_ftp("../Data/whisper/encoder/all_layers_results.csv", "../Data/whisper/decoder/all_layers_results.csv")
message("Loading subword-condition data...")
sub_ftp   <- build_final_ftp("../Data/whisper_subword/encoder/all_layers_results.csv", "../Data/whisper_subword/decoder/all_layers_results.csv")

fit_alone <- function(formula, data, file) {
  if (file.exists(paste0(file, ".rds"))) { message("  [cached] ", basename(file)); return(invisible(NULL)) }
  message("  Fitting ", basename(file), " ...")
  brm(formula = as.formula(formula), data = data,
      prior = set_prior("normal(0, 1)", class = "b"),
      iter = 6000L, warmup = 3000L, chains = 4L, cores = 4L, seed = 964L,
      backend = "cmdstanr", silent = 2L, refresh = 0L, file = file)
  invisible(NULL)
}

message("\n=== Fitting indep-condition alone+duration models (NEW) ===")
for (comp in WH_COMPONENTS) {
  fit_alone(FREQ_DURATION_FORM,   indep_ftp %>% filter(component == comp), file.path(WH_CACHE_DIR, paste0("model_freq_duration_final_", comp)))
  fit_alone(PREDIC_DURATION_FORM, indep_ftp %>% filter(component == comp), file.path(WH_CACHE_DIR, paste0("model_predic_duration_final_", comp)))
}

message("\n=== Fitting subword-condition alone+duration models (refit, was stale) ===")
for (comp in WH_COMPONENTS) {
  fit_alone(FREQ_DURATION_FORM,   sub_ftp %>% filter(component == comp), file.path(WH_CACHE_DIR, paste0("model_freq_duration_sub_final_", comp)))
  fit_alone(PREDIC_DURATION_FORM, sub_ftp %>% filter(component == comp), file.path(WH_CACHE_DIR, paste0("model_predic_duration_sub_final_", comp)))
}

# ==============================================================================
# Report
# ==============================================================================
report <- function(path, label) {
  m <- readRDS(path)
  fx <- fixef(m)
  post <- as_draws_df(m)
  p <- rownames(fx)[rownames(fx) != "Intercept" & rownames(fx) != "c_duration"]
  col <- paste0("b_", p)
  pgt0 <- mean(post[[col]] > 0) * 100
  cred <- (fx[p,"Q2.5"] > 0 || fx[p,"Q97.5"] < 0)
  cat(sprintf("  %-40s Est=%7.3f  95%% CI=[%6.3f, %6.3f]  cred=%s  %%>0=%.1f%%\n",
              label, fx[p,"Estimate"], fx[p,"Q2.5"], fx[p,"Q97.5"], cred, pgt0))
}

cat("\n\n================ ALONE MODELS (duration-controlled), FRESH DATA ================\n")
cat("\n-- Indep condition --\n")
for (comp in WH_COMPONENTS) {
  report(file.path(WH_CACHE_DIR, paste0("model_freq_duration_final_", comp, ".rds")), paste0(comp, " frequency alone"))
  report(file.path(WH_CACHE_DIR, paste0("model_predic_duration_final_", comp, ".rds")), paste0(comp, " predictability alone"))
}
cat("\n-- Subword condition --\n")
for (comp in WH_COMPONENTS) {
  report(file.path(WH_CACHE_DIR, paste0("model_freq_duration_sub_final_", comp, ".rds")), paste0(comp, " frequency alone"))
  report(file.path(WH_CACHE_DIR, paste0("model_predic_duration_sub_final_", comp, ".rds")), paste0(comp, " predictability alone"))
}

message("\nDone.")
