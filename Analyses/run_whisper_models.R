# run_whisper_models.R
#
# Temporary script to pre-fit Whisper brm() models only.
# Run from the Analyses/ directory while OLMo/BabyLM pipelines are still running.
# Models are saved to the same model_cache/whisper/ directory that
# run_models_parallel.R and analysis-script.Rmd expect.

suppressPackageStartupMessages({
  library(tidyverse)
  library(brms)
  library(future)
  library(furrr)
})

N_WORKERS <- 4L  # 4 models × 4 chains = 16 cores

WH_CACHE_DIR <- "../model_cache/whisper"
dir.create(WH_CACHE_DIR, recursive = TRUE, showWarnings = FALSE)
wh_brms_path <- function(name) file.path(WH_CACHE_DIR, name)

WH_COMPONENTS  <- c("encoder", "decoder")
WH_FINAL_LAYER <- 11L

# ---- Load Whisper data -------------------------------------------------------
message("Loading Whisper data...")
ftp_lookup <- read_csv("../Data/ftp_lookup.csv", show_col_types = FALSE) %>%
  rename(predic = ftp)

load_whisper_component <- function(path, comp) {
  read_csv(path, show_col_types = FALSE) %>%
    select(-any_of("predic")) %>%
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

whisper_final <- whisper_all %>% filter(layer == WH_FINAL_LAYER)
whisper_first <- whisper_all %>% filter(layer == 0L)

whisper_final_ftp <- whisper_final %>% filter(!is.na(log_predic)) %>%
  group_by(component) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic))) %>% ungroup()
whisper_first_ftp <- whisper_first %>% filter(!is.na(log_predic)) %>%
  group_by(component) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic))) %>% ungroup()

wh_data <- setNames(lapply(WH_COMPONENTS, function(comp) list(
  final     = whisper_final     %>% filter(component == comp),
  first     = whisper_first     %>% filter(component == comp),
  final_ftp = whisper_final_ftp %>% filter(component == comp),
  first_ftp = whisper_first_ftp %>% filter(component == comp)
)), WH_COMPONENTS)

# ---- Formulas ----------------------------------------------------------------
FREQ_FORM       <- "logit ~ log_freq + (1 | verb_up)"
PREDIC_FORM     <- "logit ~ log_predic + (1 | verb_up)"
JOINT_FORM      <- "logit ~ c_log_freq * c_log_predic + (1 | verb_up)"
POLY_JOINT_FORM <- paste0("logit ~ c_log_freq + I(c_log_freq^2) + ",
                           "c_log_predic + I(c_log_predic^2) + ",
                           "c_log_freq:c_log_predic + (1 | verb_up)")

mk <- function(formula, data, file) list(formula = formula, data = data, file = file)

# ---- Whisper model specs (16 models) -----------------------------------------
model_specs <- unlist(lapply(WH_COMPONENTS, function(comp) {
  d <- wh_data[[comp]]
  list(
    mk(FREQ_FORM,       d$final,     wh_brms_path(paste0("model_freq_final_",       comp))),
    mk(FREQ_FORM,       d$first,     wh_brms_path(paste0("model_freq_first_",       comp))),
    mk(PREDIC_FORM,     d$final_ftp, wh_brms_path(paste0("model_predic_final_",     comp))),
    mk(PREDIC_FORM,     d$first_ftp, wh_brms_path(paste0("model_predic_first_",     comp))),
    mk(JOINT_FORM,      d$final_ftp, wh_brms_path(paste0("model_joint_final_",      comp))),
    mk(JOINT_FORM,      d$first_ftp, wh_brms_path(paste0("model_joint_first_",      comp))),
    mk(POLY_JOINT_FORM, d$final_ftp, wh_brms_path(paste0("model_poly_joint_final_", comp))),
    mk(POLY_JOINT_FORM, d$first_ftp, wh_brms_path(paste0("model_poly_joint_first_", comp)))
  )
}), recursive = FALSE)

pending <- Filter(function(s) !file.exists(paste0(s$file, ".rds")), model_specs)

message(sprintf(
  "%d / %d Whisper models need fitting (rest already cached).",
  length(pending), length(model_specs)
))

if (length(pending) == 0L) {
  message("All Whisper models cached. Nothing to do.")
  quit(status = 0L)
}

# ---- Worker function ---------------------------------------------------------
run_one_model <- function(spec) {
  library(brms)
  brm(
    formula = as.formula(spec$formula),
    data    = spec$data,
    prior   = set_prior("normal(0, 1)", class = "b"),
    iter    = 6000L, warmup = 3000L, chains = 4L, cores = 4L, seed = 964L,
    backend = "cmdstanr",
    silent  = 2L,
    refresh = 0L,
    file    = spec$file
  )
  invisible(spec$file)
}

# ---- Run in parallel ---------------------------------------------------------
message(sprintf(
  "Fitting %d Whisper models: %d workers × 4 chains = %d cores.",
  length(pending), N_WORKERS, N_WORKERS * 4L
))

plan(multisession, workers = N_WORKERS)
future_map(pending, run_one_model, .progress = TRUE,
           .options = furrr_options(packages = "brms"))
plan(sequential)

message("Done. Whisper models saved to ", WH_CACHE_DIR)