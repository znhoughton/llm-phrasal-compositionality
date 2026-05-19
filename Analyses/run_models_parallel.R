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
# Hardware: 24 cores assumed — 6 parallel models × 4 chains each.
# Adjust N_WORKERS below if your core count differs.

suppressPackageStartupMessages({
  library(tidyverse)
  library(brms)
  library(future)
  library(furrr)
})

if (!requireNamespace("cmdstanr", quietly = TRUE))
  stop("cmdstanr not found. Install with:\n  install.packages('cmdstanr', repos = c('https://mc-stan.org/r-packages/', getOption('repos')))\n  cmdstanr::install_cmdstan()")

N_WORKERS <- 6L  # 6 models × 4 chains = 24 cores

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
  final     = whisper_final     %>% filter(component == comp),
  first     = whisper_first     %>% filter(component == comp),
  final_ftp = whisper_final_ftp %>% filter(component == comp),
  first_ftp = whisper_first_ftp %>% filter(component == comp)
)), WH_COMPONENTS)

# ---- Build model spec list --------------------------------------------------
FREQ_FORM       <- "logit ~ log_freq + (1 | verb_up)"
PREDIC_FORM     <- "logit ~ log_predic + (1 | verb_up)"
JOINT_FORM      <- "logit ~ c_log_freq * c_log_predic + (1 | verb_up)"
POLY_JOINT_FORM <- paste0("logit ~ c_log_freq + I(c_log_freq^2) + ",
                           "c_log_predic + I(c_log_predic^2) + ",
                           "c_log_freq:c_log_predic + (1 | verb_up)")

mk <- function(formula, data, file) list(formula = formula, data = data, file = file)

model_specs <- c(
  # ---- OLMo (16 models) -----------------------------------------------------
  list(
    # frequency
    mk(FREQ_FORM, olmo_indep_final,     olmo_brms_path("model_freq_up_independently")),
    mk(FREQ_FORM, olmo_indep_first,     olmo_brms_path("model_freq_up_independently_first_layer")),
    mk(FREQ_FORM, olmo_sub_final,       olmo_brms_path("model_freq_up_subword")),
    mk(FREQ_FORM, olmo_sub_first,       olmo_brms_path("model_freq_up_subword_first_layer")),
    # predictability
    mk(PREDIC_FORM, olmo_indep_final_ftp, olmo_brms_path("model_predic_up_independently")),
    mk(PREDIC_FORM, olmo_sub_final_ftp,   olmo_brms_path("model_predic_up_subword")),
    mk(PREDIC_FORM, olmo_indep_first_ftp, olmo_brms_path("model_predic_up_independently_first_layer")),
    mk(PREDIC_FORM, olmo_sub_first_ftp,   olmo_brms_path("model_predic_up_subword_first_layer")),
    # joint (interaction)
    mk(JOINT_FORM, olmo_indep_final_ftp,  olmo_brms_path("model_freq_predic_up_independently")),
    mk(JOINT_FORM, olmo_sub_final_ftp,    olmo_brms_path("model_freq_predic_up_subword")),
    mk(JOINT_FORM, olmo_indep_first_ftp,  olmo_brms_path("model_freq_predic_up_independently_first")),
    mk(JOINT_FORM, olmo_sub_first_ftp,    olmo_brms_path("model_freq_predic_up_subword_first")),
    # polynomial joint
    mk(POLY_JOINT_FORM, olmo_indep_final_ftp, olmo_brms_path("model_poly_joint_indep_final")),
    mk(POLY_JOINT_FORM, olmo_sub_final_ftp,   olmo_brms_path("model_poly_joint_sub_final")),
    mk(POLY_JOINT_FORM, olmo_indep_first_ftp, olmo_brms_path("model_poly_joint_indep_first")),
    mk(POLY_JOINT_FORM, olmo_sub_first_ftp,   olmo_brms_path("model_poly_joint_sub_first"))
  ),
  # ---- BabyLM (48 models: 16 groups × 3 tags) --------------------------------
  unlist(lapply(BLM_TAGS, function(tag) {
    sl <- blm_slug(tag)
    d  <- blm_data[[tag]]
    list(
      # frequency
      mk(FREQ_FORM, d$indep_final,     blm_brms_path(paste0("model_freq_indep_final_", sl))),
      mk(FREQ_FORM, d$indep_first,     blm_brms_path(paste0("model_freq_indep_first_", sl))),
      mk(FREQ_FORM, d$sub_final,       blm_brms_path(paste0("model_freq_sub_final_",   sl))),
      mk(FREQ_FORM, d$sub_first,       blm_brms_path(paste0("model_freq_sub_first_",   sl))),
      # predictability
      mk(PREDIC_FORM, d$indep_final_ftp, blm_brms_path(paste0("model_predic_indep_final_", sl))),
      mk(PREDIC_FORM, d$sub_final_ftp,   blm_brms_path(paste0("model_predic_sub_final_",   sl))),
      mk(PREDIC_FORM, d$indep_first_ftp, blm_brms_path(paste0("model_predic_indep_first_", sl))),
      mk(PREDIC_FORM, d$sub_first_ftp,   blm_brms_path(paste0("model_predic_sub_first_",   sl))),
      # joint
      mk(JOINT_FORM, d$indep_final_ftp, blm_brms_path(paste0("model_joint_indep_final_", sl))),
      mk(JOINT_FORM, d$sub_final_ftp,   blm_brms_path(paste0("model_joint_sub_final_",   sl))),
      mk(JOINT_FORM, d$indep_first_ftp, blm_brms_path(paste0("model_joint_indep_first_", sl))),
      mk(JOINT_FORM, d$sub_first_ftp,   blm_brms_path(paste0("model_joint_sub_first_",   sl))),
      # polynomial joint
      mk(POLY_JOINT_FORM, d$indep_final_ftp, blm_brms_path(paste0("model_poly_joint_indep_final_", sl))),
      mk(POLY_JOINT_FORM, d$sub_final_ftp,   blm_brms_path(paste0("model_poly_joint_sub_final_",   sl))),
      mk(POLY_JOINT_FORM, d$indep_first_ftp, blm_brms_path(paste0("model_poly_joint_indep_first_", sl))),
      mk(POLY_JOINT_FORM, d$sub_first_ftp,   blm_brms_path(paste0("model_poly_joint_sub_first_",   sl)))
    )
  }), recursive = FALSE),
  # ---- Whisper (16 models: 8 groups × 2 components) --------------------------
  unlist(lapply(WH_COMPONENTS, function(comp) {
    d <- wh_data[[comp]]
    list(
      # frequency
      mk(FREQ_FORM,       d$final,     wh_brms_path(paste0("model_freq_final_",       comp))),
      mk(FREQ_FORM,       d$first,     wh_brms_path(paste0("model_freq_first_",       comp))),
      # predictability
      mk(PREDIC_FORM,     d$final_ftp, wh_brms_path(paste0("model_predic_final_",     comp))),
      mk(PREDIC_FORM,     d$first_ftp, wh_brms_path(paste0("model_predic_first_",     comp))),
      # joint
      mk(JOINT_FORM,      d$final_ftp, wh_brms_path(paste0("model_joint_final_",      comp))),
      mk(JOINT_FORM,      d$first_ftp, wh_brms_path(paste0("model_joint_first_",      comp))),
      # polynomial joint
      mk(POLY_JOINT_FORM, d$final_ftp, wh_brms_path(paste0("model_poly_joint_final_", comp))),
      mk(POLY_JOINT_FORM, d$first_ftp, wh_brms_path(paste0("model_poly_joint_first_", comp)))
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
  .options  = furrr_options(packages = "brms")
)
plan(sequential)

message("Done. All fitted models saved to model_cache/.")
message("You can now knit analysis-script.Rmd — all brm() calls will load from cache.")