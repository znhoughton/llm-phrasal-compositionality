# prepare_results.R
# Pre-computes all data needed for writeup.qmd and saves to writeup/results/.
# Run this once (or after re-fitting models) before rendering the writeup.
#
# Can be run from any directory — it locates itself automatically.

library(tidyverse)
library(brms)
library(mgcv)
library(tidybayes)
library(jsonlite)

# ── Environment check ──────────────────────────────────────────────────────────
# This script must be run from the project root (llm-phrasal-compositionality/).
probe_file = "Data/olmo-3-7b/Data_up/all_layers_results.csv"
if (!file.exists(probe_file)) {
  stop("Working directory must be the project root (llm-phrasal-compositionality/). ",
       "Current directory: ", getwd(), "\n",
       "In RStudio: Session > Set Working Directory > To Project Directory, ",
       "then re-run.")
}
first_line = readLines(probe_file, n = 1, warn = FALSE)
lfs_data_available = !grepl("git-lfs", first_line, fixed = TRUE)
if (!lfs_data_available) {
  message("NOTE: Data files are Git LFS pointers (not yet pulled). ",
          "Sections 1-4 will be skipped. ",
          "Run 'git lfs pull' to enable them. Section 5b runs fine without raw data.")
}
rm(probe_file, first_line)

# ── Constants ──────────────────────────────────────────────────────────────────
# Paths are relative to the project root (llm-phrasal-compositionality/).
OLMO_CACHE_DIR      = "model_cache/olmo"
BLM_CACHE_DIR       = "model_cache/babylm"
WH_CACHE_DIR        = "model_cache/whisper"
RESULTS_DIR         = "writeup/results"
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

OLMO_FINAL_LAYER    = 31L
OLMO_N_LAYERS       = 32L
BLM_FINAL_LAYER_MAP = list("opt-125m" = 11L, "opt-350m" = 23L, "opt-1.3b" = 23L)
BLM_N_LAYERS        = c("opt-125m" = 12L, "opt-350m" = 24L, "opt-1.3b" = 24L)
WH_FINAL_LAYER      = 11L
WH_N_LAYERS         = 12L

BLM_TAGS      = c("opt-125m", "opt-350m", "opt-1.3b")
WH_COMPONENTS = c("encoder", "decoder")
blm_slug      = function(tag) gsub("\\.", "", gsub("opt-", "", tag))
BLM_LABELS    = c("opt-125m" = "BabyLM 125M", "opt-350m" = "BabyLM 350M", "opt-1.3b" = "BabyLM 1.3B")
WH_LABELS     = c("encoder"  = "Encoder",      "decoder"  = "Decoder")

olmo_rds = function(name) readRDS(file.path(OLMO_CACHE_DIR, paste0(name, ".rds")))
blm_rds  = function(name) readRDS(file.path(BLM_CACHE_DIR,  paste0(name, ".rds")))
wh_rds   = function(name) readRDS(file.path(WH_CACHE_DIR,   paste0(name, ".rds")))

if (lfs_data_available) {

# ── 1. Raw data ────────────────────────────────────────────────────────────────
message("Loading raw data...")

olmo_indep = read_csv("Data/olmo-3-7b/Data_up/all_layers_results.csv",
                      show_col_types = FALSE) %>%
  mutate(log_freq = log(frequency), log_predic = log(predic / (1 - predic)),
         verb_up = factor(verb_up)) %>%
  filter(!is.na(predic), is.finite(log_predic))

olmo_sub = read_csv("Data/olmo-3-7b/Data_upsubword/all_layers_results.csv",
                    show_col_types = FALSE) %>%
  mutate(log_freq = log(frequency), log_predic = log(predic / (1 - predic)),
         verb_up = factor(verb_up)) %>%
  filter(!is.na(predic), is.finite(log_predic))

load_babylm_csvs = function(tag) {
  list(
    ui = read_csv(paste0("Data/babylm/", tag, "/Data_up/all_layers_results.csv"),
                  show_col_types = FALSE) %>%
      rename_with(~ sub("^ftp$", "predic", .)) %>%
      mutate(model = tag, log_freq = log(frequency),
             log_predic = log(predic / (1 - predic)), verb_up = factor(verb_up)) %>%
      filter(!is.na(predic), is.finite(log_predic)),
    us = read_csv(paste0("Data/babylm/", tag, "/Data_upsubword/all_layers_results.csv"),
                  show_col_types = FALSE) %>%
      rename_with(~ sub("^ftp$", "predic", .)) %>%
      mutate(model = tag, log_freq = log(frequency),
             log_predic = log(predic / (1 - predic)), verb_up = factor(verb_up)) %>%
      filter(!is.na(predic), is.finite(log_predic))
  )
}
blm_raw   = setNames(lapply(BLM_TAGS, load_babylm_csvs), BLM_TAGS)
blm_indep = map_dfr(blm_raw, "ui") %>% mutate(model = factor(model, levels = BLM_TAGS))
blm_sub   = map_dfr(blm_raw, "us") %>% mutate(model = factor(model, levels = BLM_TAGS))

ftp_lookup = read_csv("Data/ftp_lookup.csv", show_col_types = FALSE) %>%
  rename(predic = ftp)
load_whisper_component = function(path, comp) {
  read_csv(path, show_col_types = FALSE) %>%
    select(-any_of(c("ftp", "predic"))) %>%
    mutate(component = comp, frequency = suppressWarnings(as.numeric(frequency)),
           log_freq = log(frequency),
           verb_up_chr = as.character(verb_up)) %>%
    left_join(ftp_lookup, by = c("verb_up_chr" = "verb_up")) %>%
    filter(!is.na(predic)) %>%
    mutate(log_predic = log(predic / (1 - predic)), verb_up = factor(verb_up_chr)) %>%
    filter(is.finite(log_predic)) %>% select(-verb_up_chr)
}
whisper_all = bind_rows(
  load_whisper_component("Data/whisper/encoder/all_layers_results.csv", "encoder"),
  load_whisper_component("Data/whisper/decoder/all_layers_results.csv", "decoder")
) %>% mutate(component = factor(component, levels = c("encoder", "decoder")))

# ── 2. Predictability-filtered subsets ─────────────────────────────────────────
blm_final_layer_lut = tibble(model = names(BLM_FINAL_LAYER_MAP),
                              final_layer = unlist(BLM_FINAL_LAYER_MAP))

olmo_indep_final_predic = olmo_indep %>% filter(layer == OLMO_FINAL_LAYER, !is.na(log_predic)) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic)))
olmo_indep_first_predic = olmo_indep %>% filter(layer == 0, !is.na(log_predic)) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic)))
olmo_sub_final_predic   = olmo_sub %>% filter(layer == OLMO_FINAL_LAYER, !is.na(log_predic)) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic)))
olmo_sub_first_predic   = olmo_sub %>% filter(layer == 0, !is.na(log_predic)) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic)))

blm_indep_final_predic = blm_indep %>% filter(!is.na(log_predic)) %>%
  left_join(blm_final_layer_lut, by = "model") %>%
  filter(layer == final_layer) %>% select(-final_layer) %>%
  group_by(model) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic))) %>% ungroup()
blm_indep_first_predic = blm_indep %>% filter(layer == 0, !is.na(log_predic)) %>%
  group_by(model) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic))) %>% ungroup()
blm_sub_final_predic = blm_sub %>% filter(!is.na(log_predic)) %>%
  left_join(blm_final_layer_lut, by = "model") %>%
  filter(layer == final_layer) %>% select(-final_layer) %>%
  group_by(model) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic))) %>% ungroup()
blm_sub_first_predic = blm_sub %>% filter(layer == 0, !is.na(log_predic)) %>%
  group_by(model) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic))) %>% ungroup()

whisper_final_predic = whisper_all %>% filter(layer == WH_FINAL_LAYER, !is.na(log_predic)) %>%
  group_by(component) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic))) %>% ungroup()
whisper_first_predic = whisper_all %>% filter(layer == 0, !is.na(log_predic)) %>%
  group_by(component) %>%
  mutate(c_log_freq = c(scale(log_freq)), c_log_predic = c(scale(log_predic))) %>% ungroup()

# ── 3. Items tables ────────────────────────────────────────────────────────────
message("Saving items tables...")

olmo_indep %>% filter(layer == 0) %>% group_by(verb_up) %>%
  summarise(frequency = first(frequency), predic = first(predic),
            log_freq = first(log_freq), log_predic = first(log_predic), .groups = "drop") %>%
  write_csv(file.path(RESULTS_DIR, "olmo_items.csv"))

blm_indep %>% filter(layer == 0, model == "opt-125m") %>% group_by(verb_up) %>%
  summarise(frequency = first(frequency), predic = first(predic),
            log_freq = first(log_freq), log_predic = first(log_predic), .groups = "drop") %>%
  write_csv(file.path(RESULTS_DIR, "blm_items.csv"))

whisper_all %>% filter(layer == 0) %>%
  group_by(component, verb_up) %>%
  summarise(frequency = first(frequency), predic = first(predic),
            log_freq = first(log_freq), log_predic = first(log_predic), .groups = "drop") %>%
  write_csv(file.path(RESULTS_DIR, "whisper_items_by_component.csv"))

# ── 4. Whisper splits table ────────────────────────────────────────────────────
message("Saving Whisper splits table...")

wh_layer_meta = lapply(setNames(WH_COMPONENTS, WH_COMPONENTS), function(comp) {
  meta = fromJSON(file.path("Data/whisper", comp, "layer_metadata.json"))
  meta$layers[meta$layers$layer == 0, ]
})
wh_test_counts = whisper_all %>%
  filter(layer == 0) %>%
  group_by(component) %>%
  summarise(test_sentences = n(), test_types = n_distinct(verb_up),
            items_w_predic = n_distinct(verb_up[!is.na(predic)]), .groups = "drop")

tibble(component = WH_COMPONENTS, Component = c("Encoder", "Decoder")) %>%
  rowwise() %>%
  mutate(
    `Train pos` = wh_layer_meta[[component]]$train_n_positive,
    `Train neg` = wh_layer_meta[[component]]$train_n_negative,
    `Val pos`   = wh_layer_meta[[component]]$val_n_positive,
    `Val neg`   = wh_layer_meta[[component]]$val_n_negative
  ) %>%
  ungroup() %>%
  left_join(wh_test_counts %>% mutate(Component = str_to_title(component)), by = "Component") %>%
  select(Component, `Train pos`, `Train neg`, `Val pos`, `Val neg`,
         `Test sentences` = test_sentences, `Test types` = test_types,
         `Items w/ predic.` = items_w_predic) %>%
  write_csv(file.path(RESULTS_DIR, "wh_splits.csv"))

# ── 5. GAM diff data ───────────────────────────────────────────────────────────
message("Loading GAMs and computing diff data...")

freq_diff_data = function(model, data, n_layers, panel) {
  layer_seq   = 0:(n_layers - 1)
  log_freq_lo = quantile(data$log_freq, 0.1)
  log_freq_hi = quantile(data$log_freq, 0.9)
  newdata = expand.grid(layer = layer_seq, freq_level = c("low", "high")) %>%
    mutate(log_freq = ifelse(freq_level == "low", log_freq_lo, log_freq_hi),
           verb_up  = data$verb_up[1])
  preds = predict(model, newdata = newdata, se.fit = TRUE, exclude = "s(verb_up)")
  newdata %>%
    mutate(fit = preds$fit, se = preds$se.fit) %>%
    select(layer, freq_level, fit, se) %>%
    pivot_wider(names_from = freq_level, values_from = c(fit, se)) %>%
    mutate(diff    = fit_high - fit_low,
           se_diff = sqrt(se_high^2 + se_low^2),
           lo95    = diff - 1.96 * se_diff, hi95 = diff + 1.96 * se_diff,
           lo80    = diff - 1.28 * se_diff, hi80 = diff + 1.28 * se_diff,
           panel   = panel)
}

predic_diff_data = function(model, data, n_layers, panel) {
  layer_seq  = 0:(n_layers - 1)
  predic_lo  = quantile(data$log_predic, 0.1, na.rm = TRUE)
  predic_hi  = quantile(data$log_predic, 0.9, na.rm = TRUE)
  newdata = expand.grid(layer = layer_seq, predic_level = c("low", "high")) %>%
    mutate(log_predic = ifelse(predic_level == "low", predic_lo, predic_hi),
           verb_up = data$verb_up[1])
  preds = predict(model, newdata = newdata, se.fit = TRUE, exclude = "s(verb_up)")
  newdata %>%
    mutate(fit = preds$fit, se = preds$se.fit) %>%
    select(layer, predic_level, fit, se) %>%
    pivot_wider(names_from = predic_level, values_from = c(fit, se)) %>%
    mutate(diff    = fit_high - fit_low,
           se_diff = sqrt(se_high^2 + se_low^2),
           lo95    = diff - 1.96 * se_diff, hi95 = diff + 1.96 * se_diff,
           lo80    = diff - 1.28 * se_diff, hi80 = diff + 1.28 * se_diff,
           panel   = panel)
}

olmo_gam_freq_indep = olmo_rds("model_freq_layer_up_independently")
olmo_gam_freq_sub   = olmo_rds("model_freq_layer_up_subword")
blm_gam_freq_indep  = setNames(lapply(BLM_TAGS, function(tag)
  blm_rds(paste0("model_freq_layer_indep_", blm_slug(tag)))), BLM_TAGS)
blm_gam_freq_sub    = setNames(lapply(BLM_TAGS, function(tag)
  blm_rds(paste0("model_freq_layer_sub_", blm_slug(tag)))), BLM_TAGS)
wh_gam_freq         = setNames(lapply(WH_COMPONENTS, function(comp)
  wh_rds(paste0("model_freq_layer_", comp))), WH_COMPONENTS)

olmo_gam_predic_indep = olmo_rds("model_predic_layer_up_independently")
olmo_gam_predic_sub   = olmo_rds("model_predic_layer_up_subword")
blm_gam_predic_indep  = setNames(lapply(BLM_TAGS, function(tag)
  blm_rds(paste0("model_predic_layer_indep_", blm_slug(tag)))), BLM_TAGS)
blm_gam_predic_sub    = setNames(lapply(BLM_TAGS, function(tag)
  blm_rds(paste0("model_predic_layer_sub_", blm_slug(tag)))), BLM_TAGS)
wh_gam_predic         = setNames(lapply(WH_COMPONENTS, function(comp)
  wh_rds(paste0("model_predic_layer_", comp))), WH_COMPONENTS)

bind_rows(
  freq_diff_data(olmo_gam_freq_indep, olmo_indep, OLMO_N_LAYERS, "OLMo-3 7B"),
  bind_rows(lapply(BLM_TAGS, function(tag)
    freq_diff_data(blm_gam_freq_indep[[tag]], blm_indep %>% filter(model == tag),
                   BLM_N_LAYERS[[tag]], BLM_LABELS[[tag]])))
) %>% write_csv(file.path(RESULTS_DIR, "ddata_freq_indep.csv"))

bind_rows(
  freq_diff_data(olmo_gam_freq_sub, olmo_sub, OLMO_N_LAYERS, "OLMo-3 7B"),
  bind_rows(lapply(BLM_TAGS, function(tag)
    freq_diff_data(blm_gam_freq_sub[[tag]], blm_sub %>% filter(model == tag),
                   BLM_N_LAYERS[[tag]], BLM_LABELS[[tag]])))
) %>% write_csv(file.path(RESULTS_DIR, "ddata_freq_sub.csv"))

bind_rows(lapply(WH_COMPONENTS, function(comp)
  freq_diff_data(wh_gam_freq[[comp]], whisper_all %>% filter(component == comp),
                 WH_N_LAYERS, WH_LABELS[[comp]]))) %>%
  write_csv(file.path(RESULTS_DIR, "ddata_freq_wh.csv"))

bind_rows(
  predic_diff_data(olmo_gam_predic_indep, olmo_indep, OLMO_N_LAYERS, "OLMo-3 7B"),
  bind_rows(lapply(BLM_TAGS, function(tag)
    predic_diff_data(blm_gam_predic_indep[[tag]], blm_indep %>% filter(model == tag),
                     BLM_N_LAYERS[[tag]], BLM_LABELS[[tag]])))
) %>% write_csv(file.path(RESULTS_DIR, "ddata_predic_indep.csv"))

bind_rows(
  predic_diff_data(olmo_gam_predic_sub, olmo_sub, OLMO_N_LAYERS, "OLMo-3 7B"),
  bind_rows(lapply(BLM_TAGS, function(tag)
    predic_diff_data(blm_gam_predic_sub[[tag]], blm_sub %>% filter(model == tag),
                     BLM_N_LAYERS[[tag]], BLM_LABELS[[tag]])))
) %>% write_csv(file.path(RESULTS_DIR, "ddata_predic_sub.csv"))

bind_rows(lapply(WH_COMPONENTS, function(comp)
  predic_diff_data(wh_gam_predic[[comp]], whisper_all %>% filter(component == comp),
                   WH_N_LAYERS, WH_LABELS[[comp]]))) %>%
  write_csv(file.path(RESULTS_DIR, "ddata_predic_wh.csv"))

} # end if (lfs_data_available) — sections 1-5

# ── 5b. GAM full prediction grids (for heatmaps) ──────────────────────────────
# STANDALONE: requires only model_cache/ RDS files. No raw data or items CSVs needed.
# Predictor ranges are extracted directly from model$model (the training frame stored
# inside each fitted GAM object). Run this block on its own if needed.
message("Computing GAM prediction grids for heatmaps...")

freq_grid_data = function(model, n_layers, panel, n_pred = 60) {
  log_freq_vec = model$model$log_freq
  dummy_verb   = model$model$verb_up[1]
  layer_seq    = 0:(n_layers - 1)
  pred_seq     = seq(quantile(log_freq_vec, 0.01, na.rm = TRUE),
                     quantile(log_freq_vec, 0.99, na.rm = TRUE), length.out = n_pred)
  newdata = expand.grid(layer = layer_seq, log_freq = pred_seq) %>%
    mutate(verb_up = dummy_verb)
  fit = as.numeric(predict(model, newdata = newdata, exclude = "s(verb_up)"))
  newdata %>% mutate(fit = fit, panel = panel) %>% select(layer, log_freq, fit, panel)
}

predic_grid_data = function(model, n_layers, panel, n_pred = 60) {
  log_predic_vec = model$model$log_predic
  dummy_verb     = model$model$verb_up[1]
  layer_seq      = 0:(n_layers - 1)
  pred_seq       = seq(quantile(log_predic_vec, 0.01, na.rm = TRUE),
                       quantile(log_predic_vec, 0.99, na.rm = TRUE), length.out = n_pred)
  newdata = expand.grid(layer = layer_seq, log_predic = pred_seq) %>%
    mutate(verb_up = dummy_verb)
  fit = as.numeric(predict(model, newdata = newdata, exclude = "s(verb_up)"))
  newdata %>% mutate(fit = fit, panel = panel) %>% select(layer, log_predic, fit, panel)
}

# Load GAM models (same as section 5; safe to re-load if already in memory)
gam_fi  = olmo_rds("model_freq_layer_up_independently")
gam_fs  = olmo_rds("model_freq_layer_up_subword")
gam_pi  = olmo_rds("model_predic_layer_up_independently")
gam_ps  = olmo_rds("model_predic_layer_up_subword")
gam_blm_fi = setNames(lapply(BLM_TAGS, function(t) blm_rds(paste0("model_freq_layer_indep_",   blm_slug(t)))), BLM_TAGS)
gam_blm_fs = setNames(lapply(BLM_TAGS, function(t) blm_rds(paste0("model_freq_layer_sub_",     blm_slug(t)))), BLM_TAGS)
gam_blm_pi = setNames(lapply(BLM_TAGS, function(t) blm_rds(paste0("model_predic_layer_indep_", blm_slug(t)))), BLM_TAGS)
gam_blm_ps = setNames(lapply(BLM_TAGS, function(t) blm_rds(paste0("model_predic_layer_sub_",   blm_slug(t)))), BLM_TAGS)
gam_wh_f   = setNames(lapply(WH_COMPONENTS, function(c) wh_rds(paste0("model_freq_layer_",   c))), WH_COMPONENTS)
gam_wh_p   = setNames(lapply(WH_COMPONENTS, function(c) wh_rds(paste0("model_predic_layer_", c))), WH_COMPONENTS)

bind_rows(
  freq_grid_data(gam_fi, OLMO_N_LAYERS, "OLMo-3 7B"),
  bind_rows(lapply(BLM_TAGS, function(t)
    freq_grid_data(gam_blm_fi[[t]], BLM_N_LAYERS[[t]], BLM_LABELS[[t]])))
) %>% write_csv(file.path(RESULTS_DIR, "grid_freq_indep.csv"))

bind_rows(
  freq_grid_data(gam_fs, OLMO_N_LAYERS, "OLMo-3 7B"),
  bind_rows(lapply(BLM_TAGS, function(t)
    freq_grid_data(gam_blm_fs[[t]], BLM_N_LAYERS[[t]], BLM_LABELS[[t]])))
) %>% write_csv(file.path(RESULTS_DIR, "grid_freq_sub.csv"))

bind_rows(lapply(WH_COMPONENTS, function(c)
  freq_grid_data(gam_wh_f[[c]], WH_N_LAYERS, WH_LABELS[[c]]))) %>%
  write_csv(file.path(RESULTS_DIR, "grid_freq_wh.csv"))

bind_rows(
  predic_grid_data(gam_pi, OLMO_N_LAYERS, "OLMo-3 7B"),
  bind_rows(lapply(BLM_TAGS, function(t)
    predic_grid_data(gam_blm_pi[[t]], BLM_N_LAYERS[[t]], BLM_LABELS[[t]])))
) %>% write_csv(file.path(RESULTS_DIR, "grid_predic_indep.csv"))

bind_rows(
  predic_grid_data(gam_ps, OLMO_N_LAYERS, "OLMo-3 7B"),
  bind_rows(lapply(BLM_TAGS, function(t)
    predic_grid_data(gam_blm_ps[[t]], BLM_N_LAYERS[[t]], BLM_LABELS[[t]])))
) %>% write_csv(file.path(RESULTS_DIR, "grid_predic_sub.csv"))

bind_rows(lapply(WH_COMPONENTS, function(c)
  predic_grid_data(gam_wh_p[[c]], WH_N_LAYERS, WH_LABELS[[c]]))) %>%
  write_csv(file.path(RESULTS_DIR, "grid_predic_wh.csv"))

# ── 5c. GAM smooth summary tables ─────────────────────────────────────────────
# Extracts EDF, F-statistic, and p-value for the te() smooth from each GAM.
# Uses the same model objects loaded in section 5b — standalone, no raw data needed.
message("Extracting GAM smooth summaries...")

gam_smooth_row = function(model, model_name, predictor) {
  st    = summary(model)$s.table
  te_row = st[grep("^te\\(", rownames(st)), , drop = FALSE]
  if (nrow(te_row) == 0) te_row = st[1, , drop = FALSE]
  tibble(
    Model     = model_name,
    Predictor = predictor,
    edf       = te_row[1, "edf"],
    f_stat    = te_row[1, "F"],
    p_value   = te_row[1, "p-value"]
  )
}

bind_rows(
  gam_smooth_row(gam_fi, "OLMo-3 7B", "Frequency"),
  bind_rows(lapply(BLM_TAGS, function(t)
    gam_smooth_row(gam_blm_fi[[t]], BLM_LABELS[[t]], "Frequency"))),
  gam_smooth_row(gam_pi, "OLMo-3 7B", "Predictability"),
  bind_rows(lapply(BLM_TAGS, function(t)
    gam_smooth_row(gam_blm_pi[[t]], BLM_LABELS[[t]], "Predictability")))
) %>% write_csv(file.path(RESULTS_DIR, "gam_smooth_indep.csv"))

bind_rows(
  gam_smooth_row(gam_fs, "OLMo-3 7B", "Frequency"),
  bind_rows(lapply(BLM_TAGS, function(t)
    gam_smooth_row(gam_blm_fs[[t]], BLM_LABELS[[t]], "Frequency"))),
  gam_smooth_row(gam_ps, "OLMo-3 7B", "Predictability"),
  bind_rows(lapply(BLM_TAGS, function(t)
    gam_smooth_row(gam_blm_ps[[t]], BLM_LABELS[[t]], "Predictability")))
) %>% write_csv(file.path(RESULTS_DIR, "gam_smooth_sub.csv"))

bind_rows(
  gam_smooth_row(gam_wh_f[["encoder"]], "Encoder", "Frequency"),
  gam_smooth_row(gam_wh_f[["decoder"]], "Decoder", "Frequency"),
  gam_smooth_row(gam_wh_p[["encoder"]], "Encoder", "Predictability"),
  gam_smooth_row(gam_wh_p[["decoder"]], "Decoder", "Predictability")
) %>% write_csv(file.path(RESULTS_DIR, "gam_smooth_wh.csv"))

# ── 5d. GAM diff data (standalone) ────────────────────────────────────────────
# Produces ddata_*.csv using model$model instead of raw data frames.
# Safe to run without LFS data — requires only the GAM RDS files from section 5b.
message("Computing GAM diff data (standalone)...")

freq_diff_sa = function(model, n_layers, panel) {
  mdat        = model$model
  log_freq_lo = quantile(mdat$log_freq, 0.1, na.rm = TRUE)
  log_freq_hi = quantile(mdat$log_freq, 0.9, na.rm = TRUE)
  newdata = expand.grid(layer = 0:(n_layers - 1), freq_level = c("low", "high")) %>%
    mutate(log_freq = ifelse(freq_level == "low", log_freq_lo, log_freq_hi),
           verb_up  = mdat$verb_up[1])
  preds = predict(model, newdata = newdata, se.fit = TRUE, exclude = "s(verb_up)")
  newdata %>%
    mutate(fit = preds$fit, se = preds$se.fit) %>%
    select(layer, freq_level, fit, se) %>%
    pivot_wider(names_from = freq_level, values_from = c(fit, se)) %>%
    mutate(diff    = fit_high - fit_low,
           se_diff = sqrt(se_high^2 + se_low^2),
           lo95    = diff - 1.96 * se_diff, hi95 = diff + 1.96 * se_diff,
           lo80    = diff - 1.28 * se_diff, hi80 = diff + 1.28 * se_diff,
           panel   = panel)
}

predic_diff_sa = function(model, n_layers, panel) {
  mdat      = model$model
  predic_lo = quantile(mdat$log_predic, 0.1, na.rm = TRUE)
  predic_hi = quantile(mdat$log_predic, 0.9, na.rm = TRUE)
  newdata = expand.grid(layer = 0:(n_layers - 1), predic_level = c("low", "high")) %>%
    mutate(log_predic = ifelse(predic_level == "low", predic_lo, predic_hi),
           verb_up    = mdat$verb_up[1])
  preds = predict(model, newdata = newdata, se.fit = TRUE, exclude = "s(verb_up)")
  newdata %>%
    mutate(fit = preds$fit, se = preds$se.fit) %>%
    select(layer, predic_level, fit, se) %>%
    pivot_wider(names_from = predic_level, values_from = c(fit, se)) %>%
    mutate(diff    = fit_high - fit_low,
           se_diff = sqrt(se_high^2 + se_low^2),
           lo95    = diff - 1.96 * se_diff, hi95 = diff + 1.96 * se_diff,
           lo80    = diff - 1.28 * se_diff, hi80 = diff + 1.28 * se_diff,
           panel   = panel)
}

bind_rows(
  freq_diff_sa(gam_fi, OLMO_N_LAYERS, "OLMo-3 7B"),
  bind_rows(lapply(BLM_TAGS, function(t)
    freq_diff_sa(gam_blm_fi[[t]], BLM_N_LAYERS[[t]], BLM_LABELS[[t]])))
) %>% write_csv(file.path(RESULTS_DIR, "ddata_freq_indep.csv"))

bind_rows(
  freq_diff_sa(gam_fs, OLMO_N_LAYERS, "OLMo-3 7B"),
  bind_rows(lapply(BLM_TAGS, function(t)
    freq_diff_sa(gam_blm_fs[[t]], BLM_N_LAYERS[[t]], BLM_LABELS[[t]])))
) %>% write_csv(file.path(RESULTS_DIR, "ddata_freq_sub.csv"))

bind_rows(lapply(WH_COMPONENTS, function(c)
  freq_diff_sa(gam_wh_f[[c]], WH_N_LAYERS, WH_LABELS[[c]]))) %>%
  write_csv(file.path(RESULTS_DIR, "ddata_freq_wh.csv"))

bind_rows(
  predic_diff_sa(gam_pi, OLMO_N_LAYERS, "OLMo-3 7B"),
  bind_rows(lapply(BLM_TAGS, function(t)
    predic_diff_sa(gam_blm_pi[[t]], BLM_N_LAYERS[[t]], BLM_LABELS[[t]])))
) %>% write_csv(file.path(RESULTS_DIR, "ddata_predic_indep.csv"))

bind_rows(
  predic_diff_sa(gam_ps, OLMO_N_LAYERS, "OLMo-3 7B"),
  bind_rows(lapply(BLM_TAGS, function(t)
    predic_diff_sa(gam_blm_ps[[t]], BLM_N_LAYERS[[t]], BLM_LABELS[[t]])))
) %>% write_csv(file.path(RESULTS_DIR, "ddata_predic_sub.csv"))

bind_rows(lapply(WH_COMPONENTS, function(c)
  predic_diff_sa(gam_wh_p[[c]], WH_N_LAYERS, WH_LABELS[[c]]))) %>%
  write_csv(file.path(RESULTS_DIR, "ddata_predic_wh.csv"))

# ── 6. brms table data ─────────────────────────────────────────────────────────
message("Loading brms models and extracting posterior summaries...")

brms_df = function(model, model_name) {
  fe     = as.data.frame(fixef(model))
  params = rownames(fe)
  rownames(fe) = NULL
  post = as_draws_df(model)
  pgt0 = sapply(params, function(p) {
    col = paste0("b_", p)
    if (col %in% names(post)) round(mean(post[[col]] > 0) * 100, 1) else NA_real_
  })
  fe %>%
    mutate(Model = model_name, Parameter = params, `% > 0` = pgt0) %>%
    select(Model, Parameter, Estimate, Est.Error, Q2.5, Q97.5, `% > 0`)
}

save_brms_table = function(model_list, filename) {
  bind_rows(mapply(brms_df, model_list, names(model_list), SIMPLIFY = FALSE)) %>%
    write_csv(file.path(RESULTS_DIR, filename))
}

# Load joint linear models
olmo_brms_joint_indep_first = readRDS(file.path(OLMO_CACHE_DIR, "model_freq_predic_up_independently_first.rds"))
olmo_brms_joint_indep_final = readRDS(file.path(OLMO_CACHE_DIR, "model_freq_predic_up_independently.rds"))
olmo_brms_joint_sub_first   = readRDS(file.path(OLMO_CACHE_DIR, "model_freq_predic_up_subword_first.rds"))
olmo_brms_joint_sub_final   = readRDS(file.path(OLMO_CACHE_DIR, "model_freq_predic_up_subword.rds"))

blm_brms_joint_indep_first = setNames(lapply(BLM_TAGS, function(tag)
  readRDS(file.path(BLM_CACHE_DIR, paste0("model_joint_indep_first_", blm_slug(tag), ".rds")))), BLM_TAGS)
blm_brms_joint_indep_final = setNames(lapply(BLM_TAGS, function(tag)
  readRDS(file.path(BLM_CACHE_DIR, paste0("model_joint_indep_final_", blm_slug(tag), ".rds")))), BLM_TAGS)
blm_brms_joint_sub_first   = setNames(lapply(BLM_TAGS, function(tag)
  readRDS(file.path(BLM_CACHE_DIR, paste0("model_joint_sub_first_", blm_slug(tag), ".rds")))), BLM_TAGS)
blm_brms_joint_sub_final   = setNames(lapply(BLM_TAGS, function(tag)
  readRDS(file.path(BLM_CACHE_DIR, paste0("model_joint_sub_final_", blm_slug(tag), ".rds")))), BLM_TAGS)

wh_brms_joint_first = setNames(lapply(WH_COMPONENTS, function(comp)
  readRDS(file.path(WH_CACHE_DIR, paste0("model_joint_first_", comp, ".rds")))), WH_COMPONENTS)
wh_brms_joint_final = setNames(lapply(WH_COMPONENTS, function(comp)
  readRDS(file.path(WH_CACHE_DIR, paste0("model_joint_final_", comp, ".rds")))), WH_COMPONENTS)

# Load joint poly models
olmo_brms_poly_joint_indep_first = readRDS(file.path(OLMO_CACHE_DIR, "model_poly_joint_indep_first.rds"))
olmo_brms_poly_joint_indep_final = readRDS(file.path(OLMO_CACHE_DIR, "model_poly_joint_indep_final.rds"))
olmo_brms_poly_joint_sub_first   = readRDS(file.path(OLMO_CACHE_DIR, "model_poly_joint_sub_first.rds"))
olmo_brms_poly_joint_sub_final   = readRDS(file.path(OLMO_CACHE_DIR, "model_poly_joint_sub_final.rds"))

blm_brms_poly_joint_indep_first = setNames(lapply(BLM_TAGS, function(tag)
  readRDS(file.path(BLM_CACHE_DIR, paste0("model_poly_joint_indep_first_", blm_slug(tag), ".rds")))), BLM_TAGS)
blm_brms_poly_joint_indep_final = setNames(lapply(BLM_TAGS, function(tag)
  readRDS(file.path(BLM_CACHE_DIR, paste0("model_poly_joint_indep_final_", blm_slug(tag), ".rds")))), BLM_TAGS)
blm_brms_poly_joint_sub_first   = setNames(lapply(BLM_TAGS, function(tag)
  readRDS(file.path(BLM_CACHE_DIR, paste0("model_poly_joint_sub_first_", blm_slug(tag), ".rds")))), BLM_TAGS)
blm_brms_poly_joint_sub_final   = setNames(lapply(BLM_TAGS, function(tag)
  readRDS(file.path(BLM_CACHE_DIR, paste0("model_poly_joint_sub_final_", blm_slug(tag), ".rds")))), BLM_TAGS)

wh_brms_poly_joint_first = setNames(lapply(WH_COMPONENTS, function(comp)
  readRDS(file.path(WH_CACHE_DIR, paste0("model_poly_joint_first_", comp, ".rds")))), WH_COMPONENTS)
wh_brms_poly_joint_final = setNames(lapply(WH_COMPONENTS, function(comp)
  readRDS(file.path(WH_CACHE_DIR, paste0("model_poly_joint_final_", comp, ".rds")))), WH_COMPONENTS)

# Save brms table CSVs
save_brms_table(list(
  "OLMo-3 7B"   = olmo_brms_joint_indep_first,
  "BabyLM 125M" = blm_brms_joint_indep_first[["opt-125m"]],
  "BabyLM 350M" = blm_brms_joint_indep_first[["opt-350m"]],
  "BabyLM 1.3B" = blm_brms_joint_indep_first[["opt-1.3b"]]
), "tbl_exp1_joint_first.csv")

save_brms_table(list(
  "OLMo-3 7B"   = olmo_brms_joint_indep_final,
  "BabyLM 125M" = blm_brms_joint_indep_final[["opt-125m"]],
  "BabyLM 350M" = blm_brms_joint_indep_final[["opt-350m"]],
  "BabyLM 1.3B" = blm_brms_joint_indep_final[["opt-1.3b"]]
), "tbl_exp1_joint_final.csv")

save_brms_table(list(
  "OLMo-3 7B"   = olmo_brms_joint_sub_first,
  "BabyLM 125M" = blm_brms_joint_sub_first[["opt-125m"]],
  "BabyLM 350M" = blm_brms_joint_sub_first[["opt-350m"]],
  "BabyLM 1.3B" = blm_brms_joint_sub_first[["opt-1.3b"]]
), "tbl_exp2_joint_first.csv")

save_brms_table(list(
  "OLMo-3 7B"   = olmo_brms_joint_sub_final,
  "BabyLM 125M" = blm_brms_joint_sub_final[["opt-125m"]],
  "BabyLM 350M" = blm_brms_joint_sub_final[["opt-350m"]],
  "BabyLM 1.3B" = blm_brms_joint_sub_final[["opt-1.3b"]]
), "tbl_exp2_joint_final.csv")

save_brms_table(list(
  "OLMo-3 7B"   = olmo_brms_poly_joint_indep_first,
  "BabyLM 125M" = blm_brms_poly_joint_indep_first[["opt-125m"]],
  "BabyLM 350M" = blm_brms_poly_joint_indep_first[["opt-350m"]],
  "BabyLM 1.3B" = blm_brms_poly_joint_indep_first[["opt-1.3b"]]
), "tbl_exp1_poly_joint_first.csv")

save_brms_table(list(
  "OLMo-3 7B"   = olmo_brms_poly_joint_indep_final,
  "BabyLM 125M" = blm_brms_poly_joint_indep_final[["opt-125m"]],
  "BabyLM 350M" = blm_brms_poly_joint_indep_final[["opt-350m"]],
  "BabyLM 1.3B" = blm_brms_poly_joint_indep_final[["opt-1.3b"]]
), "tbl_exp1_poly_joint_final.csv")

save_brms_table(list(
  "OLMo-3 7B"   = olmo_brms_poly_joint_sub_first,
  "BabyLM 125M" = blm_brms_poly_joint_sub_first[["opt-125m"]],
  "BabyLM 350M" = blm_brms_poly_joint_sub_first[["opt-350m"]],
  "BabyLM 1.3B" = blm_brms_poly_joint_sub_first[["opt-1.3b"]]
), "tbl_exp2_poly_joint_first.csv")

save_brms_table(list(
  "OLMo-3 7B"   = olmo_brms_poly_joint_sub_final,
  "BabyLM 125M" = blm_brms_poly_joint_sub_final[["opt-125m"]],
  "BabyLM 350M" = blm_brms_poly_joint_sub_final[["opt-350m"]],
  "BabyLM 1.3B" = blm_brms_poly_joint_sub_final[["opt-1.3b"]]
), "tbl_exp2_poly_joint_final.csv")

save_brms_table(list(
  "Encoder" = wh_brms_joint_first[["encoder"]],
  "Decoder" = wh_brms_joint_first[["decoder"]]
), "tbl_exp3_joint_first.csv")

save_brms_table(list(
  "Encoder" = wh_brms_joint_final[["encoder"]],
  "Decoder" = wh_brms_joint_final[["decoder"]]
), "tbl_exp3_joint_final.csv")

save_brms_table(list(
  "Encoder" = wh_brms_poly_joint_first[["encoder"]],
  "Decoder" = wh_brms_poly_joint_first[["decoder"]]
), "tbl_exp3_poly_joint_first.csv")

save_brms_table(list(
  "Encoder" = wh_brms_poly_joint_final[["encoder"]],
  "Decoder" = wh_brms_poly_joint_final[["decoder"]]
), "tbl_exp3_poly_joint_final.csv")

# ── 7. Joint surface plot data ─────────────────────────────────────────────────
# Patches existing surface CSVs with 95% CI bands.
# Uses the grid and scaling params already in each CSV plus model$data for verb_up.
# No raw LFS data needed — works from brms RDS files alone.
message("Adding 95% CI to surface CSVs (slow — calls fitted() on brms models)...")

patch_surface_ci = function(csv_path, model_map) {
  df      = read_csv(csv_path, show_col_types = FALSE)
  df$q025 = NA_real_
  df$q975 = NA_real_
  for (lbl in unique(df$label)) {
    model = model_map[[lbl]]
    if (is.null(model)) { warning("No model for label: ", lbl); next }
    idx  = which(df$label == lbl)
    grid = df[idx, c("c_log_freq", "c_log_predic"), drop = FALSE]
    if ("c_log_ftp" %in% rownames(fixef(model)))
      grid = rename(grid, c_log_ftp = c_log_predic)
    grid$verb_up = model$data$verb_up[1]
    preds        = fitted(model, newdata = grid, re_formula = NA, summary = TRUE)
    df$fit[idx]  = preds[, "Estimate"]
    df$q025[idx] = preds[, "Q2.5"]
    df$q975[idx] = preds[, "Q97.5"]
  }
  write_csv(df, csv_path)
  invisible(df)
}

message("  Exp1 linear joint (OLMo)...")
patch_surface_ci(
  file.path(RESULTS_DIR, "surface_exp1_joint_olmo.csv"),
  list(
    "OLMo — first layer" = olmo_brms_joint_indep_first,
    "OLMo — final layer" = olmo_brms_joint_indep_final
  )
)

message("  Exp1 linear joint (BabyLM)...")
patch_surface_ci(
  file.path(RESULTS_DIR, "surface_exp1_joint_blm.csv"),
  c(
    setNames(lapply(BLM_TAGS, function(tag) blm_brms_joint_indep_first[[tag]]),
             paste0(BLM_LABELS[BLM_TAGS], " — first layer")),
    setNames(lapply(BLM_TAGS, function(tag) blm_brms_joint_indep_final[[tag]]),
             paste0(BLM_LABELS[BLM_TAGS], " — final layer"))
  )
)

message("  Exp2 linear joint (OLMo)...")
patch_surface_ci(
  file.path(RESULTS_DIR, "surface_exp2_joint_olmo.csv"),
  list(
    "OLMo — first layer" = olmo_brms_joint_sub_first,
    "OLMo — final layer" = olmo_brms_joint_sub_final
  )
)

message("  Exp2 linear joint (BabyLM)...")
patch_surface_ci(
  file.path(RESULTS_DIR, "surface_exp2_joint_blm.csv"),
  c(
    setNames(lapply(BLM_TAGS, function(tag) blm_brms_joint_sub_first[[tag]]),
             paste0(BLM_LABELS[BLM_TAGS], " — first layer")),
    setNames(lapply(BLM_TAGS, function(tag) blm_brms_joint_sub_final[[tag]]),
             paste0(BLM_LABELS[BLM_TAGS], " — final layer"))
  )
)

message("  Exp3 linear joint (Whisper)...")
patch_surface_ci(
  file.path(RESULTS_DIR, "surface_exp3_joint_wh.csv"),
  c(
    setNames(lapply(WH_COMPONENTS, function(comp) wh_brms_joint_first[[comp]]),
             paste0(WH_LABELS[WH_COMPONENTS], " — first layer")),
    setNames(lapply(WH_COMPONENTS, function(comp) wh_brms_joint_final[[comp]]),
             paste0(WH_LABELS[WH_COMPONENTS], " — final layer"))
  )
)

message("  Exp1 poly joint (OLMo)...")
patch_surface_ci(
  file.path(RESULTS_DIR, "surface_exp1_poly_olmo.csv"),
  list(
    "OLMo — first layer" = olmo_brms_poly_joint_indep_first,
    "OLMo — final layer" = olmo_brms_poly_joint_indep_final
  )
)

message("  Exp1 poly joint (BabyLM)...")
patch_surface_ci(
  file.path(RESULTS_DIR, "surface_exp1_poly_blm.csv"),
  c(
    setNames(lapply(BLM_TAGS, function(tag) blm_brms_poly_joint_indep_first[[tag]]),
             paste0(BLM_LABELS[BLM_TAGS], " — first layer")),
    setNames(lapply(BLM_TAGS, function(tag) blm_brms_poly_joint_indep_final[[tag]]),
             paste0(BLM_LABELS[BLM_TAGS], " — final layer"))
  )
)

message("  Exp2 poly joint (OLMo)...")
patch_surface_ci(
  file.path(RESULTS_DIR, "surface_exp2_poly_olmo.csv"),
  list(
    "OLMo — first layer" = olmo_brms_poly_joint_sub_first,
    "OLMo — final layer" = olmo_brms_poly_joint_sub_final
  )
)

message("  Exp2 poly joint (BabyLM)...")
patch_surface_ci(
  file.path(RESULTS_DIR, "surface_exp2_poly_blm.csv"),
  c(
    setNames(lapply(BLM_TAGS, function(tag) blm_brms_poly_joint_sub_first[[tag]]),
             paste0(BLM_LABELS[BLM_TAGS], " — first layer")),
    setNames(lapply(BLM_TAGS, function(tag) blm_brms_poly_joint_sub_final[[tag]]),
             paste0(BLM_LABELS[BLM_TAGS], " — final layer"))
  )
)

message("  Exp3 poly joint (Whisper)...")
patch_surface_ci(
  file.path(RESULTS_DIR, "surface_exp3_poly_wh.csv"),
  c(
    setNames(lapply(WH_COMPONENTS, function(comp) wh_brms_poly_joint_first[[comp]]),
             paste0(WH_LABELS[WH_COMPONENTS], " — first layer")),
    setNames(lapply(WH_COMPONENTS, function(comp) wh_brms_poly_joint_final[[comp]]),
             paste0(WH_LABELS[WH_COMPONENTS], " — final layer"))
  )
)

message("Done! All results saved to ", RESULTS_DIR, "/")
