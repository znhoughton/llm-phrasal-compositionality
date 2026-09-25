# compute_suppression_table.R
#
# Recomputes writeup.tex's Statistical Suppression appendix table
# (tbl-exp3-sub-suppression, Section app-whisper-subword-suppression) on the
# current (post V+up-fix) data, for all 10 conditions in the paper.
#
# This table's original computation lived only in an ad-hoc notebook chunk
# (never saved as a standalone script -- same "orphaned code" issue already
# fixed for the GAM-fitting scripts). Reconstructed here from the writeup's
# own equations:
#
#   r_{y,weak}, r_{y,strong}: zero-order Pearson correlations of the
#     classifier logit with log_freq and log_predic (whichever has the
#     smaller/larger |r| is "weak"/"strong"), at the final layer, on the
#     same row set the joint model is fit on (predic-valid rows only).
#   r_{1,2}: correlation between log_freq and log_predic themselves.
#   Threshold = r_{y,weak} / r_{y,strong}  (Equation eq-suppression-threshold)
#   Result: "sign flip" if r_{1,2} > Threshold, else "stable"
#
# Run from Analyses/ directory.

suppressPackageStartupMessages(library(tidyverse))

OLMO_FINAL_LAYER <- 31L
BLM_FINAL_LAYER_MAP <- list("opt-125m" = 11L, "opt-350m" = 23L, "opt-1.3b" = 23L)
WH_FINAL_LAYER <- 11L

suppression_row <- function(logit, log_freq, log_predic, label) {
  r_freq   <- cor(logit, log_freq)
  r_predic <- cor(logit, log_predic)
  r_12     <- cor(log_freq, log_predic)
  if (abs(r_freq) < abs(r_predic)) {
    weak <- r_freq; strong <- r_predic
  } else {
    weak <- r_predic; strong <- r_freq
  }
  threshold <- weak / strong
  result <- if (r_12 > threshold) "sign flip" else "stable"
  tibble(Condition = label, r_freq = r_freq, r_predic = r_predic,
         r_12 = r_12, threshold = threshold, result = result)
}

rows <- list()

# ---- OLMo ----
message("OLMo...")
olmo_indep <- read_csv("../Data/olmo-3-7b/Data_up/all_layers_results.csv", show_col_types = FALSE) %>%
  mutate(log_freq = log(frequency), log_predic = log(predic / (1 - predic))) %>%
  filter(layer == OLMO_FINAL_LAYER, !is.na(predic), is.finite(log_predic))
rows$olmo_indep <- suppression_row(olmo_indep$logit, olmo_indep$log_freq, olmo_indep$log_predic, "OLMo indep")

olmo_sub <- read_csv("../Data/olmo-3-7b/Data_upsubword/all_layers_results.csv", show_col_types = FALSE) %>%
  mutate(log_freq = log(frequency), log_predic = log(predic / (1 - predic))) %>%
  filter(layer == OLMO_FINAL_LAYER, !is.na(predic), is.finite(log_predic))
rows$olmo_sub <- suppression_row(olmo_sub$logit, olmo_sub$log_freq, olmo_sub$log_predic, "OLMo subword")

# ---- BabyLM ----
message("BabyLM...")
for (tag in c("opt-125m", "opt-350m", "opt-1.3b")) {
  fl <- BLM_FINAL_LAYER_MAP[[tag]]
  label <- toupper(sub("opt-", "", tag))
  if (grepl("b$", label, ignore.case = FALSE) && !grepl("M$", label)) label <- gsub("B$", "B", label)
  label <- switch(tag, "opt-125m" = "BabyLM-125M", "opt-350m" = "BabyLM-350M", "opt-1.3b" = "BabyLM-1.3B")

  ui <- read_csv(paste0("../Data/babylm/", tag, "/Data_up/all_layers_results.csv"), show_col_types = FALSE) %>%
    rename_with(~ sub("^ftp$", "predic", .)) %>%
    mutate(log_freq = log(frequency), log_predic = log(predic / (1 - predic))) %>%
    filter(layer == fl, !is.na(predic), is.finite(log_predic))
  rows[[paste0(tag, "_indep")]] <- suppression_row(ui$logit, ui$log_freq, ui$log_predic, paste0(label, " indep"))

  us <- read_csv(paste0("../Data/babylm/", tag, "/Data_upsubword/all_layers_results.csv"), show_col_types = FALSE) %>%
    rename_with(~ sub("^ftp$", "predic", .)) %>%
    mutate(log_freq = log(frequency), log_predic = log(predic / (1 - predic))) %>%
    filter(layer == fl, !is.na(predic), is.finite(log_predic))
  rows[[paste0(tag, "_sub")]] <- suppression_row(us$logit, us$log_freq, us$log_predic, paste0(label, " subword"))
}

# ---- Whisper ----
message("Whisper...")
ftp_lookup <- read_csv("../Data/ftp_lookup.csv", show_col_types = FALSE) %>% rename(predic = ftp)

load_wh_final <- function(path) {
  read_csv(path, show_col_types = FALSE) %>%
    select(-any_of("predic")) %>%
    mutate(log_freq = log(frequency), verb_up_chr = as.character(verb_up)) %>%
    left_join(ftp_lookup, by = c("verb_up_chr" = "verb_up")) %>%
    filter(!is.na(predic)) %>%
    mutate(log_predic = log(predic / (1 - predic))) %>%
    filter(is.finite(log_predic), layer == WH_FINAL_LAYER)
}

wh_enc     <- load_wh_final("../Data/whisper/encoder/all_layers_results.csv")
wh_dec     <- load_wh_final("../Data/whisper/decoder/all_layers_results.csv")
wh_enc_sub <- load_wh_final("../Data/whisper_subword/encoder/all_layers_results.csv")
wh_dec_sub <- load_wh_final("../Data/whisper_subword/decoder/all_layers_results.csv")

rows$wh_enc     <- suppression_row(wh_enc$logit,     wh_enc$log_freq,     wh_enc$log_predic,     "Whisper encoder indep")
rows$wh_enc_sub <- suppression_row(wh_enc_sub$logit, wh_enc_sub$log_freq, wh_enc_sub$log_predic, "Whisper encoder subword")
rows$wh_dec     <- suppression_row(wh_dec$logit,     wh_dec$log_freq,     wh_dec$log_predic,     "Whisper decoder indep")
rows$wh_dec_sub <- suppression_row(wh_dec_sub$logit, wh_dec_sub$log_freq, wh_dec_sub$log_predic, "Whisper decoder subword")

tbl <- bind_rows(rows)
cat("\n\n================ RECOMPUTED SUPPRESSION TABLE (fresh, post V+up-fix data) ================\n\n")
print(tbl %>% mutate(across(where(is.numeric), ~round(., 3))), n = Inf, width = Inf)

cat("\n\n================ ORIGINAL (pre-fix) TABLE, FOR REFERENCE ================\n")
cat("OLMo indep              -0.327 -0.524 0.337 0.624 stable\n")
cat("OLMo subword             -0.284 -0.426 0.337 0.666 stable\n")
cat("BabyLM-125M indep       -0.274 -0.155 0.253 0.567 stable\n")
cat("BabyLM-125M subword     -0.205 -0.057 0.253 0.277 stable\n")
cat("BabyLM-350M indep       -0.284 -0.098 0.253 0.345 stable\n")
cat("BabyLM-1.3B indep       -0.306 -0.119 0.253 0.388 stable\n")
cat("Whisper encoder indep   -0.044 -0.043 0.358 0.984 stable\n")
cat("Whisper encoder subword -0.045 -0.068 0.358 0.655 stable\n")
cat("Whisper decoder indep   -0.110 -0.202 0.358 0.545 stable\n")
cat("Whisper decoder subword -0.032 -0.137 0.358 0.235 sign flip\n")

message("\nDone.")
