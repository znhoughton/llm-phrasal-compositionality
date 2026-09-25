# extract_whisper_final_numbers.R
# Pulls every number needed for the Experiment 3 writeup update (main text +
# appendix): joint model coefficients, GAM edf/F/p, alone-model coefficients,
# for both indep and subword conditions, formatted to match the existing
# tbl-exp1/exp2-joint-final table style.
suppressPackageStartupMessages(library(brms))

WH <- "../model_cache/whisper"

joint_row <- function(path, label) {
  m <- readRDS(path)
  fx <- fixef(m)
  post <- as_draws_df(m)
  params <- rownames(fx)
  for (p in params) {
    col <- paste0("b_", p)
    pgt0 <- round(mean(post[[col]] > 0) * 100, 1)
    cat(sprintf("%-10s %-14s Est=%6.2f Err=%5.2f CI=[%6.2f, %6.2f] %%>0=%5.1f\n",
                label, p, fx[p,"Estimate"], fx[p,"Est.Error"], fx[p,"Q2.5"], fx[p,"Q97.5"], pgt0))
  }
}

gam_row <- function(path, label) {
  m <- readRDS(path)
  s <- summary(m)$s.table
  te <- s[grep("^te\\(", rownames(s)), , drop = FALSE][1, ]
  cat(sprintf("%-10s GAM edf=%.2f F=%.2f p=%.4g\n", label, te["edf"], te["F"], te["p-value"]))
}

alone_row <- function(path, label, paramname) {
  m <- readRDS(path)
  fx <- fixef(m)
  post <- as_draws_df(m)
  col <- paste0("b_", paramname)
  pgt0 <- round(mean(post[[col]] > 0) * 100, 1)
  cat(sprintf("%-10s %-14s Est=%6.3f CI=[%6.3f, %6.3f] %%>0=%5.1f\n",
              label, paramname, fx[paramname,"Estimate"], fx[paramname,"Q2.5"], fx[paramname,"Q97.5"], pgt0))
}

cat("========== JOINT MODELS (indep) ==========\n")
joint_row(file.path(WH, "model_joint_duration_final_encoder.rds"), "encoder")
joint_row(file.path(WH, "model_joint_duration_final_decoder.rds"), "decoder")

cat("\n========== JOINT MODELS (subword) ==========\n")
joint_row(file.path(WH, "model_joint_duration_sub_final_encoder.rds"), "encoder_sub")
joint_row(file.path(WH, "model_joint_duration_sub_final_decoder.rds"), "decoder_sub")

cat("\n========== GAM MODELS (indep) ==========\n")
gam_row(file.path(WH, "model_freq_layer_duration_encoder.rds"), "encoder freq")
gam_row(file.path(WH, "model_predic_layer_duration_encoder.rds"), "encoder predic")
gam_row(file.path(WH, "model_freq_layer_duration_decoder.rds"), "decoder freq")
gam_row(file.path(WH, "model_predic_layer_duration_decoder.rds"), "decoder predic")

cat("\n========== GAM MODELS (subword) ==========\n")
gam_row(file.path(WH, "model_freq_layer_duration_sub_encoder.rds"), "encoder freq sub")
gam_row(file.path(WH, "model_predic_layer_duration_sub_encoder.rds"), "encoder predic sub")
gam_row(file.path(WH, "model_freq_layer_duration_sub_decoder.rds"), "decoder freq sub")
gam_row(file.path(WH, "model_predic_layer_duration_sub_decoder.rds"), "decoder predic sub")

cat("\n========== ALONE MODELS (indep) ==========\n")
alone_row(file.path(WH, "model_freq_duration_final_encoder.rds"), "encoder", "log_freq")
alone_row(file.path(WH, "model_predic_duration_final_encoder.rds"), "encoder", "log_predic")
alone_row(file.path(WH, "model_freq_duration_final_decoder.rds"), "decoder", "log_freq")
alone_row(file.path(WH, "model_predic_duration_final_decoder.rds"), "decoder", "log_predic")

cat("\n========== ALONE MODELS (subword) ==========\n")
alone_row(file.path(WH, "model_freq_duration_sub_final_encoder.rds"), "encoder_sub", "log_freq")
alone_row(file.path(WH, "model_predic_duration_sub_final_encoder.rds"), "encoder_sub", "log_predic")
alone_row(file.path(WH, "model_freq_duration_sub_final_decoder.rds"), "decoder_sub", "log_freq")
alone_row(file.path(WH, "model_predic_duration_sub_final_decoder.rds"), "decoder_sub", "log_predic")

message("\nDone.")
