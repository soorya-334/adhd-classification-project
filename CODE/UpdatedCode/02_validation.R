# =============================================================
# ADHD Classification Pipeline — VALIDATION (v4, file 2 of 3)
#
# >>> DECISION 10 <<< REPEATED-SPLIT VALIDATION — R1-#3 / R2-major
# Repeats the ENTIRE split -> preprocess -> train -> evaluate cycle
# (run_one_split(), from 01_main_pipeline.R) across many random
# partitions instead of trusting a single 60/40 split. Uses lighter CV
# settings (5-fold, 1 repeat, smaller SVM grid) to keep runtime
# reasonable — this is a validation-strategy check, not the primary
# result, so it doesn't need to be as exhaustively tuned as the primary
# run.
#
# REQUIRES: 01_main_pipeline.R must already be sourced in this session
# (needs run_one_split(), INCLUDE_ADHD_MEDICATION, INCLUDE_CLINICAL_FEATURES,
# med_suffix). Run:
#   source("01_main_pipeline.R")
#   source("02_validation.R")
#
# Expect this to take a while (N_REPEATS x 4 models x lighter CV each).
# =============================================================

stopifnot(exists("run_one_split"), exists("med_suffix"))

RUN_REPEATED_SPLIT_VALIDATION <- TRUE
N_REPEATS <- 30

if (RUN_REPEATED_SPLIT_VALIDATION) {
  light_svm_grid <- expand.grid(sigma = c(0.05, 0.5), C = c(1, 10))
  repeated_results <- map_dfr(1:N_REPEATS, function(i) {
    run_one_split(seed = 1000 + i,
                   INCLUDE_ADHD_MEDICATION = INCLUDE_ADHD_MEDICATION,
                   INCLUDE_CLINICAL_FEATURES = INCLUDE_CLINICAL_FEATURES,
                   cv_number = 5, cv_repeats = 1, svm_grid = light_svm_grid,
                   verbose = TRUE)
  })
  write.csv(repeated_results, paste0("Repeated_split_results", med_suffix, ".csv"), row.names = FALSE)

  summary_stats <- repeated_results %>%
    group_by(Model) %>%
    summarise(N = n(),
              Mean_Accuracy = mean(Accuracy), SD_Accuracy = sd(Accuracy),
              Median_Accuracy = median(Accuracy),
              Pct2.5 = quantile(Accuracy, 0.025), Pct97.5 = quantile(Accuracy, 0.975),
              Mean_AUC = mean(AUC, na.rm = TRUE), SD_AUC = sd(AUC, na.rm = TRUE),
              Pct_Below_Chance = mean(Accuracy < 0.5484) * 100, .groups = "drop")
  print(summary_stats)
  write.csv(summary_stats, paste0("Repeated_split_summary", med_suffix, ".csv"), row.names = FALSE)
}

cat("\n=== 02_validation.R complete ===\n")
if (RUN_REPEATED_SPLIT_VALIDATION) {
  cat("Repeated-split validation (", N_REPEATS, "splits):",
      paste0("Repeated_split_summary", med_suffix, ".csv"), "\n")
  cat("Objects available for 03_visualization.R: repeated_results, summary_stats, N_REPEATS\n")
} else {
  cat("RUN_REPEATED_SPLIT_VALIDATION was FALSE — no repeated-split objects created.\n")
  cat("The repeated-split box plot in 03_visualization.R will be skipped.\n")
}
