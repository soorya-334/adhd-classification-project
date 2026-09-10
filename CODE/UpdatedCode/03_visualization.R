# =============================================================
# ADHD Classification Pipeline — VISUALIZATION (v4, file 3 of 3)
#
# All plotting/figure-generation code lives here. Nothing in this file
# fits or evaluates a model — it only reads objects already produced by
# 01_main_pipeline.R (and, for the repeated-split box plot, 02_validation.R).
#
# REQUIRES, in this order:
#   source("01_main_pipeline.R")
#   source("02_validation.R")   # optional, only needed for the box plot
#   source("03_visualization.R")
# =============================================================

stopifnot(exists("primary"), exists("primary_full"), exists("med_suffix"))

# =============================================================
# PLOTS FOR THE PRIMARY RUN (single split, seed=76)
# =============================================================
results_df <- primary %>%
  select(Model, Accuracy, Precision, F1, Specificity, AUC) %>%
  pivot_longer(-Model, names_to = "Metric", values_to = "Value") %>%
  mutate(Model = factor(Model, levels = c("GLM", "RF", "XGB", "AttnSVM")))

p <- ggplot(results_df, aes(x = Model, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~ Metric, scales = "free") +
  theme_minimal() +
  labs(title = "Model Performance Comparison (primary run, seed=76)")
ggsave(paste0("model_performance_comparison_CORRECTED", med_suffix, ".png"), plot = p, width = 8, height = 6)

for (metric in unique(results_df$Metric)) {
  g <- ggplot(filter(results_df, Metric == metric), aes(x = Model, y = Value, fill = Model)) +
    geom_bar(stat = "identity", position = "dodge") +
    theme_minimal(base_size = 14) +
    labs(title = paste(metric, "Comparison Across Models"), y = metric, x = "Model") +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1))
  ggsave(paste0("metric_", tolower(metric), "_comparison_CORRECTED", med_suffix, ".png"), g, width = 7, height = 5)
}

cat("\n=== DONE ===\n")
cat("Primary run (seed=76, full CV):", paste0("Evaluation_metrics_CORRECTED", med_suffix, ".csv"), "\n")
if (exists("RUN_REPEATED_SPLIT_VALIDATION") && isTRUE(RUN_REPEATED_SPLIT_VALIDATION) && exists("N_REPEATS")) {
  cat("Repeated-split validation (", N_REPEATS, "splits):",
      paste0("Repeated_split_summary", med_suffix, ".csv"), "\n")
}

# =============================================================
# REPEATED-SPLIT VALIDATION BOX PLOT (DECISION 10)
# Only runs if 02_validation.R was sourced and produced repeated_results.
# =============================================================
if (exists("repeated_results") && exists("N_REPEATS")) {
  box_plot <- ggplot(repeated_results, aes(x = Model, y = Accuracy, fill = Model)) +
    geom_boxplot() +
    geom_jitter(width = 0.15, alpha = 0.4) +
    geom_hline(yintercept = 0.5484, linetype = "dashed", color = "red") +
    theme_minimal(base_size = 13) +
    labs(title = paste0("Accuracy across ", N_REPEATS, " random splits (", med_suffix, ")"),
         subtitle = "Dashed line = no-information rate (majority class baseline)",
         y = "Test Accuracy")
  ggsave(paste0("repeated_split_accuracy", med_suffix, ".png"), box_plot, width = 8, height = 6)
} else {
  cat("Skipping repeated-split box plot: source 02_validation.R first to create 'repeated_results'.\n")
}

# =============================================================
# FIGURE 3 (style): confusion-matrix percentage grid, 2x2 per model,
# faceted GLMnet / Random Forest / XGBoost / AttnSVM. Uses the REAL
# fitted models from the primary run (primary_full) -- no hardcoded
# numbers -- so it automatically stays correct if you rerun with
# different data, seeds, or toggles.
# =============================================================
X_test <- primary_full$X_test
y_test_factor <- primary_full$y_test_factor

cm_counts <- list(
  GLMnet = table(Predicted = predict(primary_full$glm_model, X_test),
                  Actual = y_test_factor),
  `Random Forest` = table(Predicted = predict(primary_full$rf_model, X_test),
                           Actual = y_test_factor),
  XGBoost = table(Predicted = predict(primary_full$xgb_model, X_test),
                   Actual = y_test_factor),
  AttnSVM = table(Predicted = primary_full$pred_attnsvm, Actual = y_test_factor)
)

cm_df_list <- lapply(names(cm_counts), function(nm) {
  m <- cm_counts[[nm]]  # rows = Predicted, cols = Actual
  # percentage WITHIN each actual class (column-normalized), matching the
  # original figure's row-reads-as-"Actual" layout
  pct <- round(100 * sweep(m, 2, colSums(m), "/"))
  d <- as.data.frame(pct)
  d$Model <- nm
  d
})
cm_df <- bind_rows(cm_df_list)
cm_df$Model <- factor(cm_df$Model, levels = c("GLMnet", "Random Forest", "XGBoost", "AttnSVM"))
cm_df$Actual <- factor(cm_df$Actual, levels = c("ADHD", "Non_ADHD"), labels = c("ADHD", "Non-ADHD"))
cm_df$Predicted <- factor(cm_df$Predicted, levels = c("Non_ADHD", "ADHD"), labels = c("Non-ADHD", "ADHD"))

p_cm <- ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = paste0(Freq, "%")), color = "white", size = 5) +
  facet_wrap(~ Model, ncol = 2) +
  scale_fill_gradient(low = "#a6c8e8", high = "#1a5490", guide = "none") +
  theme_minimal(base_size = 13) +
  theme(panel.grid = element_blank(), strip.text = element_text(face = "bold", size = 13)) +
  labs(x = "Predicted", y = "Actual")
ggsave(paste0("Figure3_confusion_matrices", med_suffix, ".tiff"), p_cm,
       width = 6, height = 6.5, dpi = 300, compression = "lzw")
ggsave(paste0("Figure3_confusion_matrices", med_suffix, ".png"), p_cm, width = 6, height = 6.5, dpi = 300)
cat("Saved Figure3_confusion_matrices", med_suffix, ".tiff/.png\n")

# =============================================================
# FIGURE 4 (style): ROC curves, all four models overlaid, using the
# REAL predicted probabilities from the primary run.
# =============================================================
p_glm <- predict(primary_full$glm_model, X_test, type = "prob")[, "ADHD"]
p_rf  <- predict(primary_full$rf_model,  X_test, type = "prob")[, "ADHD"]
p_xgb <- predict(primary_full$xgb_model, X_test, type = "prob")[, "ADHD"]
p_svm <- primary_full$p_attnsvm

roc_glm <- pROC::roc(y_test_factor, p_glm, quiet = TRUE)
roc_rf  <- pROC::roc(y_test_factor, p_rf,  quiet = TRUE)
roc_xgb <- pROC::roc(y_test_factor, p_xgb, quiet = TRUE)
roc_svm <- pROC::roc(y_test_factor, p_svm, quiet = TRUE)

cat(sprintf("\nAUC -- GLM: %.3f | RF: %.3f | XGB: %.3f | AttnSVM: %.3f\n",
            pROC::auc(roc_glm), pROC::auc(roc_rf), pROC::auc(roc_xgb), pROC::auc(roc_svm)))

roc_to_df <- function(roc_obj, model_name) {
  data.frame(x = 1 - roc_obj$specificities, y = roc_obj$sensitivities, Model = model_name)
}
roc_df <- bind_rows(
  roc_to_df(roc_glm, "GLM"), roc_to_df(roc_rf, "RF"),
  roc_to_df(roc_xgb, "XGB"), roc_to_df(roc_svm, "AttnSVM")
) %>% arrange(Model, x)
roc_df$Model <- factor(roc_df$Model, levels = c("GLM", "RF", "XGB", "AttnSVM"))

p_roc <- ggplot(roc_df, aes(x = x, y = y, color = Model)) +
  geom_step(direction = "vh", linewidth = 0.7) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
  scale_x_reverse(breaks = seq(1, 0, -0.2)) +
  scale_y_continuous(breaks = seq(0, 1, 0.2)) +
  scale_color_manual(values = c(GLM = "#F8766D", RF = "#7CAE00", XGB = "#00BFC4", AttnSVM = "#C77CFF")) +
  theme_minimal(base_size = 12) +
  theme(panel.background = element_rect(fill = "grey92", color = NA),
        panel.grid = element_line(color = "white")) +
  labs(title = "ROC Curves for ADHD Prediction", x = "1 - Specificity", y = "Sensitivity", color = "ML Model")
ggsave(paste0("Figure4_ROC_curves", med_suffix, ".tiff"), p_roc,
       width = 6, height = 5, dpi = 300, compression = "lzw")
ggsave(paste0("Figure4_ROC_curves", med_suffix, ".png"), p_roc, width = 6, height = 5, dpi = 300)
cat("Saved Figure4_ROC_curves", med_suffix, ".tiff/.png\n")

cat("\n=== 03_visualization.R complete ===\n")
