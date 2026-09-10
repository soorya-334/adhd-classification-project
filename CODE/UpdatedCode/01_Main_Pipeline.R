library(tidyverse)
library(readxl)
library(caret)
library(glmnet)
library(randomForest)
library(xgboost)
library(e1071)
library(pROC)
library(recipes)
library(reshape2)
library(cowplot)
library(scales)
library(stringr)
library(corrplot)

# ------------------------ DATA LOADING ------------------------
stat_data   <- read_excel("D:/SET PROJECT/result/Statistical_Data.xlsx")
region_data <- read_excel("D:/SET PROJECT/result/region_matters_stats.xlsx")

region_data <- region_data %>% mutate(Subject_clean = str_extract(Subject, "sub-\\d+"))

clean_med <- function(df) df %>%
  mutate(Subject = as.character(Subject) %>% trimws(),
         ADHD_medication = case_when(
           ADHD_medication %in% c("n/a", "NA", "", "N/A") ~ "No_Medication",
           TRUE ~ ADHD_medication))
stat_data   <- clean_med(stat_data)
region_data <- clean_med(region_data)

stat_wide <- stat_data %>%
  select(Subject, sex, age_ses_T1, ADHD_diagnosis, ADHD_medication, race, ethnicity,
         region, volume_mm3, mean_intensity, std_intensity, skewness, kurtosis,
         contrast, homogeneity, energy, correlation, entropy, LBP_mean, fractal_dimension) %>%
  pivot_wider(
    id_cols = c(Subject, sex, age_ses_T1, ADHD_diagnosis, ADHD_medication, race, ethnicity),
    names_from = region,
    values_from = c(volume_mm3, mean_intensity, std_intensity, skewness, kurtosis,
                    contrast, homogeneity, energy, correlation, entropy, LBP_mean, fractal_dimension),
    names_sep = "__", values_fn = mean, values_fill = NA)

region_vols <- region_data %>%
  select(Subject_clean, starts_with("GM_volume_mm3"), starts_with("WM_volume_mm3"),
         CSF_volume_mm3, Brain_volume_mm3, GM_WM_ratio, matches("prob_")) %>%
  rename(Subject = Subject_clean)

data <- stat_wide %>%
  inner_join(region_vols, by = "Subject") %>%
  filter(ADHD_diagnosis %in% c(0, 1)) %>%
  mutate(ADHD = factor(ADHD_diagnosis, levels = c(0, 1), labels = c("Non_ADHD", "ADHD")),
         race = replace_na(race, "Not_Identified"),
         ethnicity = replace_na(ethnicity, "Not_Identified"),
         sex = replace_na(sex, "Not_Identified"),
         ADHD_medication = replace_na(ADHD_medication, "No_Medication")) %>%
  filter(!is.na(age_ses_T1))

data$Race <- factor(data$race)
data$Ethnicity <- factor(data$ethnicity)
data$Medication <- factor(data$ADHD_medication)
data$AgeGroup <- cut(data$age_ses_T1, breaks = c(0, 10, 18), labels = c("0-10", "11-18"), right = FALSE)

clinical_factors_full <- c("race", "ethnicity", "ADHD_medication", "sex")

# X_imaging's exclusion list is ALWAYS the full clinical set, regardless of
# any downstream toggle — this is what keeps clinical variables from
# leaking back into the "imaging" block when medication/clinical features
# are toggled off (see DECISION 7/8). NOTE: age_ses_T1 is intentionally
# NOT excluded here — it is retained as a continuous imaging-block
# covariate, not treated as a separate clinical predictor (see header note
# on the "five clinical variables" text-vs-code mismatch).
X_imaging_base <- data %>%
  select(-any_of(c("Subject", "ADHD_diagnosis", "ADHD",
                    clinical_factors_full, "Race", "Ethnicity", "Medication", "AgeGroup")),
         -starts_with("cerebellum_mnifnirt_prob_"))
# >>> DECISION 3 <<< prefix so imaging columns can be recovered from the
# unified X_train/X_test after the shared recipe has processed everything
names(X_imaging_base) <- paste0("img__", names(X_imaging_base))
y_all <- data$ADHD

# =============================================================
# CORE PIPELINE — one full split -> preprocess -> train -> evaluate cycle.
# Used both for the single detailed "primary" run below (with plots in
# 03_visualization.R) and for the repeated-split validation loop in
# 02_validation.R (DECISION 10).
# =============================================================
run_one_split <- function(seed,
                           INCLUDE_ADHD_MEDICATION = TRUE,
                           INCLUDE_CLINICAL_FEATURES = TRUE,
                           cv_number = 10, cv_repeats = 5,
                           svm_grid = expand.grid(sigma = c(0.01, 0.1, 1), C = c(0.1, 1, 10)),
                           verbose = TRUE, return_models = FALSE) {

  clinical_factors <- if (!INCLUDE_CLINICAL_FEATURES) {
    character(0)
  } else if (INCLUDE_ADHD_MEDICATION) {
    clinical_factors_full
  } else {
    setdiff(clinical_factors_full, "ADHD_medication")
  }

  cat("ADHD_medication included as predictor:", INCLUDE_ADHD_MEDICATION, "\n")
  cat("Clinical/demographic features included:", INCLUDE_CLINICAL_FEATURES, "\n")

  X_imaging <- X_imaging_base
  y <- y_all

  set.seed(seed)
  train_idx <- createDataPartition(y, p = 0.6, list = FALSE)

  # >>> DECISION 1b <<< valid_factors from TRAINING FOLD only
  valid_factors <- if (length(clinical_factors) > 0) {
    clinical_factors[sapply(data[train_idx, clinical_factors], \(x) n_distinct(na.omit(x)) > 1)]
  } else character(0)

  X_clinical <- if (length(valid_factors) > 0) {
    model.matrix(reformulate(valid_factors), data = data) %>% as.data.frame()
  } else {
    data.frame(row.names = seq_len(nrow(data)))
  }
  # >>> DECISION 3 <<< same prefixing for clinical columns
  if (ncol(X_clinical) > 0) names(X_clinical) <- paste0("clin__", names(X_clinical))
  X_all <- bind_cols(X_imaging, X_clinical)

  # >>> DECISION 2 <<< z-score scaling added; NZV/impute/dummy all fit on
  # training fold only (DECISION 1). This SINGLE recipe is now the ONLY
  # preprocessing pathway in the whole script — GLM, RF, XGBoost, AND
  # both AttSVM streams all draw from its output (DECISION 3).
  rec <- recipe(ADHD ~ ., data = bind_cols(X_all, ADHD = y)) %>%
    step_impute_median(all_numeric_predictors()) %>%
    step_nzv(all_predictors()) %>%
    step_dummy(all_nominal_predictors()) %>%
    step_zv(all_predictors()) %>%
    step_normalize(all_numeric_predictors())
  rec_prep <- prep(rec, training = bind_cols(X_all[train_idx, ], ADHD = y[train_idx]))
  X_train <- bake(rec_prep, new_data = X_all[train_idx, ])
  X_test  <- bake(rec_prep, new_data = X_all[-train_idx, ])

  # >>> DECISION 3 <<< recover imaging vs clinical column subsets from the
  # SAME unified, already-preprocessed X_train/X_test by name prefix
  img_cols  <- grep("^img__",  names(X_train), value = TRUE)
  clin_cols <- grep("^clin__", names(X_train), value = TRUE)
  HAS_CLINICAL <- length(clin_cols) > 0

  ctrl <- trainControl(method = "repeatedcv", number = cv_number, repeats = cv_repeats,
                        classProbs = TRUE, summaryFunction = twoClassSummary,
                        savePredictions = "final")
  y_train_clean <- factor(y[train_idx], levels = c("Non_ADHD", "ADHD"))
  y_test_factor <- factor(y[-train_idx], levels = c("Non_ADHD", "ADHD"))

  glm_model <- train(x = X_train, y = y_train_clean, method = "glmnet",
                      metric = "ROC", trControl = ctrl)
  rf_model  <- train(x = X_train, y = y_train_clean, method = "rf",
                      tuneGrid = expand.grid(mtry = c(5, 10, 20)),
                      metric = "ROC", ntree = 200, trControl = ctrl)
  capture.output({
    xgb_model <- train(x = X_train, y = y_train_clean, method = "xgbTree",
                        metric = "ROC", trControl = ctrl, verbose = FALSE, nthread = 1)
  }, file = NULL)

  # ---------------- AttSVM: BOTH streams now drawn from the unified,
  # already-imputed / NZV-filtered / normalized X_train / X_test ----------------
  svm_img <- train(x = X_train[, img_cols, drop = FALSE], y = y_train_clean,
                    method = "svmRadial", tuneGrid = svm_grid, metric = "ROC", trControl = ctrl)

  if (HAS_CLINICAL) {
    svm_cli <- train(x = X_train[, clin_cols, drop = FALSE], y = y_train_clean,
                      method = "svmRadial", tuneGrid = svm_grid, metric = "ROC", trControl = ctrl)
  }

  # ---------------- Fusion weight ----------------
  # >>> DECISION 6 <<< alpha from TRAINING CV AUC only, never test AUC
  get_cv_auc <- function(model) {
    bt <- model$bestTune; res <- model$results
    match_idx <- Reduce(`&`, lapply(names(bt), function(col) res[[col]] == bt[[col]]))
    res$ROC[match_idx][1]
  }
  if (HAS_CLINICAL) {
    cv_auc_img <- get_cv_auc(svm_img)
    cv_auc_cli <- get_cv_auc(svm_cli)
    alpha <- cv_auc_cli / (cv_auc_cli + cv_auc_img)
  } else {
    cv_auc_img <- get_cv_auc(svm_img); cv_auc_cli <- NA_real_
    alpha <- 0
  }

  # ---------------- DECISION 9: data-driven threshold ----------------
  # Derive the AttnSVM decision threshold from FUSED TRAINING out-of-fold
  # predictions (Youden's index), never from test data.
  oof_img <- svm_img$pred[svm_img$pred$sigma == svm_img$bestTune$sigma &
                           svm_img$pred$C == svm_img$bestTune$C, c("rowIndex", "ADHD", "obs")]
  names(oof_img)[names(oof_img) == "ADHD"] <- "p_img"

  if (HAS_CLINICAL) {
    oof_cli <- svm_cli$pred[svm_cli$pred$sigma == svm_cli$bestTune$sigma &
                             svm_cli$pred$C == svm_cli$bestTune$C, c("rowIndex", "ADHD")]
    names(oof_cli)[names(oof_cli) == "ADHD"] <- "p_cli"
    oof <- merge(oof_img, oof_cli, by = "rowIndex")
    oof$p_fused <- alpha * oof$p_cli + (1 - alpha) * oof$p_img
  } else {
    oof <- oof_img
    oof$p_fused <- oof$p_img
  }
  roc_oof <- pROC::roc(oof$obs, oof$p_fused, levels = c("Non_ADHD", "ADHD"),
                        direction = "<", quiet = TRUE)
  thr <- as.numeric(pROC::coords(roc_oof, "best", best.method = "youden",
                                  ret = "threshold", transpose = TRUE)[1])
  if (is.na(thr)) thr <- 0.5  # fallback if Youden fails to find a unique optimum

  # ---------------- Test-set prediction (test data touched ONLY here) ----------------
  p_img <- predict(svm_img, newdata = X_test[, img_cols, drop = FALSE], type = "prob")[, "ADHD"]
  p_cli <- if (HAS_CLINICAL) predict(svm_cli, newdata = X_test[, clin_cols, drop = FALSE], type = "prob")[, "ADHD"] else rep(0, length(p_img))
  p_attnsvm <- alpha * p_cli + (1 - alpha) * p_img
  pred_attnsvm <- factor(ifelse(p_attnsvm > thr, "ADHD", "Non_ADHD"), levels = c("Non_ADHD", "ADHD"))

  # ---------------- Evaluation ----------------
  eval_metrics <- function(pred, probs, true) {
    true <- factor(true, levels = c("Non_ADHD", "ADHD"))
    pred <- factor(pred, levels = c("Non_ADHD", "ADHD"))
    cm <- caret::confusionMatrix(pred, true, positive = "ADHD")
    roc_obj <- tryCatch(pROC::roc(true, probs, quiet = TRUE), error = function(e) NULL)
    Precision <- unname(cm$byClass["Pos Pred Value"])
    F1 <- suppressWarnings(F_meas(pred, true, relevant = "ADHD"))
    list(Accuracy = as.numeric(cm$overall["Accuracy"]),
         Accuracy_CI_Lo = as.numeric(cm$overall["AccuracyLower"]),
         Accuracy_CI_Hi = as.numeric(cm$overall["AccuracyUpper"]),
         Sensitivity = as.numeric(cm$byClass["Sensitivity"]),
         Specificity = as.numeric(cm$byClass["Specificity"]),
         Precision = ifelse(is.nan(Precision), NA_real_, as.numeric(Precision)),
         F1 = ifelse(is.nan(F1), NA_real_, as.numeric(F1)),
         AUC = if (!is.null(roc_obj)) as.numeric(pROC::auc(roc_obj)) else NA_real_)
  }

  res <- bind_rows(
    c(Model = "GLM",     Seed = seed, eval_metrics(predict(glm_model, X_test), predict(glm_model, X_test, type="prob")[,"ADHD"], y[-train_idx])),
    c(Model = "RF",      Seed = seed, eval_metrics(predict(rf_model,  X_test), predict(rf_model,  X_test, type="prob")[,"ADHD"], y[-train_idx])),
    c(Model = "XGB",     Seed = seed, eval_metrics(predict(xgb_model, X_test), predict(xgb_model, X_test, type="prob")[,"ADHD"], y[-train_idx])),
    c(Model = "AttnSVM", Seed = seed, eval_metrics(pred_attnsvm, p_attnsvm, y[-train_idx]))
  ) %>% mutate(across(-Model, as.numeric),
               Predictors = ncol(X_train), ImagingPredictors = length(img_cols),
               ClinicalPredictors = length(clin_cols), Alpha = alpha, Threshold = thr,
               CV_AUC_img = cv_auc_img, CV_AUC_cli = cv_auc_cli)

  if (verbose) {
    cat(sprintf("Seed %d | total=%d (img=%d, clin=%d) | alpha=%.3f | threshold=%.3f\n",
                seed, ncol(X_train), length(img_cols), length(clin_cols), alpha, thr))
  }

  if (return_models) {
    return(list(metrics = res, glm_model = glm_model, rf_model = rf_model, xgb_model = xgb_model,
                pred_attnsvm = pred_attnsvm, p_attnsvm = p_attnsvm,
                X_test = X_test, y_test_factor = y_test_factor))
  }
  res
}

# =============================================================
# PRIMARY RUN — matches original manuscript methodology (full CV settings,
# same seed=76 as your original script), for the main reported result and
# plots (see 03_visualization.R). Set your configuration here.
# =============================================================
INCLUDE_ADHD_MEDICATION   <- TRUE   # >>> DECISION 7 <<<
INCLUDE_CLINICAL_FEATURES <- TRUE   # >>> DECISION 8 <<<

primary_full <- run_one_split(seed = 76,
                               INCLUDE_ADHD_MEDICATION = INCLUDE_ADHD_MEDICATION,
                               INCLUDE_CLINICAL_FEATURES = INCLUDE_CLINICAL_FEATURES,
                               cv_number = 10, cv_repeats = 5, return_models = TRUE)
primary <- primary_full$metrics
print(primary)
med_suffix <- paste0(if (INCLUDE_ADHD_MEDICATION) "_withMed" else "_noMed",
                      if (!INCLUDE_CLINICAL_FEATURES) "_imagingOnly" else "")
write.csv(primary, paste0("Evaluation_metrics_CORRECTED", med_suffix, ".csv"), row.names = FALSE)

cat("\n=== 01_main_pipeline.R complete ===\n")
cat("Objects available for 02_validation.R / 03_visualization.R:\n")
cat("  data, X_imaging_base, y_all, clinical_factors_full, run_one_split()\n")
cat("  INCLUDE_ADHD_MEDICATION, INCLUDE_CLINICAL_FEATURES, med_suffix\n")
cat("  primary_full (fitted models + test set), primary (metrics table)\n")
