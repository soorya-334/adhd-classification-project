# ------------------------ LIBRARIES ------------------------
library(tidyverse)
library(readxl)
library(caret)
library(glmnet)
library(randomForest)
library(xgboost)
library(e1071)
library(pROC)
library(recipes)
library(precrec)
library(reshape2)
library(cowplot)
library(scales)
library(dplyr)
library(stringr)
library(corrplot)
library(writexl)

# ------------------------ DATA LOADING ------------------------
set.seed(75)
stat_data   <- read_excel("D:/SET PROJECT/result/Statistical_Data.xlsx")
region_data <- read_excel("D:/SET PROJECT/result/region_matters_stats.xlsx")

region_data <- region_data %>%
  mutate(Subject_clean = str_extract(Subject, "sub-\\d+"))

stat_data <- stat_data %>%
  mutate(Subject = as.character(Subject) %>% trimws(),
         ADHD_medication = case_when(
           ADHD_medication %in% c("n/a", "NA", "", "N/A") ~ "No_Medication",
           TRUE ~ ADHD_medication))

region_data <- region_data %>%
  mutate(Subject = as.character(Subject) %>% trimws(),
         ADHD_medication = case_when(
           ADHD_medication %in% c("n/a", "NA", "", "N/A") ~ "No_Medication",
           TRUE ~ ADHD_medication))

# Pivot stat_data
stat_wide <- stat_data %>%
  select(Subject, sex, age_ses_T1, ADHD_diagnosis, ADHD_medication, race, ethnicity,
         region, volume_mm3, mean_intensity, std_intensity, skewness, kurtosis,
         contrast, homogeneity, energy, correlation, entropy, LBP_mean, fractal_dimension) %>%
  pivot_wider(
    id_cols = c(Subject, sex, age_ses_T1, ADHD_diagnosis, ADHD_medication, race, ethnicity),
    names_from = region,
    values_from = c(volume_mm3, mean_intensity, std_intensity, skewness, kurtosis,
                    contrast, homogeneity, energy, correlation, entropy, LBP_mean, fractal_dimension),
    names_sep = "__",
    values_fn = mean,
    values_fill = NA)

# Region Volumes
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

# Encode factors
data$Race <- factor(data$race)
data$Ethnicity <- factor(data$ethnicity)
data$Medication <- factor(data$ADHD_medication)
data$AgeGroup <- cut(data$age_ses_T1, breaks = c(0, 10, 18),labels = c("0-10", "11-18"), right = FALSE)

# ------------------------ FEATURE SPLITS ------------------------
clinical_factors <- c("race", "ethnicity", "ADHD_medication", "sex")

X_imaging <- data %>%
  select(-any_of(c("Subject", "ADHD_diagnosis", "ADHD", clinical_factors)),
         -starts_with("cerebellum_mnifnirt_prob_"))

nzv <- nearZeroVar(X_imaging)
if (length(nzv) > 0) X_imaging <- X_imaging[, -nzv]

valid_factors <- clinical_factors[sapply(data[clinical_factors], \(x) n_distinct(na.omit(x)) > 1)]
X_clinical <- model.matrix(reformulate(valid_factors), data = data) %>% as.data.frame()

X_all <- bind_cols(X_imaging, X_clinical)
X_imaging_raw <- X_imaging
X_clinical_raw <- X_clinical
y <- data$ADHD   

#------------------------Train/Test Split up --------------------------
set.seed(75)
train_idx <- createDataPartition(y, p = 0.6, list = FALSE)

rec <- recipe(ADHD ~ ., data = bind_cols(X_all, ADHD = y)) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())

rec_prep <- prep(rec, training = bind_cols(X_all[train_idx, ], ADHD = y[train_idx]))
X_train <- bake(rec_prep, new_data = X_all[train_idx, ])
X_test  <- bake(rec_prep, new_data = X_all[-train_idx, ])

y[train_idx]
y[-train_idx]
table(y[-train_idx])  # testing
table(y[train_idx]) # training 
length(train_idx)
length(-train_idx)
data$Subject[train_idx]
data$Subject[-train_idx]


# ------------------------ MODEL TRAINING ------------------------
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5, classProbs = TRUE,summaryFunction = twoClassSummary)

y_train_clean <- factor(y[train_idx], levels = c("Non_ADHD", "ADHD"))
glm_model <- train(x = X_train, y = y_train_clean, method = "glmnet", metric = "ROC",trControl = ctrl)
rf_grid <- expand.grid(mtry = c(5, 10, 20))
rf_model <- train(x = X_train, y = y_train_clean, method = "rf", tuneGrid = rf_grid, metric = "ROC",ntree = 200, trControl = ctrl)
capture.output({ xgb_model <- train(x=X_train, y=y_train_clean, method="xgbTree", metric="ROC", trControl=ctrl, verbose=FALSE, nthread=1)}, file = NULL)

# ------------------------ SVM FUSION ------------------------

# ----------------- Preprocess Imaging Data ------------------
X_imaging_raw_clean <- X_imaging_raw %>% 
  mutate(across(where(is.numeric), ~ replace_na(., median(., na.rm = TRUE))))

svm_grid <- expand.grid( sigma = c(0.01, 0.1, 1), C = c(0.1, 1, 10))

X_img_svm <- X_imaging_raw_clean %>% select(where(is.numeric))
svm_img <- train(x=X_img_svm[train_idx,], y=factor(y[train_idx],levels=c("Non_ADHD","ADHD")), method="svmRadial", tuneGrid=svm_grid, metric="ROC", trControl=ctrl);

# ----------------- Preprocess Clinical Data ------------------
nzv_cli <- nearZeroVar(X_clinical_raw)
X_cli_svm <- X_clinical_raw[, -nzv_cli]
svm_cli <- train(x=X_cli_svm[train_idx,], y=factor(y[train_idx],levels=c("Non_ADHD","ADHD")), method="svmRadial", tuneGrid=svm_grid, metric="ROC", trControl=ctrl)

# predict probabilities correctly:
p_img <- predict(svm_img, newdata = X_img_svm[-train_idx, ],type    = "prob")[, "ADHD"]
p_cli <- predict(svm_cli,newdata = X_cli_svm[-train_idx, ],type    = "prob")[, "ADHD"]

# ensure factor matches:
y_test_factor <- factor(y[-train_idx], levels = c("Non_ADHD","ADHD"))

# compute ROC/AUC
roc_obj_img <- pROC::roc(y_test_factor, p_img)
auc_img     <- pROC::auc(roc_obj_img)
roc_obj_cli <- pROC::roc(y_test_factor, p_cli)
auc_cli     <- pROC::auc(roc_obj_cli)

# -------------------------- fusion -------------------------- 
alpha       <- auc_cli/(auc_cli+auc_img)
p_attnsvm   <- alpha*p_cli + (1-alpha)*p_img
pred_attnsvm <- factor(ifelse(p_attnsvm > 0.6,"ADHD","Non_ADHD"),levels=c("Non_ADHD","ADHD"))

# ------------------------ EVALUATION ------------------------
eval_metrics <- function(pred, probs, true) {
  true <- factor(true, levels = c("Non_ADHD", "ADHD"))
  pred <- factor(pred, levels = c("Non_ADHD", "ADHD"))
  
  cm <- caret::confusionMatrix(pred, true, positive = "ADHD")
  roc_obj <- pROC::roc(true, probs)
  
  Precision   <- unname(cm$byClass["Pos Pred Value"])
  Specificity <- unname(cm$byClass["Specificity"])
  F1          <- F_meas(pred, true, relevant = "ADHD")
  
  list(
    Accuracy    = as.numeric(cm$overall["Accuracy"]),
    Precision   = ifelse(is.nan(Precision), NA_real_, as.numeric(Precision)),
    F1          = ifelse(is.nan(F1), NA_real_, as.numeric(F1)),
    Specificity = as.numeric(Specificity),
    AUC         = as.numeric(pROC::auc(roc_obj))
  )
}


results <- list(
  GLM = eval_metrics(predict(glm_model, X_test), predict(glm_model, X_test, type = "prob")[, "ADHD"], y[-train_idx]),
  RF = eval_metrics(predict(rf_model, X_test), predict(rf_model, X_test, type = "prob")[, "ADHD"], y[-train_idx]),
  XGB = eval_metrics(predict(xgb_model, X_test), predict(xgb_model, X_test, type = "prob")[, "ADHD"], y[-train_idx]),
  AttnSVM = eval_metrics(pred_attnsvm, p_attnsvm, y[-train_idx])
)

print(results)
write.csv(results, "Evaluation_matrics.csv", row.names = FALSE)

# Predictions for each model
pred_glm  <- predict(glm_model, X_test)
pred_rf   <- predict(rf_model, X_test)
pred_xgb  <- predict(xgb_model, X_test)
pred_attn <- pred_attnsvm  # already created from attention fusion

table(pred_glm, pred_rf)
table(pred_glm, pred_xgb)
table(pred_glm, pred_attn)
table(pred_rf, pred_xgb)
table(pred_rf, pred_attn)
table(pred_xgb, pred_attn)

# ------------------------ MODEL COMPARISON PLOT ------------------------
results_df <- bind_rows(results, .id = "Model") %>%
  pivot_longer(-Model, names_to = "Metric", values_to = "Value") %>%
  mutate(Model = factor(Model, levels = c("GLM", "RF", "XGB", "AttnSVM")))

# 1. Create the plot
p <- ggplot(results_df, aes(x = Model, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~ Metric, scales = "free") +
  theme_minimal() +
  labs(title = "Model Performance Comparison")

# 2. Save it
ggsave("model_performance_comparison.png", plot = p, width = 8, height = 6)

# ------------------------ FEATURE IMPORTANCE ------------------------
imp_df <- data.frame(
  Feature = rownames(varImp(glm_model)$importance),
  GLM = varImp(glm_model)$importance$Overall,
  RF = varImp(rf_model)$importance$Overall
)
write.csv(imp_df, "feature_importance.csv", row.names = FALSE)

# ------------------------ ROC CURVE ------------------------
y_test <- factor(y[-train_idx], levels = c("Non_ADHD", "ADHD"))

roc_glm <- roc(y_test, predict(glm_model, X_test, type = "prob")[, "ADHD"])
roc_rf  <- roc(y_test, predict(rf_model, X_test, type = "prob")[, "ADHD"])
roc_xgb <- roc(y_test, predict(xgb_model, X_test, type = "prob")[, "ADHD"])
roc_svm <- roc(y_test, p_attnsvm)

# Plot with custom thickness and legend formatting
roc_plot <- ggroc(
  list(GLM = roc_glm, RF = roc_rf, XGB = roc_xgb, AttnSVM = roc_svm),
  aes = c("color")  # enables automatic legend
) +
  geom_line(size = 0.5) +  # Control line thickness here
  theme_minimal() +
  labs(
    title = "ROC Curves for ADHD Prediction",
    x = "1 - Specificity",
    y = "Sensitivity",
    color = "ML Model"  # Title of the legend
  ) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.2)) +
  scale_x_reverse(breaks = seq(1, 0, by = -0.2)) +
  theme(
    plot.title = element_text(size = 8, face = "bold", hjust = 0.5),  # Title
    axis.title.x = element_text(size = 6),  # X-axis label size
    axis.title.y = element_text(size = 6),  # Y-axis label size
    axis.text = element_text(size = 6),
    legend.position = "right",           # Move legend to bottom
    legend.title = element_text(size = 6),
    legend.text = element_text(size = 6)
  )

# Save the plot
ggsave("roc_comparison.png", plot = roc_plot, width = 4, height = 3, dpi = 300)

# ------------------------ CALIBRATION ------------------------
cal <- calibration(y_test ~ p_attnsvm)
plot(cal)
ggsave("calibration_plot.png")

# ------------------------ INDIVIDUAL METRIC GRAPHS (ALL MODELS) ------------------------
metric_list <- unique(results_df$Metric)

for (metric in metric_list) {
  g <- ggplot(filter(results_df, Metric == metric), aes(x = Model, y = Value, fill = Model)) +
    geom_bar(stat = "identity", position = "dodge") +
    theme_minimal(base_size = 14) +
    labs(title = paste(metric, "Comparison Across Models"), y = metric, x = "Model") +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1))
  
  ggsave(paste0("metric_", tolower(metric), "_comparison.png"), g, width = 7, height = 5)
}

# ------------------------ COMBINED BAR PLOT FOR AttnSVM ONLY ------------------------
attnsvm_df <- results_df %>% filter(Model == "AttnSVM")

# Create the plot object
attnsvm_plot <- ggplot(attnsvm_df, aes(x = Metric, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", width = 0.6) +
  theme_minimal(base_size = 14) +
  labs(title = "AttnSVM Model Evaluation Metrics", y = "Score", x = "Metric") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1))

# Save it
ggsave("attnsvm_combined_metrics.png", plot = attnsvm_plot, width = 7, height = 5)