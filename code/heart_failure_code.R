# Packages
library(GGally)
library(gridExtra)
library(skimr)
library(themis)
library(tidymodels)
library(tidyverse)
library(vip)

theme_set(theme_bw())

# Data
hf <- read_csv("data/heart_failure_clinical_records_dataset.csv")
str(hf)
skim(hf)

hf <- hf %>% 
  rename(death = DEATH_EVENT) %>% 
  mutate(death = as.factor(death))

# Train-test split----
set.seed(123)
hf_split <- initial_split(hf, strata = death, prop = 0.7)
hf_train <- training(hf_split)
hf_test <- testing(hf_split)

train_folds <- vfold_cv(hf_train, v = 10, repeats = 2)

# Variable distributions----
hf_train %>% 
  mutate(across(c(sex, smoking, anaemia, diabetes, high_blood_pressure), ~ as.factor(.))) %>% 
  ggpairs(switch = "y")

function(data){
  num_vars <- data %>% 
  select(where(~ is.numeric(.) && max(.) > 1)) %>%
  names()

 cat_vars <- data %>% 
  select(where(~is.numeric(.) && max(.) == 1)) %>% 
  names()
 
 for (c_var in categorical_vars) {
   for (n_var in numeric_vars) {
     
     
     var_plot <- data %>% 
       select(!!sym(n_var), !!sym(c_var)) %>% 
       group_by(!!sym(c_var)) %>% 
       # calculate the daily mean value of the resp. numerical val for each level of categorical var 
       summarise(var_mean = mean(!!sym(n_var)),
                 .groups = "drop")

     
     b <- ggplot(data = var_plot,
                 aes(
                   x = !!sym(c_var),
                   y = var_mean,
                   group = !!sym(c_var),
                   color = !!sym(c_var)
                 )
     ) +
       geom_boxplot(alpha = 0.5) +
       labs(title = paste("Box plot of mean", n_var, "by", c_var),
            x = c_var,
            y = n_var) +
       theme_minimal() +
       theme(legend.position = "none")
       
     plot(b)
     
   }
   
 }
 
 
 
}




hf_train %>%
  ggplot(aes())
  
#  Model----
## Model recipe----
hf_recipe <- recipe(death ~ ., data = hf_train) %>% 
  step_log(c(creatinine_phosphokinase, serum_creatinine)) %>% 
  step_normalize(c(platelets, age, serum_sodium, time, ejection_fraction, creatinine_phosphokinase)) %>% 
  step_smote(death, over_ratio = 1)

wf <- workflow() %>% 
  add_recipe(hf_recipe)


hf_prep <- prep(hf_recipe)
juiced <- juice(hf_prep)

tibble("before" = hf_train %>%
         count(death) %>%
         mutate(prop = n * 100 / sum(n)),
       "after" = juiced %>%
         count(death) %>%
         mutate(prop = n * 100 / sum(n)),
)

summary(juiced)
juiced %>% count(death) %>% 
  mutate(prop = n * 100 / sum(n))


#  Set metrics for classification
met <- metric_set(roc_auc, mcc, precision)

## Lasso ----
tune_lasso <- logistic_reg(penalty = tune(), 
                           mixture = 1, 
                           engine = "glmnet") 

lasso_grid <- tune_grid(
  add_model(wf, tune_lasso),
  resamples = train_folds,
  grid = grid_regular(
    penalty(),
    levels = 50),
  metrics = met
)

lasso_grid %>% 
  collect_metrics() %>% 
  ggplot(aes(penalty, mean, color = .metric)) +
  geom_line(size = 1.5) +
  geom_errorbar(aes(ymax = mean + std_err, 
                    ymin = mean - std_err),
                alpha = 0.5) +
  facet_wrap(~ .metric, nrow = 3, scales = "free") +
  scale_x_log10() +
  theme_bw() +
  theme(legend.position = "none")
  
# Extract best-perfroming model
best_roc_lasso <- lasso_grid %>% select_best("roc_auc")
best_mcc_lasso <- lasso_grid %>% select_best("mcc")

final_lasso_r <- finalize_workflow(wf %>% add_model(tune_lasso), best_roc_lasso)
final_lasso_m <- finalize_workflow(wf %>% add_model(tune_lasso), best_mcc_lasso)

# Check performance on test set
last_fit(final_lasso_m, hf_split, metrics = met) %>% 
  collect_metrics()

# Variable importance ----
final_lasso %>% 
  fit(hf_train) %>% 
  extract_fit_parsnip() %>% 
  vip::vi(lambda = best_roc_lasso$penalty) %>% 
  mutate(Variable = fct_reorder(Variable, Importance)) %>% 
  ggplot(aes(Importance, Variable, color = Sign)) +
  geom_point(size = 3)

## Random forest ----
tune_rf <- rand_forest(
  mode = "classification",
  engine = "ranger",
  trees = tune(),
  mtry = tune(), 
  min_n = tune()
)

# Parameter tuning rf----
rf_grid <- tune_grid(
  add_model(wf, tune_rf),
  resamples = train_folds,
  grid = grid_regular(
    trees(c(100, 200, 1000)),
    mtry(range = c(3, 10)),
    min_n(c(2, 5)),
    levels = 100),
  metrics = met
)

rf_grid %>% 
  collect_metrics() %>% 
  filter(.metric == "roc_auc") %>% 
  select(mean, min_n, mtry) %>% 
  pivot_longer(min_n:mtry, 
               values_to = "auc",
               names_to = "parameter") %>% 
  ggplot(aes(x = auc, y = mean, color = parameter)) +
  geom_point() +
  facet_wrap(~ parameter)

# Extract model with best performance
best_roc_rf <- rf_grid %>% select_best("roc_auc")

final_model_rf <- finalize_model(tune_rf,best_roc_rf)

# Variable importance
final_model_rf %>% 
  set_engine("ranger", importance = "permutation") %>% 
  fit(death ~ .,
      data = juice(hf_prep)) %>% 
  vip(geom = "point")

# Create workflow with best model
final_wf_rf <- workflow() %>% 
  add_recipe(hf_recipe) %>% 
  add_model(final_model_rf)

final_res_rf <- last_fit(final_wf_rf, hf_split, metrics = met) 

final_res_rf %>% collect_metrics()

## Support vector machines
tune_svm <- svm_rbf(
  mode = "classification", 
  engine = "kernlab",
  cost = tune(),
)

svm_grid <- tune_grid(
  add_model(wf, tune_svm),
  resamples = train_folds,
  grid = grid_regular(
    cost(),
    levels = 50
  )
)

best_roc_svm <- svm_grid %>% select_best("roc_auc")

final_svm <- finalize_workflow(add_model(wf, tune_svm),
                               best_roc_svm)

last_fit(final_svm, hf_split, metrics = met) %>% 
  collect_metrics()

