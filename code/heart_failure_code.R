# Packages
library(corrplot)
library(GGally)
library(gridExtra)
library(Hmisc)
library(skimr)
library(themis)
library(tidymodels)
library(tidyverse)
library(vip)


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
theme_set(theme_bw())

hf_train %>% 
  mutate(across(c(sex, smoking, anaemia, diabetes, high_blood_pressure), ~ as.factor(.))) %>% 
  ggpairs(switch = "y")


num_vars <- hf_train %>% 
  select(where(~ is.numeric(.x) && max(.x) > 1)) %>% 
  names()

plots <- list()

for (i in seq_along(num_vars)) {
  
  var <- num_vars[i]
  
  if (str_detect(var, "creatinine")) {
    p <- ggplot(data = hf_train, aes(x = log(!!sym(var))))
    
    q <-  ggplot(data = hf_train, aes(sample = log(!!sym(var)))) +
      stat_qq() +
      stat_qq_line()
    
  } else {
    p <- ggplot(data = hf_train, aes(x = !!sym(var)))
    
    q <-  ggplot(data = hf_train, aes(sample = !!sym(var))) +
      stat_qq() +
      stat_qq_line()
  }
  
  h <- p + 
    geom_histogram() +
    ggtitle(var)
  
  plots <- append(plots, list(h, q))
}

do.call("grid.arrange", c(plots, ncol = 4))

# ## Correlations----
# correlation_matrix <- hf_train %>%
#   select(all_of(num_vars)) %>%
#   cor(as.matrix(.), method = "pearson")
# 
# 
# 
# corrplot::corrplot(correlation_matrix, 
#                    method = "square",
#                    type = "lower",
#                    order = "alphabet",
#                    addCoef.col = 'black',
#                    diag = FALSE)


cat_vars <- hf_train %>% 
  select(c(death, where(~is.numeric(.) && max(.) == 1))) %>% 
  names()

cplots <- list()

for (i in seq_along(cat_vars)){
  
  var = cat_vars[i]
  
  # d <- hf_train %>% 
  #   count({{ var }}) %>% 
  #   mutate(prop = n / sum(n) * 100,
  #          label = paste0(n, " (", round(prop, 1), "%)"))
  
  p <- ggplot(hf_train, aes(x = !!sym(var),
                            group = !!sym(var),
                            fill = factor(!!sym(var)))) +
    geom_bar() +
    # geom_text(data = d, aes(label = label),
    #           position = position_stack(vjust = 0.5)) +
    theme(legend.position = "none") +
    ggtitle(var)
  
  cplots <- append(cplots, list(p))
}
do.call("grid.arrange", c(cplots, ncol = 2))



#  Model----
## Model recipe----
hf_recipe <- recipe(death ~ ., data = hf_train) %>% 
  step_BoxCox(serum_creatinine) %>% 
  step_log(creatinine_phosphokinase) %>% 
  step_normalize(c(platelets, age, serum_sodium, time, ejection_fraction, creatinine_phosphokinase)) %>% 
  step_smote(death, over_ratio = 0.8)

wf <- workflow() %>% 
  add_recipe(hf_recipe)


hf_prep <- prep(hf_recipe)
juiced <- juice(hf_prep)

summary(juiced)
juiced %>% count(death) %>% 
  mutate(prop = n * 100 / sum(n))


#  Set metrics for classification
met <- metric_set(roc_auc, mcc, recall)

# Create bootstraps
set.seed(2024)
hf_boot <- bootstraps(hf_train, times = 20)


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
best_recall_lasso <- lasso_grid %>% select_best("recall")

final_lasso <- 
  finalize_workflow(add_model(wf, tune_lasso),best_roc_lasso) %>% 
  fit(hf_train) %>% 
  extract_fit_parsnip() %>% 
  vi(lambda = best_roc_lasso$penalty)

# Check performance on test set
last_fit(final_lasso, hf_split, metrics = met) %>% 
  collect_metrics()

# Variable importance ----
final_lasso %>% 
  fit(hf_train) %>% 
  extract_fit_parsnip() %>% 
  vip::vi(lambda = best_roc_lasso$penalty)


## Random forest ----
tune_rf <- rand_forest(
  mode = "classification",
  engine = "ranger",
  trees = 100,
  mtry = tune(), 
  min_n = tune()
)

# Parameter tuning rf----
rf_grid <- tune_grid(
  add_model(wf, tune_rf),
  resamples = train_folds,
  grid = grid_regular(
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

