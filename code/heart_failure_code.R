# Packages
library(corrplot)
library(gridExtra)
library(Hmisc)
library(skimr)
library(tidymodels)
library(tidyverse)
library(vip)


# Data
hf <- read.csv("../data/heart_failure_clinical_records_dataset.csv")
str(hf)
skim(hf)

hf <- hf %>% 
  rename(death = DEATH_EVENT) %>% 
  mutate(death = as.factor(death))

# Train-test split
set.seed(123)
hf_split <- initial_split(hf, strata = death, prop = 0.7)
hf_train <- training(hf_split)
hf_test <- testing(hf_split)

train_folds <- vfold_cv(hf_train, v = 10, repeats = 2)

# Variable distributions
theme_set(theme_bw())

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

## Correlations
correlation_matrix <- hf_train %>%
  select(all_of(num_vars)) %>%
  cor(as.matrix(.), method = "pearson")



corrplot::corrplot(correlation_matrix, 
                   method = "square",
                   type = "lower",
                   order = "alphabet",
                   addCoef.col = 'black',
                   diag = FALSE)


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



#  Model
## Model recipe
hf_recipe <- recipe(death ~ ., data = hf_train) %>% 
  step_log(contains("creatinine")) %>%
  step_normalize(c(platelets, age, serum_creatinine, serum_sodium, time, ejection_fraction, creatinine_phosphokinase))

wf <- workflow() %>% 
  add_recipe(hf_recipe)

met <- metric_set(roc_auc, f_meas, mcc, precision)


## Lasso
tune_lasso <- logistic_reg(penalty = tune(), mixture = 1) %>% 
  set_engine("glmnet")

lasso_grid <- tune_grid(
  add_model(wf, tune_lasso),
  resamples = train_folds,
  grid = grid_regular(
    penalty(),
    levels = 50
  )
)

best_roc_lasso <- lasso_grid %>% select_best("roc_auc")

final_lasso <- finalize_workflow(add_model(wf, tune_lasso),
                                 best_roc_lasso)

last_fit(final_lasso, hf_split, metrics = met) %>% 
  collect_metrics()

### Variable importance
final_lasso %>% 
  fit(hf_train) %>% 
  extract_fit_parsnip() %>% 
  vip::vi(lambda = best_lasso$penalty)


## Random forest
tune_rf <- rand_forest(
  mode = "classification",
  engine = "ranger",
  trees = 100, 
  mtry = tune(), 
  min_n = tune()
)

rf_grid <- tune_grid(
  add_model(wf, tune_rf),
  resamples = train_folds,
  grid = grid_regular(
    mtry(range = c(3, 8)),
    min_n(),
    levels = 50
  )
)

best_roc_rf <- rf_grid %>% select_best("roc_auc")

final_rf <- finalize_workflow(add_model(wf, tune_rf),
                              best_roc_rf)

last_fit(final_rf, hf_split, metrics = met) %>% 
  collect_metrics()

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

