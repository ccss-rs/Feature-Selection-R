# Dimensionality Reduction Series
# Part I: Feature Selection Workshop Code

## Data Prep ----

## Set Seed
set.seed(31415)

## Load Libraries
# install.packages()
library(tidyverse)
library(ggplot2)
library(stats)
library(caret)
library(haven)
library(leaps)
library(glmnet)
library(corrplot)

## Load & Examine Dataset
data <- read_csv("labor_market_discrimination.csv")

summary(data)
#attributes(data)


## Filter Method ----
colnames(data)
numeric_vars <- dplyr::select(data, "call", "n_jobs","years_exp", "frac_black","frac_white", 
                              "l_med_hh_inc","frac_dropout","frac_colp","l_inc","parent_sales",
                              "parent_emp","branch_sales","branch_emp","frac_black_emp_zip",   
                              "frac_white_emp_zip","l_med_hh_inc_emp_zip","frac_dropout_emp_zip","frac_colp_emp_zip","l_inc_emp_zip") %>% drop_na()
numeric_vars = as.data.frame(numeric_vars)


# Compute correlation matrix
cormatrix <- stats::cor(numeric_vars)
View(cormatrix)

## Can also use caret::filterVarImp() to measure and rank feature importance

## Wrapper Methods ----

#### Forward Stepwise Selection ----
fit_fwd = regsubsets(call ~ ., data = numeric_vars, nvmax = 19, method = "forward")
fit_fwd_sum = summary(fit_fwd)
fit_fwd_sum

#### Backward Stepwise Selection ----
## Can also use Recursive Feature Elimination using rfe() from the caret package
fit_bwd = regsubsets(call ~ ., data = numeric_vars, nvmax = 19, method = "backward")
fit_bwd_sum = summary(fit_bwd)
fit_bwd_sum

#### Exhaustive Selection  ----
fit_ex = regsubsets(call ~ ., data = numeric_vars, nvmax = 19, method = "exhaustive")
fit_ex_sum = summary(fit_ex)
fit_ex_sum

#### Sequential Replacement Selection  ----
fit_seq = regsubsets(call ~ ., data = numeric_vars, nvmax = 19, method = "seqrep")
fit_seq_sum = summary(fit_seq)
fit_seq_sum

#### Best subset selection ----
#### The regsubsets() function (part of the leaps library) performs best subset selection by identifying the best model that contains a given number of predictors, where best is quantified using RSS. 
fit_all = regsubsets(call ~ ., data = numeric_vars, nvmax = 19)
fit_all_sum = summary(fit_all)
fit_all_sum

# prints attributes/results calculated
names(fit_all_sum)
fit_all_sum$bic
# fit_all_sum$aic
fit_all_sum$rss
fit_all_sum$adjr2
fit_all_sum$cp

### Evaluating Feature Selection Performance  ----
par(mfrow = c(2, 2))
plot(fit_all_sum$rss, xlab = "Number of Variables", ylab = "RSS", type = "b")
best_rss = which.min(fit_all_sum$rss)
points(best_rss, fit_all_sum$rss[best_rss],
       col = "red",cex = 2, pch = 20)

plot(fit_all_sum$adjr2, xlab = "Number of Variables", ylab = "Adjusted RSq", type = "b")
best_adj_r2 = which.max(fit_all_sum$adjr2)
points(best_adj_r2, fit_all_sum$adjr2[best_adj_r2],
       col = "red",cex = 2, pch = 20)

plot(fit_all_sum$cp, xlab = "Number of Variables", ylab = "Cp", type = 'b')
best_cp = which.min(fit_all_sum$cp)
points(best_cp, fit_all_sum$cp[best_cp], 
       col = "red", cex = 2, pch = 20)

plot(fit_all_sum$bic, xlab = "Number of Variables", ylab = "BIC", type = 'b')
best_bic = which.min(fit_all_sum$bic)
points(best_bic, fit_all_sum$bic[best_bic], 
       col = "red", cex = 2, pch = 20)

## Embedded Methods ----
#### LASSO ----
# alpha: Lasso(1) or Ridge regression(0) 
# nlambda:the number of values to try for lambda, glmnet will automatically pick which specific values

numeric_vars <- as.data.frame(numeric_vars)
dependent_vars <- as.matrix(select(numeric_vars,-call))

lasso_fit <- glmnet(x=dependent_vars, y=numeric_vars$call, family = "gaussian")
par(mfrow = c(1, 1))
plot(lasso_fit)

# Get the specific values which glmnet chose for lambda 
lasso_fit$lambda

# specifying "lambda" tells R to put log(lambda) on the horizontal axis 
plot(lasso_fit,"lambda")

# gets coefs based on specific lambda's 
# REVIEW OUTPUT FOR ACCURACY
coef(lasso_fit,s=.05)
coef(lasso_fit,s=.2)

# use cross validation to choose an appropriate λ
# nfolds:specifies how many folds to use for K-fold cross validation 
cv_lasso_fit<-cv.glmnet(x=dependent_vars, y=numeric_vars$call,alpha=1,nlambda=20,nfolds=10)
cv_lasso_fit

# The two vertical lines on the plot are the values of lambda which has the lowest CV error 
plot(cv_lasso_fit)
cv_lasso_fit$lambda.min

# the largest value of lambda which has a CV error within 1SE of the lowest  CV error 
cv_lasso_fit$lambda.1se

# use lasso_fit$lambda.min to get the lambda value with the best cverror 
# REVIEW OUTPUTS FOR ACCURACY
coef(cv_lasso_fit,s=cv_lasso_fit$lambda.min)
coef(cv_lasso_fit,s=cv_lasso_fit$lambda.1se)

# the predict function takes the model we fit 'cv_lasso_fit' 
# using the 's' argument, we specify which value of lambda we want to use 
# we could use either the lambda with the smallest CV error or the largest lambda 
# with CV error within 1se of the lowest CV error 
# new x is the covariates for which we are making predictions. In this case, it's the test data 
lasso_predicted_min <-predict(cv_lasso_fit, s=cv_lasso_fit$lambda.min, newx=dependent_vars, type="response") 
lasso_predicted_min

lasso_predicted_se <-predict(cv_lasso_fit, s=cv_lasso_fit$lambda.1se, newx=dependent_vars, type="response")
lasso_predicted_se

coef(cv_lasso_fit, s=cv_lasso_fit$lambda.min)

# We can calculate the mean squared prediction error on test data 
lasso_test_error <- mean((numeric_vars$call - lasso_predicted_min)^2)
lasso_test_error

# To do the same process with ridge regression instead of LASSO, use alpha = 0 in the glmnet call


