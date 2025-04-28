library(mlbench)
library(purrr)

data("PimaIndiansDiabetes2")
ds <- as.data.frame(na.omit(PimaIndiansDiabetes2))
## fit a logistic regression model to obtain a parametric equation
logmodel <- glm(diabetes ~ .,
                data = ds,
                family = "binomial")
summary(logmodel)

cfs <- coefficients(logmodel) ## extract the coefficients
prednames <- variable.names(ds)[-9] ## fetch the names of predictors in a vector
prednames

sz <- 100000000 ## to be used in sampling
##sample(ds$pregnant, size = sz, replace = T)

dfdata <- map_dfc(prednames,
                  function(nm){ ## function to create a sample-with-replacement for each pred.
                    eval(parse(text = paste0("sample(ds$",nm,
                                             ", size = sz, replace = T)")))
                  }) ## map the sample-generator on to the vector of predictors
## and combine them into a dataframe

names(dfdata) <- prednames
dfdata

class(cfs[2:length(cfs)])

length(cfs)
length(prednames)
## Next, compute the logit values
pvec <- map((1:8),
            function(pnum){
              cfs[pnum+1] * eval(parse(text = paste0("dfdata$",
                                                     prednames[pnum])))
            }) %>% ## create beta[i] * x[i]
  reduce(`+`) + ## sum(beta[i] * x[i])
  cfs[1] ## add the intercept

## exponentiate the logit to obtain probability values of thee outcome variable
dfdata$outcome <- ifelse(1/(1 + exp(-(pvec))) > 0.5,
                         1, 0)

library(xgboost)
library(caret)

# Define dataset sizes to evaluate
dataset_sizes <- c(100, 1000, 10000, 100000, 1000000, 10000000)

# Create a results table
results <- data.frame(
  dataset_size = dataset_sizes,
  accuracy = numeric(length(dataset_sizes)),
  time_taken = numeric(length(dataset_sizes))
)

# Run evaluation for each dataset size
for(i in seq_along(dataset_sizes)) {
  size <- dataset_sizes[i]
  cat("\nEvaluating dataset size:", size, "\n")
  
  # Sample from the generated dataset
  set.seed(123)
  if(size <= nrow(dfdata)) {
    indices <- sample(1:nrow(dfdata), size)
    sample_data <- dfdata[indices, ]
  } else {
    indices <- sample(1:nrow(dfdata), size, replace = TRUE)
    sample_data <- dfdata[indices, ]
  }
  
  # Split data into training (70%) and testing (30%) sets
  train_indices <- createDataPartition(sample_data$outcome, p = 0.7, list = FALSE)
  train_data <- sample_data[train_indices, ]
  test_data <- sample_data[-train_indices, ]
  
  # Prepare data for XGBoost
  train_matrix <- as.matrix(train_data[, -ncol(train_data)])
  train_label <- train_data$outcome
  test_matrix <- as.matrix(test_data[, -ncol(test_data)])
  test_label <- test_data$outcome
  
  dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
  dtest <- xgb.DMatrix(data = test_matrix, label = test_label)
  
  # Set XGBoost parameters
  params <- list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    eta = 0.1,
    max_depth = 6
  )
  
  # Record start time
  start_time <- Sys.time()
  
  # Train model with cross-validation
  cv_model <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 100,
    nfold = 5,
    early_stopping_rounds = 10,
    verbose = 0
  )
  
  # Get optimal number of rounds
  best_nrounds <- cv_model$best_iteration
  
  # Train final model
  model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = best_nrounds,
    watchlist = list(train = dtrain, test = dtest),
    verbose = 0
  )
  
  # Record end time
  end_time <- Sys.time()
  time_taken <- difftime(end_time, start_time, units = "secs")
  
  # Make predictions
  predictions <- predict(model, dtest)
  predictions_binary <- ifelse(predictions > 0.5, 1, 0)
  
  # Calculate accuracy
  accuracy <- mean(predictions_binary == test_label)
  
  # Store results
  results$accuracy[i] <- accuracy
  results$time_taken[i] <- as.numeric(time_taken)
  
  # Print results
  cat("Method: XGBoost with simple cross-validation\n")
  cat("Dataset size:", size, "\n")
  cat("Testing-set predictive performance (accuracy):", round(accuracy, 4), "\n")
  cat("Time taken for model fitting:", round(as.numeric(time_taken), 2), "seconds\n")
}

print(results)

formatted_results <- data.frame(
   "Method used" = rep("XGBoost in R – direct use of xgboost() with simple cross-validation", length(dataset_sizes)),
  "Dataset size" = format(results$dataset_size, big.mark = ","),
  "Testing-set predictive performance" = round(results$accuracy, 4),
  "Time taken for the model to be fit" = paste0(round(results$time_taken, 2), " seconds")
)

print(formatted_results)


library(caret)
library(xgboost)

# Define dataset sizes to evaluate
dataset_sizes <- c(100, 1000, 10000, 100000, 1000000, 10000000)

# Create a results table
results <- data.frame(
  dataset_size = dataset_sizes,
  accuracy = numeric(length(dataset_sizes)),
  time_taken = numeric(length(dataset_sizes))
)

# Run evaluation for each dataset size
for(i in seq_along(dataset_sizes)) {
  size <- dataset_sizes[i]
  cat("\nEvaluating dataset size:", size, "\n")
  
  # Sample from the generated dataset
  set.seed(123)
  if(size <= nrow(dfdata)) {
    indices <- sample(1:nrow(dfdata), size)
    sample_data <- dfdata[indices, ]
  } else {
    indices <- sample(1:nrow(dfdata), size, replace = TRUE)
    sample_data <- dfdata[indices, ]
  }
  
  # Convert outcome to factor with proper level names for caret
  sample_data$outcome <- factor(sample_data$outcome, 
                                levels = c(0, 1), 
                                labels = c("neg", "pos"))
  
  # Split data into training (70%) and testing (30%) sets
  train_indices <- createDataPartition(sample_data$outcome, p = 0.7, list = FALSE)
  train_data <- sample_data[train_indices, ]
  test_data <- sample_data[-train_indices, ]
  
  # Define 5-fold cross-validation
  control <- trainControl(
    method = "cv",
    number = 5,
    verboseIter = FALSE,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  )
  
  # Set XGBoost parameters
  xgb_grid <- expand.grid(
    nrounds = 100,
    eta = 0.1,
    max_depth = 6,
    gamma = 0,
    colsample_bytree = 0.8,
    min_child_weight = 1,
    subsample = 0.8
  )
  
  # Record start time
  start_time <- Sys.time()
  
  # Train model using caret
  set.seed(123)
  xgb_model <- train(
    outcome ~ .,
    data = train_data,
    method = "xgbTree",
    trControl = control,
    tuneGrid = xgb_grid,
    metric = "ROC"
  )
  
  # Record end time
  end_time <- Sys.time()
  time_taken <- difftime(end_time, start_time, units = "secs")
  
  # Make predictions
  predictions <- predict(xgb_model, test_data)
  
  # Calculate accuracy
  cm <- confusionMatrix(predictions, test_data$outcome)
  accuracy <- cm$overall["Accuracy"]
  
  # Store results
  results$accuracy[i] <- accuracy
  results$time_taken[i] <- as.numeric(time_taken)
  
  # Print results
  cat("Method: XGBoost via caret with 5-fold CV\n")
  cat("Dataset size:", size, "\n")
  cat("Testing-set predictive performance (accuracy):", round(accuracy, 4), "\n")
  cat("Time taken for model fitting:", round(as.numeric(time_taken), 2), "seconds\n")
}

# Print final results table
print(results)

# Format table for better presentation
formatted_results <- data.frame(
  "Method used" = rep("XGBoost in R – via caret, with 5-fold CV simple cross-validation", length(dataset_sizes)),
  "Dataset size" = format(results$dataset_size, big.mark = ","),
  "Testing-set predictive performance" = round(results$accuracy, 4),
  "Time taken for the model to be fit" = paste0(round(results$time_taken, 2), " seconds")
)

# Print formatted table
print(formatted_results)