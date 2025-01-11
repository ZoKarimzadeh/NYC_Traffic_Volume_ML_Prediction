
#   RIDGE_Regression from MACHINE LEARNING1 + Generalized Additive Model from Machine Learning two 
#   SCALED ALL NUMERIC COLUMNS
#   SAMPLE TAKEN 5%, REMOVES near zero variance + HIGH CORRELATION


# 1. Libraries

library(readr)       # READING DATASET
library(dplyr)       # DATA MANIPULATION
library(ggplot2)     # VISUALIZATION
library(glmnet)      # RIDGE_REGRESSION
library(mgcv)        # Generalized Additive Models
library(Matrix)      # SPARSE MATRICES
library(caret)       # HELPING FEATURE SELECTION



set.seed(42)  # reproducibility


# 2. Loading & Inspecting  the Data

data <- read_csv("NYC_Traffic_Data.csv")
cat("Summary of the dataset:\n")
print(summary(data))

counting_missing_values <- colSums(is.na(data))
cat("\nValues that are missing(by column):\n")
print(counting_missing_values)


# 3.Cleaning THE DATASET

# (a)  COLUMN_MEAN IS REPLACED BY THE VALUES THAT ARE MISSING 
data <- data %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# (b) AND CHANGING CHARACTER INTO FACTORS
data <- data %>%
  mutate(across(where(is.character), as.factor))

# 4. DATA INTO TRAINING , VALIDATION AND TESTING(SPLIT) 

SAMPLE_SET <- sample(1:nrow(data), nrow(data))
TRAING_SET  <- SAMPLE_SET[1:floor(0.6 * nrow(data))]
VALIDATION_SET  <- SAMPLE_SET[(floor(0.6 * nrow(data)) + 1):floor(0.8 * nrow(data))]
TESTING_SET   <- SAMPLE_SET[(floor(0.8 * nrow(data)) + 1):nrow(data)]

DATA_FOR_TRAINING <- data[TRAING_SET, ]
DATA_FOR_VALIDATION <- data[VALIDATION_SET, ]
DATA_FOR_TESTING  <- data[TESTING_SET, ]


# AT FIRST, WE REDUCED TO 5%(TRAINING DATA SET BECAUSE OF THE BIG DATA WE HAVE)

DATA_FOR_TRAINING <- DATA_FOR_TRAINING[sample(1:nrow(DATA_FOR_TRAINING), size = floor(0.05 * nrow(DATA_FOR_TRAINING))), ]


#NAs ARE BEING REMOVED

DATA_FOR_TRAINING <- DATA_FOR_TRAINING[complete.cases(DATA_FOR_TRAINING), ]
DATA_FOR_VALIDATION <- DATA_FOR_VALIDATION[complete.cases(DATA_FOR_VALIDATION), ]
DATA_FOR_TESTING  <- DATA_FOR_TESTING[complete.cases(DATA_FOR_TESTING), ]



# 6. SELECTING FEATURES


# 6a. HERE BY USING nearZeroVar WE REMOVE THE DATA WHICH HAVE NO VARIABILITY BECAUSE IT WILL MAKE ISSUES IN MODELING 
# WE ALSO ENSURED BY REMOVING FROM VALIDATION AND TESTING SETS

NEAR_ZERO_VARIANCE <- nearZeroVar(DATA_FOR_TRAINING, saveMetrics = TRUE)
if (any(NEAR_ZERO_VARIANCE$nzv)) {
  cat("\nRemoving COLUMS THAT HAVE VARIANCES CLOSE TO ZERO:\n")
  print(rownames(NEAR_ZERO_VARIANCE)[NEAR_ZERO_VARIANCE$nzv])
  
  
 # WE KEPT WHICH ARE NOT NEAR ZERO VARIANCE
  
  DATA_FOR_TRAINING <- DATA_FOR_TRAINING[, !NEAR_ZERO_VARIANCE$nzv, drop = FALSE]
  
 
  # CHECKING IN OTHER DATA SETS TO MATCH WITH TRAING 
  
  DATA_FOR_VALIDATION <- DATA_FOR_VALIDATION[, intersect(names(DATA_FOR_VALIDATION), names(DATA_FOR_TRAINING))]
  DATA_FOR_TESTING  <- DATA_FOR_TESTING[,  intersect(names(DATA_FOR_TESTING),  names(DATA_FOR_TRAINING))]
}


#6b. we  made a correleTION MATRIX AND DELETE THE DUPLICATES WHICH ARE HIGHLY CORRELated

colums_are_Numeric <- names(DATA_FOR_TRAINING)[sapply(DATA_FOR_TRAINING, is.numeric)]
if (length(colums_are_Numeric) > 1) {
  matrixCorrelation <- cor(DATA_FOR_TRAINING[colums_are_Numeric], use = "pairwise.complete.obs")
  correlationHIGH   <- findCorrelation(matrixCorrelation, cutoff = 0.8)
  
  if (length(correlationHIGH) > 0) {
    cat("\nRemoving", length(correlationHIGH), " HIGHLY CORRELATED(0.8)\n")
    colums_are_Numeric <- colums_are_Numeric[-correlationHIGH]
  }
}

# ALL NON NUMERIC AND NOT HIGHLY CORRELATED NUMERIC COLUMNS

DATA_FOR_TRAINING <- DATA_FOR_TRAINING[, c(
  colums_are_Numeric,
  setdiff(names(DATA_FOR_TRAINING), names(DATA_FOR_TRAINING)[sapply(DATA_FOR_TRAINING, is.numeric)])
)]
DATA_FOR_VALIDATION <- DATA_FOR_VALIDATION[, intersect(names(DATA_FOR_VALIDATION), names(DATA_FOR_TRAINING))]
DATA_FOR_TESTING  <- DATA_FOR_TESTING[,  intersect(names(DATA_FOR_TESTING),  names(DATA_FOR_TRAINING))]



#7. SCALING NUMERIC VALUES(ALL)

# a) TAKING means AND STANDARD DEVIATION from  training DATA set

ALL_NUMERIC_TRAING <- DATA_FOR_TRAINING %>% select(where(is.numeric))
MEANS_IN_TRAINING <- sapply(ALL_NUMERIC_TRAING, mean, na.rm = TRUE)
STANDARD_DEV_TRAININGG   <- sapply(ALL_NUMERIC_TRAING, sd, na.rm = TRUE)


# AVOIDING THE ISSUSE(DIVIDING BY ZERO)
STANDARD_DEV_TRAININGG[STANDARD_DEV_TRAININGG == 0] <- 1

DATAFRAME_SCALED <- function(df, means, sds) {
  colums_are_Numeric <- sapply(df, is.numeric)
  df[colums_are_Numeric] <- mapply(
    function(x, m, s) (x - m)/s,
    df[colums_are_Numeric], 
    means[names(df[colums_are_Numeric])], 
    sds[names(df[colums_are_Numeric])]
  )
  df
}

#(b) applying in all (from traing, to testing)

DATA_FOR_TRAINING <- DATAFRAME_SCALED(DATA_FOR_TRAINING, MEANS_IN_TRAINING, STANDARD_DEV_TRAININGG)
DATA_FOR_VALIDATION <- DATAFRAME_SCALED(DATA_FOR_VALIDATION, MEANS_IN_TRAINING, STANDARD_DEV_TRAININGG)
DATA_FOR_TESTING  <- DATAFRAME_SCALED(DATA_FOR_TESTING, MEANS_IN_TRAINING, STANDARD_DEV_TRAININGG)

# Double-checking

cat("\n'Vol' in DATA_FOR_TRAINING:\n")
str(DATA_FOR_TRAINING$Vol)


# 8. applying Ridge Regression ( from ML1) onthe that are already Scaled 

TRAINING_X <- sparse.model.matrix(Vol ~ ., data = DATA_FOR_TRAINING)[, -1]
TRAINING_Y <- DATA_FOR_TRAINING$Vol

TESTING_X  <- sparse.model.matrix(Vol ~ ., data = DATA_FOR_TESTING)[, -1]
TESTING_Y  <- DATA_FOR_TESTING$Vol

if (nrow(TRAINING_X) != length(TRAINING_Y)) stop("THERE is a  Mismatch: TRAINING_X / TRAINING_Y")
if (nrow(TESTING_X)  != length(TESTING_Y))  stop("There is a Mismatch: TESTING_X / TESTING_Y")

cat("\nRidge Regression (training on scaled data)...\n")
ridge_regression_cv <- cv.glmnet(TRAINING_X, TRAINING_Y, alpha = 0, nfolds = 5)
selecting_best_lambda <- ridge_regression_cv$lambda.min

model_ridge_regression <- glmnet(TRAINING_X, TRAINING_Y, alpha = 0, lambda = selecting_best_lambda)
pred_ridge_regression <- predict(model_ridge_regression, newx = TESTING_X)
MSE_ridge_regression   <- mean((TESTING_Y - pred_ridge_regression)^2)


# 9. Generalized Additive Model TAKEN FROM MACHINE EARNING 2 

# ADAPTING THE FORMULA to THE actual columns  


FORMULA_FOR_MODEL <- Vol ~ s(M, k = 5) + Boro

cat("\nTraining GAM to the  scaled data :\n")
print(FORMULA_FOR_MODEL)

GAM_MODEL <- gam(
  formula = FORMULA_FOR_MODEL,
  data    = DATA_FOR_TRAINING,
  method  = "REML"   
)

# PREDICTION ON TEST SET
PREDICTION_OF_GAM <- predict(GAM_MODEL, newdata = DATA_FOR_TESTING)
MSE_OF_GAM   <- mean((TESTING_Y - PREDICTION_OF_GAM)^2, na.rm = TRUE)


# 10. Comparing the  Models

cat("\nPerformance of the Model:\n")
cat("MSE of the Ridge Regression :", MSE_ridge_regression, "\n")
cat("MSE of the Generalized Additive Model:", MSE_OF_GAM, "\n")

# Visualization of the  Predictions
par(mfrow = c(1, 2))

plot(TESTING_Y, pred_ridge_regression, col = "blue", pch = 16,
     xlab = "Actual_Traffic_scaled", ylab = "Predicted _Traffic_scaled",
     main = "scaled_Ridge_Regression(Predictions)")
abline(0, 1, col = "red", lwd = 2)

plot(TESTING_Y, PREDICTION_OF_GAM, col = "green", pch = 16,
     xlab = "scaled and real Traffic ", ylab = "Predicted Traffic scaled",
     main = "scaled Generalized Additive Model Predictions ")
abline(0, 1, col = "red", lwd = 2)


# 11. Saving the  Results to a File

dataframe_results <- data.frame(
  Actual_Scaled       = as.vector(TESTING_Y),
  Ridge_Prediction    = as.vector(pred_ridge_regression),
  GAM_Prediction      = as.vector(PREDICTION_OF_GAM)
)
write.csv(dataframe_results, "Model_Results.csv", row.names = FALSE)

cat("\nAnalysis Complete. Results saved to 'Model_Results.csv'.\n")
