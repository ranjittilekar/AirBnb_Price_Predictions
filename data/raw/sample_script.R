# Load libraries

library(randomForest)
library(magrittr)
library(dplyr)

# Load data
path = ""
train <- read.csv(paste0(path,"train.csv"))
test <- read.csv(paste0(path,"test.csv"))
ids = test$id
test$log_price <- NA

all_data <- rbind(train,test)
train_set = 1:nrow(train)
test_set <- (nrow(train)+1):(nrow(train) + nrow(test))

# Select a subset of the data

keep_cols <- c('property_type','room_type','bed_type','cancellation_policy','city',
               'accommodates','bathrooms','latitude','longitude',
               'number_of_reviews','review_scores_rating','log_price')

all_data <- all_data[,keep_cols]

# Impute missing values with 0

fillna <- function(column) {
  column[is.na(column)] <- 0
  return(column)
}

col_type <- sapply(all_data,class)
numeric_type <- !(col_type %in% c("character","factor"))
all_data[,numeric_type] <- sapply(all_data[,numeric_type], fillna)

# Train a Random Forest model with cross-validation

cv_folds <- sample(1:3, size = nrow(train), replace = TRUE)

for(i in 1:3) {
  # Train the model using the training sets
  fit <- randomForest(log_price ~ .,
                      data = all_data[train_set[cv_folds !=i],],
                      ntree = 10)
  
  # Make predictions using the testing set
  preds <- predict(fit, all_data[train_set[cv_folds == i],])
  
  # Calculate RMSE for current cross-validation split
  print(mean((preds - all_data[train_set[cv_folds == i],'log_price'])^2)^.5)
}

# Create submission file

fit <- randomForest(log_price ~ ., data = all_data[train_set,], ntree = 1)
prediction <- predict(fit, all_data[test_set,])

sample_submission <- data.frame(id = ids, log_price = prediction)
write.csv(sample_submission, "sample_submission.csv", row.names = FALSE)
