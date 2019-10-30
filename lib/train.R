###########################################################
### Train a classification model with training features ###
###########################################################
train <- function(feature_df = pairwise_data, alpha = 1, n_folds = 5, cv_measure = 'class') {
  library(glmnet)
  ### Train a multinomial lasso model using processed features from training images
  ### With K fold cross validation selecting penalty term that minimizes classification error 
  
  ### Input:
  ### - a data frame containing features and labels
  ### - a parameter list
  ### Output: trained model
  
  X_train <- model.matrix(emotion_idx~., data = feature_df)
  Y_train <- feature_df$emotion_idx

  cv_out <- cv.glmnet(x = X_train,
                      y = Y_train,
                      alpha = alpha,
                      type.measure = cv_measure,
                      nfolds = n_folds,
                      family = 'multinomial')
  
  
  return(cv_out)
}

