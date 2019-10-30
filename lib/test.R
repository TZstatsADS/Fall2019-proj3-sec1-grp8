########################################
### Classification with testing data ###
########################################

test <- function(model, dat_test){
  
  ### Input: 
  ###  - the fitted classification model using training data
  ###  - processed features from testing images 
  ### Output: training model specification
  X_test <- model.matrix(emotion_idx~., data = dat_test)
  
  probabilities <- as.data.frame(predict(cv_models, s = 'lambda.min', newx = X_test, type = 'response'))
  predictions <- as.factor(unname(sapply(X = colnames(probabilities)[apply(probabilities, 1, which.max)], as.integer)))
  return(predictions)
}
