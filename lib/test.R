########################################
### Classification with testing data ###
########################################

test <- function(model, dat_test){
  
  ### Input: 
  ###  - the fitted classification model using training data
  ###  - processed features from testing images 
  ### Output: training model specification
  if ('emotion_idx' %in% colnames(dat_test)) {
    X_test <- model.matrix(emotion_idx~., dat_test)
  } else {
    dat_test$emotion_idx <- rep(1, nrow(dat_test))
    X_test <- model.matrix(emotion_idx ~., dat_test)
  }
  probabilities <- as.data.frame(predict(model, s = 'lambda.min', newx = X_test, type = 'response'))
  predictions <- as.factor(unname(sapply(X = colnames(probabilities)[apply(probabilities, 1, which.max)], as.integer)))
  return(predictions)
}
