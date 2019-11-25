#############################################################
### Construct features and responses for training images  ###
#############################################################

feature <- function(input_list = fiducial_pt_list, index, train = TRUE){
  
  ### Construct process features for training images 
  
  ### Input: a list of images or fiducial points; index: train index or test index

  ### Output: a data frame containing: features and a column of label
  
  ### here is an example of extracting pairwise distances between fiducial points
  ### Step 1: Write a function pairwise_dist to calculate pairwise distance of items in a vector
  pairwise_dist <- function(vec) {
    ### input: a vector(length n), output: a vector containing pairwise distances(length n(n-1)/2)
    return(as.vector(dist(vec)))
  }
  
  ### Step 2: Write a function pairwise_dist_result to apply function in Step 1 to column of a matrix
  ### Instead of a len n(n-1) output, the output is simply n(n-1)/2 length vector of the euclidean distances between the ordered (x,y) pairs
  pairwise_dist_result <- function(mat){
    euclidean_distances <- NA
    mat_result <- as.vector(apply(mat, 2, pairwise_dist))
    for (i in 1:(length(mat_result)/2)) {
      euclidean_distances[i] <- round(sqrt(mat_result[i]^2 + mat_result[i + (length(mat_result)/2)]^2), 0)
    }
    return(euclidean_distances) 
  }
  
  ### Step 3: Apply function in Step 2 to selected index of input list, output: a feature matrix with ncol = n(n-1) = 78*77 = 6006
  pairwise_dist_feature <- t(sapply(input_list[index], pairwise_dist_result))
  
  ### Step 4: construct a dataframe containing features and label with nrow = length of index
  ### column bind feature matrix in Step 3 and corresponding features
  
  if ('emotion_idx' %in% colnames(info)) {
    pairwise_data <- cbind(pairwise_dist_feature, info$emotion_idx[index])
    pairwise_data <- as.data.frame(pairwise_data)
    colnames(pairwise_data) <- c(paste("feature", 1:(ncol(pairwise_data) - 1), sep = ""), 'emotion_idx')
    pairwise_data$emotion_idx <- as.factor(pairwise_data$emotion_idx)
  } else {
    pairwise_data <- as.dataframe(pairwise_dist_feature)
    colnames(pairwise_data) <- c(paste("feature", 1:ncol(pairwise_data), sep = ""))
  }

  return(feature_df = pairwise_data)
}
