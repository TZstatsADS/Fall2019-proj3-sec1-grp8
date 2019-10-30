# Project: Can you recognize the emotion from an image of a face? 
<img src="figs/CE.jpg" alt="Compound Emotions" width="500"/>

(Image source: https://www.pnas.org/content/111/15/E1454)

### [Full Project Description](doc/project3_desc.md)

Term: Fall 2019

+ Team ## Group 8
+ Team members
  + Gao, Jason yg2583@columbia.edu
  + Hadzic, Samir sh3586@columbia.edu
  + He, Chongyu ch3379@columbia.edu
  + Zhang, Jerry jz2966@columbia.edu

[Presentation](https://github.com/TZstatsADS/Fall2019-proj3-sec1--proj3-sec1-grp8/blob/master/doc/Predictive%20Modelling.pptx)

### **Project summary**:  
+ In this project, we created an classification program that can recognize emotion from an image of a face. We 1) employed NMF and PCA as dimensionality reduction technique to improve classification accuracy and reduce model runtime, 2) implemented GBM as basline model, 3) implemented random forest, SVM and neural network, specifically, there are two parts in the neural network model, firstly we train a binary classification model, then for each category we train different models to classify corresponding emotions and 3) evaluated the performance gain of your proposed improvement against the baseline. We utilized tensorflow in python for improved model.
+ The accuracy rate for binary classifier and multiclass classifier in baseline GBM model are 0.65 and 0.37 respectively.For the improved SVM model the accuracy rate are 0.98 and 0.56 respectively, and the neural network model are 0.98 and 0.63 respectively, which are better than the baseline models.

### Contribution Statement
*Jerry Zhang* is responsible for the calculation of pairwise euclidean distance, implementation of neural network model, cross validation, and final evaluation.

*Chongyu He* is responsble for baseline GBM model, SVM, and hyperparameter tuning of NN model.

*Samir Hadzic* is responsble for ridge and lasso model shrinkage.

*Jason Gao* is responsible for PCA and NMF dimensionality reduction, organizing Readme file and final presentation.

Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.