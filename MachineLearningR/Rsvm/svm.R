# SVM
# data
data("iris")
str(iris)
library(ggplot2)
qplot(Petal.Length,Petal.Width,data = iris,
      color=Species)

# svm library
library(e1071)
mymodel <- svm(Species~.,data=iris)
summary(mymodel) # SVM-Type:  C-classification , if response variable would have been 
# continuous then this would have been regression

plot(mymodel, data=iris,
     Petal.Width~Petal.Length,
     slice = list(Sepal.Width=3, Sepal.Length=4))

# confusion matrix and misclassification error
pred <- predict(mymodel,iris)
tab <- table(Predicted=pred,Actual=iris$Species)
tab
# accuracy can be predicted as (50+48+48)/(150), and error rate as 1-accuracy
1- sum(diag(tab))/sum(tab)

# our svm used radial kernel, we can also use linear kernel, or polynomial, or sigmoid
mymodel <- svm(Species~.,data=iris,
               kernel="sigmoid")
summary(mymodel) # SVM-Type:  C-classification , if response variable would have been 
# continuous then this would have been regression

plot(mymodel, data=iris,
     Petal.Width~Petal.Length,
     slice = list(Sepal.Width=3, Sepal.Length=4))

# confusion matrix and misclassification error
pred <- predict(mymodel,iris)
tab <- table(Predicted=pred,Actual=iris$Species)
tab
# accuracy can be predicted as (50+48+48)/(150), and error rate as 1-accuracy
1- sum(diag(tab))/sum(tab)

# best is radial kernel, worst is sigmoid kernel

# fine tune our svm
mymodel <- svm(Species~.,data=iris)
summary(mymodel) # SVM-Type:  C-classification , if response variable would have been 
# continuous then this would have been regression

plot(mymodel, data=iris,
     Petal.Width~Petal.Length,
     slice = list(Sepal.Width=3, Sepal.Length=4))

# TUNING OCCURS HERE
# epsilon: 0,0.1,0.2,0.3......1
# cost: 2^2,2^3,2^4,......2^9 Default is 1
set.seed(1234)
tmodel <- tune(svm, Species~.,data=iris,
     ranges = list(epsilon=seq(0,1,0.1),cost=2^(2:7)))   # don't do this for large
# data set because we can see the combinations value is 88 for this, time-consuming
plot(tmodel) # darker regions means better performance, i.e i can change cost to upto
# 2^7 and re run
summary(tmodel)

# choose best model
mymodel <- tmodel$best.model
summary(mymodel) # generally for classification type problems radial kernel is best
plot(mymodel, data=iris,
     Petal.Width~Petal.Length,
     slice = list(Sepal.Width=3, Sepal.Length=4))

# confusion matrix and misclassification error
pred <- predict(mymodel,iris)
tab <- table(Predicted=pred,Actual=iris$Species)
tab
# accuracy can be predicted as (50+48+48)/(150), and error rate as 1-accuracy
1- sum(diag(tab))/sum(tab)

# much better
