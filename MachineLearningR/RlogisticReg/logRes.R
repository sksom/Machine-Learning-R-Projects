# Logistic Regression
# Differnce between classification and Regression:
# for regression: response variable must be continuous
# for classification: response variable must be classes (factors basically)
library(nnet)

# getting data
data <- read.csv(file.choose(),header = T)
str(data)

# model 
mymodel <- multinom(admit~.,data=data)

# prediction
pred <- predict(mymodel,data)
tab <- table(Predicted=pred, Actual=data$admit)

# error rate
1- sum(diag(tab))/sum(tab)

# model performance evaluation
library(ROCR)
pred2 <- predict(mymodel,data, type = "prob")
pred2 <- prediction(pred2,data$admit)
eval <- performance(pred2, "acc")
plot(eval)
# insert horizontal and vertical line in the graph
abline(h=0.71,v=0.42)

# identifying best values on x and y axis in this graph
maxOnY <- which.max(slot(eval,"y.values")[[1]])
# accuracy at this index
acc <- slot(eval,"y.values")[[1]][maxOnY]
cutOff <- slot(eval,"x.values")[[1]][maxOnY]
print(c(Accuracy=acc, CutOff=cutOff))

# ROC curve gives us the best balance between true positive and false positive
pred3 <- predict(mymodel,data, type = "prob")
pred3 <- prediction(pred3,data$admit)
roc <- performance(pred3,"tpr","fpr")
plot(roc,
     colorize=T) # color is based on the cutoff values max is 0.72
abline(a=0,b=1)

# we can calculate area under the curve
auc <- performance(pred3, "auc")
auc <- unlist(slot(auc,"y.values"))
auc <- round(auc,4)
legend(0.7,0.4,auc,title = "AUC" ,cex = 1) # x-cordinate=0.7, y-cordinate=0.4,cex=size
