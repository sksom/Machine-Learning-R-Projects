# Testing for naive bayes
# best for checking
#  Email: spam not spam
#  tweets: positive or negative
#  face recognition
#  classify news article

library(naivebayes)
library(dplyr)
library(ggplot2)
library(psych)

# Data
data <- read.csv(file.choose(),header = T)
str(data)
# cross tabulation between admit and rank i.e admit vs rank
xtabs(~admit+rank, data= data)
# convert admit and rank as factors bcz that's what they actually are
data$rank <- as.factor(data$rank)
data$admit <- as.factor(data$admit)
str(data)

# visualization
pairs.panels(data[,-1]) # since first one is the response variable
# not too much correlation between variables  gre vs gpa is 0.38

# let's create some box plot to check further correlation
data %>%
  ggplot(aes(x=admit, y=gpa, fill= admit))+
  geom_boxplot()+
  ggtitle("Box Plot")

data %>%
  ggplot(aes(x=gre, fill= admit))+
  geom_density(alpha=0.8,color= "black")+
  ggtitle("Density Plot")

data %>%
  ggplot(aes(x=gre, fill= admit))+
  geom_histogram(binwidth = 100)+
  ggtitle("GRE")

# start training, enough investigation
# data partition
set.seed(1234)
ind <- sample(2, nrow(data), replace = T, prob = c(0.8,0.2))
training <- data[ind==1,]
test <- data[ind==2,]

# naive bayes model
model <- naive_bayes(admit~.,data = training,usekernel = T)
model

# we can verify the data from the model
training %>%
  filter(admit=="0") %>%
  summarise(mean(gre),sd(gre))

plot(model)

# Predict
pred <- model %>%
  predict(training,type= "prob")
head(cbind(pred,training))

# confusion matrix - train data
pred2 <- model %>%
  predict(training)
tab1 <- table(Prediction=pred2,Actual=training$admit)
1- sum(diag(tab1))/sum(tab1)

# confusion matrix - test data
pred3 <- model %>%
  predict(test)
tab2 <- table(Prediction=pred3,Actual=test$admit)
1- sum(diag(tab2))/sum(tab2)   # 0.32, not that good

# kernel based densities may work better when numerical values are not normally 
# distributed, How useKernel=T, in model creation
# improved to 0.30
