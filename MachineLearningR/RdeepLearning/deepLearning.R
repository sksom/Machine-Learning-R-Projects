# Deep learning using keras
library(keras)

# import data
data <- read.csv(file.choose(),header = T)
str(data)

# change data to matrix
data <- as.matrix(data)
dimnames(data) <- NULL
str(data) # num [1:2126, 1:22]

# Normalize data except the response variable
data[,1:21] <- normalize(data[,1:21])
data[,22] <- as.numeric(data[,22])-1
summary(data)

# data partition
set.seed(1234)
ind <- sample(2,nrow(data),replace = T,prob = c(0.7,0.3))
training <- data[ind==1,1:21]
test <- data[ind==2,1:21]
trainingTarget <- data[ind==1,22]
testTarget <- data[ind==2,22]

# one-hot encoding
trainLabels <- to_categorical(trainingTarget)
testLabels <- to_categorical(testTarget)
print(testLabels)

# model, units=8(experimantal), input_shape= 21, bcz 21 input variables, next units=3
# bcz three response classes 0,1,2
model <- keras_model_sequential()
model %>%
  layer_dense(units = 8, activation = "relu",input_shape = c(21)) %>%
  layer_dense(units = 3, activation = "softmax")
summary(model)

# compile, for only two response classes use "binary_crossentropy"
model %>%
  compile(loss = "categorical_crossentropy", optimizer = "adam", 
          metrics = "accuracy")

# fit the model
history1 <- model %>%
  fit(training,trainLabels,epochs = 200,batch_size = 32,validation_split = 0.2)
plot(history1)

# evaluate model for test data
model %>% evaluate(test,testLabels)

# Prediction and confusion matrix - test data
pred <- model %>%
  predict_classes(test)
prob <- model %>%
  predict_proba(test)
table(Predicted=pred,Actual=testTarget)
cbind(prob,Predicted=pred,Actual=testTarget)

# Fine tune our model let's add one more layer in the mid
model <- keras_model_sequential()
model %>%
  layer_dense(units = 100, activation = "relu",input_shape = c(21)) %>%
  layer_dense(units = 30,activation = "relu") %>%
  layer_dense(units = 3, activation = "softmax")
summary(model)

# compile, for only two response classes use "binary_crossentropy"
model %>%
  compile(loss = "categorical_crossentropy", optimizer = "adam", 
          metrics = "accuracy")

# fit the model
history1 <- model %>%
  fit(training,trainLabels,epochs = 200,batch_size = 32,validation_split = 0.2)
plot(history1)

# Prediction and confusion matrix - test data
pred <- model %>%
  predict_classes(test)
prob <- model %>%
  predict_proba(test)
table(Predicted=pred,Actual=testTarget)
cbind(prob,Predicted=pred,Actual=testTarget)

# So like this we can fine tune our model