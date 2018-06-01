# trying to perform image classification
source("http://bioconductor.org/biocLite.R")
biocLite()
biocLite("EBImage") 
library("EBImage")
library(keras)

# read images
pics <- c("p1.jpg","p2.jpg","p3.jpg","p4.jpg","p5.jpg","p6.jpg",
          "c1.jpg","c2.jpeg","c3.jpg","c4.jpg","c5.jpg","c6.jpeg")
mypic <- list()
for (i in 1:12) {
  mypic[[i]] <- readImage(pics[i])
}

# explore
print(mypic[[1]])
display(mypic[[1]])
summary(mypic[[1]])
hist(mypic[[1]])
str(mypic)

# all pics have different dimensions, let's re-size them
for (i in 1:12) {
  mypic[[i]] <- resize(mypic[[i]],28,28)
}
str(mypic)

#Reshaping the images as single vector of 28*28*3= 2352
for (i in 1:12) {
  mypic[[i]] <- array_reshape(mypic[[i]],c(28,28,3))
}
str(mypic)

# row bind
trainx <- NULL
# first five planes for train 1 for test 
for (i in 1:5) {
  trainx <- rbind(trainx,mypic[[i]])
}
# first five cars for train 1 for test
for (i in 7:11) {
  trainx <- rbind(trainx,mypic[[i]])
}
str(trainx)
# now testing images
testx <- rbind(mypic[[6]],mypic[[12]])
str(testx)
# response variable creation 0-plane, 1-car
trainy <- c(0,0,0,0,0,1,1,1,1,1)
testy <- c(0,1)

# one hot encoding i.e. converting response variable to categorical
trainLabels <- to_categorical(trainy)
testLabels <- to_categorical(testy)

# model, %>% it's a pipe operator which passes everything on it's left to it's right
# i.e anything given to model will be passed on to the layers
model <- keras_model_sequential()
# designing fully connected NN
# units: no of neurons in hidden layer, input_shape: no of neurons in input layer
# which must be equal to the number of pixels/features in each row 
# we have 3 layers as a typical NN has
model %>%
  layer_dense(units=256,activation = "relu", input_shape = c(2352)) %>%
  layer_dense(units = 128,activation = "relu") %>%
  layer_dense(units = 2, activation = "softmax")
# only two in the last bcz we hv only two classes to predict, 0 and 1
summary(model)

# compile, binary_crossentropy bcz we hv only 2 response variables
model %>%
  compile(loss = "binary_crossentropy",
          optimizer = optimizer_rmsprop(),
          metrics = c("accuracy"))

# fit model
history1 <- model %>%
  fit(trainx,
      trainLabels,
      epochs = 30,
      batch_size = 32,
      validation_split = 0.2)
plot(history1)

# Evaluation and prediction - training data
model %>%
  evaluate(trainx,trainLabels)
pred <- model %>% predict_classes(trainx)
table(Prediction=pred,Actual=trainy)
prob <- model %>% predict_proba(trainx)
cbind(prob, Predicted= pred, Actual= trainy)

# so model predicts 9th pic wrong
display(mypic[[9]])

# Evaluation and prediction - test data
model %>%
  evaluate(testx,testLabels)
pred2 <- model %>% predict_classes(testx)
table(Prediction=pred2,Actual=testy)
prob2 <- model %>% predict_proba(testx)
cbind(prob2, Predicted= pred2, Actual= testy)
