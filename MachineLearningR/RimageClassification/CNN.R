# trying to perform image classification
source("http://bioconductor.org/biocLite.R")
biocLite()
biocLite("EBImage") 
library("EBImage")
library(keras)

# read images
pic1 <- c("p1.jpg","p2.jpg","p3.jpg","p4.jpg","p5.jpg",
          "c1.jpg","c2.jpeg","c3.jpg","c4.jpg","c5.jpg",
          "b1.jpg","b2.jpg","b3.jpg","b4.jpeg","b5.jpg")
train <- list()
for (i in 1:15) {
  train[[i]] <- readImage(pic1[i])
}

pic2 <- c("p6.jpg","c6.jpeg","b6.jpg")
test <- list()
for (i in 1:3) {
  test[[i]] <- readImage(pic2[i])
}

# explore
print(train[[1]])
display(train[[1]])
summary(train[[1]])
hist(train[[1]])
str(train)

# plotting all images at once
par(mfrow=c(3,5))
for (i in 1:15) {
  plot(train[[i]])
}

# restoring the plot settings
par(mfrow=c(1,1))

# all pics have different dimensions, let's re-size them
for (i in 1:15) {
  train[[i]] <- resize(train[[i]],100,100)
}
str(train)
for (i in 1:3) {
  test[[i]] <- resize(test[[i]],100,100)
}
str(test)

train <- combine(train)
str(train) # num [1:100, 1:100, 1:3, 1:15]  1:15 bcz of 15 images
# why combine(train) bcz our CNN need data as 15x100x100x3
x <- tile(train,5)
display(x,title="Pictures")

test <- combine(test)
y <- tile(test,3)
display(y,title="Pics")

# to supply into CNN we need to reorder our data by shuffling the dimensions above
# 15 bcmes 1st dim, then 1,2,3
train <- aperm(train, c(4,1,2,3))
test <- aperm(test,c(4,1,2,3))
str(train)  # reordered : [1:15, 1:100, 1:100, 1:3]

# response
trainy <- c(0,0,0,0,0,1,1,1,1,1,2,2,2,2,2)
testy <- c(0,1,2)

# One-hot encoding
trainLabels <- to_categorical(trainy)
testLabels <- to_categorical(testy)

# model
model <- keras_model_sequential()
model %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3),activation = "relu",
                input_shape = c(100,100,3)) %>% 
# as we can see here I specified dim for only 1 image nd not the no of images
  layer_conv_2d(filters = 32, kernel_size = c(3,3),activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3),activation = "relu") %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3),activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 3, activation = "softmax") %>%
  compile(loss = "categorical_crossentropy",
          optimizer = optimizer_sgd(lr=0.01,decay = 1e-6, momentum = 0.9,
                                    nesterov = T),
          metrics = c("accuracy"))

summary(model)

# fit model
history1 <- model %>%
  fit(train,trainLabels,epochs = 60, batch_size = 32, validation_split = 0.2)
plot(history1)

# evaluation and prediction - train data
model %>% evaluate(train, trainLabels)
pred <- model %>% predict_classes(train)
# confusion matrix
table(Prediction=pred,Actual=trainy)

prob <- model %>% predict_proba(train)
cbind(prob, Predicted =pred, Actual =trainy)

# evaluation and prediction - test data
model %>% evaluate(test, testLabels)
pred2 <- model %>% predict_classes(test)
# confusion matrix
table(Prediction=pred2,Actual=testy)

prob2 <- model %>% predict_proba(test)
cbind(prob2, Predicted =pred2, Actual =testy)
