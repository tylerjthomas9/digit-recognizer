#https://www.kaggle.com/c/digit-recognizer/data
library(keras)


train <- as.matrix(read.csv("C:\\Users\\tyler\\Google Drive\\Kaggle projects\\digit-recognizer\\train.csv"))
test <- as.matrix(read.csv("C:\\Users\\tyler\\Google Drive\\Kaggle projects\\digit-recognizer\\test.csv"))
dim(train)


x_train <- train[,-1]
y_train <- train[,1]
x_test <- test


# reshape (The x data is a 3-d array (images,width,height))
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
# rescale
x_train <- x_train / 255
x_test <- x_test / 255



#recode y
y_train <- to_categorical(y_train, 10)




#defining the model
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model)

#compile the model with appropriate loss function, optimizer, and metrics:
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)


history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)

plot(history)

model %>% evaluate(x_train, y_train)

predictions <- cbind(1:28000, model %>% predict_classes(x_test))
colnames(predictions) <- c("ImageID",	"Label")
write.csv(predictions, "C:\\Users\\tyler\\Google Drive\\Kaggle projects\\digit-recognizer\\keras1.csv", row.names=FALSE)


pred <- model %>% predict_classes(x_train)
wrong <- which(pred != train[,1])


