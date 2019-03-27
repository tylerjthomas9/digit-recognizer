library(keras)
train <- as.matrix(read.csv("C:\\Users\\tyler\\Google Drive\\Kaggle projects\\digit-recognizer\\train.csv"))
test <- as.matrix(read.csv("C:\\Users\\tyler\\Google Drive\\Kaggle projects\\digit-recognizer\\test.csv"))
#train <- as.matrix(read.csv("/Users/tylerthomas/Google Drive/Kaggle projects/digit-recognizer/train.csv"))
#test <- as.matrix(read.csv("/Users/tylerthomas/Google Drive/Kaggle projects/digit-recognizer/test.csv"))

dim(train)


x_train <- train[,-1]
y_train <- train[,1]
x_test <- test


# reshape (The x data is a 3-d array (images,width,height))
x_train <- array_reshape(x_train, c(nrow(x_train), c(28,28,1)))
x_test <- array_reshape(x_test, c(nrow(x_test), c(28,28,1)))
# rescale
x_train <- x_train / 255
x_test <- x_test / 255
dim(y_train)


#recode 
y_train <- to_categorical(y_train, 10)


#import pretrained model
conv_base <- application_inception_v3(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(84, 84, 3)
)

conv_base
model <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu",
              input_shape =1*1* 2048) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = optimizer_rmsprop(lr = 2e-5),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history <- model %>% fit(
  x_train, y_train,
  epochs = 30,
  batch_size = 20,
)
