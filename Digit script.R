#https://www.kaggle.com/c/digit-recognizer/data
library(keras)
#install_keras(tensorflow="gpu")

train <- as.matrix(read.csv("C:\\Users\\tyler\\Google Drive\\Kaggle projects\\digit-recognizer\\train.csv"))
test <- as.matrix(read.csv("C:\\Users\\tyler\\Google Drive\\Kaggle projects\\digit-recognizer\\test.csv"))
#train <- as.matrix(read.csv("/Users/tylerthomas/Google Drive/Kaggle projects/digit-recognizer/train.csv"))
#test <- as.matrix(read.csv("/Users/tylerthomas/Google Drive/Kaggle projects/digit-recognizer/test.csv"))


dim(train)


x_train <- train[,-1]
y_train <- train[,1]
x_test <- test


# reshape (The x data is a 3-d array (images,width,height))
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
class(x_train)
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
  epochs = 9, batch_size = 128, 
  validation_split = 0.2
)

plot(history)

model %>% evaluate(x_train, y_train)

predictions <- cbind(1:28000, model %>% predict_classes(x_test))
colnames(predictions) <- c("ImageID",	"Label")
write.csv(predictions, "C:\\Users\\tyler\\Google Drive\\Kaggle projects\\digit-recognizer\\keras1.csv", row.names=FALSE)


pred <- model %>% predict_classes(x_train)
wrong <- which(pred != train[,1])


##########################################################convolutional NN
library(keras)
train <- as.matrix(read.csv("C:\\Users\\tyler\\Google Drive\\Kaggle projects\\digit-recognizer\\train.csv", header=T))
test <- as.matrix(read.csv("C:\\Users\\tyler\\Google Drive\\Kaggle projects\\digit-recognizer\\test.csv", header=T))
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



#recode 
y_train <- to_categorical(y_train, 10)




#simple cnn
#layer_conv_2d(output_depth, c(window_height, window_width))
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu")

#flatten
model <- model %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")


#test CNN
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 512, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(28, 28, 1)) %>%
  layer_batch_normalization()%>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 1024, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 1024, kernel_size = c(3, 3), activation = "relu")


model <- model %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1024, activation = "relu") %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")


#adding classifier
model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)
model %>% fit(
  x_train, y_train,
  epochs = 10, batch_size=64
)

#test model training
model %>% fit(
  x_train, y_train,
  epochs = 10, batch_size=256
)
model %>% fit(
  x_train, y_train,
  epochs = 10, batch_size=1024
)
model %>% fit(
  x_train, y_train,
  epochs = 10, batch_size=5000
)

predictions <- cbind(1:28000, model %>% predict_classes(x_test))
colnames(predictions) <- c("ImageID",	"Label")
write.csv(predictions, "C:\\Users\\tyler\\Google Drive\\Kaggle projects\\digit-recognizer\\keras1.csv", row.names=FALSE)


#############################################################new method
set.seed(111)

if (!require("pacman")) install.packages("pacman") 

pacman::p_load(tidyverse, keras, tensorflow)


train <- as.matrix(read.csv("C:\\Users\\tyler\\Google Drive\\Kaggle projects\\digit-recognizer\\train.csv", header=T))
test <- as.matrix(read.csv("C:\\Users\\tyler\\Google Drive\\Kaggle projects\\digit-recognizer\\test.csv", header=T))


## Preprocess the data

train.label<-train[,1] %>% to_categorical()
train.feature<-train[,-1] %>% normalize()
test.feature<-test %>% normalize()
dim(train.feature)<-c(nrow(train.feature),28,28,1)
dim(test.feature)<-c(nrow(test.feature),28,28,1)



## Building A Simple Convolutional Neural Network
model<-keras_model_sequential()



model %>% 
  
  layer_conv_2d(filters = 32, kernel_size = c(5,5),padding = 'Valid',
                
                activation = 'relu', input_shape = c(28,28,1))%>%
  
  layer_batch_normalization()%>%
  
  layer_conv_2d(filters = 32, kernel_size = c(5,5),padding = 'Same',
                
                activation = 'relu')%>%
  
  layer_batch_normalization()%>%
  
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  
  layer_dropout(rate = 0.2) %>% 
  
  
  
  layer_conv_2d(filters = 64, kernel_size = c(3,3),padding = 'Same',
                
                activation = 'relu')%>%
  
  layer_batch_normalization()%>%
  
  layer_conv_2d(filters = 64, kernel_size = c(3,3),padding = 'Same',
                
                activation = 'relu')%>%
  
  layer_batch_normalization()%>%
  
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  
  layer_dropout(rate = 0.2) %>%
  
  
  
  layer_flatten() %>% 
  
  layer_dense(units=1024,activation='relu')%>%
  
  layer_dense(units=512,activation='relu')%>%
  
  layer_dense(units=256,activation='relu')%>%
  
  layer_dense(units=10,activation='softmax')



model%>%compile(
  
  loss='categorical_crossentropy',
  
  optimizer='adam',
  
  metrics='accuracy'
  
)


datagen <- image_data_generator(
  
  featurewise_center = F,
  
  samplewise_center=F,
  
  featurewise_std_normalization = F,
  
  samplewise_std_normalization=F,
  
  zca_whitening=F,
  
  horizontal_flip = F,
  
  vertical_flip = F,
  
  width_shift_range = 0.15,
  
  height_shift_range = 0.15,
  
  zoom_range = 0.15,
  
  rotation_range = .15,
  
  shear_range = 0.15
  
)

datagen %>% fit_image_data_generator(train.feature)



#training
history<-model %>%
  
  fit_generator(flow_images_from_data(train.feature, train.label, datagen, batch_size = 64),
                
                steps_per_epoch = nrow(train.feature)/64, epochs = 30)

plot(history)


model %>% fit(
  train.feature, train.label,
  epochs = 10, batch_size=128
)


predictions <- cbind(1:28000, model %>% predict_classes(test.feature))
colnames(predictions) <- c("ImageID",	"Label")
write.csv(predictions, "C:\\Users\\tyler\\Google Drive\\Kaggle projects\\digit-recognizer\\keras1.csv", row.names=FALSE)









model<-keras_model_sequential()


model %>% 
  
  layer_conv_2d(filters = 128, kernel_size = c(5,5),padding = 'Valid',
                
                activation = 'relu', input_shape = c(28,28,1))%>%
  
  layer_batch_normalization()%>%
  
  layer_conv_2d(filters = 128, kernel_size = c(5,5),padding = 'Same',
                
                activation = 'relu')%>%
  
  layer_batch_normalization()%>%
  
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  
  layer_dropout(rate = 0.2) %>% 
  
  
  
  layer_conv_2d(filters = 256, kernel_size = c(3,3),padding = 'Same',
                
                activation = 'relu')%>%
  
  layer_batch_normalization()%>%
  
  layer_conv_2d(filters = 256, kernel_size = c(3,3),padding = 'Same',
                
                activation = 'relu')%>%
  
  layer_batch_normalization()%>%
  
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  
  layer_dropout(rate = 0.2) %>%
  
  
  
  layer_flatten() %>% 
  
  layer_dense(units=1024,activation='relu')%>%
  
  layer_dense(units=512,activation='relu')%>%
  layer_dropout(rate = 0.2) %>%
  
  layer_dense(units=256,activation='relu')%>%
  
  layer_dense(units=10,activation='softmax')

