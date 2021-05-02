# load libraries
library(tidyverse)
library(tensorflow)
library(keras)
getwd()
# path to image folders
train_image_file<- "fruits-360/Training"
# list of fruits to model
fruit_list <- c("Kiwi", "Banana", "Apricot", "Avocado", "Cocos", "Clementine", "Mandarine", "Orange",
                "Limes", "Lemon", "Peach", "Plum", "Raspberry", "Strawberry", "Pineapple", "Pomegranate")
# number of fruit classes (i.e. fruits)
number_of_class <- length(fruit_list)
number_of_class

#Train : Validation = 8:2
train_data_generator <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2)


# training images
train_image_array <- flow_images_from_directory(train_image_file, 
                                                    train_data_generator,
                                                    subset = 'training',
                                                    target_size = c(32,32),
                                                    class_mode = "categorical",
                                                    classes = fruit_list,
                                                    batch_size = 32,
                                                    seed = 123)
#Found 6171 images

# validation images
valid_image_array <- flow_images_from_directory(train_image_file, 
                                                    train_data_generator,
                                                    subset = 'validation',
                                                    target_size = c(32,32),
                                                    class_mode = "categorical",
                                                    classes = fruit_list,
                                                    batch_size = 32,
                                                    seed = 123)
#Found 1538 images

#Number of images in each classes
table(train_image_array$classes)

table(valid_image_array$classes)


# number of training samples
train_samples <- train_image_array$n
train_samples
# number of validation samples
valid_samples <- valid_image_array$n

# define number of epochs
epochs <- 10


model <- keras_model_sequential()
#padding = "same" --> apply padding --> restrict reducing the image size. 
model %>%
  layer_conv_2d(filter = 16, kernel_size = c(3,3), padding = "same", input_shape = c(32, 32, 3), activation = "relu") %>%
  layer_batch_normalization() %>%

  # Second hidden layer
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same", activation = "relu") %>%
  layer_batch_normalization() %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
   #Bath norm vs Dropout
  #Batch increase to speed up optimization
  #dropout -> to control the overfitting 
  
  
  # Flatten max filtered output into feature vector and feed into dense layer
  layer_flatten() %>%
  #layer_flatten() is converting the data into a 1-d array for inputting it to the next layer. 
  #we flatten the output of the convolutional layers to create a single long feature vector. 
  layer_dense(128) %>%
  #layer_dense = fully connected layer
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  #dropout prevents overfitting the model during training
  
  # Outputs from dense layer are projected onto output layer
  layer_dense(number_of_class) %>% 
  layer_activation("softmax")
#softmax is usually used in last layer for normalizing the probability in order to decision. 
#sigmoid is used for binary classification 
#softmax is used for multiple classification 
# compile
model %>% compile(
  loss = "categorical_crossentropy",
  #optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  #optimizer = 'adam', --> loss:0.116, ac : 0.968
  optimizer = 'sgd',
  metrics = "accuracy"
)

model_fit <- model %>% fit(
  # training data
  train_image_array,
  steps_per_epoch = as.integer(train_samples / 32), 
  
  # validation data
  validation_data = valid_image_array,
  validation_steps = as.integer(valid_samples / 32)
)
#reticulate::py_install("pillow")
#reticulate::py_install("Scipy")
plot(model_fit)


#TEST SET
test_image_file <- "fruits-360/Test"
test_data_generator <- image_data_generator(rescale = 1/255)
test_generator <- flow_images_from_directory(
  test_image_file,
  test_data_generator,
  target_size = c(32,32),
  class_mode = "categorical",
  classes = fruit_list,
  batch_size = 1,
  shuffle = FALSE,
  seed = 123)
#Found 2592 images

model %>%
  evaluate(test_generator, 
                     steps = as.integer(test_generator$n))
