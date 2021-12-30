import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from keras import models, layers

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

train_data = train_data.astype('float32') / 255
test_data = test_data.astype('float32') / 255

train_data = train_data.reshape(-1, 28, 28, 1)
test_data = test_data.reshape(-1, 28, 28, 1)


CNN_model = models.Sequential()

CNN_model.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
CNN_model.add(layers.BatchNormalization())
CNN_model.add(layers.Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', activation ='relu'))
CNN_model.add(layers.BatchNormalization())
CNN_model.add(layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
CNN_model.add(layers.BatchNormalization())
CNN_model.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
CNN_model.add(layers.BatchNormalization())
CNN_model.add(layers.Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', activation ='relu'))
CNN_model.add(layers.BatchNormalization())
CNN_model.add(layers.MaxPool2D(pool_size=(3,3)))
CNN_model.add(layers.Dropout(0.2))

CNN_model.add(layers.Dense(128, activation = "relu"))
CNN_model.add(layers.Flatten())
CNN_model.add(layers.Dense(64, activation = "relu"))
CNN_model.add(layers.BatchNormalization())
CNN_model.add(layers.Dense(10, activation = "softmax"))

if __name__ == "__main__":
    CNN_model.compile(optimizer = "rmsprop" , loss = "sparse_categorical_crossentropy", metrics=["accuracy"])
    CNN_model.fit(train_data, train_labels, epochs = 10, batch_size = 64, validation_data=(test_data, test_labels))

    CNN_model.save('classification_model.h5')