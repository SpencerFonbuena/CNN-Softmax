import tensorflow as tf
import numpy as np
from PIL import Image
import csv 
import mplfinance as mpf
import pandas as pd
from matplotlib import pyplot as plt
import os


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data = tf.keras.utils.image_dataset_from_directory('Images', image_size=(512,512), label_mode='categorical')

#standardizes and allows us to use the data papeline
data = data.map(lambda x, y: (x/255, y))
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

#partition of portions of data to use for train/dev/test sets
train_size = int(len(data) * .95) # 95 percent training set
dev_size = int(len(data) * .025) # 2.5 percent development set
test_size = int(len(data) * .025) # 2.5 percent test set

#Actually take each data batch and process it
train = data.take(train_size)
development = data.skip(train_size).take(dev_size)
test = data.skip(train_size + dev_size).take(test_size)

#initialize the type of model it is
model = tf.keras.Sequential()

#Model Infrastructure | 150 million parameters
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1 ,padding='same', activation='relu', input_shape=(512,512,3) )) #(512,512,64)
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1 ,padding='same', activation='relu')) #(512,512,64)
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)) #(256,256,64)

model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1 ,padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))) #(256,256,128)
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1 ,padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))) #(256,256,128)
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)) #(128,128,128)

model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1 ,padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))) #(128,128, 256)
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1 ,padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))) #(128,128,256)
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1 ,padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))) #(128,128,256)
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)) #(64,64,256)

model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1 ,padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))) #(64,64,512)
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1 ,padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))) #(64,64,512)
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1 ,padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))) #(64,64,512)
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)) #(32, 32, 512)

model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1 ,padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))) #(32,32,512)
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1 ,padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))) #(32,32,512)
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1 ,padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))) #(32,32,512)
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)) #(16,16,512)

model.add(tf.keras.layers.Flatten()) #(0, 131072)
model.add(tf.keras.layers.Dense(units=150, activation='relu'))
model.add(tf.keras.layers.Dense(units=150, activation='relu'))
model.add(tf.keras.layers.Dense(units=4, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01)))

#Back Propogation
model.compile(optimizer='adam', loss= tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

#Make a log of the training
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

#Train the model
hist = model.fit(train, epochs=20, validation_data=development, callbacks=[tensorboard_callback])

#evaluate
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()
accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)

precision.update_state(y, yhat)
recall.update_state(y, yhat)
accuracy.update_state(y, yhat)

model.save('cnn_mach1.h5')