import tensorflow as tf
import numpy as np
from PIL import Image
import csv 
import mplfinance as mpf
import pandas as pd
from matplotlib import pyplot as plt
import os


#HYPER-PARAMETERS
NUM_CANDLES = 90

#Read in the data
df = pd.read_csv('Raw_Data/nq5.txt', header = None, names = ['Date', 'open', 'high', 'low', 
    'close', 'volume'], sep=',',index_col=0,parse_dates=True)


#initialize number of candles to be shown on graph
A = 0
B = A + NUM_CANDLES
C = 0

#This for loop creates the images and stores the where ts.keras will be able to parse them into labels and feed them in the NN
#for i in range(0, len(df['close']) - NUM_CANDLES + 45):
for i in range(0, 36000):

    #store the number of candles you want shown on the graph
    storage = df.iloc[A:B]
    #create the graph
    mc = mpf.make_marketcolors(up='g',down='r')
    s  = mpf.make_mpf_style(marketcolors=mc)


    #find 1 percent and 2 percent above and below
    one_low = df['close'][B] * .99
    two_low = df['close'][B] * .98
    one_high = df['close'][B] * 1.01
    two_high = df['close'][B] * 1.02
    #initialize the label counter
    label_counter = B

    #this is to make sure that once it either enters the "gone up by one percent" or "gone down by 1 percent"
    #it doesn't enter the other while loops
    pathway = 0

    try:
        #look for the instance when the price increases or decreases by 1 percent
        #look for the instance when the price increases or decreases by 1 percent
        while df['low'][label_counter] >= one_low and df['high'][label_counter] <= one_high:
            label_counter += 1

        #If the price moved up 1 pecent first, this while loop will trigger and check if it is a two to one, or a one to one trade
        while df['low'][label_counter] >= one_low and df['high'][label_counter] <= two_high:
            label_counter += 1
            pathway = 1
        #Check if price has increased two percent
        if df['high'][label_counter] >= two_high and pathway == 1:
            file_path = 'Images/two_up/' + str(i) + '.png'
            mpf.plot(storage,type='candle',savefig=file_path , warn_too_much_data = 10000000, style = s, volume=True)
            #this resized the image to be a 512x512 resolution imgae
            sized_image = Image.open('Images/two_up/' + str(i) + '.png').crop((177,40,689,552)).save('Images/two_up/' + str(i) + '.png')
        
        #check if price has reversed back down to the one percent marker
        elif df['low'][label_counter] <= one_low and pathway == 1:
            file_path = 'Images/one_up/' + str(i) + '.png'
            mpf.plot(storage,type='candle',savefig=file_path , warn_too_much_data = 10000000, style = s, volume=True)
            #this resized the image to be a 512x512 resolution imgae
            sized_image = Image.open('Images/one_up/' + str(i) + '.png').crop((177,40,689,552)).save('Images/one_up/' + str(i) + '.png')
        
        #if the price moved down 1 pecent first, this will check if it is a two to one, or a one to one trade
        while df['high'][label_counter] <= one_high and df['low'][label_counter] >= two_low and pathway != 1:
            label_counter += 1
            pathway = 2
    
        #check if the price has continued down two percent
        if df['low'][label_counter] <= two_low and pathway == 2:
            file_path = 'Images/two_down/' + str(i) + '.png'
            mpf.plot(storage,type='candle',savefig=file_path , warn_too_much_data = 10000000, style = s, volume=True)
            #this resized the image to be a 512x512 resolution imgae
            sized_image = Image.open('Images/two_down/' + str(i) + '.png').crop((177,40,689,552)).save('Images/two_down/' + str(i) + '.png')
        
        #check if price reversed back up to the 1 percent above marker
        elif df['high'][label_counter] >= one_high and pathway == 2:
            file_path = 'Images/one_down/' + str(i) + '.png'
            mpf.plot(storage,type='candle',savefig=file_path , warn_too_much_data = 10000000, style = s, volume=True)
            #this resized the image to be a 512x512 resolution imgae
            sized_image = Image.open('Images/one_down/' + str(i) + '.png').crop((177,40,689,552)).save('Images/one_down/' + str(i) + '.png')
    except:
        break
       
    #increment the graph by one 15 minute interval 
    A += 1
    B += 1
    C += 1

#Creates the data pipeline        
data = tf.keras.utils.image_dataset_from_directory('Images', image_size=(512,512))

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

model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1 ,padding='same', activation='relu')) #(256,256,128)
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1 ,padding='same', activation='relu', )) #(256,256,128)
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)) #(128,128,128)

model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1 ,padding='same', activation='relu')) #(128,128, 256)
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1 ,padding='same', activation='relu')) #(128,128,256)
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1 ,padding='same', activation='relu')) #(128,128,256)
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)) #(64,64,256)

model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1 ,padding='same', activation='relu')) #(64,64,512)
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1 ,padding='same', activation='relu')) #(64,64,512)
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1 ,padding='same', activation='relu')) #(64,64,512)
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)) #(32, 32, 512)

model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1 ,padding='same', activation='relu')) #(32,32,512)
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1 ,padding='same', activation='relu')) #(32,32,512)
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1 ,padding='same', activation='relu')) #(32,32,512)
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)) #(16,16,512)

model.add(tf.keras.layers.Flatten()) #(0, 131072)
model.add(tf.keras.layers.Dense(units=1000, activation='relu'))
model.add(tf.keras.layers.Dense(units=1000, activation='relu'))
model.add(tf.keras.layers.Dense(units=4, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01)))

#Back Propogation
model.compile(optimizer='adam', loss= tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy', 'FalseNegatives', 'FalsePositives', 'precision', 'sparsecategoricalaccuracy'])

#Make a log of the training
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

#Train the model
hist = model.fit(train, epochs=1000, validation_data=development, callbacks=[tensorboard_callback])

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