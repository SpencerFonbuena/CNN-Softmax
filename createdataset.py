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



