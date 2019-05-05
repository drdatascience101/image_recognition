#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:46:08 2019

@author: zxs
"""

# Import the required libraries
import os
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import keras
import tensorflow as tf

# Set the working directory
train = '/Users/zxs/Documents/code/kaggle/dog_breeds/stanford/data/train/'
val = '/Users/zxs/Documents/code/kaggle/dog_breeds/stanford/data/validate/'
wd = '/Users/zxs/Documents/code/kaggle/dog_breeds/stanford/data/'

'''
    CSV conversion
'''

# Initialize lists to store the data
train_imgs, train_lbls, val_imgs, val_lbls = [], [], [], []

# Initialize counters
train_n, val_n = 0, 0

'''
    Training Data
'''

# Process the validation directory
train_folders = os.listdir(train)

os.chdir(train)

# Iterate the folders
for folder in train_folders:
    
    # Navigate
    os.chdir(folder)
    
    # Iterate the files
    for file in os.listdir():
        
        # Skip bad file
        if file != '.DS_Store':
            
            try:
                
                # Process the individual file
                img = Image.open(file)
                img1 = img.resize((224, 224))
                img2 = np.asarray(img1, dtype = np.float32) 
            
                # Update the master lists
                train_imgs.append(img2)
                train_lbls.append(folder)
                train_n += 1
            
            except:
                
                # PNG exceptions
                img = Image.open(file)
                rgb = img.convert('RGB')
                rgb1 = rgb.resize((224, 224))
                
                rgb2 = np.asarray(rgb1, dtype = np.float32) 
                
                # Update the master lists
                train_imgs.append(rgb2)
                train_lbls.append(folder)
                train_n += 1
                
    os.chdir(train)

# Navigate
os.chdir(wd)

# Reshape
train_imgs1 = np.asarray(train_imgs)

'''
    Validation Data
'''
    
# Process the validation directory
val_folders = os.listdir(val)

os.chdir(val)

# Iterate the folders
for folder in val_folders:
    
    # Navigate
    os.chdir(folder)
    
    # Iterate the files
    for file in os.listdir():
        
        # Skip bad file
        if file != '.DS_Store':
            
            # Process the individual file
            img = Image.open(file)
            img1 = img.resize((224, 224))
            img2 = np.asarray(img1, dtype = np.float32) 
            
            if img2.shape == (224, 224, 3):
                
                # Update the master lists
                val_imgs.append(img2)
                val_lbls.append(folder)
                val_n += 1
            
            else:
                
                pass
            
    os.chdir(val)

# Navigate
os.chdir(wd)

# Reshape
val_imgs1 = np.asarray(val_imgs)

'''
    Additional Processing
'''

# Initialize the label encoder
le = LabelEncoder()
le.fit(train_lbls)

y = le.transform(train_lbls)
val_y = le.transform(val_lbls)

from keras.utils import to_categorical

y_binary = to_categorical(y)
val_y_binary = to_categorical(val_y)

'''
    KERAS NN
'''

# Import the required libraries
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

# Initialize the model
mod = Sequential()

# Add layers
mod.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (224, 224, 3)))
mod.add(Conv2D(32, (3, 3), activation = 'relu'))
mod.add(MaxPooling2D(pool_size = (2, 2)))

mod.add(Conv2D(64, (3, 3), activation = 'relu'))
mod.add(Conv2D(64, (3, 3), activation = 'relu'))
mod.add(MaxPooling2D(pool_size = (2, 2)))

mod.add(Flatten())
mod.add(Dense(256, activation = 'relu'))
mod.add(Dense(120, activation = 'softmax'))

# Add optimizer
sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)

# Compile
mod.compile(loss = 'categorical_crossentropy', optimizer = sgd)

# Train
mod.fit(train_imgs1, y_binary)

# Evaluate sample of 5
preds = mod.predict(val_imgs1[:1000])
