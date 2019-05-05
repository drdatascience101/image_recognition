# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Import the required libraries
import os
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras import layers
from keras.preprocessing.image import save_img
from keras.utils.vis_utils import model_to_dot
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.applications.xception import Xception
from keras.applications.nasnet import NASNetMobile
from keras.models import Sequential,Input,Model
from keras.layers import Dense,Flatten,Dropout,Concatenate,GlobalAveragePooling2D,Lambda,ZeroPadding2D
from keras.layers import SeparableConv2D,BatchNormalization,MaxPooling2D,Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau,LearningRateScheduler

# Set the working directory
wd = '/Users/zxs/Documents/code/kaggle/dog_breeds/stanford/data/Images'
os.chdir(wd)

# Initialize empty lists to store the data
imgs = []
lbls = []

# Iterate the directory
for folder in os.listdir():
    
    # Navigate to folder
    if os.path.isdir(folder):
        
        os.chdir(folder)
        
        # Iterate pictures
        for file in os.listdir():
            
            # Open image
            img = image.load_img(file, target_size = (224, 224))            
            img1 = image.img_to_array(img)
            
            # Update results
            imgs.append(img1)
            lbls.append(folder)
        
        # Navigate back to main directory
        os.chdir(wd)

# Encode the labels
le = LabelEncoder()

lbls1 = le.fit_transform(lbls)

# Separate the variabbles and data
train_x, test_x, train_y, test_y = train_test_split(imgs, lbls1, test_size = .3, random_state = 100)
val_x, test_x, val_y, test_y = train_test_split(test_x, test_y, test_size = .3, random_state = 100)

# Initialize the data generator
dg = ImageDataGenerator()

dg.fit(train_x)

# Initialize the NN
base_model = VGG16(include_top = False, input_shape = (150, 150, 3), weights = 'imagenet')

for layer in base_model.layers:
    
    layer.trainable = False
    
for layer in base_model.layers:
    
    print(layer,layer.trainable)

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(10, activation ='softmax'))
model.summary()

# Build
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

history = model.fit_generator(dg.flow(train_x, train_y, batch_size = 16), 
                              validation_data  = (val_x, val_y), 
                              validation_steps = 1000, 
                              steps_per_epoch  = 1000,
                              epochs = 20, 
                              verbose = 1)   