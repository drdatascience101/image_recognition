# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Import the required libraries
import os
import tarfile

# Set the working directory
wd = '/Users/zxs/Documents/code/kaggle/dog_breeds/stanford/data/'
img_dir = wd + 'Images'

os.chdir(wd)

#Load the tar file
imgs = tarfile.open('images.tar', 'r')

# Read
imgs.extractall()

os.chdir(img_dir)

folders = os.listdir()

def convert_folders(x):

    x1 = x.replace('-', '_')
    
    x2 = x1.split('_', 1)
    
    x3 = x2[1]
    
    x4 = x3.lower() 
    
    return x4

new_folders = [convert_folders(x) for x in folders]

new_folders1 = [os.rename(x[0], x[1]) for x in zip(folders, new_folders)]

# Function to convert the folder names
def convert_files(folder):
    
    # Navigate
    os.chdir(folder)
    
    # Initialize a counter
    n = 0

    # List files
    files = os.listdir()
    
    # Iterate
    for file in files:
        
        if file != '.DS_Store':
            
            # Update counter
            n += 1
        
            # Rename file
            fn, ext = os.path.splitext(file)
        
            new_fn = folder + str(n)
        
            new_fn = new_fn + ext
        
            os.rename(file, new_fn)
    
    # Navigate    
    os.chdir(img_dir)

## Apply function to folders    
for folder in os.listdir():
    
    convert_files(folder)

# Create directories to store training / validation data        
train_dir = '/Users/zxs/Documents/code/kaggle/dog_breeds/stanford/data/train/' 
val_dir = '/Users/zxs/Documents/code/kaggle/dog_breeds/stanford/data/validate/'         

os.chdir(wd)  
os.mkdir(train_dir)
os.mkdir(val_dir)

# Make the training / validation folders
def mk_folder(orig_folder):
    
    # Create training folder if not already present
    os.chdir(train_dir)
    
    try:
        
        os.mkdir(orig_folder)
        
    except:
        
        pass

    # Create validation folder if not already present    
    os.chdir(val_dir)
    
    try:
        
        os.mkdir(orig_folder)
        
    except:
        
        pass
    
    # Navigate back to directory
    os.chdir(img_dir)

# Iterate the original folders
for folder in new_folders:
    
    # Create destinations
    mk_folder(folder)

# Copy files to training / validation folders    
for folder in new_folders:
    
    os.chdir(img_dir)
    os.chdir(folder)
    
    # Initialize a counter
    n = 0    
    
    # List files
    files = os.listdir()
    num_files = len(files)

    # Determine length of split    
    train_len = int(num_files * .8)

    # Iterate
    for file in files:
        
        # Check condition for training
        if n <= train_len:
            
            # Update counter
            n += 1
            
            # Define file paths
            src = img_dir + '/' + folder + '/' + file
            dest = train_dir + ('{}/'.format(folder)) + file
            
            # Move
            os.rename(src, dest)
        
        # Condition for validation
        elif n > train_len:
            
            # Update counter
            n += 1
            
            # Define file paths
            src = img_dir + '/' + folder + '/' + file
            dest = val_dir + ('{}/'.format(folder)) + file
            
            # Move
            os.rename(src, dest)
    
    # Navigate back to top
    os.chdir(img_dir)
       