"""
Code to generate a dataset to train an autoencoder.
This dataset will contain Config.TRAINING_DATA_SIZE patches with size Config.PATCH_IMG_SIZE extracted from Config.TRAINING_DATASET_PATH
In order to get a lesser number of files, we will generate .npy files wih Config.TRAINING_DATA_PER_FILE paches in each file.
So we will save Config.TRAINING_DATA_SIZE/Config.TRAINING_DATA_PER_FILE files wih Config.TRAINING_DATA_PER_FILE x Config.PATCH_IMG_SIZE instances.
All will be saved in Config.NETWORK_TRAINING_DATA_PATH with names defined by Config.NETWORK_TRAINING_DATA_FILES_NAME_STRUCTURE.
"""

import numpy as np
import cv2
import os
import random
from glob import glob

from config import Config
import images_utils

assert Config.TRAINING_DATA_SIZE % Config.TRAINING_DATA_PER_FILE == 0                           # We check that the total number of data and the data per file make sense.

if not os.path.isdir(Config.NETWORK_TRAINING_DATA_PATH):                                        # We generate the folder to contain the data if it does not exist.
    os.makedirs(Config.NETWORK_TRAINING_DATA_PATH)

all_images_paths = glob(os.path.join(Config.TRAINING_DATASET_PATH,"*"))                         # We get the list of images paths.
number_of_training_data_files = int(Config.TRAINING_DATA_SIZE / Config.TRAINING_DATA_PER_FILE)  # We get the number of training data files that we will generate.

for training_data_file_index in range(number_of_training_data_files):
    training_data_in_file = None                                                                # Matrix that contains the data that will be saved in this file.
    
    for patch_index in range(Config.TRAINING_DATA_PER_FILE):                                        
        img_index = random.randint(0,len(all_images_paths)-1)                                           # We get the image index to use.
        img_path = all_images_paths[img_index]                                                          # We get the image path.
        assert img_path[-4:] == ".jpg" or img_path[-4:] == ".png"                                       # We esure img_path is a image.
        img = cv2.imread(img_path)                                                                      # We load the image.
        
        while img.shape[0] < Config.PATCH_IMG_SIZE[0] or img.shape[1] < Config.PATCH_IMG_SIZE[1]:       # We ensure the image has enough size to get the patch.
            img_index = random.randint(0,len(all_images_paths)-1)                                           # We get the image index to use.
            img_path = all_images_paths[img_index]                                                          # We get the image path.
            assert img_path[-4:] == ".jpg" or img_path[-4:] == ".png"                                       # We esure img_path is a image.
            img = cv2.imread(img_path)                                                                      # We load the image.
            
        random_patch_from_image = images_utils.get_random_patch_from_image(img, Config.PATCH_IMG_SIZE)  # We get the random patch from the image
        
        if training_data_in_file is None:                                                               # If training_data_in_file is not initialized.
            patch_channels = random_patch_from_image.shape[-1]                                              # We get the channels from patch (and so, it should be the number of channels of img).
            training_data_in_file = np.zeros(
                shape = (
                    Config.TRAINING_DATA_PER_FILE, 
                    Config.PATCH_IMG_SIZE[0], 
                    Config.PATCH_IMG_SIZE[1], 
                    patch_channels))                                                                        # We initialize training_data_in_file.
        
        training_data_in_file[patch_index] = random_patch_from_image                                    # We insert the patch into the matrix that will be saved to file.

    assert training_data_in_file.shape[0] == Config.TRAINING_DATA_PER_FILE                      # We check training_data_inf_file has the correct shape.
    assert training_data_in_file.shape[1] == Config.PATCH_IMG_SIZE[0]
    assert training_data_in_file.shape[2] == Config.PATCH_IMG_SIZE[1]
    
    file_name = os.path.join(
        Config.NETWORK_TRAINING_DATA_PATH, 
        Config.NETWORK_TRAINING_DATA_FILES_NAME_STRUCTURE.format(training_data_file_index))     # We generate the file name.
    print(file_name)
    print(training_data_in_file.shape)
    np.save(file_name, training_data_in_file)                                                   # We save the file.
