"""
Code to generate datasets to train an autoencoder.
This dataset will contain Config.SPECIFIC_SEQUENCE_TRAINING_DATA_SIZE patches with size Config.PATCH_IMG_SIZE extracted from Config.CHANGEDETECTON_DATASET_PATH
In order to get a lesser number of files, we will generate .npy files wih Config.TRAINING_DATA_PER_FILE paches in each file.
So we will save Config.SPECIFIC_SEQUENCE_TRAINING_DATA_SIZE/Config.TRAINING_DATA_PER_FILE files wih Config.TRAINING_DATA_PER_FILE instances.
All will be saved in Config.NETWORK_TRAINING_DATA_PATH with names defined by Config.NETWORK_TRAINING_DATA_FILES_NAME_STRUCTURE.
"""

import numpy as np
import cv2
import os
import random
from tqdm import tqdm
from glob import glob

import config
import images_utils
import datasets_utils

configuration = config.Config()

assert configuration.SPECIFIC_SEQUENCE_TRAINING_DATA_SIZE % configuration.TRAINING_DATA_PER_FILE == 0                           # We check that the total number of data and the data per file make sense.

if not os.path.isdir(configuration.NETWORK_TRAINING_DATA_PATH):                                        # We generate the folder to contain the data if it does not exist.
    os.makedirs(configuration.NETWORK_TRAINING_DATA_PATH)

all_images_paths = glob(os.path.join(configuration.TRAINING_DATASET_PATH,"*"))                         # We get the list of images paths.
number_of_training_data_files = int(configuration.SPECIFIC_SEQUENCE_TRAINING_DATA_SIZE / configuration.TRAINING_DATA_PER_FILE)  # We get the number of training data files that we will generate.

for (category, video_name) in datasets_utils.get_change_detection_categories_and_videos_list():
    print(category)
    print(video_name)
    if category in configuration.CATEGORIES_TO_TEST:
        video_images_list, video_initial_roi_frame, video_last_roi_frame = datasets_utils.get_original_change_detection_data(video_name)
        print(f"Initial ROI frame: {video_initial_roi_frame}")
        for training_data_file_index in range(number_of_training_data_files):
            training_data_in_file = None                                                                                    # Matrix that contains the data that will be saved in this file.
            for patch_index in tqdm(range(configuration.TRAINING_DATA_PER_FILE)):
                img_index = random.randint(0,video_initial_roi_frame-1)                                                     # We get the image index to use.
                img_path = video_images_list[img_index]                                                                     # We get the image path.
                assert img_path[-4:] == ".jpg" or img_path[-4:] == ".png"                                                   # We esure img_path is a image.
                img = cv2.imread(img_path)                                                                                  # We load the image.

                while img.shape[0] < configuration.PATCH_IMG_SIZE[0] or img.shape[1] < configuration.PATCH_IMG_SIZE[1]:     # We ensure the image has enough size to get the patch.
                    img_index = random.randint(0,video_initial_roi_frame-1)                                                 # We get the image index to use.
                    img_path = video_images_list[img_index]                                                                 # We get the image path.
                    assert img_path[-4:] == ".jpg" or img_path[-4:] == ".png"                                               # We esure img_path is a image.
                    img = cv2.imread(img_path)                                                                              # We load the image.

                random_patch_from_image = images_utils.get_random_patch_from_image(img, configuration.PATCH_IMG_SIZE)       # We get the random patch from the image

                if training_data_in_file is None:                                                                   # If training_data_in_file is not initialized.
                    patch_channels = random_patch_from_image.shape[-1]                                              # We get the channels from patch (and so, it should be the number of channels of img).
                    training_data_in_file = np.zeros(
                        shape = (
                            configuration.TRAINING_DATA_PER_FILE, 
                            configuration.PATCH_IMG_SIZE[0], 
                            configuration.PATCH_IMG_SIZE[1], 
                            patch_channels))                                                                        # We initialize training_data_in_file.
            
                training_data_in_file[patch_index] = random_patch_from_image                                    # We insert the patch into the matrix that will be saved to file.

            assert training_data_in_file.shape[0] == configuration.TRAINING_DATA_PER_FILE                      # We check training_data_inf_file has the correct shape.
            assert training_data_in_file.shape[1] == configuration.PATCH_IMG_SIZE[0]
            assert training_data_in_file.shape[2] == configuration.PATCH_IMG_SIZE[1]

            folder_name = os.path.join(
                configuration.NETWORK_TRAINING_DATA_PATH, 
                f"{video_name}")                                                                            # We generate the folder name.

            if not os.path.isdir(folder_name):
                os.makedirs(folder_name)

            file_name = os.path.join(
                folder_name,
                configuration.NETWORK_TRAINING_DATA_FILES_NAME_STRUCTURE.format(training_data_file_index))     # We generate the file name.
            print(file_name)
            print(training_data_in_file.shape)
            np.save(file_name, training_data_in_file)                                                           # We save the file.

    else:
        print(f"{category} is not included in test list..")
