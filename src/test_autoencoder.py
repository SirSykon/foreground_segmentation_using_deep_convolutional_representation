import os
import cv2
import time
import numpy as np
import random
import math
from glob import glob
import tensorflow as tf
import tensorflow.keras.layers as layers
from config import Config
import GPU_utils as GPU_utils               # pylint: disable=no-name-in-module
import data_utils                           # pylint: disable=no-name-in-module
import datasets_utils
import Autoencoder


def autoencode_video(video_images_paths, num_training_images, autoencoder, video_name, category):
    for index, video_image_path in enumerate(video_images_paths):
        image = cv2.imread(video_image_path)/255.
        print(video_image_path)
        autoencoded_image = autoencoder.autoencode(tf.expand_dims(image,0))[0]
        segmented_image_path = os.path.join(Config.MODEL_FOLDER_PATH,
                                                    "aut_test", 
                                                    category, 
                                                    video_name, 
                                                    "aut{:0>6}.png".format(index+1))    # We generate the new autoencoded image path.
        print(segmented_image_path)
                                                    
        cv2.imwrite(segmented_image_path, autoencoded_image.numpy()*255.)             # We save the autoencoded image.

"""
GENERAL INITIALIZATION
"""

GPU_utils.tensorflow_2_x_dark_magic_to_restrict_memory_use(Config.GPU_TO_USE)

autoencoder = Autoencoder.Convolutional_Autoencoder(Config.MODEL_FOLDER_PATH, load = True)

dataset_path = Config.TEST_DATASET_PATH

for (category, video_name) in datasets_utils.get_change_detection_categories_and_videos_list(filter_value = "canoe"):
    print(category)
    print(video_name)
    video_images_list, video_initial_roi_frame, video_last_roi_frame = datasets_utils.get_original_change_detection_data(video_name)
    print("ROI")
    print(video_initial_roi_frame)
    autoencode_folder = os.path.join(Config.MODEL_FOLDER_PATH, 
                                        "aut_test",
                                        category, 
                                        video_name)
    if not os.path.isdir(autoencode_folder):
        os.makedirs(autoencode_folder)
                    
    autoencode_video(video_images_list, video_initial_roi_frame-1, autoencoder, video_name, category)
