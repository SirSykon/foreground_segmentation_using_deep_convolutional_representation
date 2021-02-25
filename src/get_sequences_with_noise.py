import os
import cv2
import time
import numpy as np
import random
import math
from glob import glob
import tensorflow as tf
import tensorflow.keras.layers as layers
import config
import GPU_utils as GPU_utils               # pylint: disable=no-name-in-module
import data_utils                           # pylint: disable=no-name-in-module
import datasets_utils

configuration = config.Config()

def add_noise_to_video(video_images_paths, video_name, category):
    for index, video_image_path in enumerate(video_images_paths):
        image = cv2.imread(video_image_path)/255.
        print(video_image_path)
        autoencoded_image = autoencoder.autoencode(tf.expand_dims(image,0))[0]
        image_with_noise_path = os.path.join(configuration.MODEL_FOLDER_PATH,
                                                    "aut_test", 
                                                    category, 
                                                    video_name, 
                                                    "img{:0>6}.png".format(index+1))    # We generate the new autoencoded image path.
        print(image_with_noise_path)
                                                    
        cv2.imwrite(image_with_noise_path, autoencoded_image.numpy()*255.)             # We save the autoencoded image.

"""
GENERAL INITIALIZATION
"""

dataset_path = configuration.CHANGEDETECTON_DATASET_PATH

for (category, video_name) in datasets_utils.get_change_detection_categories_and_videos_list(dataset = dataset_path, filter_value = None):
    print(category)
    print(video_name)
    video_images_list, video_initial_roi_frame, video_last_roi_frame = datasets_utils.get_original_change_detection_data(video_name)
    print("ROI")
    print(video_initial_roi_frame)
                    
    autoencode_viadd_noise_to_videodeo(video_images_list, video_name, category)
