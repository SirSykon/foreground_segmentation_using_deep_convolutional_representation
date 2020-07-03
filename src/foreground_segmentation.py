import os
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

"""
FUNCTIONS
"""

"""
Function to update Welford's online algorithm to obtain mean and variance.
"""
@tf.function
def welford_update(existingAggregate, newValue):
    if existingAggregate is None:
        count = 0
        mean = newValue * 0
        M2 = newValue * 0
    else:
        (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2

    return (count, mean, M2)


"""
Function to finalize Welford's online algorithm to obtain mean and variance.
"""
@tf.function
def welford_finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    if count < 2:
        return None
    else:
       (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
       return (mean, variance, sampleVariance)

"""
"""
@tf.function
def welford_algorithm(image, autoencoder, existingAggregate, finalize):

    if finalize:                                                                    # We finalize Welford's online algorithm.
        mean, variance, sampleVariance = welford_finalize(existingAggregate)        # We finalize Welford's online algorithm to get mean and variance.
        return mean, variance, sampleVariance
                   
    else:                                                                           # We do not finalize.
        encoded_image = autoencoder.encode(image)
        existingAggregate = welford_update(existingAggregate, encoded_image)        # We update Welford's online algorihtm.
        return existingAggregate

"""
"""
@tf.function
def image_segmentation(image, autoencoder, background_model):
    
    
"""

"""
def video_segmentation(video_images_paths, num_training_images, autoencoder):
    existingAggregate = None
    background_model = None
    for index, video_image_path in enumerate(video_images_paths):
        image = cv2.imread(video_image_path)
        _, image_name = os.path.split(image)        
        
        training = index < num_training_images
                
        if training:                                                                    # We are still getting images to initialize background model.                     
            finalize_welford_algorithm = False
            existingAggregate = welford_algorithm(image, 
                                                    autoencoder, 
                                                    existingAggregate, 
                                                    finalize_welford_algorithm)         # We update Welford's online algorithm.
        
        else:
            if background_model is None:                                                # This is the first image to get a foreground segmentation
                finalize_welford_algorithm = True                                                                      
                mean, variance, sampleVariance = welford_algorithm(None, 
                                                    None, 
                                                    existingAggregate, 
                                                    finalize_welford_algorithm)         # We will finalize Welford's online algorithm                
            
            segmented_image, background_model = image_segmentation(image, 
                                                                    autoencoder, 
                                                                    background_model)   # We get the segmented image and the updated background model.
            segmented_image_path = os.path.join(Config.SEGMENTATION_OUTPUT_FOLDER, 
                                                    category, 
                                                    video_name, 
                                                    "seg{:0>6}.png".format(index+1))    # We generate the new segmented image path.
            cv2.imwrite(segmented_image_path, segmented_image)                          # We save the segmented image.
        


"""
GENERAL INITIALIZATION
"""

GPU_utils.tensorflow_2_x_dark_magic_to_restrict_memory_use(Config.GPU_TO_USE)

autoencoder = Autoencoder.Convolutional_Autoencoder(Config.MODEL_FOLDER_PATH, load = True)

dataset_path = Config.TEST_DATASET

for (category, video_name) in datasets_utils.get_change_detection_categories_and_videos_list():
    print(category)
    print(video_name)
    video_images_list, video_initial_roi_frame, video_last_roi_frame = datasets_utils.get_original_change_detection_data(video_name)
    video_segmentation(video_images_list, video_initial_roi_frame, autoencoder, video_name, category)
