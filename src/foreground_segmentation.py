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

"""
FUNCTIONS
"""

"""
Function to update Welford's online algorithm to obtain mean and variance.
"""
@tf.function
def welford_update(existingAggregate, newValue):
    if existingAggregate is None:
        count = 0.
        mean = newValue * 0
        M2 = newValue * 0
    else:
        (count, mean, M2) = existingAggregate

    print(mean.dtype)
    print(M2.dtype)
    count += 1
    delta = tf.subtract(newValue, mean)
    mean += tf.divide(delta, count)
    delta2 = tf.subtract(newValue, mean)
    M2 += tf.multiply(delta, delta2)

    return (count, mean, M2)


"""
Function to finalize Welford's online algorithm to obtain mean and variance.
"""
@tf.function
def welford_finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    if count < 2:
        return (mean, tf.zeros(shape=mean.shape), tf.zeros(shape=mean.shape))       # We return values to evade tensorflow error but should not happen.
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
        encoded_image = autoencoder.encode(tf.expand_dims(image,0))[0]
        existingAggregate = welford_update(existingAggregate, encoded_image)        # We update Welford's online algorihtm.
        return existingAggregate

"""
"""
@tf.function
def segmentate_image(image, autoencoder, background_model, vol_h = 1, L = 16, alpha = 0.001, pi_Fore = 0.5, pi_Back = 0.5):
    v = autoencoder.encode(tf.expand_dims(image,0))[0]                                                                                  # We encode the image.
    (mean, var) = background_model                                                                                                      # We get the background model.
    print("mean")
    print(mean.shape)
    print("var")
    print(var.shape)
    det_sigma = tf.reduce_prod(var, 2)                                                                                                  # eq 21
    print("det_sigma")
    print(det_sigma.shape)
    print(det_sigma.numpy())
    aux = tf.divide(tf.pow((v - mean),2), var)                                                                                          # eq 22.1
    print("aux")
    print(aux.shape)
    print(aux.numpy())
    aux = tf.reduce_sum(aux,2)                                                                                                          # eq 22.2
    print("aux")
    print(aux.shape)
    print(aux.numpy())
    K_v = ((2*math.pi)**(-1*L/2.)) * tf.multiply(tf.pow(det_sigma, -1/2), tf.exp((-1/2)*aux))                                           # eq 9
    print("K_v")
    print(K_v.shape)
    
    U_v = 1/vol_h                                                                                                                       # eq 10
    p_v_Back = K_v                                                                                                                      # eq 7
    p_v_Fore = U_v                                                                                                                      # eq 8
    R_Fore = tf.divide(np.multiply(pi_Fore, p_v_Fore), tf.add(tf.multiply(pi_Back, p_v_Back), tf.multiply(pi_Fore, p_v_Fore)))          # eq 17
    print("R_Fore")
    print(R_Fore.shape)

    R_Back = 1 - R_Fore
    new_mean = tf.add(tf.multiply((1 - alpha*R_Back)[:,:,None], mean), tf.multiply((alpha*R_Back)[:,:,None], v))                        # eq 18
    new_var = tf.add(tf.multiply((1 - alpha*R_Back)[:,:,None], var), tf.multiply((alpha*R_Back)[:,:,None], tf.pow(v - mean, 2)))        # eq 19
    print("background model")
    print(new_mean.shape)
    print(new_var.shape)
    quit()
    return R_Fore, (new_mean, new_var)
    
    
"""

"""
def video_segmentation(video_images_paths, num_training_images, autoencoder, video_name, category):
    existingAggregate = None
    background_model = None
    for index, video_image_path in enumerate(video_images_paths):
        image = cv2.imread(video_image_path)/255.
        print(video_image_path)
        _, image_name = os.path.split(video_image_path)
        print(image_name)
        
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
                background_model = (mean, variance)           
                print(mean.numpy())
                print(variance.numpy())
            
            segmented_image, background_model = segmentate_image(image, 
                                                                    autoencoder, 
                                                                    background_model)   # We get the segmented image and the updated background model.
            print(segmented_image.numpy())
            segmented_image_path = os.path.join(Config.SEGMENTATION_OUTPUT_FOLDER, 
                                                    category, 
                                                    video_name, 
                                                    "seg{:0>6}.png".format(index+1))    # We generate the new segmented image path.
            print(segmented_image_path)
            print(np.max(segmented_image.numpy()))

            cv2.imwrite(segmented_image_path, segmented_image.numpy()*255.)             # We save the segmented image.
        


"""
GENERAL INITIALIZATION
"""

GPU_utils.tensorflow_2_x_dark_magic_to_restrict_memory_use(Config.GPU_TO_USE)

autoencoder = Autoencoder.Convolutional_Autoencoder(Config.MODEL_FOLDER_PATH, load = True)

dataset_path = Config.TEST_DATASET_PATH

for (category, video_name) in datasets_utils.get_change_detection_categories_and_videos_list():
    print(category)
    print(video_name)
    video_images_list, video_initial_roi_frame, video_last_roi_frame = datasets_utils.get_original_change_detection_data(video_name)
    print("ROI")
    print(video_initial_roi_frame)
    segmentation_folder = os.path.join(Config.SEGMENTATION_OUTPUT_FOLDER, 
                                        category, 
                                        video_name)
    if not os.path.isdir(segmentation_folder):
        os.makedirs(segmentation_folder)
                    
    video_segmentation(video_images_list, video_initial_roi_frame-1, autoencoder, video_name, category)
