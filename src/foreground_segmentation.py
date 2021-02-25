import os
import cv2
import time
import numpy as np
import random
import math
import tensorflow as tf
import tensorflow.keras.layers as layers
from config import Config
import GPU_utils as GPU_utils               # pylint: disable=no-name-in-module
import data_utils                           # pylint: disable=no-name-in-module
import datasets_utils
import Autoencoder
import time

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
        count = tf.cast(count, tf.float64)
        mean = newValue * 0.
        mean = tf.cast(mean, tf.float64)
        M2 = newValue * 0.
        M2 = tf.cast(M2, tf.float64)
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
        return (mean, tf.zeros(shape=mean.shape, dtype=tf.float64), tf.zeros(shape=mean.shape, dtype=tf.float64))       # We return values to evade tensorflow error but should not happen.
    else:
       (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))                                          # We get the values online calculated.
       return (mean, variance, sampleVariance)

"""
Function to apply Weldfor algorithm.
"""
#@tf.function
def welford_algorithm(image, autoencoder, existingAggregate, finalize):

    if finalize:                                                                    # We finalize Welford's online algorithm.
        mean, variance, sampleVariance = welford_finalize(existingAggregate)        # We finalize Welford's online algorithm to get mean and variance.
        return mean, variance, sampleVariance
                   
    else:                                                                           # We do not finalize.
        encoded_image = autoencoder.encode(tf.expand_dims(image,0))[0]              # We get the encoded image.
        print(encoded_image)
        encoded_image = tf.cast(encoded_image, tf.float64)                          # We ensure is a tf double tensor.
        existingAggregate = welford_update(existingAggregate, encoded_image)        # We update Welford's online algorihtm.
        return existingAggregate

"""
Function to segmentate image.
"""
#@tf.function
def segmentate_image(image, autoencoder, background_model, vol_h = 1, L = 16, alpha = 0.001, pi_Fore = 0.5, pi_Back = 0.5, min_var = 0.001):
    v = autoencoder.encode(tf.expand_dims(image,0))[0]                                                                                  # We encode the image.
    assert L == v.shape[-1]                                                                                                             # We ensure L is the same as the channel depth.
    v = tf.cast(v, tf.float64)                                                                                                          # We ensure is a tf double tensor.
    (mean, var) = background_model                                                                                                      # We get the background model.
    #print("mean")
    #print(mean.shape)
    #print(mean)
    #print(mean.dtype)
    #print("v")
    #print(v)
    #print(v.dtype)
    #print("diff")
    #print(v-mean)
    #print("mean")
    #print(var)
    #print(tf.reduce_sum(tf.cast(tf.math.is_nan(mean),tf.int32)).numpy())
    #print("var")
    #print(var.shape)
    #print(tf.reduce_sum(tf.cast(tf.math.is_nan(var),tf.int32)).numpy())
    det_sigma = tf.reduce_prod(var, 2)                                                                                                  # eq 21
    #print("det_sigma")
    #print(det_sigma.shape)
    #print(det_sigma)
    #print(tf.reduce_sum(tf.cast(tf.math.is_nan(det_sigma),tf.int32)).numpy())
    aux = tf.divide(tf.pow((v - mean),2), var)                                                                                          # eq 22.1
    #print("aux")
    #print(aux.shape)
    #print(tf.reduce_sum(tf.cast(tf.math.is_nan(aux),tf.int32)).numpy())
    aux = tf.reduce_sum(aux,2)                                                                                                          # eq 22.2
    #print("aux")
    #print(aux.shape)
    #print(tf.reduce_sum(tf.cast(tf.math.is_nan(aux),tf.int32)).numpy())
    #print(tf.reduce_max(aux).numpy())
    #print(tf.reduce_min(aux).numpy())
    K_v_aux_1 = ((2*math.pi)**(-1*L/2.))
    #print("K_v_aux_1")
    #print(K_v_aux_1)
    K_v_aux_2 = tf.pow(det_sigma, -1/2)
    #print("K_v_aux_2")
    #print(K_v_aux_2.shape)
    #print(tf.reduce_sum(tf.cast(tf.math.is_nan(K_v_aux_2),tf.int32)).numpy())
    #print(tf.reduce_max(K_v_aux_2).numpy())
    #print(tf.reduce_min(K_v_aux_2).numpy())
    K_v_aux_3 = tf.exp((-1/2)*aux)
    #print("K_v_aux_3")
    #print(K_v_aux_3.shape)
    #print(tf.reduce_sum(tf.cast(tf.math.is_nan(K_v_aux_3),tf.int32)).numpy())
    #print(tf.reduce_max(K_v_aux_3).numpy())
    #print(tf.reduce_min(K_v_aux_3).numpy())
    K_v = K_v_aux_1 * tf.multiply(K_v_aux_2, K_v_aux_3)                                                                                 # eq 9
    #print("K_v")
    #print(K_v.shape)
    #print(tf.reduce_sum(tf.cast(tf.math.is_nan(K_v),tf.int32)).numpy())
    
    U_v = tf.cast(1./vol_h, tf.float64)                                                                                                 # eq 10
    p_v_Back = K_v                                                                                                                      # eq 7
    p_v_Fore = U_v                                                                                                                      # eq 8
    R_Fore = tf.divide(np.multiply(pi_Fore, p_v_Fore), tf.add(tf.multiply(pi_Back, p_v_Back), tf.multiply(pi_Fore, p_v_Fore)))          # eq 17
    #print("R_Fore")
    #print(R_Fore.shape)
    #print(tf.reduce_sum(tf.cast(tf.math.is_nan(R_Fore),tf.int32)).numpy())

    R_Back = 1 - R_Fore
    new_mean = tf.add(tf.multiply((1 - alpha*R_Back)[:,:,None], mean), tf.multiply((alpha*R_Back)[:,:,None], v))                        # eq 18
    new_var = tf.add(tf.multiply((1 - alpha*R_Back)[:,:,None], var), tf.multiply((alpha*R_Back)[:,:,None], tf.pow(v - mean, 2)))        # eq 19
    corrected_var = tf.clip_by_value(new_var, clip_value_min=min_var, clip_value_max = np.inf)                                          # We ensure that var is min_var as minimum. 

    #print("background model")
    #print(new_mean.shape)
    #print(tf.reduce_sum(tf.cast(tf.math.is_nan(new_mean),tf.int32)).numpy())
    #print(new_var.shape)
    #print(tf.reduce_sum(tf.cast(tf.math.is_nan(new_var),tf.int32)).numpy())
    #print(corrected_var.shape)
    #print(tf.reduce_sum(tf.cast(tf.math.is_nan(corrected_var),tf.int32)).numpy())
    
    return R_Fore, (new_mean, corrected_var)
    
    
"""

"""
def video_segmentation(video_images_paths, num_training_images, autoencoder, segmentation_folder, min_var = 0.001):
    existingAggregate = None
    background_model = None
    processing_time = []
    for index, video_image_path in enumerate(video_images_paths):
        start_time = time.time()
        image = cv2.imread(video_image_path)/255.
        print("Video image path: {}".format(video_image_path))
        _, image_name = os.path.split(video_image_path)
        print("Image name: {}".format(image_name))
        
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
                                                    
                variance = tf.clip_by_value(variance, clip_value_min=min_var, clip_value_max = np.inf)   # We ensure that var is min_var as minimum. 
                background_model = (mean, variance)           
                #print(mean.numpy())
                #print(variance.numpy())
            
            segmented_image, background_model = segmentate_image(image, 
                                                                    autoencoder, 
                                                                    background_model,
                                                                    min_var = min_var)   # We get the segmented image and the updated background model.
            #print(segmented_image.numpy())
            segmented_image_path = os.path.join(segmentation_folder, 
                                                    "seg{:0>6}.png".format(index+1))    # We generate the new segmented image path.
            print("Output image path: {}".format(segmented_image_path))
            #print(np.max(segmented_image.numpy()))
            assert np.sum(np.isnan(segmented_image.numpy())) == 0
                
            resized_segmented_image = cv2.resize(segmented_image.numpy()*255., (image.shape[1], image.shape[0]))    # We resize the segmented image to ensure it has the same size as the original image.
            
            cv2.imwrite(segmented_image_path, resized_segmented_image)                   # We save the segmented image.
        
        processing_time.append(time.time()-start_time)
        
    return np.array(processing_time)

"""
GENERAL INITIALIZATION
"""

GPU_utils.tensorflow_2_x_dark_magic_to_restrict_memory_use(Config.GPU_TO_USE)

autoencoder = Autoencoder.Autoencoder(Config.MODEL_FOLDER_PATH, load = True)                    # We lad a generic autoencoder defined by the model path given as argument.

for (noise, category, video_name) in datasets_utils.get_change_detection_noises_categories_and_videos_list(filter_value = None):
    print(noise)
    print(category)
    print(video_name)
    if category in Config.CATEGORIES_TO_TEST:
        video_images_list, video_initial_roi_frame, video_last_roi_frame = datasets_utils.get_noise_change_detection_data(video_name, noise)
        print("ROI")
        print(video_initial_roi_frame)
        segmentation_folder = os.path.join(Config.SEGMENTATION_OUTPUT_FOLDER, 
                                            noise,
                                            category, 
                                            video_name)

        print("segmentation folder: {}.".format(segmentation_folder))

        if not os.path.isdir(segmentation_folder):
            os.makedirs(segmentation_folder)

        processing_time = video_segmentation(video_images_list, (video_initial_roi_frame-1), autoencoder, segmentation_folder)
        bck_train_processing_time = processing_time[:(video_initial_roi_frame-1)]
        seg_img_processing_time = processing_time[(video_initial_roi_frame-1):]
        print("Background training average time: {} with fps {}".format(np.average(bck_train_processing_time),1./np.average(bck_train_processing_time)))
        print("Image segmentation average time: {} with fps {}".format(np.average(seg_img_processing_time),1./np.average(seg_img_processing_time)))
    else:
        print("We skip category {}.".format(category))
