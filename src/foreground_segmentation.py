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
import Autoencoder

GPU_utils.tensorflow_2_x_dark_magic_to_restrict_memory_use(Config.GPU_TO_USE)

autoencoder = Autoencoder.Convolutional_Autoencoder(Config.MODEL_FOLDER_PATH, load = True)

def video_segmentation(video_images, num_training_images):

    for index, image in enumerate(video_images):
        
        if index < num_training_images:             # background training
        
        
        else:                                       # foreground segmentation
        
        
       

