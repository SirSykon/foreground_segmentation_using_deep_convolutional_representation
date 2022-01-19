"""
General functions to deal with images.
"""

import random
import numpy as np

"""
Function to get a random patch from a given image.
inputs:
    image_matrix : np array -> Image from wich the patch will be obtained.
    patch_size : tuple -> tuple with shape (height, width)
returns:
    patch : np array with shape (height, width, channels). Note that height and width are defined by patch_size while channels is defined by image_matrix.
"""

def get_random_patch_from_image(image_matrix, patch_size):
    image_matrix_height = image_matrix.shape[0]
    image_matrix_width = image_matrix.shape[1]
    patch_height = patch_size[0]
    patch_width = patch_size[1]
    
    random_height = random.randint(0,image_matrix_height-patch_height-1)
    random_width = random.randint(0,image_matrix_width-patch_width-1)
    
    patch = image_matrix[random_height:random_height+patch_height, random_width:random_width+patch_width, :]
    
    assert patch.shape[0] == patch_height and patch.shape[1] == patch_width
    
    return patch
