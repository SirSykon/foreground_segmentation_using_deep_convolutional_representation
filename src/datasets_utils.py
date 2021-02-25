import os
from glob import glob
from config import Config

"""
Function to get information from change detection video given its name,
Inputs:
    video_to_get_name : str -> video name to get information from.
Outputs:
    list_images : list -> List of images from chosen video.
    initial_roi_frame -> initial image in temporal Region of Interest.
    last_roi_frame -> last image in temporal Region of Interest (usually the last image in list_images).
"""

def get_original_change_detection_data(video_to_get_name):
    dataset_path = Config.CHANGEDETECTON_DATASET_PATH                                           # We use default dataset position.
    
    categories_path = glob(os.path.join(dataset_path,"*"))                                      # We get categories path.
    for category_path in categories_path:                                                       # For each category...
        videos_in_category_path = glob(os.path.join(category_path, "*"))                        # We get the videos in category path.
        for video_in_category in videos_in_category_path:                                       # For each video in category path...
            _, video_name = os.path.split(video_in_category)                                    # We get the video name
            if video_name == video_to_get_name :                                                # If video name is equal to the video name we are looking for,
                with open(os.path.join(video_in_category, "temporalROI.txt")) as open_file:     # We obtain temporal Region of Interest data.
                    data = open_file.readline()
                    splitted_data = data.split()
                    initial_roi_frame = int(splitted_data[0])
                    last_roi_frame = int(splitted_data[1])
                    
                list_images = sorted(glob(os.path.join(video_in_category,"input","*.jpg")))     # We get the list of images paths.
                
                return list_images, initial_roi_frame, last_roi_frame                           # We return the data
                
"""
Function to get information from change detection video given its name and noise,
Inputs:
    video_to_get_name : str -> video name to get information from.
    noise : stri -> noise added to the video to get data from.
Outputs:
    list_images : list -> List of images from chosen video.
    initial_roi_frame -> initial image in temporal Region of Interest.
    last_roi_frame -> last image in temporal Region of Interest (usually the last image in list_images).
"""

def get_noise_change_detection_data(video_to_get_name, noise):
    dataset_path = Config.NOISE_CHANGEDETECTON_DATASET_PATH                            # We use default dataset position.

    noises_path = glob(os.path.join(dataset_path, "*"))                                 # We get the categories path.
    
    noises_categories_and_videos_list = []
    
    for noise_path in noises_path:                                                                      #For each noise path...
        _, noise_name = os.path.split(noise_path)                                                       # We get the noise name.
        if noise_name == noise:
            print("We get data images from noise {}.".format(noise))
            categories_path = glob(os.path.join(noise_path,"*"))                                        # We get categories path.
            for category_path in categories_path:                                                       # For each category...
                videos_in_category_path = glob(os.path.join(category_path, "*"))                        # We get the videos in category path.
                for video_in_category in videos_in_category_path:                                       # For each video in category path...
                    _, video_name = os.path.split(video_in_category)                                    # We get the video name
                    if video_name == video_to_get_name :                                                # If video name is equal to the video name we are looking for,
                        _, initial_roi_frame, last_roi_frame = get_original_change_detection_data(video_to_get_name)    # We get roi indexes from original dataset.
                        
                        list_images = None
                        
                        if os.path.isdir(os.path.join(video_in_category, "input")):                     # If images are in "input" subfolder.
                            list_images = sorted(glob(os.path.join(video_in_category, "input","*.jpg")))    # We get the list of images paths.
                        else:
                            list_images = sorted(glob(os.path.join(video_in_category,"*.png")))         # We get the list of images paths.
                            if len(list_images) == 0:
                                list_images = sorted(glob(os.path.join(video_in_category,"*.jpeg")))         # We get the list of images paths.
                            
                        return list_images, initial_roi_frame, last_roi_frame                           # We return the data
                
"""
Function to get ground truth images from change detection video given its name,
Inputs:
    video_name_to_get_gt : str -> video name to get gt from.
Outputs:
    list_images : list -> List of images from chosen video.
"""

def get_original_change_detection_gt_data(video_name_to_get_gt):
    dataset_path = Config.CHANGEDETECTON_DATASET_PATH                                  # We use default dataset position.
    
    categories_path = glob(os.path.join(dataset_path,"*"))                                      # We get categories path.
    for category_path in categories_path:                                                       # For each category...
        videos_in_category_path = glob(os.path.join(category_path, "*"))                        # We get the videos in category path.
        for video_in_category in videos_in_category_path:                                       # For each video in category path...
            _, video_name = os.path.split(video_in_category)                                    # We get the video name
            if video_name == video_to_get_name :                                                # If video name is equal to the video name we are looking for,
                    
                list_images = sorted(glob(os.path.join(video_in_category,"groundtruth","*.png")))     # We get the list of images paths.
                
                return list_images                                                              # We return the data

"""
Function to get the list of changedetection videos in dataset_path.
Inputs:
    dataset_path : str -> Path to change detection dataset or a variant one.
Outputs:
    cateogires_and_videos_list : list -> A list with structure [(category_name, video_name)].
"""

def get_change_detection_categories_and_videos_list(dataset_path = None, filter_value = None):

    if dataset_path is None:                                                            # If dataset_path is None, we set it as default.
        dataset_path = Config.CHANGEDETECTON_DATASET_PATH

    categories_path = glob(os.path.join(dataset_path, "*"))                             # We get the categories path.
    
    categories_and_videos_list = []
    
    for category_path in categories_path:                                               # For each category name...
        _, category_name = os.path.split(category_path)                                 # We get the category name
        videos_in_category_path = glob(os.path.join(category_path, "*"))                # We get the videos in category path.
        for video_in_category in videos_in_category_path:                               # For each video in category path...
            _, video_name = os.path.split(video_in_category)                            # We get the video name
            if filter_value is None or category_name==filter_value or video_name==filter_value:
                categories_and_videos_list.append((category_name, video_name))          # We add the tuple to the list.
            
    return categories_and_videos_list                                                   # We return the data
    
"""
Function to get the list of changedetection videos in dataset_path taking in account noises.
Inputs:
    dataset_path : str -> Path to change detection dataset or a variant one.
Outputs:
    cateogires_and_videos_list : list -> A list with structure [(category_name, video_name)].
"""

def get_change_detection_noises_categories_and_videos_list(dataset_path = None, filter_value = None):

    if dataset_path is None:                                                            # If dataset_path is None, we set it as default.
        dataset_path = Config.NOISE_CHANGEDETECTON_DATASET_PATH

    noises_path = glob(os.path.join(dataset_path, "*"))                                 # We get the categories path.
    
    noises_categories_and_videos_list = []
    
    for noise_path in noises_path:                                                                      #For each noise path...
        if os.path.isdir(noise_path):
            _, noise_name = os.path.split(noise_path)                                                       # We get the noise name.
            categories_in_noise_paths = glob(os.path.join(noise_path, "*"))                                 # We get the categories in video path.
            for category_name, video_name in get_change_detection_categories_and_videos_list():             # For each combination of category and video...
                        noises_categories_and_videos_list.append((noise_name, category_name, video_name))   # We add the tuple to the list.
        else:
            print("{} is not folder.".format(noise_path))
            
    return noises_categories_and_videos_list                                                            # We return the data