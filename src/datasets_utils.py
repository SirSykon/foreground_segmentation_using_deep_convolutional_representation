import os
from glob import glob


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
    dataset_path = "/usr/share/Data1/Datasets/changeDetection"                                  # We use default dataset position.
    
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
Function to get the list of changedetection videos in dataset_path.
Inputs:
    dataset_path : str -> Path to change detection dataset or a variant one.
Outputs:
    cateogires_and_videos_list : list -> A list with structure [(category_name, video_name)].
"""

def get_change_detection_categories_and_videos_list(dataset_path = None):

    if dataset_path is None:                                                            # If dataset_path is None, we set it as default.
        dataset_path = "/usr/share/Data1/Datasets/changeDetection"

    categories_path = glob(os.path.join(dataset_path, "*"))                             # We get the categories path.
    
    categories_and_videos_list = []
    
    for category_path in categories_path:                                               # For each category name...
        _, category_name = os.path.split(category_path)                                 # We get the category name
        videos_in_category_path = glob(os.path.join(category_path, "*"))                # We get the videos in category path.
        for video_in_category in videos_in_category_path:                               # For each video in category path...
            _, video_name = os.path.split(video_in_category)                            # We get the video name
            categories_and_videos_list.append((category_name, video_name))              # We add the tuple to the list.
            
    return categories_and_videos_list                                                   # We return the data
        
print(get_change_detection_categories_and_videos_list())        
a, b, c = get_original_change_detection_data("pedestrians")
print(a)
print(b)
print(c)


