import os
import config
import data_utils                           # pylint: disable=no-name-in-module
import datasets_utils
import sys
import pkgutil

configuration = config.Config()

sys.path.append(configuration.BGS_FOLDER)
sys.path.append(os.path.join(configuration.BGS_FOLDER, "build"))
print(sys.path)
search_path = ['.'] # set to None to see all modules importable from sys.path
all_modules = [x[1] for x in pkgutil.iter_modules(path=search_path)]
print(all_modules)
from bgs_handler import apply_bgs_to_frame_sequence
"""
GENERAL INITIALIZATION
"""
print(configuration.CATEGORIES_TO_TEST)
for (noise, category, video_name) in datasets_utils.get_change_detection_noises_categories_and_videos_list():
    print(noise)
    print(category)
    print(video_name)

    if category in configuration.CATEGORIES_TO_TEST and noise in configuration.NOISES_LIST:
        video_folder = os.path.join(configuration.NOISE_CHANGEDETECTON_DATASET_PATH, noise, category, video_name)
        print(video_folder)
        segmentation_folder_suffix = os.path.join(noise,
                                            category, 
                                            video_name)

        print("segmentation folder suffix: {}.".format(segmentation_folder_suffix))
        
        apply_bgs_to_frame_sequence(video_folder, configuration.RIVALS_MAIN_OUTPUT_FOLDER, segmentation_folder_suffix)

    else:
        print("We skip category {}.".format(category))
