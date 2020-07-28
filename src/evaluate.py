import evaluation_utils
import datasets_utils
from glob_import glob
import os
import pandas as pd
import numpy as np

changedetection_dataset_path = "/usr/share/Data1/Datasets/changeDetection"                                          # We use default dataset position.

all_data_df = pd.DataFrame, columns = ["noise", "category", "video", "TP", "TN", "FP", "FN", "precision", "recall", "TNR", "FNR", "f-measure"])         # We initialize the all data dataframe.

for (category, video_name) in datasets_utils.get_change_detection_categories_and_videos_list(filter_value = None):              # For each category and video.
    print(category)
    print(video_name)
    _, video_initial_roi_frame, video_last_roi_frame = datasets_utils.get_original_change_detection_data(video_name)            # We get video dataset information.
    print("ROI")
    print(video_initial_roi_frame)
    segmentation_folder = os.path.join(Config.SEGMENTATION_OUTPUT_FOLDER, 
                                        category, 
                                        video_name)                                                                             # We generate segmentation folder path.
    segmented_images_structure = os.path.join(segmentation_folder, "seg{:0>6}.png")                                             # We generate segmentation images path structure.
    gt_images_structure = os.path.join(changedetection_dataset_path, category, video_name, "groundtruth", "gt{:0>6}.png")       # We generate ground truth images path structure.
    
    TP_video_sum, TN_video_sum, FP_video_sum, FN_video_sum, precision_video_sum, recall_video_sum, TNR_video_sum, FNR_video_sum, f_measure_video_sum = evaluation_utils.compare_video(segmented_images_structure, gt_images_structure, video_initial_roi_frame, video_last_roi_frame)       # We get the comparison between gt and segmentation images.
                                        
    video_row_serie = pd.Serie([noise, category, video_name, TP_video_sum, TN_video_sum, FP_video_sum, FN_video_sum, precision_video_sum, recall_video_sum, TNR_video_sum, FNR_video_sum, f_measure_video_sum], index = all_data_df.columns)                                                      # We initialize the serie.
    video_row_serie.name = noise+"_"+category+"_"+video_name                                            # We set a name to the serie.
    
    all_data_df = all_data_df.append(video_row_serie)                                                   # We append the row with evaluation data to the all data dataframe.
    
    evaluation_folder = Config.EVALUATION_OUTPUT_FOLDER
    
if not os.path.isdir(evaluation_folder):
    os.makedirs(evaluation_folder)
    
    all_data_df.to_csv(path_or_buf = os.path.join(evaluation_folder, "evaluation.csv"))
        
    
    
    
    
        
