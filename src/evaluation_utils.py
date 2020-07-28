import numpy as np
import pandas as pd

"""
Function to perform the comparison between a segmented image and a groundtruth. We gess both are grayscale images with channels between 0. and 255.
    inputs:
        img : numpy array -> segmented image matrix.
        gt :  numpy array -> groundtruth image matrix.
        img_negative_threshold -> The image value treshold we will consider as negative.
        img_positive_threshold -> The image value trshold we will consider as positive.
        gt_negative_threshold -> The ground truth value treshold we will consider as negative.
        gt_positive_threshold -> The ground truth value trshold we will consider as positive.
    outputs:
        tuple with TP, TN, FP, FN
"""

def compare_gt_and_img(img, gt, img_negative_threshold = 0., img_positive_threshold = 255.,  gt_negative_threshold = 0., gt_positive_threshold = 255.):

    assert len(img.shape) < 4
    assert len(img.shape) > 1

    if len(img.shape) == 3:
        img_to_use = img[:,:,0].copy()
    else:
        img_to_use = img.copy()
        
    print(img_to_use)
    assert len(gt.shape) < 4
    assert len(gt.shape) > 1

    if len(gt.shape) == 3:
        gt_to_use = gt[:,:,0]
    else:
        gt_to_use = gt.copy() 
        
    print(gt_to_use)
    assert img.shape == gt.shape

    img_negative = img_to_use <= img_negative_threshold
    print("img_negative")
    print(img_negative)
    img_positive = img_to_use >= img_positive_threshold
    print("img_positive")
    print(img_positive)

    gt_negative = gt_to_use <= gt_negative_threshold
    print("gt_negative")
    print(gt_negative)
    gt_positive = gt_to_use >= gt_positive_threshold
    print("gt_positive")
    print(gt_positive)

    TP_mask = np.logical_and(img_positive, gt_positive)*1.
    TN_mask = np.logical_and(img_negative, gt_negative)*1.
    FP_mask = np.logical_and(img_positive, gt_negative)*1.
    FN_mask = np.logical_and(img_negative, gt_positive)*1.

    print("TP mask")
    print(TP_mask)

    print("TN mask")
    print(TN_mask)

    print("FP_mask")
    print(FP_mask)

    print("FN_mask")
    print(FN_mask)

    TP = np.sum(TP_mask)
    TN = np.sum(TN_mask)
    FP = np.sum(FP_mask)
    FN = np.sum(FN_mask)

    return TP, TN, FP, FN

"""
Function to get statistics from foreground segmentation video compared to groundtruth.
    inputs:
        img_path_structure : str -> string with the image path structure. For example: /some/path/seg_img_{:0>6}.png
        gt_path_structure : str -> string with the groundtruth path structure. For example: /some/path/gt_structure_{:0>6}.png
        temporal_roi_start : int -> The number of the first image within temporal roi.
        temporal_roi_end : int -> The number of the last image within temporal roi.
"""
def compare_video(img_path_structure, gt_path_structure, temporal_roi_start, temporal_roi_end):

    video_statistics = []
    for offset in range(temporal_roi_end - temporal_roi_start):
        image_index = temporal_roi_start + offset
        img_path = img_path_structure.format(image_index)
        gt_path = gt_path_structure.format(image_index)
        print(img_path)
        print(gt_path)
        
        img = cv2.imread(img_path)
        gt = cv2.imread(gt_path)
        
        TP, TN, FP, FN = compare_video(img, gt)
        
        video_statistics.append(np.array([TP, TN, FP, FN]))
        
    video_statistics = np.array(video_statistics)
    
    video_statistics_sum = np.sum(video_statistics, axis = 0)
    
    print(video_statistics_sum)
    
    TP_video_sum = video_statistics_sum[0]
    TN_video_sum = video_statistics_sum[1]
    FP_video_sum = video_statistics_sum[2]
    FN_video_sum = video_statistics_sum[3]
    
    precision_video_sum = TP_video_sum/(TP_video_sum+FP_video_sum)
    recall_video_sum = TP_video_sum/(TP_video_sum+FN_video_sum)
    TNR_video_sum = TN_video_sum/(TN_video_sum+FP_video_sum)
    FNR_video_sum = FN_video_sum/(FN_video_sum+TP_video_sum)
    
    f_measure_video_sum = (2*precision_video_sum*recall_video_sum)/(precision_video_sum + recall_video_sum)
    
    return TP_video_sum, TN_video_sum, FP_video_sum, FN_video_sum, precision_video_sum, recall_video_sum, TNR_video_sum, FNR_video_sum, f_measure_video_sum
