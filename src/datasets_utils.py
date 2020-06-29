import os
from glob import glob


def get_original_change_detection_data(video_to_get_name):
    dataset_path = "/usr/share/Data1/Datasets/changeDetection"
    
    categories_path = glob(os.path.join(dataset_path,"*"))    
    for category_path in categories_path:
        videos_in_category_path = glob(os.path.join(category_path, "*"))        
        for video_in_category in videos_in_category_path:
            _, video_name = os.path.split(video_in_category)            
            if video_name == video_to_get_name :
                with open(os.path.join(video_in_category, "temporalROI.txt")) as open_file:
                    data = open_file.readline()
                    splitted_data = data.split()
                    initial_roi_frame = int(splitted_data[0])
                    last_roi_frame = int(splitted_data[1])
                    
                list_images = sorted(glob(os.path.join(video_in_category,"input","*.jpg")))
                
                return list_images, initial_roi_frame, last_roi_frame
                
a, b, c = get_original_change_detection_data("pedestrians")

print(a)
print(b)
print(c)
