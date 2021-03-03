import os
import cv2
import config
import data_utils                           # pylint: disable=no-name-in-module
import datasets_utils

configuration = config.Config()

def add_noise_to_video(video_images_paths, add_noise, output_folder):
    for index, video_image_path in enumerate(video_images_paths):
        print(f"input: {video_image_path}")
        image = cv2.imread(video_image_path)/255.        
        image_with_noise = add_noise(image)
        image_with_noise_path = os.path.join(output_folder, 
                                                "img{:0>6}.png".format(index+1))                # We generate the new image path.
        print(f"output: {image_with_noise_path}")
                                                    
        cv2.imwrite(image_with_noise_path, image_with_noise*255.)                                # We save the new image.

"""
GENERAL INITIALIZATION
"""

dataset_path = configuration.CHANGEDETECTON_DATASET_PATH

for noise_name in configuration.NOISES_LIST:                                                    # For each noise in configuration noises list.
    configuration.set_noise(noise_name)                                                         # We set configuration to use this noise.
    for (category, video_name) in datasets_utils.get_change_detection_categories_and_videos_list(dataset_path = dataset_path, filter_value = None):
        print(noise_name)
        print(category)
        print(video_name)
        video_images_list, video_initial_roi_frame, video_last_roi_frame = datasets_utils.get_original_change_detection_data(video_name)    # We get the change detection sequence data.
        print("ROI")
        print(video_initial_roi_frame)

        # Now we generate the function to add noise to the images.
        add_noise_function = None
        if "gaussian" in configuration.NOISE:
            add_noise_function = data_utils.get_add_gaussian_noise_function(
                                                                gaussian_noise_mean = configuration.GAUSSIAN_NOISE_MEAN, 
                                                                gaussian_noise_standard_desviation = configuration.GAUSSIAN_NOISE_STANDARD_DESVIATION, 
                                                                min_value = 0, 
                                                                max_value = 1)
        if "uniform" in configuration.NOISE:
            add_noise_function = data_utils.get_add_uniform_noise_function(
                                                                uniform_min_value = configuration.UNIFORM_MIN_VALUE, 
                                                                uniform_max_value = configuration.UNIFORM_MAX_VALUE, 
                                                                min_value = 0, 
                                                                max_value = 1)

        output_folder = os.path.join(configuration.NOISE_CHANGEDETECTON_DATASET_PATH,
                                configuration.NOISE,
                                category,
                                video_name)                                                         # We generate the output images folder.

        if not os.path.isdir(output_folder):                                                        # We generate the folder path if it doesn't exist.
            os.makedirs(output_folder)
                        
        add_noise_to_video(video_images_list, add_noise_function, output_folder)                    # We add the noise to the video.
