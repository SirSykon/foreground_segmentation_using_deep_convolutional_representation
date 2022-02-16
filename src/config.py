import Autoencoder
import os

import data_utils

class Config:
    """
    General configuration.
    """
    CONFIG_FILE_PATH = os.path.abspath("./config.py")
    TRAINING = False                                                                                                        # Configuration for training.
    NETWORK_MODEL_NAME = "conv7"                                                                                           # Network model name.
    CATEGORIES_TO_TEST = ["baseline","dynamicBackground","badWeather","shadow", "lowFramerate", "intermittentObjectMotion", "nightVideos", "turbulence", "thermal"]
    CATEGORIES_TO_TEST = ["baseline","dynamicBackground","badWeather", "nightVideos", "turbulence", "thermal"]
    CATEGORIES_TO_TEST = ["baseline","badWeather", "thermal"]
    CATEGORIES_TO_TEST = ["dynamicBackground","baseline"]

    EVALUATE_METHODS_LIST = [   "conv5/generic",
                                "conv5/specific_to_sequence",
                                "conv5/specific_to_original_video_with_added_noise", 
                                "conv7/generic",
                                "conv7/specific_to_sequence",
                                "conv7/specific_to_original_video_with_added_noise",
                                "conv9/generic",
                                "conv9/specific_to_sequence",
                                "conv9/specific_to_original_video_with_added_noise", 
                                "LOBSTER",
                                "PAWCS",
                                "SuBSENSE"]                                                                 # Methods names to be evaluated.

    NOISES_LIST = ["gaussian_1", "gaussian_2", "gaussian_3", "uniform_1"]                                                   # List of noises to use.
    #NOISES_LIST = ["uniform_1"]                                                                                            # List of noises to use.

    TYPE_OF_AUTOENCODER = "SPECIFIC_TO_SEQUENCE"                                                     # Type of AE to be trained. Options:
                                                                                                                            #   SPECIFIC_TO_ORIGINAL_VIDEO_WITH_ADDED_NOISE - One autoencoder for each sequence by adding noise on the fly to the original video.
                                                                                                                            #   SPECIFIC_TO_SEQUENCE - One autoencoder for each sequence with included noise.
                                                                                                                            #   GENERIC - One autoencoder for all sequences. Trained with imagenet.

    NOISE_TO_ADD_TO_INPUT_DATA = None
    NOISE_TO_ADD_TO_OUTPUT_DATA = None

    """
    Folders configuration.
    """

    MAIN_OUTPUT_FOLDER = "../output/"                                                                                       # Main output folder to save all data.
    SEGMENTATION_OUTPUT_FOLDER = MAIN_OUTPUT_FOLDER + "segmentation/" + NETWORK_MODEL_NAME + "/"
    TRAINING_OUTPUT_SUBFOLDER_PATH = MAIN_OUTPUT_FOLDER + "training_output_subfolder/"                                      # Subfolder to save testing output debugging data.
    TESTING_OUTPUT_SUBFOLDER_PATH = MAIN_OUTPUT_FOLDER + "testing_output_subfolder/"                                        # Subfolder to save testing output debugging data.
    EVALUATION_OUTPUT_FOLDER = MAIN_OUTPUT_FOLDER + "evaluation/" + NETWORK_MODEL_NAME + "/"                                # Subfolder to save evaluation.
    MODELS_FOLDER = MAIN_OUTPUT_FOLDER + "models/"                                                                          # Path to models folder.
    MODEL_FOLDER_PATH = MODELS_FOLDER+NETWORK_MODEL_NAME+"/"                                                                # Path to network model folder.

    TRAINING_DATASET_PATH = "/media/jrggcgz/Maxtor/Files/Work/datasets/COCO/images/train2017"                               # Training dataset folder path.
    CHANGEDETECTON_DATASET_PATH = "../../datasets/change_detection/dataset2014/dataset"                                     # We use default dataset position..
    NOISE_CHANGEDETECTON_DATASET_PATH = "../../datasets/noise_change_detection"                                             # We use default dataset position..

    NETWORK_TRAINING_DATA_PATH = MAIN_OUTPUT_FOLDER + "network_training_data/"                                              # Path to network training data.
    NETWORK_TRAINING_DATA_FILES_NAME_STRUCTURE = "training_data_file_{}.npy"                                                # Autoencoder training data files structures.

    BGS_FOLDER = os.path.abspath("../../bgslibrary/")

    RIVALS_MAIN_OUTPUT_FOLDER = "../output/rivals/"

    """
    Training configuration.
    """
    
    BATCH_SIZE = 64                                                                                                         # Network training batch size.
    EPOCHS = 20                                                                                                             # Network training epochs.
    PATCH_IMG_SIZE = (64,64)                                                                                                # Patches image shape.
    NUM_CHANNELS = 3                                                                                                        # Number of channels.
    TRAINING_DATA_SIZE = 400000                                                                                             # How many instances should we get to train the autoencoder?
    TRAINING_DATA_PER_FILE = 5000                                                                                           # Ho many isntances should we introduce in each file?
    SPECIFIC_SEQUENCE_TRAINING_DATA_SIZE = 40000
    GPU_TO_USE = 0                                                                                                          # GPU to use.
    VALIDATION_DATA_SPLIT_FOR_NETWORK_TRAINING = 0.2                                                                        # Validation split to use during network training.

    def __init__(self):
        self.AUTOENCODER_MODEL = None
        if self.NETWORK_MODEL_NAME == "conv1":
            self.AUTOENCODER_MODEL = Autoencoder.Convolutional_Autoencoder_2_encoding_decoding_layers_3x3_filters
            self.L = 32
        if self.NETWORK_MODEL_NAME == "conv2":
            self.AUTOENCODER_MODEL = Autoencoder.Convolutional_Autoencoder_2_encoding_decoding_layers_5x5_filters
            self.L = 32
        if self.NETWORK_MODEL_NAME == "conv3":
            self.AUTOENCODER_MODEL = Autoencoder.Convolutional_Autoencoder_3_encoding_decoding_layers_3x3_filters
            self.L = 16
        if self.NETWORK_MODEL_NAME == "conv4":
            self.AUTOENCODER_MODEL = Autoencoder.Convolutional_Autoencoder_3_encoding_decoding_layers_5x5_filters
            self.L = 16
        if self.NETWORK_MODEL_NAME == "conv5":
            self.AUTOENCODER_MODEL = Autoencoder.Convolutional_Autoencoder_4_encoding_decoding_layers_3x3_filters
            self.L = 8
        if self.NETWORK_MODEL_NAME == "conv6":
            self.AUTOENCODER_MODEL = Autoencoder.Convolutional_Autoencoder_4_encoding_decoding_layers_5x5_filters
            self.L = 8
        if self.NETWORK_MODEL_NAME == "conv7":
            self.AUTOENCODER_MODEL = Autoencoder.Convolutional_Autoencoder_5_encoding_decoding_layers_3x3_filters
            self.L = 4
        if self.NETWORK_MODEL_NAME == "conv8":
            raise(NotImplementedError)
        if self.NETWORK_MODEL_NAME == "conv9":
            self.AUTOENCODER_MODEL = Autoencoder.Convolutional_Autoencoder_6_encoding_decoding_layers_3x3_filters
            self.L = 2
        if self.NETWORK_MODEL_NAME == "conv10":
            self.AUTOENCODER_MODEL = Autoencoder.Convolutional_Autoencoder_6_encoding_decoding_layers_5x5_filters
            self.L = 2
        if self.NETWORK_MODEL_NAME == "conv11":
            self.AUTOENCODER_MODEL = Autoencoder.Convolutional_Autoencoder_2_encoding_decoding_layers_with_maxpool_3x3_filters
            self.L=32
        if self.NETWORK_MODEL_NAME == "conv_orig":
            self.AUTOENCODER_MODEL = Autoencoder.Convolutional_Autoencoder
        if self.NETWORK_MODEL_NAME == "clean_conv1":
            raise(NotImplementedError)
            self.AUTOENCODER_MODEL = Autoencoder.Convolutional_Autoencoder_2_encoding_decoding_layers_3x3_filters  
        if self.NETWORK_MODEL_NAME == "clean_conv2":
            raise(NotImplementedError)
            self.AUTOENCODER_MODEL = Autoencoder.Convolutional_Autoencoder_3_encoding_decoding_layers_3x3_filters 
        if self.NETWORK_MODEL_NAME == "enc1":
            raise(NotImplementedError)
            self.SUPPORT_MODEL_NAME = "clean_conv1"
            self.SUPPORT_MODEL_FOLDER_PATH = self.MODELS_FOLDER+self.SUPPORT_MODEL_NAME+"/"                                          # Path to network model folder.
            self.ENCODER_MODEL = Autoencoder.Convolutional_Encoder_2_encoding_decoding_layers_3x3_filters
        if self.NETWORK_MODEL_NAME == "enc2":
            raise(NotImplementedError)
            self.SUPPORT_MODEL_NAME = "clean_conv2"
            self.SUPPORT_MODEL_FOLDER_PATH = self.MODELS_FOLDER+self.SUPPORT_MODEL_NAME+"/"                                     # Path to network model folder.
            self.ENCODER_MODEL = Autoencoder.Convolutional_Encoder_3_encoding_decoding_layers_3x3_filters    
        
        if self.TRAINING:
            self.NETWORK_PROCESSED_REGIONS_OUTPUT_PATH = self.TRAINING_OUTPUT_SUBFOLDER_PATH + self.NETWORK_MODEL_NAME + "/"    # Path to folder to save network processed regions. 
        
        else:
            self.NETWORK_PROCESSED_REGIONS_OUTPUT_PATH = self.TESTING_OUTPUT_SUBFOLDER_PATH + self.NETWORK_MODEL_NAME + "/"     # Path to folder to save network processed regions.

        if self.TYPE_OF_AUTOENCODER == "GENERIC":
            self.STEPS_PER_EPOCH = self.TRAINING_DATA_SIZE//self.BATCH_SIZE                                                     # How many batches do we use to train within each epoch?
        else:
            self.STEPS_PER_EPOCH = self.SPECIFIC_SEQUENCE_TRAINING_DATA_SIZE//self.BATCH_SIZE                                   # How many batches do we use to train within each epoch?
        
        self.set_noise(self.NOISES_LIST[0])

        self.ALL_SEGMENTATION_OUTPUT_FOLDER = [os.path.join(self.MAIN_OUTPUT_FOLDER, "segmentation/", method_name) for method_name in self.EVALUATE_METHODS_LIST]


    def set_noise(self, noise_name):
        self.NOISE = noise_name

        if self.NOISE == "gaussian_1":
            self.GAUSSIAN_NOISE_MEAN = 0
            self.GAUSSIAN_NOISE_STANDARD_DESVIATION = 0.1

        if self.NOISE == "gaussian_2":
            self.GAUSSIAN_NOISE_MEAN = 0
            self.GAUSSIAN_NOISE_STANDARD_DESVIATION = 0.2

        if self.NOISE == "gaussian_3":
            self.GAUSSIAN_NOISE_MEAN = 0
            self.GAUSSIAN_NOISE_STANDARD_DESVIATION = 0.3

        if self.NOISE == "uniform_1":
            self.UNIFORM_MIN_VALUE = -0.5
            self.UNIFORM_MAX_VALUE = 0.5

    def get_noise_function(self, noise_name):

        if noise_name == "gaussian_1":
            return data_utils.get_add_gaussian_noise_function(gaussian_noise_mean = 0, gaussian_noise_standard_desviation = 0.1)

        if noise_name == "gaussian_2":
            return data_utils.get_add_gaussian_noise_function(gaussian_noise_mean = 0, gaussian_noise_standard_desviation = 0.2)

        if noise_name == "gaussian_3":
            return data_utils.get_add_gaussian_noise_function(gaussian_noise_mean = 0, gaussian_noise_standard_desviation = 0.3)

        if noise_name == "uniform_1":
            return data_utils.get_add_uniform_noise_function(uniform_min_value=-0.5, uniform_max_value=0.5)

        return None
            

