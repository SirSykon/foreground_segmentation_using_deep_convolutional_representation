import Autoencoder

class Config:
    TRAINING = False                                                                                                     # Configuration for training.

    TRAINING_DATASET_PATH = "/media/sykon/Maxtor/Files/Work/datasets/COCO/images/train2017"                             # Training dataset folder path.
    CHANGEDETECTON_DATASET_PATH = "../../datasets/change_detection/dataset2014/dataset"                                # We use default dataset position..
    NOISE_CHANGEDETECTON_DATASET_PATH = "../output/noise_change_detection"                                              # We use default dataset position..
    CATEGORIES_TO_TEST = ["baseline","dynamicBackground"]
    
    MAIN_OUTPUT_FOLDER = "../output/"                                                                                   # Main output folder to save all data.
    NETWORK_MODEL_NAME = "conv11"                                                                                        # Network model name.
    SEGMENTATION_OUTPUT_FOLDER = MAIN_OUTPUT_FOLDER + "segmentation/" + NETWORK_MODEL_NAME + "/"
    TRAINING_OUTPUT_SUBFOLDER = "training_output_subfolder/"                                                            # Subfolder to save training output debugging data.
    TRAINING_OUTPUT_SUBFOLDER_PATH = MAIN_OUTPUT_FOLDER + TRAINING_OUTPUT_SUBFOLDER                                     # Subfolder to save testing output debugging data.
    TESTING_OUTPUT_SUBFOLDER = "testing_output_subfolder/"                                                              # Subfolder to save testing output debugging data.
    TESTING_OUTPUT_SUBFOLDER_PATH = MAIN_OUTPUT_FOLDER + TESTING_OUTPUT_SUBFOLDER                                       # Subfolder to save testing output debugging data.
    EVALUATION_OUTPUT_FOLDER = MAIN_OUTPUT_FOLDER + "evaluation/" + NETWORK_MODEL_NAME + "/"
    MODELS_FOLDER = MAIN_OUTPUT_FOLDER + "models/"                                                                      # Path to models folder.
    MODEL_FOLDER_PATH = MODELS_FOLDER+NETWORK_MODEL_NAME+"/"                                                            # Path to network model folder.

    NETWORK_TRAINING_DATA_PATH = MAIN_OUTPUT_FOLDER + "network_training_data/"                                          # Path to network training data.
    NETWORK_TRAINING_DATA_FILES_NAME_STRUCTURE = "training_data_file_{}.npy"

    NOISE = "gaussian_1"                                                                                                # Default noise to use.
    NOISES_LIST = ["gaussian_1", "gaussian_2", "gaussian_3", "uniform_1"]                                               # List of noises to use.
    
    BATCH_SIZE = 64                                                                                                     # Network training batch size.
    EPOCHS = 20                                                                                                         # Network training epochs.
    PATCH_IMG_SIZE = (64,64)                                                                                            # Patches image shape.
    NUM_CHANNELS = 3                                                                                                    # Number of channels.
    TRAINING_DATA_SIZE = 400000                                                                                         # How many instances should we get to train the autoencoder?
    TRAINING_DATA_PER_FILE = 5000                                                                                       # Ho many isntances should we introduce in each file?
    GPU_TO_USE = 0                                                                                                      # GPU to use.
    VALIDATION_DATA_SPLIT_FOR_NETWORK_TRAINING = 0.2                                                                    # Validation split to use during network training.
    STEPS_PER_EPOCH = TRAINING_DATA_SIZE//BATCH_SIZE                                                                    # How many batches do we use to train within each epoch?

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
            self.AUTOENCODER_MODEL = Autoencoder.Convolutional_Autoencoder_2_encoding_decoding_layers_3x3_filters  
        if self.NETWORK_MODEL_NAME == "clean_conv2":
            self.AUTOENCODER_MODEL = Autoencoder.Convolutional_Autoencoder_3_encoding_decoding_layers_3x3_filters 
        if self.NETWORK_MODEL_NAME == "enc1":
            self.SUPPORT_MODEL_NAME = "clean_conv1"
            self.SUPPORT_MODEL_FOLDER_PATH = self.MODELS_FOLDER+SUPPORT_MODEL_NAME+"/"                                          # Path to network model folder.
            self.ENCODER_MODEL = Autoencoder.Convolutional_Encoder_2_encoding_decoding_layers_3x3_filters
        if self.NETWORK_MODEL_NAME == "enc2":
            self.SUPPORT_MODEL_NAME = "clean_conv2"
            self.SUPPORT_MODEL_FOLDER_PATH = self.MODELS_FOLDER+self.SUPPORT_MODEL_NAME+"/"                                     # Path to network model folder.
            self.ENCODER_MODEL = Autoencoder.Convolutional_Encoder_3_encoding_decoding_layers_3x3_filters    
        
        if self.TRAINING:
            self.NETWORK_PROCESSED_REGIONS_OUTPUT_PATH = self.TRAINING_OUTPUT_SUBFOLDER_PATH + self.NETWORK_MODEL_NAME + "/"    # Path to folder to save network processed regions. 
        
        else:
            self.NETWORK_PROCESSED_REGIONS_OUTPUT_PATH = self.TESTING_OUTPUT_SUBFOLDER_PATH + self.NETWORK_MODEL_NAME + "/"     # Path to folder to save network processed regions.

        self.set_noise(self.NOISE)

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