import Autoencoder

class Config:
    TRAINING = False                                                                                                     # Configuration for training.

    TRAINING_DATASET_PATH = "/media/sykon/Maxtor/Files/Work/datasets/COCO/images/train2017"                                   # Training dataset folder path.
    CHANGEDETECTON_DATASET_PATH = "/media/sykon/Maxtor/Files/Work/datasets/change_detection/dataset2014/dataset"        # We use default dataset position..
    NOISE_CHANGEDETECTON_DATASET_PATH = "/usr/share/Data1/Datasets/changeDetection_noise"                               # We use default dataset position..
    CATEGORIES_TO_TEST = ["baseline","dynamicBackground"]
    
    MAIN_OUTPUT_FOLDER = "../output/"                                                                                   # Main output folder to save all data.
    NETWORK_MODEL_NAME = "conv5"                                                                                        # Network model name.
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
        if self.NETWORK_MODEL_NAME == "conv2":
            self.AUTOENCODER_MODEL = Autoencoder.Convolutional_Autoencoder_2_encoding_decoding_layers_5x5_filters
        if self.NETWORK_MODEL_NAME == "conv3":
            self.AUTOENCODER_MODEL = Autoencoder.Convolutional_Autoencoder_3_encoding_decoding_layers_3x3_filters
        if self.NETWORK_MODEL_NAME == "conv4":
            self.AUTOENCODER_MODEL = Autoencoder.Convolutional_Autoencoder_3_encoding_decoding_layers_5x5_filters
        if self.NETWORK_MODEL_NAME == "conv5":
            self.AUTOENCODER_MODEL = Autoencoder.Convolutional_Autoencoder_4_encoding_decoding_layers_3x3_filters
        if self.NETWORK_MODEL_NAME == "conv6":
            self.AUTOENCODER_MODEL = Autoencoder.Convolutional_Autoencoder_4_encoding_decoding_layers_5x5_filters
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
