import Autoencoder

class Config:
    TRAINING = False                                                                                                     # Configuration for training.

    TRAINING_DATASET_PATH = "/usr/share/Data1/Datasets/COCO-dataset/images/train2017"                                   # Training dataset folder path.
    CHANGEDETECTON_DATASET_PATH = "/usr/share/Data1/Datasets/changeDetection"                                           # We use default dataset position..
    NOISE_CHANGEDETECTON_DATASET_PATH = "/usr/share/Data1/Datasets/changeDetection_noise"                               # We use default dataset position..
    CATEGORIES_TO_TEST = ["baseline","dynamicBackground"]
    
    MAIN_OUTPUT_FOLDER = "../output/"                                                                                   # Main output folder to save all data.
    NETWORK_MODEL_NAME = "enc1"                                                                                        # Network model name.
    SEGMENTATION_OUTPUT_FOLDER = MAIN_OUTPUT_FOLDER + "segmentation/" + NETWORK_MODEL_NAME + "/"
    TRAINING_OUTPUT_SUBFOLDER = "training_output_subfolder/"                                                            # Subfolder to save training output debugging data.
    TRAINING_OUTPUT_SUBFOLDER_PATH = MAIN_OUTPUT_FOLDER + TRAINING_OUTPUT_SUBFOLDER                                     # Subfolder to save testing output debugging data.
    TESTING_OUTPUT_SUBFOLDER = "testing_output_subfolder/"                                                              # Subfolder to save testing output debugging data.
    TESTING_OUTPUT_SUBFOLDER_PATH = MAIN_OUTPUT_FOLDER + TESTING_OUTPUT_SUBFOLDER                                       # Subfolder to save testing output debugging data.
    EVALUATION_OUTPUT_FOLDER = MAIN_OUTPUT_FOLDER + "evaluation/" + NETWORK_MODEL_NAME + "/"
    MODELS_FOLDER = MAIN_OUTPUT_FOLDER + "models/"                                                                      # Path to models folder.
    MODEL_FOLDER_PATH = MODELS_FOLDER+NETWORK_MODEL_NAME+"/"                                                            # Path to network model folder.
    
    AUTOENCODER_MODEL = None
    if NETWORK_MODEL_NAME == "conv1":
        AUTOENCODER_MODEL = Autoencoder.Convolutional_Autoencoder_2_encoding_decoding_layers_3x3_filters
    if NETWORK_MODEL_NAME == "conv2":
        AUTOENCODER_MODEL = Autoencoder.Convolutional_Autoencoder_2_encoding_decoding_layers_5x5_filters
    if NETWORK_MODEL_NAME == "conv3":
        AUTOENCODER_MODEL = Autoencoder.Convolutional_Autoencoder_3_encoding_decoding_layers_3x3_filters
    if NETWORK_MODEL_NAME == "conv4":
        AUTOENCODER_MODEL = Autoencoder.Convolutional_Autoencoder_3_encoding_decoding_layers_5x5_filters
    if NETWORK_MODEL_NAME == "conv5":
        AUTOENCODER_MODEL = Autoencoder.Convolutional_Autoencoder_4_encoding_decoding_layers_3x3_filters
    if NETWORK_MODEL_NAME == "conv6":
        AUTOENCODER_MODEL = Autoencoder.Convolutional_Autoencoder_4_encoding_decoding_layers_5x5_filters
    if NETWORK_MODEL_NAME == "conv_orig":
        AUTOENCODER_MODEL = Autoencoder.Convolutional_Autoencoder
    if NETWORK_MODEL_NAME == "clean_conv1":
        AUTOENCODER_MODEL = Autoencoder.Convolutional_Autoencoder_2_encoding_decoding_layers_3x3_filters  
    if NETWORK_MODEL_NAME == "clean_conv2":
        AUTOENCODER_MODEL = Autoencoder.Convolutional_Autoencoder_3_encoding_decoding_layers_3x3_filters 
    if NETWORK_MODEL_NAME == "enc1":
        SUPPORT_MODEL_NAME = "clean_conv1"
        SUPPORT_MODEL_FOLDER_PATH = MODELS_FOLDER+SUPPORT_MODEL_NAME+"/"                                                    # Path to network model folder.
        ENCODER_MODEL = Autoencoder.Convolutional_Encoder_2_encoding_decoding_layers_3x3_filters
    if NETWORK_MODEL_NAME == "enc2":
        SUPPORT_MODEL_NAME = "clean_conv2"
        SUPPORT_MODEL_FOLDER_PATH = MODELS_FOLDER+SUPPORT_MODEL_NAME+"/"                                                    # Path to network model folder.
        ENCODER_MODEL = Autoencoder.Convolutional_Encoder_3_encoding_decoding_layers_3x3_filters    
    
    if TRAINING:
        NETWORK_PROCESSED_REGIONS_OUTPUT_PATH = TRAINING_OUTPUT_SUBFOLDER_PATH + NETWORK_MODEL_NAME + "/"               # Path to folder to save network processed regions. 
    
    else:
        NETWORK_PROCESSED_REGIONS_OUTPUT_PATH = TESTING_OUTPUT_SUBFOLDER_PATH + NETWORK_MODEL_NAME + "/"                # Path to folder to save network processed regions.
        
    NETWORK_TRAINING_DATA_PATH = MAIN_OUTPUT_FOLDER + "network_training_data/"                                          # Path to network training data.
    NETWORK_TRAINING_DATA_FILES_NAME_STRUCTURE = "training_data_file_{}.npy"
    
    BATCH_SIZE = 64                                                                                                     # Network training batch size.
    EPOCHS = 20                                                                                                         # Network training epochs.
    STEPS_PER_EPOCH = 5000                                                                                              # How many batches do we use to train within each epoch?
    PATCH_IMG_SIZE = (64,64)                                                                                            # Patches image shape.
    NUM_CHANNELS = 3                                                                                                    # Number of channels.
    TRAINING_DATA_SIZE = 400000                                                                                         # How many instances should we get to train the autoencoder?
    TRAINING_DATA_PER_FILE = 10000                                                                                      # Ho many isntances should we introduce in each file?
    GPU_TO_USE = 1                                                                                                      # GPU to use.
    VALIDATION_DATA_SPLIT_FOR_NETWORK_TRAINING = 0.2                                                                    # Validation split to use during network training.
