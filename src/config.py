
class Config:
    TRAINING = True                                                                                                     # Configuration for training.

    TRAINING_DATASET_PATH = "/usr/share/Data1/Datasets/COCO-dataset/images/train2017"                                   # Training dataset folder path.
    TEST_DATASET_PATH = "/home/jorgegarcia/Documents/experiments/data/avenue/testing/frames"                            # Test dataset folder path.
    
    MAIN_OUTPUT_FOLDER = "../output/segment1/"                                                                          # Main output folder to save all data.
    TRAINING_OUTPUT_SUBFOLDER = "training_output_subfolder/"                                                            # Subfolder to save training output debugging data.
    TRAINING_OUTPUT_SUBFOLDER_PATH = MAIN_OUTPUT_FOLDER + TRAINING_OUTPUT_SUBFOLDER                                     # Subfolder to save testing output debugging data.
    TESTING_OUTPUT_SUBFOLDER = "testing_output_subfolder/"                                                              # Subfolder to save testing output debugging data.
    TESTING_OUTPUT_SUBFOLDER_PATH = MAIN_OUTPUT_FOLDER + TESTING_OUTPUT_SUBFOLDER                                       # Subfolder to save testing output debugging data.
    
    if TRAINING:
        NETWORK_PROCESSED_REGIONS_OUTPUT_PATH = TRAINING_OUTPUT_SUBFOLDER_PATH + "autoencoded_regions/"                 # Path to folder to save network processed regions. 
    
    else:
        NETWORK_PROCESSED_REGIONS_OUTPUT_PATH = TESTING_OUTPUT_SUBFOLDER_PATH + "autoencoded_regions/"                  # Path to folder to save network processed regions.
        
    NETWORK_TRAINING_DATA_PATH = MAIN_OUTPUT_FOLDER + "network_training_data/"                                          # Path to network training data.
    NETWORK_TRAINING_DATA_FILES_NAME_STRUCTURE = "training_data_file_{}.npy"
    NETWORK_MODEL_NAME = "network_1"                                                                                    # Network model name.
    MODELS_FOLDER = MAIN_OUTPUT_FOLDER + "models/"                                                                      # Path to models folder.
    MODEL_FOLDER_PATH = MODELS_FOLDER+NETWORK_MODEL_NAME+"/"                                                            # Path to network model folder.
    
    BATCH_SIZE = 64                                                                                                     # Network training batch size.
    EPOCHS = 1000                                                                                                       # Network training epochs.
    STEPS_PER_EPOCH = 500                                                                                               # How many batches do we use to train within each epoch?
    PATCH_IMG_SIZE = (64,64)                                                                                            # Patches image shape.
    TRAINING_DATA_SIZE = 400000                                                                                         # How many instances should we get to train the autoencoder?
    TRAINING_DATA_PER_FILE = 10000                                                                                      # Ho many isntances should we introduce in each file?
    GPU_TO_USE = 1                                                                                                      # GPU to use.
    VALIDATION_DATA_SPLIT_FOR_NETWORK_TRAINING = 0.2                                                                    # Validation split to use during network training.