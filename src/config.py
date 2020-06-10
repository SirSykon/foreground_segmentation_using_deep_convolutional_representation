
class Config:
    TRAINING = True                                                                                                         # Configuration for training.
    DATASET = "avenue"                                                                                                      # Dataset name to work.
    
    KNOWN_DATASET = False
    if DATASET == "ped2":
        TEST_DATASET_PATH = "/home/jorgegarcia/Documents/Work/experiments/data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test"     # Test dataset folder path.
        TRAINING_DATASET_PATH ="/home/jorgegarcia/Documents/Work/experiments/data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train" # Training dataset folder path.
        KNOWN_DATASET = True
        
    if DATASET == "avenue":
        TEST_DATASET_PATH = "/home/jorgegarcia/Documents/experiments/data/avenue/testing/frames"                        # Test dataset folder path.
        TRAINING_DATASET_PATH = "/home/jorgegarcia/Documents/experiments/data/avenue/training/frames"                   # Training dataset folder path.
        KNOWN_DATASET = True    
    
    if not KNOWN_DATASET:
        raise(NotImplementedError)
    
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
    NETWORK_MODEL_NAME = "network_1"                                                                                    # Network model name.
    MODELS_FOLDER = MAIN_OUTPUT_FOLDER + "models/"                                                                      # Path to models folder.
    MODEL_FOLDER_PATH = MODELS_FOLDER+NETWORK_MODEL_NAME+"/"                                                            # Path to network model folder.
    
    BATCH_SIZE = 64                                                                                                     # Network training batch size.
    EPOCHS = 1000                                                                                                       # Network training epochs.
    STEPS_PER_EPOCH = 500                                                                                               # How many batches do we use to train within each epoch?
    RESHAPE_IMG_SIZE = (64,64)                                                                                          # Resize image shape.
    GPU_TO_USE = 1                                                                                                      # GPU to use.
    VALIDATION_DATA_SPLIT_FOR_NETWORK_TRAINING = 0.2                                                                    # Validation split to use during network training.
