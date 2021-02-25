"""
Utilities to deal with data.
"""

import random
import numpy as np

def load_dataset(dataset_path, split = True, normalization_range = (0,1), number_of_files_to_load = 100):
"""
Function to load an images dataset from files.
inputs:
    dataset_path : str -> Path to the data files.
    validation_split : float in range (0,1) -> Validation split.
    split : boolean -> Do we split the dataset into training and validation set?
    normalization_range : tuple (0,1) or (-1,1) -> Range to noormalize the data into. All data is supossed to be stored in range (0,255)
    number_of_files_to_load : int -> Maximum number of files to load.
returns:
    If split is True, two numpy arrays are returned. Else, only one with all data.
    
"""

    all_dataset = None                                                                                      # Variable to store the complete dataset.
    list_of_data_paths_for_training = sorted(glob(os.path.join(dataset_path, "*")))                         # We get the complete list of training data files path.
    for data_file_path_index, data_file_path in enumerate(list_of_data_paths_for_training):                 # For each file paths...
        print("Loading {}.".format(data_file_path))
        data_file = np.load(data_file_path)                                                                 # We load the file
        if all_dataset is None:                                                                             # If the variable all_dataset has neot been initalized, 
            all_dataset = data_file                                                                         # we initialize it with the data loaded.
        else:                                                                                               # If the variable all_dataset has been already initialized,
            all_dataset = np.concatenate((all_dataset, data_file), axis = 0)                                # We concatenate the data to the data already store in the variable.

        if data_file_path_index == number_of_files_to_load - 1:                                             # If we have already loaded the indicated number of files to load,
            break                                                                                           # W break the loop. This is ugly but easy, don't blame me, blame the system.

    dataset_size = all_dataset.shape[0]                                                                     # We get the dataset size.

    if normalization_range == (0,1):                                                                        # We normalize the dataset to the range indicated.
        all_dataset = all_dataset/255.
    if normalization_range == (-1,1):
        all_dataset = (all_dataset/127.5)-1    
    
    if split:                                                                                               # If we must divide the dataset intotraining and validation set,
        training_dataset_size = math.floor(dataset_size*(1-validation_splitG))                              # We calculate how much data will be used as training set and get it.
        print("Dataset size")
        print(dataset_size)
        training_dataset = all_dataset[:training_dataset_size]
        
        print("Training dataset shape is {}.".format(training_dataset.shape))
        print(np.max(training_dataset))
        print(np.min(training_dataset))

        validation_dataset = all_dataset[training_dataset_size:]                                            # We get all non training data as validation set.
        print("Validation dataset shape is {}.".format(validation_dataset.shape))
        print(np.max(validation_dataset))
        print(np.min(validation_dataset))
        
        return training_dataset, validation_dataset                                                         # We return training and validation data sets.
    
    else:                                                                                                   # If the data must not be splitted, we return all the data.
        return all_dataset
        
def load_random_file(data_files_paths):
"""
Method to load a random file from data_files_pathss and return it.
"""
    number_of_files = len(data_files_paths)                 # We get the number of files paths.
    next_file_index = random.randint(0, number_of_files-1)  # We get a random next file to load index.
    return np.load(data_files_paths[next_file_index])       # We load the npy file and return it.
    
def data_generator(data_files_paths = None, batch_size = 64, change_file_after_getting_x_data_batches = 1):
"""
This generator will get a list of paths to files with data. The generator will load a random file, get X data from it and load another random file.
inputs:    
    data_files_paths : list of strs -> paths to dataset files to load.
    change_file_after_getting_x_data_batches : int >=1 -> Number of data batches that we will get before load another file randomly.
    batch_size : int >=1 -> Number of data instances to return in each iteration.
output:
    bacth : np array -> Data matrix.
"""
    
    current_data = load_random_file(data_files_paths)                                                       # We load  a random file from data_files_paths
    number_of_batches_extracted_after_last_file_load = 0                                                    # We initialize the counter to know how many batches we have extracted after last file load.
        
    while True:                                                                                             # We start to generate...
        indices = np.random.randint(current_data.shape[0], size=batch_size)                                 # We get the random indices.
        batch = current_data[indices]                                                                       # We get extract the batch
        number_of_batches_extracted_after_last_file_load += 1                                               # We increase the counter.
            
        if number_of_batches_extracted_after_last_file_load == change_file_after_getting_x_data_batches-1:  # if the counter have reached change_file_after_getting_x_data_batches,
            current_data = load_random_file(data_files_paths)                                               # We load a new random file.
            number_of_batches_extracted_after_last_file_load = 0                                            # We reset the counter.
                
        yield np.array(batch)                                                                               # We return the batch.

def autoencoder_data_generator(source_generator, preprocessing_function = None, x_preprocessing_function = None, y_preprocessing_function = None):
"""
This function is a generator that uses another generator, to adapt its output to an autoencoder training.
inputs:
    source_generator : generator -> a generator to get batches of numpy arays.
    proprocessing_function : function -> function to apply to the data obtained from source_generator.
    y_preprocessing_function : function -> function to apply to the y_batch.

output:
    x_data : batch -> Batch of data after apply proprocessing_function (if exists) to data from source_generator.
    y_data : batch -> Batch of data after apply y_proprocessing_function (if exists) to x_data.

"""

    for data in source_generator:                       # For each data from the source_generator...
        x_data = data.copy()                            # We copy the batch.
        if not preprocessing_function is None:          # If the preprocessing function exists,
            x_data = preprocessing_function(x_data)     # We apply it to the batch.
        y_data = x_data.copy()                          # We copy the batch as objective y_batch.
        if not x_preprocessing_function is None:        # If x_preprocessing_function_exists,
            x_data = x_preprocessing_function(x_data)   # We apply it to the x_batch.
        if not y_preprocessing_function is None:        # If y_preprocessing_function_exists,
            y_data = y_preprocessing_function(y_data)   # We apply it to the y_batch.
            
        yield (x_data, y_data)                         # We return both x and y batches.
        
"""
NOISES FUNCTIONS
"""

def add_gaussian_noise(matrix, gaussian_noise_mean = 0, gaussian_noise_standard_desviation = 0.2, min_value = 0, max_value = 1):
"""
This function returns a matrix modified by gaussian noise.
inputs:
    matrix : numpy matrix -> A matrix to add gaussian noise.
    gaussian_noise_mean : int -> Gaussian noise mean.
    gaussian_noise_standard_desviation : float -> Gaussian noise standard desviation.
    min_value : int -> minimum value in the output matrix.
    max_value : int maximum value in the output matrix.

outputs:
    clipped_noise_added_matrix : numpy matrix -> Matrix output from adding gaussian noise to input matrix but ensuring no value lower than min_value and no value greater than max_value.
"""
    gaussian_matrix = np.random.normal(loc = gaussian_noise_mean, scale = gaussian_noise_standard_desviation, size = matrix.shape)
    noise_added_matrix = matrix + gaussian_matrix
    clipped_noise_added_matrix = np.clip(noise_added_matrix, min_value, max_value)
    return clipped_noise_added_matrix

def add_uniform_noise(matrix, uniform_min_value = -0.5, uniform_max_value = 0.5, min_value = 0, max_value = 1):
"""
This function returns a matrix modified by uniform noise.
inputs:
    matrix : numpy matrix -> A matrix to add gaussian noise.
    uniform_min_value : float -> Minimum value to use to generate a Uniform distribution.
    uniform_max_value : float -> Maximum value to use to generate a Univorm distribution.
    min_value : int -> minimum value in the output matrix.
    max_value : int maximum value in the output matrix.

outputs:
    clipped_noise_added_matrix : numpy matrix -> Matrix output from adding uniform noise to input matrix but ensuring no value lower than min_value and no value greater than max_value.
"""
    uniform_matrix = np.random.normaluniform(low = uniform_min_value, high = uniform_max_value, size = matrix.shape)
    noise_added_matrix = matrix + uniform_matrix
    clipped_noise_added_matrix = np.clip(noise_added_matrix, min_value, max_value)
    return clipped_noise_added_matrix
        
