import os
import time
import numpy as np
import random
import shutil
from tqdm import tqdm
from glob import glob
import tensorflow as tf
import tensorflow.keras.layers as layers
import config
import GPU_utils as GPU_utils               # pylint: disable=no-name-in-module
import data_utils                           # pylint: disable=no-name-in-module
import datasets_utils
from contextlib import redirect_stdout

configuration = config.Config()
GPU_utils.tensorflow_2_x_dark_magic_to_restrict_memory_use(configuration.GPU_TO_USE)

if not os.path.isdir(configuration.MODEL_FOLDER_PATH):             # If the folder to save the model does not exist,
    os.makedirs(configuration.MODEL_FOLDER_PATH)                   # We create it.

"""
Function to normalize.
inputs:
    data : numpy array -> Data to normalize.
    normalization_range -> Range to normalize data into.
returns:
    normalized data
"""
def normalize_data(data, normalization_range=(0,1)):
    if normalization_range == (0,1):
        assert np.max(data) <= 255.
        assert np.min(data) >= 0.
        return data/255.
    
    if normalization_range == (-1,1):
        assert np.max(data) <= 255.
        assert np.min(data) >= 0.
        return data/127.5 - 1

"""
Function to denormalize.
inputs:
    data : numpy array -> Data to denormalize.
    normalization_range -> Range to denormalize data from.
returns:
    denormalized data
"""
def denormalize_data(data, normalization_range=(0,1)):
    if normalization_range == (0,1):
        assert np.max(data) <= 1.
        assert np.min(data) >= 0.
        return data*255.
    
    if normalization_range == (-1,1):
        assert np.max(data) <= 1.
        assert np.min(data) >= -1.
        return (data + 1)*127.5

reconstruction_loss = tf.keras.losses.MeanSquaredError()    # We define the reconstruction loss function.

"""
Function to create a convolutional autoencoder model
"""
def make_convolutional_autoencoder_model():
    return configuration.AUTOENCODER_MODEL(configuration.MODEL_FOLDER_PATH, load = False)

"""
Function to execute a train step.
inputs:
    images_batch : numpy array -> batch of images.
    autoencoder : model -> autoencoder model to train.
    aut_optimizer : optimizer -> optimizer to apply gradients to the model.
"""

def train_step(x_batch, y_batch, autoencoder, aut_optimizer):

    with tf.GradientTape() as aut_tape:
        reconstructed_x_batch = autoencoder(x_batch, training=True)                         # We apply the model to the batch.
        aut_loss = reconstruction_loss(y_batch, reconstructed_x_batch)                      # We calculate the autoencoded batch reconstruction loss.
    
    gradients_of_aut = aut_tape.gradient(aut_loss, autoencoder.trainable_variables)         # We get the gradients.
    aut_optimizer.apply_gradients(zip(gradients_of_aut, autoencoder.trainable_variables))   # We use the optimizer to apply the gradients to the model.

    return aut_loss, reconstructed_x_batch                                                  # We return the autoencoder loss and the reconstruction.

"""
Function to execute the training loop.
inputs:
    training_dataset_generator : generator -> Generator to obtain batches to train.
    validation_dataset_generator : generator -> Generator to obtain batches to validate.
    autoencoder : model -> Autoencoder model.
    aut_optimizer : optimizer -> autoencoder to use with autoencoder.
    checkpoint : checkpoint
    training_steps_per_epoch : Number of training steps for each epoch.
    validation_steps_per_epoch : Number of validation steps for each epoch.
    vistual_test_clip : image -> Image to apply the model and save the output as visual reference.
"""
def train(training_dataset_generator, validation_dataset_generator, autoencoder, autoencoder_optimizer, checkpoint, checkpoint_prefix, training_steps_per_epoch, validation_steps_per_epoch, visual_test_clip=None):
    #train_step_tf = tf.function(train_step).get_concrete_function(w, x, y, autoencoder_optimizer)
    
    for epoch in range(configuration.EPOCHS):                                                                      # For each epoch...
        start = time.time()                                                                                 # We get the time reference.
        print("Epoch {}.".format(epoch+1))                                          
        training_loss_vector = np.zeros(shape=(training_steps_per_epoch), dtype=np.float32)                 # We initialize a loss vector to save training losses.
        validation_loss_vector = np.zeros(shape=(validation_steps_per_epoch), dtype=np.float32)             # We initialize a loss vector to save validation losses.

        # Training phase.
        for batch_index, training_data in tqdm(enumerate(training_dataset_generator), total=training_steps_per_epoch):  # For each batch we obtain...
            (x_batch, y_batch) = training_data
            rec_loss, reconstructed_x_batch = train_step(x_batch, y_batch, autoencoder, autoencoder_optimizer)      # We apply a train step.
            training_loss_vector[batch_index] = rec_loss                                                    # We save the loss into the training loss vector.

            if batch_index == training_steps_per_epoch - 1:                                                 # If we reach the number of training_steps_per_epoch.
                break                                                                                       # We break the loop. This is ugly but don't blame me, blame the system.
        """
        print("x batch original")
        print(x_batch)
        print("x batch reconstructed")
        print(reconstructed_x_batch)
        print("x batch encoded")
        print(autoencoder.encode(x_batch))
        """
        # Validation phase.
        for batch_index, validation_data in enumerate(validation_dataset_generator):                        # For each batch we obtain...
            (x_batch, y_batch) = validation_data
            reconstructed_x_batch = autoencoder(x_batch)                                                    # We apply the autoencoder to the batch.
            rec_loss = reconstruction_loss(y_batch, reconstructed_x_batch)                                  # We calculate the reconstruction error loss.
            validation_loss_vector[batch_index] = rec_loss                                                  # We save the loss into the validation loss vector.
            
            if batch_index == validation_steps_per_epoch - 1:                                               # If we reach the number of validation_steps_per_epoch.
                break                                                                                       # We break the loop. This is ugly but don't blame me, blame the system.
        
        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Time for epoch {} is {} sec with average training loss {} after {} training steps and average validation loss {} after {} validation steps.'.format(epoch + 1, time.time()-start, np.mean(training_loss_vector), training_steps_per_epoch, np.mean(validation_loss_vector), validation_steps_per_epoch))
        
        autoencoder.save()
        with open(os.path.join(configuration.MODEL_FOLDER_PATH,'history.txt'), 'a') as f:
                with redirect_stdout(f):
                    print('Time for epoch {} is {} sec with average training loss {} after {} training steps and average validation loss {} after {} validation steps.'.format(epoch + 1, time.time()-start, np.mean(training_loss_vector), training_steps_per_epoch, np.mean(validation_loss_vector), validation_steps_per_epoch))
                    
                    
        """
        if not visual_test_clip is None:
            print("Saving test clip.")
            print("Visual test shape: {}".format(visual_test_clip.shape))
            reconstruction = autoencoder(np.expand_dims(visual_test_clip, axis=0))
            print("Prediction shape: {}".format(prediction.shape))
            image_utils.print_image((prediction[0]+1)*127.5, os.path.join(configuration.MAIN_OUTPUT_FOLDER, "testAutoencoder0"), image_preffix="reconstructed_image_{}".format(epoch+1))
            print("Reconstruction loss: {}".format(reconstruction_loss(np.expand_dims(visual_test_clip, axis=0), prediction)))
            #print("Discriminator: {}".format(discriminator(prediction)))
            
        """


"""
Function to execute the training.
inputs:
    configuration : config.Config -> configuration class with the information to 
    network_training_data_path : str -> Folder with the training data files to use.
"""
def train_autoencoder(configuration, network_training_data_path=None):
    
    if network_training_data_path is None:
        data_files_paths = glob(os.path.join(configuration.NETWORK_TRAINING_DATA_PATH, "*"))
    else:
        data_files_paths = glob(os.path.join(network_training_data_path, "*"))

    # We will copy the configuration file.
    if not os.path.isdir(configuration.MODEL_FOLDER_PATH):
        os.makedirs(configuration.MODEL_FOLDER_PATH)
    config_path = os.path.join(configuration.MODEL_FOLDER_PATH, "config.py")
    shutil.copyfile(configuration.CONFIG_FILE_PATH, config_path)
    print(f"Copying configuration file to {config_path}.")
    
    random.shuffle(data_files_paths)
    number_of_files = len(data_files_paths)

    validation_number_of_files = int(number_of_files*configuration.VALIDATION_DATA_SPLIT_FOR_NETWORK_TRAINING)

    basic_training_data_generator = data_utils.data_generator(
        data_files_paths = data_files_paths[validation_number_of_files:], 
        batch_size = configuration.BATCH_SIZE, 
        change_file_after_getting_x_data_batches = configuration.TRAINING_DATA_PER_FILE//configuration.BATCH_SIZE)
    autoencoder_training_generator = data_utils.autoencoder_data_generator(
        basic_training_data_generator, 
        preprocessing_function = normalize_data, 
        x_preprocessing_function = configuration.FUNCTION_TO_APPLY_TO_X_DATA,
        y_preprocessing_function = configuration.FUNCTION_TO_APPLY_TO_Y_DATA)

    basic_validation_data_generator = data_utils.data_generator(
        data_files_paths = data_files_paths[:validation_number_of_files], 
        batch_size = configuration.BATCH_SIZE, 
        change_file_after_getting_x_data_batches = configuration.TRAINING_DATA_PER_FILE//configuration.BATCH_SIZE)    
    autoencoder_validation_generator = data_utils.autoencoder_data_generator(
        basic_validation_data_generator, 
        preprocessing_function = normalize_data,
        x_preprocessing_function = configuration.FUNCTION_TO_APPLY_TO_X_DATA,
        y_preprocessing_function = configuration.FUNCTION_TO_APPLY_TO_Y_DATA)
        
        # data_utils.add_gaussian_noise

    autoencoder_optimizer = tf.keras.optimizers.Adam(1e-3)

    autoencoder = make_convolutional_autoencoder_model()
    print(autoencoder)
    print(len(autoencoder.trainable_variables))
    print(autoencoder.trainable_variables[0].shape)

    checkpoint_prefix = os.path.join(configuration.MODEL_FOLDER_PATH, "AUT_ckpt")
    checkpoint = tf.train.Checkpoint(autoencoder=autoencoder,
                                    autoencoder_optimizer=autoencoder_optimizer)

    """
    for x in clip_generator_class.batch_generator():
        print(x)
        print(x[0].shape)
    quit()
    """

    #predictor.compile("adam", "mse")
    #predictor.fit(clip_dataset, epochs = configuration.epochs)

    #visual_test_clip = np.load(os.path.join(configuration.clip_dataset_path, "clip_0000001.npy"))
    #visual_test_clip = all_dataset[0]
    #print("Saving test clip.")
    #print("Visual test batch shape: {}".format(visual_test_clip.shape))
    #reconstruction = autoencoder.predict(np.expand_dims(visual_test_clip, axis=0))
    #print("Prediction shape: {}".format(prediction.shape))
    #image_utils.print_image((prediction[0]+1)*127.5, os.path.join(configuration.MAIN_OUTPUT_FOLDER, "testAutoencoder0"), image_preffix="reconstructed_image_{}".format(epoch+1))
    #print("Reconstruction loss: {}".format(reconstruction_loss(np.expand_dims(visual_test_clip, axis=0), prediction)))

    train(
        autoencoder_training_generator, 
        autoencoder_validation_generator, 
        autoencoder,
        autoencoder_optimizer,
        checkpoint,
        checkpoint_prefix,
        configuration.STEPS_PER_EPOCH,
        int(configuration.STEPS_PER_EPOCH*configuration.VALIDATION_DATA_SPLIT_FOR_NETWORK_TRAINING), 
        visual_test_clip=None)
        
    autoencoder.save()


if configuration.TRAIN_SPECIFIC_AUTOENCODERS_FOR_EACH_SCENE_AND_NOISE:
    main_model_saving_folder = configuration.MODEL_FOLDER_PATH
    for (noise, category, video_name) in datasets_utils.get_change_detection_noises_categories_and_videos_list():
        print(noise)
        print(category)
        print(video_name)
        if category in configuration.CATEGORIES_TO_TEST and noise in configuration.NOISES_LIST:
            configuration.MODEL_FOLDER_PATH = os.path.join(main_model_saving_folder, f"{video_name}_{noise}")
            training_data_folder = os.path.join(configuration.NETWORK_TRAINING_DATA_PATH, f"{video_name}_{noise}")
            train_autoencoder(configuration, network_training_data_path=training_data_folder)
        else:
            print(f"{noise} noise or {category} category combination is not in configuration.")
    configuration.MODEL_FOLDER_PATH = main_model_saving_folder

else:

    train_autoencoder()

    
