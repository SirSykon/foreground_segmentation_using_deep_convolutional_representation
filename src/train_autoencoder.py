import os
import time
import numpy as np
import random
import math
from glob import glob
import tensorflow as tf
import tensorflow.keras.layers as layers
import utils.clip_utils as clip_utils
from config import Config
import utils.GPU_utils as GPU_utils             # pylint: disable=no-name-in-module
import utils.datagen_utils as datagen_utils     # pylint: disable=no-name-in-module
import Predictor

GPU_utils.tensorflow_2_x_dark_magic_to_restrict_memory_use(Config.GPU_TO_USE)

def load_dataset(split = True, normalization_range = (0,1), number_of_files_to_load = 100):
    all_dataset = None
    list_of_data_paths_for_training = sorted(glob(os.path.join(Config.NETWORK_TRAINING_DATA_PATH, "*")))
    for data_file_path_index, data_file_path in enumerate(list_of_data_paths_for_training):
        print("Loading {}.".format(data_file_path))
        data_file = np.load(data_file_path)
        if all_dataset is None:
            all_dataset = data_file
        else:
            all_dataset = np.concatenate((all_dataset, data_file), axis = 0)

        if data_file_path_index == number_of_files_to_load - 1:
            break

    dataset_size = all_dataset.shape[0]

    if normalization_range == (0,1):
        all_dataset = all_dataset/255.
    if normalization_range == (-1,1):
        all_dataset = (all_dataset/127.5)-1

    training_dataset_size = math.floor(dataset_size*(1-Config.VALIDATION_DATA_SPLIT_FOR_NETWORK_TRAINING))
    print("Dataset size")
    print(dataset_size)
    training_dataset = all_dataset[:training_dataset_size]
    
    if split:
        print("Training dataset shape is {}.".format(training_dataset.shape))
        print(np.max(training_dataset))
        print(np.min(training_dataset))

        validation_dataset = all_dataset[training_dataset_size:]
        print("Validation dataset shape is {}.".format(validation_dataset.shape))
        print(np.max(validation_dataset))
        print(np.min(validation_dataset))
        
        return training_dataset, validation_dataset
    
    else:
        return all_dataset

if not os.path.isdir(Config.MODEL_FOLDER_PATH):
    os.makedirs(Config.MODEL_FOLDER_PATH)

reconstruction_loss = tf.keras.losses.MeanSquaredError()    
    
def make_convolutional_autoencoder_model():
    return autoencoder

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images_batch, autoencoder, aut_optimizer):

    with tf.GradientTape() as aut_tape::
        reconstructed_images_batch = autoencoder(images_batch, training=True)        
        aut_loss = reconstruction_loss(images_batch, reconstructed_images_batch)
    
    gradients_of_aut = aut_tape.gradient(aut_loss, autoencoder.trainable_variables)    
    aut_optimizer.apply_gradients(zip(gradients_of_aut, autoencoder.trainable_variables))    
    rec_loss = reconstruction_loss(fake_next_frame_batch, real_next_frame_batch)

    return rec_loss, reconstructed_images_batch

def train(training_dataset_generator, autoencoder, checkpoint, steps_per_epoch, visual_test_clip=None):
    for epoch in range(Config.EPOCHS):
        start = time.time()
        print("Epoch {}.".format(epoch+1))
        loss_vector = np.zeros(shape=(Config.STEPS_PER_EPOCH, 1), dtype=np.float32)

        for batch_index, clip_batch in enumerate(training_dataset_generator):
            rec_loss, reconstructed_images_batch = train_step(images_batch, autoencoder, aut_optimizer)
            loss_vector[batch_index, 0] = rec_loss
            
            assert np.array_equal(real_clip_batch, clip_batch)
            assert np.array_equal(fake_clip_batch[:,:-1], clip_batch[:,:-1])
            
            if batch_index == steps_per_epoch - 1:
                break

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Time for epoch {} is {} sec with generator loss {}, global discriminator loss {}, local discriminator loss {} and reconstruction loss (calculate but not used in training) {}.'.format(epoch + 1, time.time()-start, np.mean(loss_vector[:, 0]),np.mean(loss_vector[:, 1]),np.mean(loss_vector[:, 2]),np.mean(loss_vector[:, 3])))

        if not visual_test_clip is None:
            print("Saving test clip.")
            print("Visual test batch shape: {}".format(visual_test_clip.shape))
            reconstruction = autoencoder.predict(np.expand_dims(visual_test_clip, axis=0))
            print("Prediction shape: {}".format(prediction.shape))
            image_utils.print_image((prediction[0]+1)*127.5, os.path.join(Config.MAIN_OUTPUT_FOLDER, "testAutoencoder0"), image_preffix="reconstructed_image_{}".format(epoch+1))
            print("Reconstruction loss: {}".format(reconstruction_loss(np.expand_dims(visual_test_clip, axis=0), prediction)))
            #print("Discriminator: {}".format(discriminator(prediction)))

autoencoder_optimizer = tf.keras.optimizers.Adam(1e-4)

autoencoder = make_convolutional_autoencoder_model()
print(len(autoencoder.trainable_variables))
print(autoencoder.trainable_variables[0].shape)

checkpoint_prefix = os.path.join(Config.MODEL_FOLDER_PATH, "AUT_ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 global_discriminator=global_discriminator,
                                 local_discriminator=local_discriminator)

all_dataset = load_dataset(split = False, normalization_range = (-1,1), number_of_files_to_load = 7)
clip_generator_class = datagen_utils.clip_generator(all_dataset[2:], Config.BATCH_SIZE, load_from_memory = True)

"""
for x in clip_generator_class.batch_generator():
    print(x)
    print(x[0].shape)
quit()
"""

#predictor.compile("adam", "mse")
#predictor.fit(clip_dataset, epochs = Config.epochs)

#visual_test_clip = np.load(os.path.join(Config.clip_dataset_path, "clip_0000001.npy"))
visual_test_clip = all_dataset[0]
print("Saving test clip.")
print("Visual test batch shape: {}".format(visual_test_clip.shape))
reconstruction = autoencoder.predict(np.expand_dims(visual_test_clip, axis=0))
print("Prediction shape: {}".format(prediction.shape))
image_utils.print_image((prediction[0]+1)*127.5, os.path.join(Config.MAIN_OUTPUT_FOLDER, "testAutoencoder0"), image_preffix="reconstructed_image_{}".format(epoch+1))
print("Reconstruction loss: {}".format(reconstruction_loss(np.expand_dims(visual_test_clip, axis=0), prediction)))
    
train(
    clip_generator_class, 
    generator, 
    global_discriminator,
    local_discriminator, 
    generator_optimizer, 
    discriminator_optimizer, 
    checkpoint, 
    Config.STEPS_PER_EPOCH, 
    visual_test_clip=visual_test_clip)
