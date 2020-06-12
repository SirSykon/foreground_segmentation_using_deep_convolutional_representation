import os
import time
import numpy as np
import random
import math
from glob import glob
import tensorflow as tf
import tensorflow.keras.layers as layers
from config import Config

"""
Class to define and use an autoencoder model and its parts.
"""

class Autoencoder(tf.keras.Model):
    encoder = None     # Encoder network
    decoder = None     # Decoder network
    autoencoder = None # Autoencoder network.
    models_path = None # Path to the folder where we save or from where we load the models

    """
    Constructor
        models_path : string
            Path to the folder where we save or from where we load the models
        load : boolean
            Do we load a model or do we train it?
    """
    def __init__(self, models_path, load = False):
    
        super(Autoencoder,self).__init__()
        
        self.models_path = models_path
        if load:
            print("Path to load: {}".format(models_path))
            self.load_encoder()
            self.load_decoder()
        else:
            print("Path to save: {}".format(models_path))
            self.optimizer = optimizer
            self.loss_object = loss_object
            self.define_encoder()
            self.encoder.summary()
            self.define_decoder()
            self.decoder.summary()
            
        self.define_autoencoder()
        self.autoencoder.summary()
        
    """
    Method use the autoencoder.
    """
    def call(self, input_batch):

        reconstruction = self.autoencoder(input_batch)
        return reconstruction
    
    """
    Method to load the encoder model.
    """
    def load_encoder(self):
        print("Loading encoder from {}.".format(self.models_path))
        json_file = open(os.path.join(self.models_path, "encoder.json"), "r")
        loaded_model_json = json_file.read()
        json_file.close()
        self.encoder = keras.models.model_from_json(loaded_model_json)
        encoder_path = os.path.join(self.models_path, "encoder.h5")  
        self.encoder.load_weights(encoder_path)
    
    """
    Method to load the decoder model.
    """
    def load_decoder(self):
        print("Loading decoder from {}.".format(self.models_path))
        json_file = open(os.path.join(self.models_path, "decoder.json"), "r")
        loaded_model_json = json_file.read()
        json_file.close()
        self.decoder = keras.models.model_from_json(loaded_model_json)
        decoder_path = os.path.join(self.models_path, "decoder.h5")  
        self.decoder.load_weights(decoder_path)

    """
    Method to define the encoder model.
    """
    def define_encoder(self):
        raise(NotImplementedError)

    """
    Method to define the decoder model.
    """
    def define_decoder(self):
        raise(NotImplementedError)

    """
    Method to define the decoder model.
    """
    def define_autoencoder(self):
        print(self.encoder.input)
        print(self.decoder.output)
        autoencoder_output = self.decoder(self.encoder(self.encoder.input))
        self.autoencoder = tf.keras.Model(self.encoder.input, autoencoder_output)

    """
    Method to predict using the encoder model.
        patches : list
            List of numpy arrays to serve as input. Their shape should be self.input_shape
        ---
        returns : list
            List of predictions.
    """
    def encode(self, patches):
        return self.encoder(patches)
    
    """
    Method to predict using the decoder model.
        patches : list
            List of numpy arrays to serve as input. Their shape should be self.input_shape
        ---
        returns : list
            List of predictions.
    """
    def decode(self, patches):
        return self.decoder(patches)

    """
    Method to predict using the autoencoder model.
        patches : list
            List of numpy arrays to serve as input. Their shape should be self.input_shape
        ---
        returns : list
            List of predictions.
    """
    def autoencode(self, patches):
        return self.autoencoder(patches)
        
    """
    Method to save models in self.models_path.
    """
    def save(self):
        if not os.path.isdir(self.models_path):
            os.makedirs(self.models_path)

        encoder_path = os.path.join(self.models_path, "encoder.h5")
        encoder_json = self.encoder.to_json()
        with open(os.path.join(self.models_path, "encoder.json"), "w") as encoder_json_file:
            encoder_json_file.write(encoder_json)
        self.encoder.save_weights(encoder_path)
        
        decoder_path = os.path.join(self.models_path, "decoder.h5")
        decoder_json = self.decoder.to_json()
        with open(os.path.join(self.models_path, "decoder.json"), "w") as decoder_json_file:
            decoder_json_file.write(decoder_json)
        self.decoder.save_weights(decoder_path)
        
class Convolutional_Autoencoder(Autoencoder):

    """
    Method to define the encoder model.
    """
    def define_encoder(self):
        encoder_input_layer = layers.Input(shape=(None, None, Config.NUM_CHANNELS))     # We define the input with no defined height(H) and width(W). We wil guess the input is (None,64,64,3)
        x = layers.Conv2D(64, (3,3), strides=(1,1), padding="valid", activation=tf.keras.layers.LeakyReLU())(input_layer)
        assert x.shape.as_list() == [None, None, None, 64]                              # Here the output should be (None, 62, 62, 64)
        x = layers.Conv2D(32, (3,3), strides=(1,1), padding="valid", activation=tf.keras.layers.LeakyReLU())(x)
        assert x.shape.as_list() == [None, None, None, 32]                              # Here the output should be (None, 60, 60, 32)
        codification = layers.Conv2D(16, (3,3), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert codification.shape.as_list() == [None, None, None, 16]                   # Here the output should be (None, 58, 58, 16)
        
        self.encoder = tf.keras.Model(encoder_input_layer, codification)
        print("ENCODER")
        print(self.encoder.summary())

    """
    Method to define the decoder model.
    """
    def define_decoder(self):
        decoder_input_layer = layers.Input(shape=self.encoder.output_shape[1:])
        layers.Conv2DTranspose(32, (3,3), strides=(1,1), padding="valid", activation=tf.keras.layers.LeakyReLU())(input_layer)
        assert x.shape.as_list() == [None, None, None, 32]                               # Here the output should be (None, 60, 60, 32)
        layers.Conv2DTranspose(64, (3,3), strides=(1,1), padding="valid", activation=tf.keras.layers.LeakyReLU())(input_layer)
        assert x.shape.as_list() == [None, 62, 62, 64]                                   # Here the output should be (None, 62, 62, 64)
        decodification = layers.Conv2D(Config.NUM_CHANNELS, (3,3), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert decodification.shape.as_list() == [None, None, None, Config.NUM_CHANNELS] # Here the output should be (None, 64, 64, 3)
        
        self.decoder = tf.keras.Model(decoder_input_layer, decodification)
        print("DECODER")
        print(self.decoder.summary())
