import os
import time
import numpy as np
import random
import math
from glob import glob
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras as keras
from contextlib import redirect_stdout

class DummyLayer(layers.Layer):
    def __init__(self):
        super(DummyLayer, self).__init__()
    
    def call(self, inputs):
        return inputs


"""
Class to define and use an autoencoder model and its parts.
"""

class Autoencoder(tf.keras.Model):

    """
    Constructor
        models_path : string
            Path to the folder where we save or from where we load the models
        load : boolean
            Do we load a model or do we train it?
    """
    def __init__(self, models_path, load = False):
    
        super(Autoencoder,self).__init__(name = "aut_model")
        
        self.models_path = models_path
        if load:
            print("Path to load: {}".format(models_path))
            self.load_encoder()
            self.load_decoder()
        else:
            print("Path to save: {}".format(models_path))
            if not os.path.isdir(self.models_path):
                os.makedirs(self.models_path)
                
            self.encoder = self.define_encoder()
            print("Encoder defined")
            self.encoder.summary()
            
            with open(os.path.join(self.models_path,'encoder.txt'), 'w') as f:
                with redirect_stdout(f):
                    self.encoder.summary()
                    
            self.decoder = self.define_decoder()
            print("Decoder defined")
            self.decoder.summary()
            
            with open(os.path.join(self.models_path,'decoder.txt'), 'a') as f:
                with redirect_stdout(f):
                    self.decoder.summary()
            
        self.autoencoder = self.define_autoencoder()
        print("Autoencoder defined")
        self.autoencoder.summary()
        
        with open(os.path.join(self.models_path,'autoencoder.txt'), 'a') as f:
                with redirect_stdout(f):
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
        return tf.keras.Model(self.encoder.input, autoencoder_output)

    """
    Method to predict using the encoder model.
        patches : list
            List of numpy arrays to serve as input.
        ---
        returns : list
            List of predictions.
    """
    def encode(self, patches):
        return self.encoder(patches)
    
    """
    Method to predict using the decoder model.
        patches : list
            List of numpy arrays to serve as input.
        ---
        returns : list
            List of predictions.
    """
    def decode(self, patches):
        return self.decoder(patches)

    """
    Method to predict using the autoencoder model.
        patches : list
            List of numpy arrays to serve as input.
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
        
class Convolutional_Autoencoder_2_encoding_decoding_layers_3x3_filters(Autoencoder):

    """
    Method to define the encoder model.
    """
    def define_encoder(self):
        encoder_input_layer = layers.Input(shape=(None, None, 3))     # We define the input with no defined height(H) and width(W). We wil guess the input is (None,64,64,3)
        x = layers.Conv2D(64, (3,3), strides=(1,1), padding="valid", activation="relu")(encoder_input_layer)
        assert x.shape.as_list() == [None, None, None, 64]                              # Here the output should be (None, 62, 62, 64)
        codification = layers.Conv2D(16, (3,3), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert codification.shape.as_list() == [None, None, None, 32]                   # Here the output should be (None, 60, 60, 32)
        
        return tf.keras.Model(encoder_input_layer, codification, name = "encoder_model")

    """
    Method to define the decoder model.
    """
    def define_decoder(self):
        decoder_input_layer = layers.Input(shape=self.encoder.output_shape[1:])
        x = layers.Conv2DTranspose(32, (3,3), strides=(1,1), padding="valid", activation="relu")(decoder_input_layer)
        assert x.shape.as_list() == [None, None, None, 32]                               # Here the output should be (None, 62, 62, 64)
        decodification = layers.Conv2DTranspose(3, (3,3), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert decodification.shape.as_list() == [None, None, None, 3] # Here the output should be (None, 64, 64, 3)
        
        return tf.keras.Model(decoder_input_layer, decodification, name = "decoder_model")

class Convolutional_Autoencoder_2_encoding_decoding_layers_with_maxpool_3x3_filters(Autoencoder):

    """
    Method to define the encoder model.
    """
    def define_encoder(self):
        encoder_input_layer = layers.Input(shape=(None, None, 3))     # We define the input with no defined height(H) and width(W). We wil guess the input is (None,64,64,3)
        x = layers.Conv2D(64, (3,3), strides=(1,1), padding="valid", activation="relu")(encoder_input_layer)
        assert x.shape.as_list() == [None, None, None, 64]                              # Here the output should be (None, 62, 62, 64)
        x = layers.MaxPooling2D()(x)
        assert x.shape.as_list() == [None, None, None, 64]                              # Here the output should be (None, 31, 31, 64)
        codification = layers.Conv2D(32, (3,3), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert codification.shape.as_list() == [None, None, None, 32]                   # Here the output should be (None, 29, 29, 32)
        
        return tf.keras.Model(encoder_input_layer, codification, name = "encoder_model")

    """
    Method to define the decoder model.
    """
    def define_decoder(self):
        decoder_input_layer = layers.Input(shape=self.encoder.output_shape[1:])
        x = layers.Conv2DTranspose(64, (3,3), strides=(1,1), padding="valid", activation="relu")(decoder_input_layer)
        assert x.shape.as_list() == [None, None, None, 64]                               # Here the output should be (None, 31, 31, 64)
        x = layers.UpSampling2D()(x)
        assert x.shape.as_list() == [None, None, None, 64]                               # Here the output should be (None, 62, 62, 64)
        decodification = layers.Conv2DTranspose(3, (3,3), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert decodification.shape.as_list() == [None, None, None, 3] # Here the output should be (None, 64, 64, 3)
        
        return tf.keras.Model(decoder_input_layer, decodification, name = "decoder_model")
        
class Convolutional_Encoder_2_encoding_decoding_layers_3x3_filters(Autoencoder):
    """
    Method to define the encoder model.
    """
    def define_encoder(self):
        encoder_input_layer = layers.Input(shape=(None, None, 3))     # We define the input with no defined height(H) and width(W). We wil guess the input is (None,64,64,3)
        x = layers.Conv2D(64, (3,3), strides=(1,1), padding="valid", activation="relu")(encoder_input_layer)
        assert x.shape.as_list() == [None, None, None, 64]                              # Here the output should be (None, 62, 62, 64)
        codification = layers.Conv2D(16, (3,3), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert codification.shape.as_list() == [None, None, None, 16]                   # Here the output should be (None, 60, 60, 16)
        
        return tf.keras.Model(encoder_input_layer, codification, name = "encoder_model")
        
    """
    Method to define a dummy decoder.
    """
    def define_decoder(self):
        decoder_input_layer = layers.Input(shape=self.encoder.output_shape[1:])
        decodification = DummyLayer()(decoder_input_layer)
        
        return tf.keras.Model(decoder_input_layer, decodification, name = "decoder_model")
    
        
class Convolutional_Autoencoder_2_encoding_decoding_layers_5x5_filters(Autoencoder):

    """
    Method to define the encoder model.
    """
    def define_encoder(self):
        encoder_input_layer = layers.Input(shape=(None, None, 3))     # We define the input with no defined height(H) and width(W). We wil guess the input is (None,64,64,3)
        x = layers.Conv2D(64, (5,5), strides=(1,1), padding="valid", activation="relu")(encoder_input_layer)
        assert x.shape.as_list() == [None, None, None, 64]                              # Here the output should be (None, 62, 62, 64)
        codification = layers.Conv2D(16, (5,5), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert codification.shape.as_list() == [None, None, None, 16]                   # Here the output should be (None, 60, 60, 16)
        
        return tf.keras.Model(encoder_input_layer, codification, name = "encoder_model")

    """
    Method to define the decoder model.
    """
    def define_decoder(self):
        decoder_input_layer = layers.Input(shape=self.encoder.output_shape[1:])
        x = layers.Conv2DTranspose(32, (5,5), strides=(1,1), padding="valid", activation="relu")(decoder_input_layer)
        assert x.shape.as_list() == [None, None, None, 32]                               # Here the output should be (None, 62, 62, 64)
        decodification = layers.Conv2DTranspose(3, (5,5), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert decodification.shape.as_list() == [None, None, None, 3] # Here the output should be (None, 64, 64, 3)
        
        return tf.keras.Model(decoder_input_layer, decodification, name = "decoder_model")
        
class Convolutional_Autoencoder_3_encoding_decoding_layers_3x3_filters(Autoencoder):

    """
    Method to define the encoder model.
    """
    def define_encoder(self):
        encoder_input_layer = layers.Input(shape=(None, None, 3))     # We define the input with no defined height(H) and width(W). We wil guess the input is (None,64,64,3)
        x = layers.Conv2D(64, (3,3), strides=(1,1), padding="valid", activation="relu")(encoder_input_layer)
        assert x.shape.as_list() == [None, None, None, 64]                              # Here the output should be (None, 62, 62, 64)
        x = layers.Conv2D(32, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 32]                              # Here the output should be (None, 60, 60, 32)
        codification = layers.Conv2D(16, (3,3), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert codification.shape.as_list() == [None, None, None, 16]                   # Here the output should be (None, 58, 58, 16)
        
        return tf.keras.Model(encoder_input_layer, codification, name = "encoder_model")

    """
    Method to define the decoder model.
    """
    def define_decoder(self):
        decoder_input_layer = layers.Input(shape=self.encoder.output_shape[1:])
        x = layers.Conv2DTranspose(32, (3,3), strides=(1,1), padding="valid", activation="relu")(decoder_input_layer)
        assert x.shape.as_list() == [None, None, None, 32]                               # Here the output should be (None, 60, 60, 32)
        x = layers.Conv2DTranspose(64, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 64]                               # Here the output should be (None, 62, 62, 64)
        decodification = layers.Conv2DTranspose(3, (3,3), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert decodification.shape.as_list() == [None, None, None, 3] # Here the output should be (None, 64, 64, 3)
        
        return tf.keras.Model(decoder_input_layer, decodification, name = "decoder_model")
        
class Convolutional_Encoder_3_encoding_decoding_layers_3x3_filters(Autoencoder):
    """
    Method to define the encoder model.
    """
    def define_encoder(self):
        encoder_input_layer = layers.Input(shape=(None, None, 3))     # We define the input with no defined height(H) and width(W). We wil guess the input is (None,64,64,3)
        x = layers.Conv2D(64, (3,3), strides=(1,1), padding="valid", activation="relu")(encoder_input_layer)
        assert x.shape.as_list() == [None, None, None, 64]                              # Here the output should be (None, 62, 62, 64)
        x = layers.Conv2D(32, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 32]                              # Here the output should be (None, 60, 60, 32)
        codification = layers.Conv2D(16, (3,3), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert codification.shape.as_list() == [None, None, None, 16]                   # Here the output should be (None, 58, 58, 16)
        
        return tf.keras.Model(encoder_input_layer, codification, name = "encoder_model")
        
    """
    Method to define a dummy decoder.
    """
    def define_decoder(self):
        decoder_input_layer = layers.Input(shape=self.encoder.output_shape[1:])
        decodification = DummyLayer()(decoder_input_layer)
        
        return tf.keras.Model(decoder_input_layer, decodification, name = "decoder_model")
        
class Convolutional_Autoencoder_3_encoding_decoding_layers_5x5_filters(Autoencoder):

    """
    Method to define the encoder model.
    """
    def define_encoder(self):
        encoder_input_layer = layers.Input(shape=(None, None, 3))     # We define the input with no defined height(H) and width(W). We wil guess the input is (None,64,64,3)
        x = layers.Conv2D(64, (5,5), strides=(1,1), padding="valid", activation="relu")(encoder_input_layer)
        assert x.shape.as_list() == [None, None, None, 64]                              # Here the output should be (None, 62, 62, 64)
        x = layers.Conv2D(32, (5,5), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 32]                              # Here the output should be (None, 60, 60, 32)
        codification = layers.Conv2D(16, (5,5), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert codification.shape.as_list() == [None, None, None, 16]                   # Here the output should be (None, 58, 58, 16)
        
        return tf.keras.Model(encoder_input_layer, codification, name = "encoder_model")

    """
    Method to define the decoder model.
    """
    def define_decoder(self):
        decoder_input_layer = layers.Input(shape=self.encoder.output_shape[1:])
        x = layers.Conv2DTranspose(32, (5,5), strides=(1,1), padding="valid", activation="relu")(decoder_input_layer)
        assert x.shape.as_list() == [None, None, None, 32]                               # Here the output should be (None, 60, 60, 32)
        x = layers.Conv2DTranspose(64, (5,5), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 64]                               # Here the output should be (None, 62, 62, 64)
        decodification = layers.Conv2DTranspose(3, (5,5), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert decodification.shape.as_list() == [None, None, None, 3] # Here the output should be (None, 64, 64, 3)
        
        return tf.keras.Model(decoder_input_layer, decodification, name = "decoder_model")
        
class Convolutional_Autoencoder_4_encoding_decoding_layers_3x3_filters(Autoencoder):

    """
    Method to define the encoder model.
    """
    def define_encoder(self):
        encoder_input_layer = layers.Input(shape=(None, None, 3))     # We define the input with no defined height(H) and width(W). We wil guess the input is (None,64,64,3)
        x = layers.Conv2D(64, (3,3), strides=(1,1), padding="valid", activation="relu")(encoder_input_layer)
        assert x.shape.as_list() == [None, None, None, 64]                              # Here the output should be (None, 62, 62, 64)
        x = layers.Conv2D(32, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 32]                              # Here the output should be (None, 60, 60, 32)
        x = layers.Conv2D(16, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 16]                              # Here the output should be (None, 58, 58, 16)
        codification = layers.Conv2D(8, (3,3), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert codification.shape.as_list() == [None, None, None, 8]                   # Here the output should be (None, 56, 56, 8)
        
        return tf.keras.Model(encoder_input_layer, codification, name = "encoder_model")

    """
    Method to define the decoder model.
    """
    def define_decoder(self):
        decoder_input_layer = layers.Input(shape=self.encoder.output_shape[1:])
        x = layers.Conv2DTranspose(16, (3,3), strides=(1,1), padding="valid", activation="relu")(decoder_input_layer)
        assert x.shape.as_list() == [None, None, None, 16]                               # Here the output should be (None, 58, 58, 16)
        x = layers.Conv2DTranspose(32, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 32]                               # Here the output should be (None, 60, 60, 32)
        x = layers.Conv2DTranspose(64, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 64]                               # Here the output should be (None, 62, 62, 64)
        decodification = layers.Conv2DTranspose(3, (3,3), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert decodification.shape.as_list() == [None, None, None, 3] # Here the output should be (None, 64, 64, 3)
        
        return tf.keras.Model(decoder_input_layer, decodification, name = "decoder_model")
        
class Convolutional_Autoencoder_4_encoding_decoding_layers_5x5_filters(Autoencoder):

    """
    Method to define the encoder model.
    """
    def define_encoder(self):
        encoder_input_layer = layers.Input(shape=(None, None, 3))     # We define the input with no defined height(H) and width(W). We wil guess the input is (None,64,64,3)
        x = layers.Conv2D(64, (5,5), strides=(1,1), padding="valid", activation="relu")(encoder_input_layer)
        assert x.shape.as_list() == [None, None, None, 64]                              # Here the output should be (None, 62, 62, 64)
        x = layers.Conv2D(32, (5,5), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 32]                              # Here the output should be (None, 60, 60, 32)
        x = layers.Conv2D(16, (5,5), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 16]                              # Here the output should be (None, 58, 58, 16)
        codification = layers.Conv2D(8, (5,5), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert codification.shape.as_list() == [None, None, None, 8]                   # Here the output should be (None, 56, 56, 8)
        
        return tf.keras.Model(encoder_input_layer, codification, name = "encoder_model")

    """
    Method to define the decoder model.
    """
    def define_decoder(self):
        decoder_input_layer = layers.Input(shape=self.encoder.output_shape[1:])
        x = layers.Conv2DTranspose(16, (5,5), strides=(1,1), padding="valid", activation="relu")(decoder_input_layer)
        assert x.shape.as_list() == [None, None, None, 16]                               # Here the output should be (None, 58, 58, 16)
        x = layers.Conv2DTranspose(32, (5,5), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 32]                               # Here the output should be (None, 60, 60, 32)
        x = layers.Conv2DTranspose(64, (5,5), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 64]                               # Here the output should be (None, 62, 62, 64)
        decodification = layers.Conv2DTranspose(3, (5,5), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert decodification.shape.as_list() == [None, None, None, 3] # Here the output should be (None, 64, 64, 3)
        
        return tf.keras.Model(decoder_input_layer, decodification, name = "decoder_model")

class Convolutional_Autoencoder_5_encoding_decoding_layers_3x3_filters(Autoencoder):

    """
    Method to define the encoder model.
    """
    def define_encoder(self):
        encoder_input_layer = layers.Input(shape=(None, None, 3))     # We define the input with no defined height(H) and width(W). We wil guess the input is (None,64,64,3)
        x = layers.Conv2D(64, (3,3), strides=(1,1), padding="valid", activation="relu")(encoder_input_layer)
        assert x.shape.as_list() == [None, None, None, 64]                              # Here the output should be (None, 62, 62, 64)
        x = layers.Conv2D(32, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 32]                              # Here the output should be (None, 60, 60, 32)
        x = layers.Conv2D(16, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 16]                              # Here the output should be (None, 58, 58, 16)
        x = layers.Conv2D(8, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 8]                   # Here the output should be (None, 56, 56, 8)
        codification = layers.Conv2D(4, (3,3), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert codification.shape.as_list() == [None, None, None, 4]                   # Here the output should be (None, 54, 54, 4)
        
        return tf.keras.Model(encoder_input_layer, codification, name = "encoder_model")

    """
    Method to define the decoder model.
    """
    def define_decoder(self):
        decoder_input_layer = layers.Input(shape=self.encoder.output_shape[1:])
        x = layers.Conv2DTranspose(8, (3,3), strides=(1,1), padding="valid", activation="relu")(decoder_input_layer)
        assert x.shape.as_list() == [None, None, None, 8]                                # Here the output should be (None, 56, 56, 8)
        x = layers.Conv2DTranspose(16, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 16]                               # Here the output should be (None, 58, 58, 16)
        x = layers.Conv2DTranspose(32, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 32]                               # Here the output should be (None, 60, 60, 32)
        x = layers.Conv2DTranspose(64, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 64]                               # Here the output should be (None, 62, 62, 64)
        decodification = layers.Conv2DTranspose(3, (3,3), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert decodification.shape.as_list() == [None, None, None, 3] # Here the output should be (None, 64, 64, 3)
        
        return tf.keras.Model(decoder_input_layer, decodification, name = "decoder_model")

class Convolutional_Autoencoder_6_encoding_decoding_layers_3x3_filters(Autoencoder):

    """
    Method to define the encoder model.
    """
    def define_encoder(self):
        encoder_input_layer = layers.Input(shape=(None, None, 3))     # We define the input with no defined height(H) and width(W). We wil guess the input is (None,64,64,3)
        x = layers.Conv2D(64, (3,3), strides=(1,1), padding="valid", activation="relu")(encoder_input_layer)
        assert x.shape.as_list() == [None, None, None, 64]                              # Here the output should be (None, 62, 62, 64)
        x = layers.Conv2D(32, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 32]                              # Here the output should be (None, 60, 60, 32)
        x = layers.Conv2D(16, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 16]                              # Here the output should be (None, 58, 58, 16)
        x = layers.Conv2D(8, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 8]                               # Here the output should be (None, 56, 56, 8)
        x = layers.Conv2D(4, (3,3), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert x.shape.as_list() == [None, None, None, 4]                               # Here the output should be (None, 54, 54, 4)
        codification = layers.Conv2D(2, (3,3), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert codification.shape.as_list() == [None, None, None, 2]                   # Here the output should be (None, 52, 52, 2)
        
        return tf.keras.Model(encoder_input_layer, codification, name = "encoder_model")

    """
    Method to define the decoder model.
    """
    def define_decoder(self):
        decoder_input_layer = layers.Input(shape=self.encoder.output_shape[1:])
        x = layers.Conv2DTranspose(4, (3,3), strides=(1,1), padding="valid", activation="relu")(decoder_input_layer)
        assert x.shape.as_list() == [None, None, None, 4]                                # Here the output should be (None, 56, 56, 4)
        x = layers.Conv2DTranspose(8, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 8]                                # Here the output should be (None, 56, 56, 8)
        x = layers.Conv2DTranspose(16, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 16]                               # Here the output should be (None, 58, 58, 16)
        x = layers.Conv2DTranspose(32, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 32]                               # Here the output should be (None, 60, 60, 32)
        x = layers.Conv2DTranspose(64, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 64]                               # Here the output should be (None, 62, 62, 64)
        decodification = layers.Conv2DTranspose(3, (3,3), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert decodification.shape.as_list() == [None, None, None, 3] # Here the output should be (None, 64, 64, 3)
        
        return tf.keras.Model(decoder_input_layer, decodification, name = "decoder_model")

class Convolutional_Autoencoder_6_encoding_decoding_layers_5x5_filters(Autoencoder):

    """
    Method to define the encoder model.
    """
    def define_encoder(self):
        encoder_input_layer = layers.Input(shape=(None, None, 3))     # We define the input with no defined height(H) and width(W). We wil guess the input is (None,64,64,3)
        x = layers.Conv2D(64, (5,5), strides=(1,1), padding="valid", activation="relu")(encoder_input_layer)
        assert x.shape.as_list() == [None, None, None, 64]                              # Here the output should be (None, 62, 62, 64)
        x = layers.Conv2D(32, (5,5), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 32]                              # Here the output should be (None, 60, 60, 32)
        x = layers.Conv2D(16, (5,5), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 16]                              # Here the output should be (None, 58, 58, 16)
        x = layers.Conv2D(8, (5,5), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 8]                               # Here the output should be (None, 56, 56, 8)
        x = layers.Conv2D(4, (5,5), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert x.shape.as_list() == [None, None, None, 4]                               # Here the output should be (None, 54, 54, 4)
        codification = layers.Conv2D(2, (5,5), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert codification.shape.as_list() == [None, None, None, 2]                   # Here the output should be (None, 52, 52, 2)
        
        return tf.keras.Model(encoder_input_layer, codification, name = "encoder_model")

    """
    Method to define the decoder model.
    """
    def define_decoder(self):
        decoder_input_layer = layers.Input(shape=self.encoder.output_shape[1:])
        x = layers.Conv2DTranspose(4, (5,5), strides=(1,1), padding="valid", activation="relu")(decoder_input_layer)
        assert x.shape.as_list() == [None, None, None, 4]                                # Here the output should be (None, 56, 56, 4)
        x = layers.Conv2DTranspose(8, (5,5), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 8]                                # Here the output should be (None, 56, 56, 8)
        x = layers.Conv2DTranspose(16, (5,5), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 16]                               # Here the output should be (None, 58, 58, 16)
        x = layers.Conv2DTranspose(32, (5,5), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 32]                               # Here the output should be (None, 60, 60, 32)
        x = layers.Conv2DTranspose(64, (5,5), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 64]                               # Here the output should be (None, 62, 62, 64)
        decodification = layers.Conv2DTranspose(3, (5,5), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert decodification.shape.as_list() == [None, None, None, 3] # Here the output should be (None, 64, 64, 3)
        
        return tf.keras.Model(decoder_input_layer, decodification, name = "decoder_model")
        
class Convolutional_Autoencoder(Autoencoder):

    """
    Method to define the encoder model.
    """
    def define_encoder(self):
        encoder_input_layer = layers.Input(shape=(None, None, 3))     # We define the input with no defined height(H) and width(W). We wil guess the input is (None,64,64,3)
        x = layers.Conv2D(64, (3,3), strides=(1,1), padding="valid", activation="relu")(encoder_input_layer)
        assert x.shape.as_list() == [None, None, None, 64]                              # Here the output should be (None, 62, 62, 64)
        x = layers.Conv2D(64, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 64]                              # Here the output should be (None, 60, 60, 64)
        x = layers.Conv2D(64, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 64]                              # Here the output should be (None, 58, 58, 64)
        x = layers.Conv2D(32, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 32]                              # Here the output should be (None, 56, 56, 32)
        x = layers.Conv2D(32, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 32]                              # Here the output should be (None, 54, 64, 32)
        x = layers.Conv2D(32, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 32]                              # Here the output should be (None, 52, 52, 32)
        codification = layers.Conv2D(16, (3,3), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert codification.shape.as_list() == [None, None, None, 16]                   # Here the output should be (None, 50, 50, 16)
        
        return tf.keras.Model(encoder_input_layer, codification, name = "encoder_model")

    """
    Method to define the decoder model.
    """
    def define_decoder(self):
        decoder_input_layer = layers.Input(shape=self.encoder.output_shape[1:])
        x = layers.Conv2DTranspose(32, (3,3), strides=(1,1), padding="valid", activation="relu")(decoder_input_layer)
        assert x.shape.as_list() == [None, None, None, 32]                               # Here the output should be (None, 52, 52, 32)
        x = layers.Conv2DTranspose(32, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 32]                               # Here the output should be (None, 54, 54, 32)
        x = layers.Conv2DTranspose(32, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 32]                               # Here the output should be (None, 56, 56, 32)
        x = layers.Conv2DTranspose(64, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 64]                               # Here the output should be (None, 58, 58, 64)
        x = layers.Conv2DTranspose(64, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 64]                               # Here the output should be (None, 60, 60, 64)
        x = layers.Conv2DTranspose(64, (3,3), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 64]                               # Here the output should be (None, 62, 62, 64)
        decodification = layers.Conv2DTranspose(3, (3,3), strides=(1,1), padding="valid", activation="sigmoid")(x)
        #decodification = layers.Conv2D(3, (3,3), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert decodification.shape.as_list() == [None, None, None, 3] # Here the output should be (None, 64, 64, 3)
        
        return tf.keras.Model(decoder_input_layer, decodification, name = "decoder_model")
        
class Convolutional_Autoencoder2(Autoencoder):

    """
    Method to define the encoder model.
    """
    def define_encoder(self):
        encoder_input_layer = layers.Input(shape=(None, None, 3))     # We define the input with no defined height(H) and width(W). We wil guess the input is (None,64,64,3)
        x = layers.Conv2D(64, (5,5), strides=(1,1), padding="valid", activation="relu")(encoder_input_layer)
        assert x.shape.as_list() == [None, None, None, 64]                              # Here the output should be (None, 60, 60, 64)
        x = layers.Conv2D(64, (5,5), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 64]                              # Here the output should be (None, 56, 56, 64)
        x = layers.Conv2D(64, (5,5), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 64]                              # Here the output should be (None, 52, 52, 64)
        x = layers.Conv2D(32, (5,5), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 32]                              # Here the output should be (None, 48, 48, 32)
        x = layers.Conv2D(32, (5,5), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 32]                              # Here the output should be (None, 44, 44, 32)
        x = layers.Conv2D(32, (5,5), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 32]                              # Here the output should be (None, 40, 40, 32)
        codification = layers.Conv2D(16, (5,5), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert codification.shape.as_list() == [None, None, None, 16]                   # Here the output should be (None, 36, 36, 16)
        
        return tf.keras.Model(encoder_input_layer, codification, name = "encoder_model")

    """
    Method to define the decoder model.
    """
    def define_decoder(self):
        decoder_input_layer = layers.Input(shape=self.encoder.output_shape[1:])
        x = layers.Conv2DTranspose(32, (5,5), strides=(1,1), padding="valid", activation="relu")(decoder_input_layer)
        assert x.shape.as_list() == [None, None, None, 32]                               # Here the output should be (None, 40, 40, 32)
        x = layers.Conv2DTranspose(32, (5,5), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 32]                               # Here the output should be (None, 44, 44, 32)
        x = layers.Conv2DTranspose(32, (5,5), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 32]                               # Here the output should be (None, 48, 48, 32)
        x = layers.Conv2DTranspose(64, (5,5), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 64]                               # Here the output should be (None, 52, 52, 64)
        x = layers.Conv2DTranspose(64, (5,5), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 64]                               # Here the output should be (None, 56, 56, 64)
        x = layers.Conv2DTranspose(64, (5,5), strides=(1,1), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 64]                               # Here the output should be (None, 60, 60, 64)
        decodification = layers.Conv2DTranspose(3, (5,5), strides=(1,1), padding="valid", activation="sigmoid")(x)
        #decodification = layers.Conv2D(3, (3,3), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert decodification.shape.as_list() == [None, None, None, 3] # Here the output should be (None, 64, 64, 3)
        
        return tf.keras.Model(decoder_input_layer, decodification, name = "decoder_model")
        
class Convolutional_Autoencoder3(Autoencoder):

    """
    Method to define the encoder model.
    """
    def define_encoder(self):
        encoder_input_layer = layers.Input(shape=(None, None, 3))     # We define the input with no defined height(H) and width(W). We wil guess the input is (None,64,64,3)
        x = layers.Conv2D(64, (3,3), strides=(2,2), padding="valid", activation="relu")(encoder_input_layer)
        assert x.shape.as_list() == [None, None, None, 64]                              # Here the output should be (None, 62, 62, 64)
        x = layers.Conv2D(32, (3,3), strides=(2,2), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 32]                              # Here the output should be (None, 56, 56, 32)
        codification = layers.Conv2D(16, (3,3), strides=(2,2), padding="valid", activation="sigmoid")(x)
        assert codification.shape.as_list() == [None, None, None, 16]                   # Here the output should be (None, 50, 50, 16)
        
        return tf.keras.Model(encoder_input_layer, codification, name = "encoder_model")

    """
    Method to define the decoder model.
    """
    def define_decoder(self):
        decoder_input_layer = layers.Input(shape=self.encoder.output_shape[1:])
        x = layers.Conv2DTranspose(32, (3,3), strides=(2,2), padding="valid", activation="relu")(decoder_input_layer)
        assert x.shape.as_list() == [None, None, None, 32]                               # Here the output should be (None, 52, 52, 32)
        x = layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding="valid", activation="relu")(x)
        assert x.shape.as_list() == [None, None, None, 64]                               # Here the output should be (None, 62, 62, 64)
        decodification = layers.Conv2DTranspose(3, (3,3), strides=(2,2), padding="valid", activation="sigmoid")(x)
        #decodification = layers.Conv2D(3, (3,3), strides=(1,1), padding="valid", activation="sigmoid")(x)
        assert decodification.shape.as_list() == [None, None, None, 3] # Here the output should be (None, 64, 64, 3)
        
        return tf.keras.Model(decoder_input_layer, decodification, name = "decoder_model")

