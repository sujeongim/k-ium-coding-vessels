
from keras.layers import Layer, Conv2D, MaxPooling2D, Flatten, Input, LSTM, Dense, TimeDistributed, Add, Reshape, Multiply, Activation, Lambda
from keras.applications import ResNet50
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Layer, Conv2D, MaxPooling2D, Flatten, Input, LSTM, Dense, TimeDistributed, Add, Reshape, Multiply, Activation, Lambda
from utils.SpatialAttention import SpatialAttentionLayer
import tensorflow as tf
import logging
import time
import os
import pandas as pd
from utils.ImagePreprocess import preprocess, crop
import glob
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from PIL import Image


# Define the spatial attention mechanism as a custom layer
class SpatialAttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(SpatialAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv = Conv2D(1, (1, 1), activation='sigmoid')
        super(SpatialAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        conv = self.conv(inputs)
        multiplied = Multiply()([inputs, conv])
        return multiplied

    def compute_output_shape(self, input_shape):
        return input_shape


def build_model(input_shape: tuple, num_labels: int=21) -> Model:
    """
    Build model instance for training or loading saved weights.
    Uses ResNet50, Attention mechanism, and LSTM; it takes 8 images in a sequence 
    and attention mechanism is applied to each image. Then it flattens them out
    and pass them to LSTM layer.
    
    # Parameters
    - `input_shape`: (pixel, pixel, rgb/grayscale channel)
    - `num_labels`: number of output labels; for aneurysm diagnosis task, it is fixed to 22
    
    """

    # Create the ResNet50 model
    resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape, pooling=None, classes=2)

    # Define the attention model
    input_layer = Input(shape=input_shape)
    resnet_output = resnet(input_layer)
    attention_output = Conv2D(256, (1, 1), activation='relu')(resnet_output)
    attention_output = SpatialAttentionLayer()(attention_output)

    attention_model = Model(inputs=input_layer, outputs=attention_output)

    # Create the final model
    input_sequence = Input(shape=(8,) + input_shape)
    time_distributed_attention = TimeDistributed(attention_model)(input_sequence)
    time_distributed_maxpool = TimeDistributed(MaxPooling2D())(time_distributed_attention)
    time_distributed_flatten = TimeDistributed(Flatten())(time_distributed_maxpool)
    lstm_output = LSTM(64)(time_distributed_flatten)
    dense_output = Dense(128, activation='relu')(lstm_output)
    final_output = Dense(num_labels, activation='sigmoid')(dense_output)

    model = Model(inputs=input_sequence, outputs=final_output)

    optimizer = Adam(learning_rate=1e-5)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'binary_accuracy'])
    
    return model
