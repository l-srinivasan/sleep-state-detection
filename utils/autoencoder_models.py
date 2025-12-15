#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 10:18:12 2025
@author: Leela Srinivasan
"""


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def train_autoencoder_validation(data, validation_data, encoding_dim=55, epochs=250, batch_size=64):
    """
    Autoencoder written by the authors.
    
    Parameters
    ----------
    data : numpy array
        2d array.
    validation_data : numpy array
        2d array..

    Returns
    -------
    autoencoder : model class
        compiled autoencoder.
    history : dict
        fit history.

    """
    # Define the autoencoder structure
    autoencoder = Sequential()
    autoencoder.add(Dense(encoding_dim, activation='relu', input_shape=(data.shape[1],)))
    autoencoder.add(Dense(data.shape[1], activation='linear'))
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    history = autoencoder.fit(data, data, validation_data=(validation_data, validation_data), epochs=epochs, batch_size=batch_size, shuffle=True)
    return autoencoder, history