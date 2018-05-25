#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    DeepCCS: CCS prediction from SMILES using deep neural network

    Copyright (C) 2018 Pier-Luc

    https://github.com/plpla/DeepCCS

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


import logging
import numpy as np
from keras.models import load_model, save_model
from .encoders import SmilesToOneHotEncoder, AdductToOneHotEncoder

class DeepCCSModel(object):

    def __init__(self):
        self.model = None
        self.adduct_encoder = AdductToOneHotEncoder()
        self.smiles_encoder = SmilesToOneHotEncoder()
        self._is_fit = False

    def fit_smiles_encoder(self, X):
	self.smiles_encoder = SmilesToOneHotEncoder.fit(X)


    def fit_adducts_encoder(self, X):
	self.adduct_encoder = AdductToOneHotEncoder.fit(X)


    def load_model_from_file(self, filename, adduct_encoder_file, smiles_encoder_file):
        self.model = load_model(filename)
        self.adduct_encoder.load_encoder(adduct_encoder_file)
        self.smiles_encoder.load_encoder(smiles_encoder_file)
        self._is_fit = True
        logging.debug("Model loaded from file {}".format(filename))

    def save_model_to_file(self, filename, adduct_encoder_file, smiles_encoder_file):
        save_model(filename)
        self.adduct_encoder.save_encoder(adduct_encoder_file)
        self.smiles_encoder.save_encoder(smiles_encoder_file)

    def predict(self, X_smiles, X_adducts):
        if not self._is_fit:
            raise ValueError("Model must be load or fit first")
        X_smiles = self.smiles_encoder.transform(X_smiles)
        X_adducts = self.adduct_encoder.transform(X_adducts)

        y_pred = self.model.predict([X_smiles, X_adducts])
        return y_pred

    def create_model():
	"""
        Builds a neural net using a set of arguments
        """
        smile_input_layer = Input(shape=(250, input_width=len(self.smiles_encoder)), name="smile")
	
	conv = Conv1D(conv_width=64, kernel_size=4, activation='relu', kernel_initializer='normal')(smile_input_layer)
        previous = conv
	
	for i in range(7-1):
            conv = Conv1D(conv_width=64, kernel_size=4, activation='relu', kernel_initializer='normal')(previous)
            if i == 7-2:
                pool = MaxPooling1D(pool_size=2, strides=2)(conv)
            else:
                pool = MaxPooling1D(pool_size=2, strides=1)(conv)
            previous = pool

        flat = Flatten()(previous)
        previous = flat

        adduct_input_layer = Input(shape=(len(self.adducts_mapper),), name="adduct")
        
	remix_layer = keras.layers.concatenate([previous, adduct_input_layer], axis=-1)
        previous = remix_layer

        for i in range(2):
            dense_layer = Dense(dense_width=384, activation="relu", kernel_initializer='normal')(previous)
            previous = dense_layer

        output = Dense(dense_width=1, activation="linear")(previous)
        
	opt = getattr(keras.optimizers, optimizer='adam')
        opt = opt(lr=0.0001)
        model = Model(input=[smile_input_layer, adduct_input_layer], outputs=output)
        model.compile(optimizer=opt, loss='mean_squared_error')
        
	self.model = model


    def train_model(X1_train, X2_train, Y_train, X1_valid, X2_valid, Y_valid, nbr_epochs):

	self.model.fit([X1_train, X2_train], Y_train, epochs=nbr_epochs, batch_size=2, 
			validation_data=([X1_valid, X2_valid], Y_valid), verbose=1, callbacks=[model_checkpoint])
	
	self._is_fit = True











