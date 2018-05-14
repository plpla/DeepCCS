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





