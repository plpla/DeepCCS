#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    DeepCCS: CCS prediction using deep neural network

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


class DeepCCSModel(object):

    def __init__(self):
        self.model = None
        self._is_trained = False;

    def load_model_from_file(self, filename):
        self.model = load_model(filename)
        self._is_trained = True
        logging.debug("Model loaded from file {}".format(filename))

    def save_model_to_file(self, filename):
        save_model(filename)

    def predict(self, X_smiles, X_adducts=[]):
        pass






