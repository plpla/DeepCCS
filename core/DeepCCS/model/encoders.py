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

import json
import numpy as np
from ..utils import split_smiles

class BaseEncoder(object):
    """
    Base encoder class to encode data in a particular way
    """
    def __init__(self):
        self.converter = {}
        self._is_fit = False

    def load_encoder(self, json_file):
        with open(json_file, "r") as fi:
            self.converter = json.load(fi)
        if len(self.converter) > 0:
            self._is_fit = True

    def save_encoder(self, file_name):
        with open(file_name, "w") as fo:
            json.dump(self.converter, fo)

    def fit(self, X):
        self._fit(X)
        self._is_fit = True

    def _fit(self, X):
        pass

    def transform(self, X):
        if self._is_fit:
            return self._transform(X)
        else:
            raise RuntimeError("Encoder must be fit first")

    def _transform(self, X):
        pass


class AdductToOneHotEncoder(BaseEncoder):

    def _fit(self, X):
        """
        X : array of all adducts
        """
        for i, j in enumerate(set(X)):
            self.converter[j] = i

    def _transform(self, X):
        number_of_element = X.shape[0]
        X_encoded = np.zeros((number_of_element, len(self.converter)))
        for i, adduct in enumerate(X):
            X_encoded[i, self.converter[adduct]] = 1
        return X_encoded


class SmilesToOneHotEncoder(BaseEncoder):
    def __init__(self):
        BaseEncoder.__init__(self)
        self._max_length = -1

    def _fit(self, X):
        """
        X : array of smiles
        """
        splitted_smiles = [split_smiles(s) for s in X]
        lengths = [len(s) for s in splitted_smiles]
        chars = [char for s in splitted_smiles for char in s]

        if len(set(lengths)) != 1:
            print(lengths)
            raise ValueError("Items in X must be all of the same length")
        else:
            self._max_length = lengths[0]

        for i, j in enumerate(set(chars)):
            self.converter[j] = i

    def _transform(self, X):
        number_of_element = len(X)
        X_encoded = np.zeros((number_of_element, self._max_length, len(self.converter)))
        for i, smiles in enumerate(X):
            for j, letter in enumerate(split_smiles(smiles)):
                    X_encoded[i, j, self.converter[letter]] = 1
        return X_encoded




