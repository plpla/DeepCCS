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

import json
import numpy as np
#from ..utils import split_smiles
from ..parameters import *

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
    def __init__(self, max_smiles_length=-1):
        BaseEncoder.__init__(self)
        self._max_length = max_smiles_length

    def _fit(self, X):
        """
        X : array of smiles
        """
        splitted_smiles = [self._split_smiles(s) for s in X]
        padded_splitted_smiles = [self._pad_smiles(s) for s in splitted_smiles]
        lengths = [len(s) for s in padded_splitted_smiles]
        chars = [char for s in padded_splitted_smiles for char in s]

        if len(set(lengths)) != 1:
            print(lengths)
            raise ValueError("Items in X must be all of the same length")
        else:
            self._max_length = lengths[0]

        for i, j in enumerate(set(chars)):
            self.converter[j] = i

    def load_encoder(self, json_file):
        BaseEncoder.load_encoder(self, json_file)
        self._max_length = MAX_SMILES_LENGTH

    def _transform(self, X):
        number_of_element = len(X)
        X_encoded = np.zeros((number_of_element, self._max_length, len(self.converter)))
        for i, smiles in enumerate(X):
            for j, letter in enumerate(self._split_smiles(smiles)):
                    X_encoded[i, j, self.converter[letter]] = 1
        return X_encoded

    def _split_smiles(self, smiles):
        splitted_smiles = []
        for j, k in enumerate(smiles):
            if j == 0:
                if k.isupper() and smiles[j + 1].islower() and smiles[j + 1] != "c":
                    splitted_smiles.append(k + smiles[j + 1])
                else:
                    splitted_smiles.append(k)
            elif j != 0 and j < len(smiles) - 1:
                if k.isupper() and smiles[j + 1].islower() and smiles[j + 1] != "c":
                    splitted_smiles.append(k + smiles[j + 1])
                elif k.islower() and smiles[j - 1].isupper():
                    pass
                else:
                    splitted_smiles.append(k)
            elif j == len(smiles) - 1:
                if k.islower() and smiles[j - 1].isupper() and k != "c":
                    pass
                else:
                    splitted_smiles.append(k)
        return splitted_smiles

    def _pad_smiles(self, smiles, padding_char=" "):
        to_pad = int((self._max_length - len(smiles)) / 2)
        s_padded_left = ([padding_char] * to_pad) + smiles
        return s_padded_left + ([padding_char] * (self._max_length - len(s_padded_left)))









