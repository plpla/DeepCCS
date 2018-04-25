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

    def _fit(self, X, n_items=-1):
    """
    X : array of all smiles
    """
	chars = []
	for smiles in X:
            for i, letter in enumerate(smiles):
	        if letter.islower() and smiles[i-1].isupper():
	            chars.append(smiles[i-1]+letter)
	        else:
	            chars.append(letter)

	for i, j in enumerate(set(chars)):
	    self.converter[j] = i



    def _transform(self, X):
    
	number_of_element = X.shape[0]
    	X_encoded = np.zeros((number_of_element, MAX_SMILES_LENGTH, len(self.converter))) # -------- MAX_SMILES_LENGTH
    	for i, smiles in enumerate(X):
        	for j, letter in enumerate(smiles):
	            if letter.isupper() and smiles[j+1].islower():
	                X_encoded[i, j, self.converter[letter+smiles[j+1]]] = 1
	            elif letter.islower() and smiles[j-1].isupper():
	                pass
	            else:
        	        X_encoded[i, j, self.converter[letter]] = 1

    	return X_encoded




