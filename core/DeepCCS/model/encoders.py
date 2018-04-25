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
        return self._transform(X)

    def _transform(self, X):
        pass


class AdductToOneHotEncoder(BaseEncoder):
    def _fit(self, X):
        pass

    def _transform(self, X):
        pass


class SmilesToOneHotEncoder(BaseEncoder):
    def _fit(self, X, n_items=-1):
        pass

    def _transform(self, X):
        pass
