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

from .parameters import *
import numpy as np
import logging


def filter_data(data_table):
    """
    Filter data table using a set of constraints defined by global vars.
    :param data_table: A pandas dataframe with at least 2 columns: 'SMILE' and 'Adduct'
    :return: A copy of the data frame with some columns removed.
    """
    # Remove smiles that are too long
    logging.debug("{} items before filtering".format(len(data_table)))
    data = data_table[np.array([len(str(i)) for i in data_table["SMILES"]]) <= MAX_SMILES_LENGTH]
    # Remove empty smiles
    data = data[np.array([len(str(i)) for i in data["SMILES"]]) > 0]
    data = data.dropna(axis=0, how="any", subset=["SMILES"])
    # Remove unhandled adducts
    data = data[np.array([(i in ADDUCTS_TO_KEEP) for i in data["Adducts"]])]
    logging.debug("{} items after filtering".format(len(data)))
    return data