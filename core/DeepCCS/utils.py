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
import pandas as pd
import h5py as h5
import sys

def filter_data(data_table):
    """
    Filter data table using a set of constraints defined by global vars.
    :param data_table: A pandas dataframe with at least 2 columns: 'SMILE' and 'Adduct'
    :return: A copy of the data frame with some columns removed.
    """
    # Remove smiles that are too long
    logging.debug("{} items before filtering".format(len(data_table)))
    pre_filter = len(data_table)
    data = data_table[np.array([len(str(i)) for i in data_table["SMILES"]]) <= MAX_SMILES_LENGTH]
    # Remove empty smiles
    data = data[np.array([len(str(i)) for i in data["SMILES"]]) > 0]
    data = data.dropna(axis=0, how="any", subset=["SMILES"])
    # Remove unhandled adducts
    data = data[np.array([(i in ADDUCTS_TO_KEEP) for i in data["Adducts"]])]
    logging.debug("{} items after filtering".format(len(data)))
    post_filter = len(data)
    sys.stdout.write("--> {} SMILES and adducts were removed.\n".format(pre_filter-post_filter))
    return data

def percentile_90(Y_true, Y_pred):
    percentile = np.percentile((abs(Y_pred-Y_true)/Y_true)*100, 90)
    return percentile

def percentile_95(Y_true, Y_pred):
    percentile = np.percentile((abs(Y_pred-Y_true)/Y_true)*100, 95)
    return percentile

def relative_mean(Y_true, Y_pred):
    mean = np.mean((abs(Y_pred-Y_true)/Y_true)*100)
    return mean

def relative_median(Y_true, Y_pred):
    med = np.median((abs(Y_pred-Y_true)/Y_true)*100)
    return med


def create_datasets_compil(path_to_templates):
    df_MCCS_pos = pd.read_csv(path_to_templates+"/MetCCS1_Template.csv").fillna(" ")
    df_MCCS_neg = pd.read_csv(path_to_templates+"/MetCCS2_Template.csv").fillna(" ")
    df_A_pos = pd.read_csv(path_to_templates+"/MetCCS3_Template.csv").fillna(" ")
    df_A_neg = pd.read_csv(path_to_templates+"/MetCCS4_Template.csv").fillna(" ")
    df_W_pos = pd.read_csv(path_to_templates+"/MetCCS5_Template.csv").fillna(" ")
    df_W_neg = pd.read_csv(path_to_templates+"/MetCCS6_Template.csv").fillna(" ")
    df_PNL = pd.read_csv(path_to_templates+"/PNL_Template.csv").fillna(" ")
    df_McLean = pd.read_csv(path_to_templates+"/McLean_Lab_Template.csv").fillna(" ")
    df_CBM = pd.read_csv(path_to_templates+"/CBM2018_Template.csv").fillna(" ")
    print("Templates loaded")

    dfs = [df_MCCS_pos, df_MCCS_neg, df_A_pos, df_A_neg, df_W_pos, df_W_neg, df_PNL, df_McLean, df_CBM]
    names = ["MetCCS_pos", "MetCCS_neg", "Agilent_pos", "Agilent_neg", "Waters_pos", "Waters_neg", "PNL", "McLean", "CBM"] 

    print("Starting writing in h5")
    f = h5.File(path_to_templates+'/DATASETS.h5', 'w')
    dt = h5.special_dtype(vlen=unicode)
    for i, j in enumerate(dfs):
        print(names[i])
	f.create_dataset(names[i]+'/Compound', data=np.array(j["Compound"]), dtype=dt)
        print("compound done")
	f.create_dataset(names[i]+'/CAS', data=np.array(j["CAS"]), dtype=dt)
	print("cas done")
        f.create_dataset(names[i]+'/SMILES', data=np.array(j["SMILES"]), dtype=dt)
	print("smiles done")
        f.create_dataset(names[i]+'/Mass', data=np.array(j["Mass"]))
	print("Mass done")
        f.create_dataset(names[i]+'/Adducts', data=np.array(j["Adducts"]), dtype=dt)
	print("adducts done")
        f.create_dataset(names[i]+'/CCS', data=np.array(j["CCS"]))
	print("ccs done")
        f.create_dataset(names[i]+'/Metadata', data=np.array(j["Metadata"]), dtype=dt)
	print("metadata done")
    f.close()







