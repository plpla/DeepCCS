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
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error
from .model.splitter import SMILESsplitter

def filter_data(data_table):
    """
    Filter data table using a set of constraints defined by global vars.
    :param data_table: A pandas dataframe with at least 2 columns: 'SMILE' and 'Adduct'
    :return: A copy of the data frame with some columns removed.
    """
    # Remove smiles that are too long
    logging.debug("{} items before filtering".format(len(data_table)))
    pre_filter = len(data_table)
    smiles_splitter = SMILESsplitter()
    data = data_table[np.array([len(smiles_splitter.split(i)) for i in data_table["SMILES"]]) <= MAX_SMILES_LENGTH]
    # Remove empty smiles
    data = data[np.array([len(str(i)) for i in data["SMILES"]]) > 0]
    data = data.dropna(axis=0, how="any", subset=["SMILES"])
    # Remove unhandled adducts
    data = data[np.array([(i in ADDUCTS_TO_KEEP) for i in data["Adducts"]])]
    logging.debug("{} items after filtering".format(len(data)))
    post_filter = len(data)
    logging.debug("{} SMILES and adducts were removed by filter.\n".format(pre_filter - post_filter))
    return data


def percentile_90(Y_true, Y_pred):
    percentile = np.percentile((abs(Y_pred - Y_true) / Y_true) * 100, 90)
    return percentile


def percentile_95(Y_true, Y_pred):
    percentile = np.percentile((abs(Y_pred - Y_true) / Y_true) * 100, 95)
    return percentile


def relative_mean(Y_true, Y_pred):
    mean = np.mean((abs(Y_pred - Y_true) / Y_true) * 100)
    return mean


def relative_median(Y_true, Y_pred):
    med = np.median((abs(Y_pred - Y_true) / Y_true) * 100)
    return med


def _create_h5(path):
    if not path.isdir(path):
        raise IOError("Path of templates cannot be found.")
    df_MCCS_pos = pd.read_csv(path + "/MetCCS1_Template.csv").fillna("")
    df_MCCS_neg = pd.read_csv(path + "/MetCCS2_Template.csv").fillna("")
    df_A_pos = pd.read_csv(path + "/MetCCS3_Template.csv").fillna("")
    df_A_neg = pd.read_csv(path + "/MetCCS4_Template.csv").fillna("")
    df_W_pos = pd.read_csv(path + "/MetCCS5_Template.csv").fillna("")
    df_W_neg = pd.read_csv(path + "/MetCCS6_Template.csv").fillna("")
    df_PNL = pd.read_csv(path + "/PNL_Template.csv").fillna("")
    df_McLean = pd.read_csv(path + "/McLean_Lab_Template.csv").fillna("")
    df_CBM = pd.read_csv(path + "/CBM2018_Template.csv").fillna("")
    logging.debug("Templates loaded")

    dfs = [df_MCCS_pos, df_MCCS_neg, df_A_pos, df_A_neg, df_W_pos, df_W_neg, df_PNL, df_McLean, df_CBM]
    names = ["MetCCS_pos", "MetCCS_neg", "Agilent_pos", "Agilent_neg", "Waters_pos", "Waters_neg", "PNL", "McLean",
             "CBM"]

    logging.debug("Starting writing in h5")
    f = h5.File(path + '/DATASETS.h5', 'w')
    if sys.version_info >= (3, 0):
        dt = h5.special_dtype(vlen=str)
    else:
        dt = h5.special_dtype(vlen=unicode)
    for i, j in enumerate(dfs):
        print(names[i])
        f.create_dataset(names[i] + '/Compound', data=np.array(j["Compound"]), dtype=dt)
        print("compound done")
        f.create_dataset(names[i] + '/CAS', data=np.array(j["CAS"]), dtype=dt)
        print("cas done")
        f.create_dataset(names[i] + '/SMILES', data=np.array(j["SMILES"]), dtype=dt)
        print("smiles done")
        f.create_dataset(names[i] + '/Mass', data=np.array(j["Mass"]))
        print("Mass done")
        f.create_dataset(names[i] + '/Adducts', data=np.array(j["Adducts"]), dtype=dt)
        print("adducts done")
        f.create_dataset(names[i] + '/CCS', data=np.array(j["CCS"]))
        print("ccs done")
        f.create_dataset(names[i] + '/Metadata', data=np.array(j["Metadata"]), dtype=dt)
        print("metadata done")
    f.close()


def output_results(Ifile_name, smiles, adducts, ccs_pred, Ofile_name):
    if Ifile_name[-4:] == ".csv":
        table = pd.read_csv(Ifile_name, sep=",", header=0)
    elif Ifile_name[-5:] == ".xlsx" or Ifile_name[-4:] == ".xls":
        table = pd.read_excel(Ifile_name, header=0)

    if not all(i in list(table.columns.values) for i in ["SMILES", "Adducts"]):
        raise ValueError("Supplied file must contain at leat 2 columns named 'SMILES' and 'Adducts'. "
                         "Use the provided template if needed.")

    results_table = pd.DataFrame({"SMILES": smiles, "Adducts": adducts, "CCS_DeepCCS": ccs_pred})
    results_table.transpose()

    out_df = pd.merge(left=table, right=results_table, on=["SMILES", "Adducts"], how='left')

    pd.options.display.max_colwidth = 2000
    out_df_string = out_df.to_string(header=True)

    if Ofile_name is None:
        sys.stdout.write(out_df_string + "\n")
    else:
        out_df.to_csv(Ofile_name, encoding='utf-8', index=False)


def output_global_stats(ccs_ref, ccs_pred):
    if len(ccs_pred) != len(ccs_ref):
        raise ValueError("The two arrays should have the same length.")
    ccs_ref = np.array(ccs_ref)
    ccs_pred = np.array(ccs_pred)

    mean = round(mean_absolute_error(ccs_ref, ccs_pred), 2)
    med = round(median_absolute_error(ccs_ref, ccs_pred), 2)
    rel_mean = round(relative_mean(ccs_ref, ccs_pred), 2)
    rel_med = round(relative_median(ccs_ref, ccs_pred), 2)
    r2 = round(r2_score(ccs_ref, ccs_pred), 2)
    perc_90 = round(percentile_90(ccs_ref, ccs_pred), 2)
    perc_95 = round(percentile_95(ccs_ref, ccs_pred), 2)

    print("---------------------------------")
    print("    Number of items :  {} ".format(len(ccs_pred)))
    print("     Absolute Mean  :  {} ".format(mean))
    print("     Relative Mean  :  {}% ".format(rel_mean))
    print("    Absolute Median :  {} ".format(med))
    print("    Relative Median :  {}% ".format(rel_med))
    print("           R2       :  {} ".format(r2))
    print("     90e Percentile :  {}% ".format(perc_90))
    print("     95e Percentile :  {}% ".format(perc_95))
    print("---------------------------------")


def read_dataset(h5_path, dataset_name):
    # Choices of dataset_name are : MetCCS_pos, MetCCS_neg, Agilent_pos, Agilent_neg, Waters_pos, Waters_neg, PNL, McLean, CBM

    # Create df
    pd_df = pd.DataFrame(columns=["Compound", "CAS", "SMILES", "Mass", "Adducts", "CCS", "Metadata"])

    # Open reference file and retrieve data corresponding to the dataset name 
    f = h5.File(h5_path, 'r')
    pd_df["Compound"] = f[dataset_name + '/Compound']
    pd_df["CAS"] = f[dataset_name + '/CAS']
    pd_df["SMILES"] = f[dataset_name + '/SMILES']
    pd_df["Mass"] = f[dataset_name + '/Mass']
    pd_df["Adducts"] = f[dataset_name + '/Adducts']
    pd_df["CCS"] = f[dataset_name + '/CCS']
    pd_df["Metadata"] = f[dataset_name + '/Metadata']
    f.close()
    pd_df = filter_data(pd_df)
    return pd_df


def read_input_table(file_name):
    if file_name[-4:] == ".csv":
        table = pd.read_csv(file_name, sep=",", header=0)
    elif file_name[-5:] == ".xlsx" or file_name[-4:] == ".xls":
        table = pd.read_excel(file_name, header=0)

    if not all(i in list(table.columns.values) for i in ["SMILES", "Adducts"]):
        raise ValueError("Supplied file must contain at leat 2 columns named 'SMILES' and 'Adducts'. "
                         "use the provided template if needed.")
    table = filter_data(table)
    smiles = np.array(table['SMILES'])
    adducts = np.array(table['Adducts'])
    return smiles, adducts


def read_reference_table(file_name):
    # Useful to read a reference table containing the ccs values corresponding to SMILES and adducts

    if file_name[-4:] == ".csv":
        table = pd.read_csv(file_name, sep=",", header=0)
    elif file_name[-5:] == ".xlsx" or file_name[-4:] == ".xls":
        table = pd.read_excel(file_name, header=0)

    if not all(i in list(table.columns.values) for i in ["SMILES", "Adducts", "CCS"]):
        raise ValueError("Supplied file must contain at leat 3 columns named 'SMILES', 'Adducts' and 'CCS'. "
                         "use the provided template if needed.")
    table = filter_data(table)
    ccs = np.array(table['CCS'])
    smiles = np.array(table['SMILES'])
    adducts = np.array(table['Adducts'])
    return smiles, adducts, ccs

