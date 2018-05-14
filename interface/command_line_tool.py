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


from sys import argv
from os import path
import argparse
import logging
import pandas as pd
from DeepCCS.utils import filter_data
import numpy as np

DESCRIPTION = "DeepCCS: CCS prediction from SMILES using deep neural network"
VERSION = "0.0.1"


class CommandLineInterface(object):
    def __init__(self):

        ###################
        # Base arg parser #
        ###################

        self.available_commands = ['predict']

        self.parser = argparse.ArgumentParser(description=DESCRIPTION)
        self.parser.add_argument('--license', action='store_true', help='Show license')
        self.parser.add_argument('--version', action='store_true', help='Show version')

        self.subparser = self.parser.add_subparsers(help='sub-command help')

        ###########
        # predict #
        ###########

        self.parser_predict = self.subparser.add_parser("predict",
                                                        help="Predict CCS for some SMILES and adducts using a pretrained model.")

        self.parser_predict.add_argument("-m", help="path to model directory", default="default")
        self.parser_predict.add_argument("-i", help="input file name", required=True)
        self.parser_predict.add_argument("-o", help="Output file name. If not specified, stdout will be used",
                                         default="")
        self.parser_predict.set_defaults(func=self.predict)

        #########
        # train #
        #########

        # TODO

        ############
        # evaluate #
        ############

        # TODO

        #########
        # Parse #
        #########

        if len(argv) == 1:
            self.parser.print_help()
            return

        if "--" == argv[1][:2]:
            args = vars(self.parser.parse_args(argv[1:]))
            print("-- was used. Here are the args" + str(args))
        else:
            args = self.parser.parse_args(argv[1:])
            print("-- was not used. Here are the args:" + str(vars(args)))
            args.func(args)

    def predict(self, args):
        print("Starting prediction tool with the following args:" + str(args))
        if not path.isdir(args.m):
            raise IOError("Model directory cannot be found")
        if not path.isfile(path.join(args.m, "model.h5")):
            raise IOError("Model file is missing from directory")
        if not path.isfile(path.join(args.m, "adductEncoder.json")):
            raise IOError("adductEncoder.json is missing from the model directory")
        if not path.isfile(path.join(args.m, "smilesEncoder.json")):
            raise IOError("smilesEncoder.json is missing from the model directory")

        from DeepCCS.model import DeepCCS
        model = DeepCCS.DeepCCSModel()
        model.load_model_from_file(model_file=path.join(args.m, "model.h5"),
                                   adduct_encoder_file=path.join(args.m, "adductEncoder.json"),
                                   smiles_encoder_file=path.join(args.m, "smilesEncoder.json"))

        X_smiles, X_adducts = self.read_input_table(args.i)
        model.smiles_encoder.transform(X_smiles)
        #model.adduct_encoder.transform(X_adducts)
        #ccs_pred = model.predict(X_smiles, X_adducts)
        #print(ccs_pred)

    def read_input_table(self, file_name):
        if file_name[-4:] == ".csv":
            table = pd.read_csv(file_name, sep=",", header=0)
        elif file_name[-5:] == ".xlsx" or file_name[-4:] == ".xls":
            table = pd.read_excel(file_name, header=0)
        print(list(table.columns.values))
        if not all(i in list(table.columns.values) for i in ["SMILES", "Adducts"]):
            raise ValueError("Supplied file must contain at leat 2 columns named 'SMILES' and 'Adducts'. "
                             "use the provided template if needed.")
        table = filter_data(table)
        smiles = np.array(table['SMILES'])
        adducts = np.array(table['Adducts'])
        return smiles, adducts


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s.%(msecs)d %(levelname)s %(module)s - %(process)d - %(funcName)s: %(message)s")
    CommandLineInterface()