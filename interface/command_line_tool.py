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

import datetime
from sys import argv, exit
from os import path, makedirs
import argparse
import logging

import numpy as np
import pandas as pd

# Seed for shuffling
np.random.seed(13)

DESCRIPTION = "DeepCCS: CCS prediction from SMILES using deep neural network"
VERSION = "0.0.1"


class CommandLineInterface(object):
    def __init__(self):

        ###################
        # Base arg parser #
        ###################

        self.available_commands = ["predict", "evaluate", "compare", "train"]

        parser = argparse.ArgumentParser(description=DESCRIPTION)
        parser.add_argument('--license', action='store_true', help='Show license')
        parser.add_argument('--version', action='store_true', help='Show version')
        parser.add_argument('command', help='Available commands', choices=self.available_commands)

        if len(argv) == 1:
            parser.print_help()
            exit()

        if "--" == argv[1][:2]:  # An option was used. Parse it immediatly
            args = parser.parse_args([argv[1], "predict"])  # Add predict to avoid error message.

            logging.debug("-- was used. Here are the args status:\n" + str(args))
            if args.license:
                self.print_license()
            if args.version:
                print("DeepCCS V{}".format(VERSION))
            else:
                print("Not a valid option.")
                print(parser.print_help())
                exit()

        else:
            args = parser.parse_args(argv[1:2])
            logging.debug("-- was not used. Here are the args:" + str(vars(args)))
            if args.command not in self.available_commands:
                print("Not a valid command.")
                print(parser.print_help())
                exit(1)
            getattr(self, args.command)()

    def predict(self):
        parser = argparse.ArgumentParser(prog='DeepCCS predict',
                                         description="Predict CCS for some SMILES and adducts using a pre-trained " +
                                                     "model.")
        parser.add_argument("-mp", help="Path to model directory", default="../saved_models/default/")
        parser.add_argument("-ap", help="Path to adducts_encoder directory", default="../saved_models/default/")
        parser.add_argument("-sp", help="Path to smiles_encoder directory", default="../saved_models/default/")
        parser.add_argument("-i", help="Input file name with SMILES and adduct columns", required=True)
        parser.add_argument("-o", help="Output file name (ex: MyFile.csv). If not specified, stdout will be used",
                            default="")

        if len(argv) <= 2:
            parser.print_help()
            exit()

        args = parser.parse_args(argv[2:])

        from DeepCCS.utils import read_input_table, output_results
        from DeepCCS.model import DeepCCS

        print("Starting prediction tool with the following args:" + str(args))
        if not path.isdir(args.mp):
            raise IOError("Model directory cannot be found")
        if not path.isdir(args.ap):
            raise IOError("adducts_encoder directory cannot be found")
        if not path.isdir(args.sp):
            raise IOError("smiles_encoder directory cannot be found")

        if not path.isfile(path.join(args.mp, "model.h5")):
            raise IOError("Model file is missing from model directory")
        if not path.isfile(path.join(args.ap, "adducts_encoder.json")):
            raise IOError("adduct_encoder.json is missing from the adducts_encoder directory directory")
        if not path.isfile(path.join(args.sp, "smiles_encoder.json")):
            raise IOError("smiles_encoder.json is missing from the smiles_encoder directory directory")

        model = DeepCCS.DeepCCSModel()
        model.load_model_from_file(filename=path.join(args.mp, "model.h5"),
                                   adduct_encoder_file=path.join(args.ap, "adducts_encoder.json"),
                                   smiles_encoder_file=path.join(args.sp, "smiles_encoder.json"))
        X_smiles, X_adducts = read_input_table(args.i)
        ccs_pred = model.predict(X_smiles, X_adducts)

        ccs_pred = ccs_pred.flatten()

        out_file_name = None
        if args.o != "":
            out_file_name = args.o
        output_results(args.i, X_smiles, X_adducts, ccs_pred, out_file_name)

    def train(self):
        parser = argparse.ArgumentParser(prog='DeepCCS train',
                                         description="Train a new model.")
        parser.add_argument("-ap", help="path to adducts_encoder directory", default=None)
        parser.add_argument("-sp", help="path to smiles_encoder directory", default=None)

        parser.add_argument("-mtrain", help="Use MetCCS train datasets to create the model", default=False,
                            action="store_true", dest="mtrain")
        parser.add_argument("-mtestA", help="Use MetCCS Agilent test datasets to create the model", default=False,
                            action="store_true", dest="mtestA")
        parser.add_argument("-mtestW", help="MetCCS Waters test datasets to create the model", default=False,
                            action="store_true", dest="mtestW")

        parser.add_argument("-pnnl", help="PNNL dataset to create the model", default=False, action="store_true",
                            dest="pnnl")
        parser.add_argument("-cbm", help="CBM2018 dataset to create the model", default=False, action="store_true",
                            dest="cbm")
        parser.add_argument("-mclean", help="McLean dataset to create the model", default=False, action="store_true",
                            dest="mclean")
        parser.add_argument("-f", help="h5 file containing all source datasets", required=True)

        parser.add_argument("-nd", help="New Data to create the model, list of template file (file1.csv,file2.csv,...)",
                            default=None)
        parser.add_argument("-nepochs", help="Number of epochs", default=150)
        parser.add_argument("-verbose", help="Keras verbosity (1 or 0)", default=1)
        parser.add_argument("-test", help="Proportion of the datasets to put in the testing set", default=0.2,
                            type=float)
        parser.add_argument("-o", help="Output directory for model and mappers", default="./")
        parser.set_defaults(func=self.train)

        if len(argv) <= 2:
            parser.print_help()
            exit()

        args = parser.parse_args(argv[2:])

        if 0 >= args.test >= 1:
            raise ValueError("Proportion in test set must be between 0 and 1. Recommended: 0.2")

        logging.debug("\nCondition is: {}".format(not(args.mtrain or args.mtestA or args.mtestW or args.pnnl or args.cbm or args.mclean)))
        if not (args.mtrain or args.mtestA or args.mtestW or args.pnnl or args.cbm or args.mclean or
                args.nd is not None):
            raise ValueError("At least one datafile must be used to train a model.")

        from DeepCCS.model import DeepCCS
        from DeepCCS.utils import read_dataset, read_reference_table, output_global_stats
        from keras.callbacks import ModelCheckpoint

        logging.debug("Starting prediction tool with the following args:" + str(args))
        if not path.isdir(args.o):
            raise IOError("Directory for output model cannot be found")
        if not path.isfile(args.f):
            raise IOError("h5 file of source datasets cannot be found")

        # Initialize lists
        training_datasets = []
        testing_datasets = []
        dt_list = []
        name_test_dataset = []

        date = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%M")  # ex:2018-05-25_14h40

        model_directory = args.o + "/" + date
        if not path.exists(model_directory):
            makedirs(model_directory)

        # ---> Exception !!!
        # MetCCS datasets are the only possible exception to the 80-20 rule

        # Load source datasets according to args
        if args.mtrain:
            for d in ["MetCCS_pos", "MetCCS_neg"]:
                df_dt = read_dataset(args.f, d)
                smiles = df_dt["SMILES"]
                adducts = df_dt["Adducts"]
                ccs = df_dt["CCS"]
                training_datasets.append([smiles, adducts, ccs])

        if args.mtestA:
            dt_list.extend(["Agilent_pos", "Agilent_neg"])
            name_test_dataset.extend(["Agilent_pos", "Agilent_neg"])
        else:
            for d in ["Agilent_pos", "Agilent_neg"]:
                df_dt = read_dataset(args.f, d)
                smiles = df_dt["SMILES"]
                adducts = df_dt["Adducts"]
                ccs = df_dt["CCS"]
                testing_datasets.append([smiles, adducts, ccs])
            name_test_dataset.extend(["Agilent_pos", "Agilent_neg"])

        if args.mtestW:
            dt_list.extend(["Waters_pos", "Waters_neg"])
            name_test_dataset.extend(["Waters_pos", "Waters_neg"])
        else:
            for d in ["Waters_pos", "Waters_neg"]:
                df_dt = read_dataset(args.f, d)
                smiles = df_dt["SMILES"]
                adducts = df_dt["Adducts"]
                ccs = df_dt["CCS"]
                testing_datasets.append([smiles, adducts, ccs])
            name_test_dataset.extend(["Waters_pos", "Waters_neg"])

        if args.pnnl:
            dt_list.append("PNL")
            
        if args.mclean:
            dt_list.append("McLean")
            
        if args.cbm:
            dt_list.append("CBM")
            
        logging.debug("Number of training dataset: {}".format(len(training_datasets)))

        logging.debug("Training dataset list: {}".format(dt_list))

        # Divide source dataset(s) using the specified proportions
        train_fraction = 1 - args.test
        for d in dt_list:
            name_test_dataset.append(d)
            data = read_dataset(args.f, d)
            train = data.sample(frac=train_fraction)
            test = data.drop(train.index)

            train_smiles = train["SMILES"]
            train_adducts = train["Adducts"]
            train_ccs = train["CCS"]

            test_smiles = test["SMILES"]
            test_adducts = test["Adducts"]
            test_ccs = test["CCS"]

            training_datasets.append([train_smiles, train_adducts, train_ccs])
            testing_datasets.append([test_smiles, test_adducts, test_ccs])
            logging.debug("\tTrain set shape: {}".format(train.shape))
            logging.debug("\tTest set shape: {}".format(test.shape))

        logging.debug(len(training_datasets))
        logging.debug("len testdt {}".format(len(testing_datasets)))

        # Load personnal dataset(s) given by -nd arg
        if args.nd is not None:
            new_datasets = args.nd.split(",")
        else:
            new_datasets = []

        # Divide new dataset(s) by the same rule as before
        if len(new_datasets) > 0:
            for f in new_datasets:
                name_test_dataset.append(f.split("/")[-1].split(".")[0])
                smiles, adducts, ccs = read_reference_table(f)

                mask_train = np.zeros(len(smiles), dtype=int)
                mask_train[:int(len(smiles) * train_fraction)] = 1
                np.random.shuffle(mask_train)
                mask_test = 1 - mask_train
                mask_train = mask_train.astype(bool)
                mask_test = mask_test.astype(bool)

                train_smiles = smiles[mask_train]
                train_adducts = adducts[mask_train]
                train_ccs = ccs[mask_train]

                test_smiles = smiles[mask_test]
                test_adducts = adducts[mask_test]
                test_ccs = ccs[mask_test]

                training_datasets.append([train_smiles, train_adducts, train_ccs])
                testing_datasets.append([test_smiles, test_adducts, test_ccs])

        # Format training_dataset arrays for learning
        training_datasets = np.concatenate(training_datasets, axis=1)

        smiles = training_datasets[0]
        adducts = training_datasets[1]
        ccs = training_datasets[2]

        # Divide training data by this rule : 90% in train, 10% in validation set
        mask_t = np.zeros(len(smiles), dtype=int)
        mask_t[:int(len(smiles) * 0.9)] = 1
        np.random.shuffle(mask_t)
        mask_v = 1 - mask_t
        mask_t = mask_t.astype(bool)
        mask_v = mask_v.astype(bool)

        X1_train = smiles[mask_t]
        X2_train = adducts[mask_t]
        Y_train = ccs[mask_t]

        X1_valid = smiles[mask_v]
        X2_valid = adducts[mask_v]
        Y_valid = ccs[mask_v]

        logging.debug("len X1_train  {}".format(len(X1_train)))
        logging.debug("len X1_valid  {}".format(len(X1_valid)))

        # Format testing_datasets (smiles and adducts) to include them in mappers creation
        test_concat = np.concatenate(testing_datasets, axis=1)
        X1_test = test_concat[0]
        X2_test = test_concat[1]

        logging.debug("len X1_test  {}".format(len(X1_test)))

        # Import DeepCCS and initialize model
        new_model = DeepCCS.DeepCCSModel()

        if args.ap is None:
            new_model.fit_adduct_encoder(np.concatenate([X2_train, X2_valid, X2_test]))
        elif args.ap == "d":
            if not path.isfile(path.join("../saved_models/default/", "adducts_encoder.json")):
                raise IOError("adduct_encoder.json is missing from the adducts_encoder directory directory")
            new_model.adduct_encoder.load_encoder("../saved_models/default/adducts_encoder.json")
        else:
            if not path.isfile(path.join(args.ap, "adducts_encoder.json")):
                raise IOError("adduct_encoder.json is missing from the adducts_encoder directory directory")
            new_model.adduct_encoder.load_encoder(args.ap + "adducts_encoder.json")

        if args.sp is None:
            new_model.fit_smiles_encoder(np.concatenate([X1_train, X1_valid, X1_test]))
        elif args.sp == "d":
            if not path.isfile(path.join("../saved_models/default/", "smiles_encoder.json")):
                raise IOError("smiles_encoder.json is missing from the smiles_encoder directory directory")
            new_model.smiles_encoder.load_encoder("../saved_models/default/smiles_encoder.json")
        else:
            if not path.isfile(path.join(args.sp, "smiles_encoder.json")):
                raise IOError("smiles_encoder.json is missing from the smiles_encoder directory directory")
            logging.debug("Loading SMILES encoder")
            new_model.smiles_encoder.load_encoder(args.sp + "smiles_encoder.json")

        logging.debug(new_model.smiles_encoder.converter)

        # Encode smiles and adducts
        X1_train_encoded = new_model.smiles_encoder.transform(X1_train)
        X1_valid_encoded = new_model.smiles_encoder.transform(X1_valid)

        X2_train_encoded = new_model.adduct_encoder.transform(X2_train)
        X2_valid_encoded = new_model.adduct_encoder.transform(X2_valid)

        # Create model structure
        new_model.create_model()

        model_file = model_directory + "/" + "model_checkpoint.model"
        model_checkpoint = ModelCheckpoint(model_file, save_best_only=True, save_weights_only=True)

        if args.verbose:
            print(new_model.model.summary())

        # Train model
        new_model.train_model(X1_train_encoded, X2_train_encoded, Y_train,
                              X1_valid_encoded, X2_valid_encoded, Y_valid,
                              model_checkpoint, int(args.nepochs), args.verbose)

        new_model.model.load_weights(model_file)

        # Save model
        new_model.save_model_to_file(model_directory + "/" + "model.h5", model_directory + "/" + "adducts_encoder.json",
                                     model_directory + "/" + "smiles_encoder.json")

        # Test the new model on each testing datasets independantly and output metrics on the performance of the model
        if not new_model._is_fit:
            raise ValueError("Model must be load or fit first")

        for i, dt in enumerate(testing_datasets):
            dt_name = name_test_dataset[i]
            X1 = dt[0]
            X2 = dt[1]
            Y = dt[2]

            Y_pred = new_model.predict(X1, X2)
            Y_pred = Y_pred.flatten()

            print(" ")
            print("> Testing on " + dt_name + " dataset:")
            print("-----------Model stats-----------")
            output_global_stats(Y, Y_pred)



    def evaluate(self):
        parser = argparse.ArgumentParser(prog='DeepCCS evaluate',
                                         description="Evaluate the model performances using SMILES and adducts for " +
                                                     "which the CCS was measured.")

        parser.add_argument("-mp", help="Path to model directory", default="../saved_models/default/")
        parser.add_argument("-ap", help="Path to adducts_encoder directory", default="../saved_models/default/")
        parser.add_argument("-sp", help="Path to smiles_encoder directory", default="../saved_models/default/")
        parser.add_argument("-i", help="Input file name. Must contain columns SMILES, adducts and CCS", required=True)
        parser.add_argument("-o", help="Output file name (ex: MyFile.csv). If not specified, stdout will be used." +
                                       " If 'none', only global stats will be shown.",
                            default="")

        if len(argv) <= 2:
            parser.print_help()
            exit()

        args = parser.parse_args(argv[2:])

        from DeepCCS.utils import read_reference_table, output_results, output_global_stats
        from DeepCCS.model import DeepCCS

        print("Starting evaluation tool with the following args:" + str(args))
        if not path.isdir(args.mp):
            raise IOError("Model directory cannot be found")
        if not path.isdir(args.ap):
            raise IOError("adducts_encoder directory cannot be found")
        if not path.isdir(args.sp):
            raise IOError("smiles_encoder directory cannot be found")

        if not path.isfile(args.i):
            raise IOError("Input file cannot be found")
        if not path.isfile(path.join(args.mp, "model.h5")):
            raise IOError("Model file is missing from model directory")
        if not path.isfile(path.join(args.ap, "adducts_encoder.json")):
            raise IOError("adducts_encoder.json is missing from the adducts_encoder directory directory")
        if not path.isfile(path.join(args.sp, "smiles_encoder.json")):
            raise IOError("smiles_encoder.json is missing from the smiles_encoder directory directory")

        model = DeepCCS.DeepCCSModel()
        model.load_model_from_file(filename=path.join(args.mp, "model.h5"),
                                   adduct_encoder_file=path.join(args.ap, "adducts_encoder.json"),
                                   smiles_encoder_file=path.join(args.sp, "smiles_encoder.json"))

        X_smiles, X_adducts, X_ccs = read_reference_table(args.i)
        ccs_pred = model.predict(X_smiles, X_adducts)

        ccs_pred = ccs_pred.flatten()

        out_file_name = None
        if args.o != "":
            out_file_name = args.o

        if out_file_name is None or out_file_name.lower() != "none":
            output_results(args.i, X_smiles, X_adducts, ccs_pred, out_file_name)

        print("-----------Model stats-----------")
        output_global_stats(X_ccs, ccs_pred)

    def compare(self):
        parser = argparse.ArgumentParser(prog='DeepCCS compare',
                                         description="Compare the CCS values in a file with the value used to create " +
                                                     "train this predictive model. No predictions involved in the " +
                                                     "process, only comparaison.")

        parser.add_argument("-i", help="Input file name", required=True)
        parser.add_argument("-o", help="Prefix of output file name (ex: MyFile_). If not specified, stdout will be " +
                                       "used. If 'none', onlyt the stats will be shown.", default="")
        parser.add_argument("-f", help="hdf5 file containing the datasets used to create this algoritm", required=True)
        parser.add_argument("-d", help="List of datasets to compare to separated by coma (dtA,dtB,dtC)", default=None)

        if len(argv) <= 2:
            parser.print_help()
            exit()

        args = parser.parse_args(argv[2:])

        from DeepCCS.utils import read_dataset, read_reference_table, output_results, output_global_stats

        logging.debug("Starting comparaison tool with the following args:" + str(args))
        if not path.isfile(args.i):
            raise IOError("Reference file cannot be found")
        if not path.isfile(args.f):
            raise IOError("h5 file cannot be found")

        # Output prefix, if none : output to stdout
        out_file_name_prefix = None
        if args.o != "":
            out_file_name_prefix = args.o

        # Data used to create algorithm
        if args.d is not None:
            dt_list = args.d.split(", ")
        else:
            dt_list = ["MetCCS_pos", "MetCCS_neg", "Agilent_pos", "Agilent_neg", "Waters_pos", "Waters_neg", "PNL",
                       "McLean", "CBM"]

        # Get a pandas dataframe for each dataset asked for comparaison
        # output another table with all the original values + the ccs given by user in an extra column
        # print general stats on the compaison
        logging.debug("Starting iterating on the dataset list of comparaison")
        for i in dt_list:
            df_dt = read_dataset(args.f, i)
            smiles = df_dt["SMILES"]
            adducts = df_dt["Adducts"]
            ccs = df_dt["CCS"]

            out_file_name = None
            if out_file_name_prefix is not None:
                out_file_name = out_file_name_prefix + i + ".txt"

            if out_file_name is None or out_file_name_prefix.lower() != "none":
                output_results(args.i, smiles, adducts, ccs, out_file_name)

            smiles_u, adducts_u, ccs_u = read_reference_table(args.i)

            df_user = pd.DataFrame({"SMILES": smiles_u,
                                    "Adducts": adducts_u,
                                    "CCS": ccs_u})

            df_ref = pd.DataFrame({"SMILES": smiles,
                                   "Adducts": adducts,
                                   "CCS_DeepCCS": ccs})

            merged_df = pd.merge(left=df_user, right=df_ref, on=["SMILES", "Adducts"], how='inner')

            if len(merged_df["CCS"]) == 0:
                print(i)
                print("No corresponding molecule, moving to next dataset.")
                continue
            else:
                print("{} dataset :".format(i))
                print("=> {} molecules used for comparaison".format(len(merged_df["CCS"])))
                print("--------Comparaison stats--------")
                output_global_stats(merged_df["CCS_DeepCCS"], merged_df["CCS"])

    def print_license(self):
        print("""
        DeepCCS: CCS predictions directly from SMILES
        Copyright (C) 2018  Pier-Luc Plante
        
        GNU General Public License version 3
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
        """)


if __name__ == '__main__':
    #logging.basicConfig(level=logging.DEBUG,
    #                    format="%(asctime)s.%(msecs)d %(levelname)s %(module)s - %(process)d - %(funcName)s: %(message)s")
    CommandLineInterface()
