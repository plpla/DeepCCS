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
from sys import argv
from os import path, makedirs
import h5py as h5
import argparse
import logging
import pandas as pd
from DeepCCS.utils import *
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error
from keras.callbacks import TensorBoard, ModelCheckpoint
from DeepCCS.model import DeepCCS

# Seed for shuffling
np.random.seed(13)



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

        self.parser_predict.add_argument("-mp", help="path to model directory", default="../saved_models/default/")
	self.parser_predict.add_argument("-ap", help="path to adducts_encoder directory", default="../saved_models/default/")
	self.parser_predict.add_argument("-sp", help="path to smiles_encoder directory", default="../saved_models/default/")

        self.parser_predict.add_argument("-i", help="input file name", required=True)
        self.parser_predict.add_argument("-o", help="Output file name (MyFile.csv). If not specified, stdout will be used",
                                         default="")
        self.parser_predict.set_defaults(func=self.predict)

        #########
        # train #
        #########
	
        self.parser_predict = self.subparser.add_parser("train",
                                                        help="Train a new model.")
        self.parser_predict.add_argument("-ap", help="path to adducts_encoder directory", default=None)
        self.parser_predict.add_argument("-sp", help="path to smiles_encoder directory", default=None)

	self.parser_predict.add_argument("-mtrain", help="MetCCS train datasets to create the model", default=None)
	self.parser_predict.add_argument("-mtestA", help="MetCCS Agilent test datasets to create the model", default=None)
	self.parser_predict.add_argument("-mtestW", help="MetCCS Waters test datasets to create the model", default=None)

	self.parser_predict.add_argument("-p", help="PNNL dataset to create the model", default=None)
	self.parser_predict.add_argument("-c", help="CBM2018 dataset to create the model", default=None)
	self.parser_predict.add_argument("-mcl", help="McLean dataset to create the model", default=None)	
	self.parser_predict.add_argument("-f", help="h5 file containing all source datasets", required=True)

        self.parser_predict.add_argument("-nd", help="New Data to create the model, list of template file (file1.csv,file2.csv,...)", default=None)
        self.parser_predict.add_argument("-nepochs", help="Number of epochs", default=150)
        self.parser_predict.add_argument("-o", help="Output directory for model and mappers", default="./")
        self.parser_predict.set_defaults(func=self.train)


        ############
        # evaluate #
        ############

        self.parser_predict = self.subparser.add_parser("evaluate",
                                                        help="Predict CCS for some SMILES and adducts using a pretrained model and ouput stats on the predictions.")

        self.parser_predict.add_argument("-mp", help="path to model directory", default="../saved_models/default/")
        self.parser_predict.add_argument("-ap", help="path to adducts_encoder directory", default="../saved_models/default/")
        self.parser_predict.add_argument("-sp", help="path to smiles_encoder directory", default="../saved_models/default/")

        self.parser_predict.add_argument("-r", help="reference file name", required=True)
	self.parser_predict.add_argument("-o", help="Output file name (MyFile.csv). If not specified, stdout will be used",
                                         default="")
        self.parser_predict.set_defaults(func=self.evaluate)

	###########
	# compare #
	###########

        self.parser_predict = self.subparser.add_parser("compare",
                                                        help="Compare for some SMILES and adducts the given CCS value with the value used to create this algoritm.")

        self.parser_predict.add_argument("-r", help="reference file name", required=True)
        self.parser_predict.add_argument("-o", help="prefix for output file name (MyFile_). If not specified, stdout will be used",
                                         default="")
	self.parser_predict.add_argument("-f", help="h5 file containing the datasets used to create this algoritm", required=True)
	self.parser_predict.add_argument("-d", help="List of datasets to compare to separated by coma (dtA,dtB,dtC)", default=None)
        self.parser_predict.set_defaults(func=self.compare)


        #########
        # Parse #
        #########

        if len(argv) == 1:
            self.parser.print_help()
            

        if "--" == argv[1][:2]:
            args = vars(self.parser.parse_args(argv[1:]))
            print("-- was used. Here are the args" + str(args))
        else:
            args = self.parser.parse_args(argv[1:])
            print("-- was not used. Here are the args:" + str(vars(args)))
            args.func(args)
    
    



    def compare(self, args):
    # Useful to compare given reference data to the ones used to create this algorithm.
    
        print("Starting comparaison tool with the following args:" + str(args))
        if not path.isfile(args.r):
            raise IOError("Reference file cannot be found")
        if not path.isfile(args.f):
            raise IOError("h5 file cannot be found")
	
        # Output prefix, if none : output to stdout
        if args.o != "":
            Ofile_name = args.o
        else:
            Ofile_name = None

	# Data used to create algorithm
	if args.d != None:
            dt_list = args.d.split(",")
	else:
            dt_list = ["MetCCS_pos", "MetCCS_neg", "Agilent_pos", "Agilent_neg", "Waters_pos", "Waters_neg", "PNL", "McLean", "CBM"]
	


	# Get a pandas dataframe for each dataset asked for comparaison 
	# output another table with all the original values + the ccs given by user in an extra column
	# print general stats on the compaison
	print("--> Starting iterating on the dataset list of comparaison")
	for i in dt_list:
	    df_dt = read_datasets(args.f, i)
	    smiles = df_dt["SMILES"] 
	    adducts = df_dt["Adducts"]
	    ccs = df_dt["CCS"] 

	    print("--> h5 file of datasets : red")
	    output_results(args.r, smiles, adducts, ccs, Ofile_name+i+".txt")
	    print("--> output table : generated")

	    smiles_u, adducts_u, ccs_u = read_reference_table(args.r)
	    ccs_user = []
	    ccs_ref = []
	    for j, smi in enumerate(smiles):
		for l, smi_u in enumerate(smiles_u):
		    if smi == smi_u and adducts[j] == adducts_u[l]:
			ccs_user.append(ccs_u[l])
			ccs_ref.append(ccs[j])
	    print("--> List of similar smiles : done")

	    if len(ccs_user) == 0:
		print(i)
		print("No corresponding molecule, moving to next dataset.")
		continue
	    else :
	        print(ccs_user[0:10])
	        print("___")
	        print(ccs_ref[0:10])
                print("{} dataset :".format(i))
		print("=> {} molecules used for comparaison".format(len(ccs_user)))
	        print("--------Comparaison stats--------")
                output_global_stats(ccs_ref, ccs_user)

	

    
    def evaluate(self, args):
    # Useful to evaluate the performances of the model, the theoritical ccs values must be known.
    
	print("Starting evaluation tool with the following args:" + str(args))
        if not path.isdir(args.mp):
            raise IOError("Model directory cannot be found")
	if not path.isdir(args.ap):
            raise IOError("adducts_encoder directory cannot be found")
	if not path.isdir(args.sp):
            raise IOError("smiles_encoder directory cannot be found")

	if not path.isfile(args.r):
            raise IOError("Reference file cannot be found")
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

        X_smiles, X_adducts, X_ccs = read_reference_table(args.r)
        ccs_pred = model.predict(X_smiles, X_adducts)

	ccs_pred = np.array([i[0] for i in ccs_pred])

	print(X_ccs[0:10])
	print("---")
	print(ccs_pred[0:10])
	

	
	if args.o != "":
            Ofile_name = args.o
	else:
	    Ofile_name = None

        output_results(args.r, X_smiles, X_adducts, ccs_pred, Ofile_name)

	print("-----------Model stats-----------")
	output_global_stats(X_ccs, ccs_pred)
	



    def predict(self, args):
    # Useful for predicting unknown ccs values

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

	ccs_pred = np.array([i[0] for i in ccs_pred])

	if args.o != "":
	    Ofile_name = args.o
	else:
	    Ofile_name = None
        
	output_results(args.i, X_smiles, X_adducts, ccs_pred, Ofile_name)



    def train(self, args):
	print("Starting prediction tool with the following args:" + str(args))
        if not path.isdir(args.o):
            raise IOError("Directory for output model cannot be found")
	if not path.isfile(args.f):
            raise IOError("h5 file of source datasets cannot be found")

	# Initialize lists
	training_datasets = []
	testing_datasets = []
	dt_list = []
	name_test_dataset = []	
	
	date = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%M") # 2018-05-25_14h40

        model_directory = args.o+"/"+date
        if not path.exists(model_directory):
            makedirs(model_directory)

	
	# ---> Exception !!!
	# MetCCS datasets are the only possible exception to the 80-20 rule

	# Load source datasets according to args	
	if args.mtrain == "y":
	    for d in ["MetCCS_pos", "MetCCS_neg"]:
		df_dt = read_datasets(args.f, d)
                smiles = df_dt["SMILES"]
                adducts = df_dt["Adducts"]
                ccs = df_dt["CCS"]
		training_datasets.append([smiles, adducts, ccs])
	    

	if args.mtestA == "y":
	    dt_list.extend(["Agilent_pos", "Agilent_neg"])
	    name_test_dataset.extend(["Agilent_pos", "Agilent_neg"])
	else:
	    for d in ["Agilent_pos", "Agilent_neg"]:
                df_dt = read_datasets(args.f, d)
                smiles = df_dt["SMILES"]
                adducts = df_dt["Adducts"]
                ccs = df_dt["CCS"]
                testing_datasets.append([smiles, adducts, ccs])
	    name_test_dataset.extend(["Agilent_pos", "Agilent_neg"])

	if args.mtestW == "y":
	    dt_list.extend(["Waters_pos", "Waters_neg"])
	    name_test_dataset.extend(["Waters_pos", "Waters_neg"])
	else:
	    for d in ["Waters_pos", "Waters_neg"]:
                df_dt = read_datasets(args.f, d)
                smiles = df_dt["SMILES"]
                adducts = df_dt["Adducts"]
                ccs = df_dt["CCS"]
                testing_datasets.append([smiles, adducts, ccs])
	    name_test_dataset.extend(["Waters_pos", "Waters_neg"])

	if args.p == "y":
	    dt_list.append("PNL")
	    name_test_dataset.append("PNL")
	    
	if args.mcl == "y":
	    dt_list.append("McLean")
	    name_test_dataset.append("McLean")

	if args.c == "y":
            dt_list.append("CBM")
	    name_test_dataset.append("CBM")
	print(len(training_datasets))    
        
	print(dt_list)
	# Divide source dataset(s) by this rule : 80% in train, 20% in test
	for d in dt_list:
	    name_test_dataset.append(d)
            data = read_datasets(args.f, d)
            train = data.sample(frac=0.8)
            test = data.drop(train.index)
	    
	    train_smiles = train["SMILES"]
	    train_adducts = train["Adducts"]
	    train_ccs = train["CCS"]

	    test_smiles = test["SMILES"] 
	    test_adducts = test["Adducts"]
	    test_ccs = test["CCS"]
            
	    training_datasets.append([train_smiles, train_adducts, train_ccs])
	    testing_datasets.append([test_smiles, test_adducts, test_ccs])
            print("\tTrain: {}".format(train.shape))
            print("\tTest: {}".format(test.shape))

	print(len(training_datasets))
	print("len testdt {}".format(len(testing_datasets)))

	# Load personnal dataset(s) given by -nd arg
	if args.nd != None:
	    new_datasets = args.nd.split(",")
	else:
	    new_datasets = []



	# Divide new dataset(s) by this rule : 80% in train, 20% in test
	if len(new_datasets) > 0 :
    	    for f in new_datasets:
	        name_test_dataset.append(f.split("/")[-1].split(".")[0])
      	        smiles, adducts, ccs = read_reference_table(f)
	    
	        mask_train = np.zeros(len(smiles), dtype=int)
	        mask_train[:int(len(smiles)*0.8)] = 1
	        np.random.shuffle(mask_train)
	        mask_test = 1-mask_train
	        mask_train = mask_train.astype(bool)
	        mask_test = mask_test.astype(bool)

    	        train_smiles = smiles[mask_train]
                train_adducts = adducts[mask_train]
                train_ccs = ccs[mask_train]

                test_smiles = smiles[mask_test]
                test_adducts = adducts[mask_test]
                test_ccs = ccs[mask_test]
    	    
    	        training_datasets.append([train_smiles, test_adducts, train_ccs])
                testing_datasets.append([test_smiles, test_adducts, test_ccs])



	# Format training_dataset arrays for learning
	training_datasets = np.concatenate(training_datasets, axis=1)

	smiles = training_datasets[0] 
	adducts = training_datasets[1]
	ccs = training_datasets[2]

	# Divide training data by this rule : 90% in train, 10% in validation set
	mask_t = np.zeros(len(smiles), dtype=int)
        mask_t[:int(len(smiles)*0.9)] = 1
        np.random.shuffle(mask_t)
        mask_v = 1-mask_t
        mask_t = mask_t.astype(bool)
        mask_v = mask_v.astype(bool)

        X1_train = smiles[mask_t]
        X2_train = adducts[mask_t]
        Y_train = ccs[mask_t]

        X1_valid = smiles[mask_v]
        X2_valid = adducts[mask_v]
        Y_valid = ccs[mask_v]

	print("len X1_train  {}".format(len(X1_train)))
	print("len X1_valid  {}".format(len(X1_valid)))
	

	
	# Format testing_datasets (smiles and adducts) to include them in mappers creation
	test_concat = np.concatenate(testing_datasets, axis=1)
	X1_test = test_concat[0]
	X2_test = test_concat[1]
	
	print("len X1_test  {}".format(len(X1_test)))

	# Import DeepCCS and initialize model
        new_model = DeepCCS.DeepCCSModel()

	
	
	if args.ap == None:
	    new_model.fit_adduct_encoder(np.concatenate([X2_train, X2_valid, X2_test]))
	elif args.ap == "d":
	    if not path.isfile(path.join("../saved_models/default/", "adducts_encoder.json")):
                raise IOError("adduct_encoder.json is missing from the adducts_encoder directory directory")
	    self.adduct_encoder.load_encoder("../saved_models/default/adducts_encoder.json")
	else:
	    if not path.isfile(path.join(args.ap, "adducts_encoder.json")):
                raise IOError("adduct_encoder.json is missing from the adducts_encoder directory directory")
	    self.adduct_encoder.load_encoder(args.ap)

	if args.sp == None:
	    new_model.fit_smiles_encoder(np.concatenate([X1_train, X1_valid, X1_test]))
        elif args.sp == "d":
            if not path.isfile(path.join("../saved_models/default/", "smiles_encoder.json")):
                raise IOError("smiles_encoder.json is missing from the smiles_encoder directory directory")
	    self.smiles_encoder.load_encoder("../saved_models/default/smiles_encoder.json")
        else:
            if not path.isfile(path.join(args.sp, "smiles_encoder.json")):
                raise IOError("smiles_encoder.json is missing from the smiles_encoder directory directory")
	    self.adduct_encoder.load_encoder(args.sp)
	
	print(new_model.smiles_encoder.converter)

	# Encode smiles and adducts
	X1_train_encoded = new_model.smiles_encoder.transform(X1_train)
	X1_valid_encoded = new_model.smiles_encoder.transform(X1_valid)

	X2_train_encoded = new_model.adduct_encoder.transform(X2_train)
        X2_valid_encoded = new_model.adduct_encoder.transform(X2_valid)


	# Create model structure
	new_model.create_model()
	
	model_file = model_directory+"/"+"model_checkpoint.model"
	model_checkpoint = ModelCheckpoint(model_file, save_best_only=True, save_weights_only=True)

	
	print(new_model.model.summary())

	# Train model
	new_model.train_model(X1_train_encoded, X2_train_encoded, Y_train, 
				X1_valid_encoded, X2_valid_encoded, Y_valid, model_checkpoint, int(args.nepochs))
	
	new_model.model.load_weights(model_file)	

	# Save model
	new_model.save_model_to_file(model_directory+"/"+"model.h5", model_directory+"/"+"adducts_encoder.json", model_directory+"/"+"smiles_encoder.json")
	

	# Test the new model on each testing datasets independantly and output metrics on the performance of the model
	if not new_model._is_fit:
	    raise ValueError("Model must be load or fit first")
	
	for i, dt in enumerate(testing_datasets):
	    dt_name = name_test_dataset[i]
	    X1 = dt[0] 
	    X2 = dt[1]
	    Y = dt[2]
	
	    Y_pred = new_model.predict(X1, X2)
	    Y_pred = np.array([i[0] for i in Y_pred])
	    
	    print(" ")
	    print("> Testing on "+dt_name+" dataset:")
	    print("-----------Model stats-----------")
	    output_global_stats(Y, Y_pred)
	    


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s.%(msecs)d %(levelname)s %(module)s - %(process)d - %(funcName)s: %(message)s")
    CommandLineInterface()
