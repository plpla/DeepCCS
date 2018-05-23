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
import h5py as h5
import argparse
import logging
import pandas as pd
from DeepCCS.utils import *
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error

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
        self.parser_predict.add_argument("-o", help="Output file name (MyFile.csv). If not specified, stdout will be used",
                                         default="")
        self.parser_predict.set_defaults(func=self.predict)

        #########
        # train #
        #########

        # TODO

        ############
        # evaluate #
        ############

        self.parser_predict = self.subparser.add_parser("evaluate",
                                                        help="Predict CCS for some SMILES and adducts using a pretrained model and ouput stats on the predictions.")

        self.parser_predict.add_argument("-m", help="path to model directory", default="default")
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
        self.parser_predict.add_argument("-o", help="prefix for output file name. If not specified, stdout will be used",
                                         default="")
	self.parser_predict.add_argument("-f", help="h5 file containing the datasets used to create this algoritm", required=True)
	self.parser_predict.add_argument("-d", help="List of datasets to compare to separated by coma (dtA,dtB,dtC)", default=None)
        self.parser_predict.set_defaults(func=self.compare)

	#############
	# create h5 #
	#############

	self.parser_predict = self.subparser.add_parser("create_h5",
                                                        help="Create the h5 file of all the datasets used to create the algorithm.")

        self.parser_predict.add_argument("-p", help="Path to the datatsets template", required=True)
	self.parser_predict.set_defaults(func=self.create_h5)



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
    
    
    def create_h5(self, args):
        print("Starting creating tool with the following args:" + str(args))
        if not path.isdir(args.p):
            raise IOError("Path of templates cannot be found.")

	create_datasets_compil(args.p)


    def output_global_stats(self, ccs_ref, ccs_pred):
	
	ccs_ref = np.array(ccs_ref)
	ccs_pred = np.array(ccs_pred)
	
	mean = round(mean_absolute_error(ccs_ref, ccs_pred),2)
        med = round(median_absolute_error(ccs_ref, ccs_pred),2)
        rel_mean = round(relative_mean(ccs_ref, ccs_pred),2)
        rel_med = round(relative_median(ccs_ref, ccs_pred),2)
        r2 = round(r2_score(ccs_ref, ccs_pred),2)
        perc_90 = round(percentile_90(ccs_ref, ccs_pred),2)
        perc_95 = round(percentile_95(ccs_ref, ccs_pred),2)

        print("---------------------------------")
        print("     Absolute Mean  :  {} ".format(mean))
        print("     Relative Mean  :  {}% ".format(rel_mean))   
        print("    Absolute Median :  {} ".format(med))   
        print("    Relative Median :  {}% ".format(rel_med))  
        print("           R2       :  {} ".format(r2))  
        print("     90e Percentile :  {}% ".format(perc_90))
        print("     95e Percentile :  {}% ".format(perc_95))  
        print("---------------------------------")



    def read_datasets(self, h5_path, dataset_name):
    # Choices of dataset_name are : MetCCS_pos, MetCCS_neg, Agilent_pos, Agilent_neg, Waters_pos, Waters_neg, PNL, McLean, CBM
      
	# Create df
	pd_df = pd.DataFrame(columns=["Compound", "CAS", "SMILES", "Mass", "Adducts", "CCS", "Metadata"])
	
	# Open reference file and retrieve data corresponding to the dataset name 
	f = h5.File(h5_path, 'r')
	pd_df["Compound"] = f[dataset_name+'/Compound']
	pd_df["CAS"] = f[dataset_name+'/CAS']
	pd_df["SMILES"] = f[dataset_name+'/SMILES']
	pd_df["Mass"] = f[dataset_name+'/Mass']
	pd_df["Adducts"] = f[dataset_name+'/Adducts']
	pd_df["CCS"] = f[dataset_name+'/CCS']
	pd_df["Metadata"] = f[dataset_name+'/Metadata']
	f.close()

	return pd_df
	


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
	    df_dt = self.read_datasets(args.f, i)
	    smiles = df_dt["SMILES"] 
	    adducts = df_dt["Adducts"]
	    ccs = df_dt["CCS"] 

	    print("--> h5 file of datasets : red")
	    self.output_results(args.r, smiles, adducts, ccs, Ofile_name+i+".txt")
	    print("--> output table : generated")

	    smiles_u, adducts_u, ccs_u = self.read_reference_table(args.r)
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
                self.output_global_stats(ccs_ref, ccs_user)

	
	
	

    
    def evaluate(self, args):
    # Useful to evaluate the performances of the model, the theoritical ccs values must be known.
    
	print("Starting evaluation tool with the following args:" + str(args))
        if not path.isdir(args.m):
            raise IOError("Model directory cannot be found")
	if not path.isfile(args.r):
            raise IOError("Reference file cannot be found")
        if not path.isfile(path.join(args.m, "model.h5")):
            raise IOError("Model file is missing from directory")
        if not path.isfile(path.join(args.m, "adducts_encoder.json")):
            raise IOError("adduct_encoder.json is missing from the model directory")
        if not path.isfile(path.join(args.m, "smiles_encoder.json")):
            raise IOError("smiles_encoder.json is missing from the model directory")
         
        from DeepCCS.model import DeepCCS
        model = DeepCCS.DeepCCSModel()
        model.load_model_from_file(filename=path.join(args.m, "model.h5"),
                                   adduct_encoder_file=path.join(args.m, "adducts_encoder.json"),
                                   smiles_encoder_file=path.join(args.m, "smiles_encoder.json"))

        X_smiles, X_adducts, X_ccs = self.read_reference_table(args.r)
        ccs_pred = model.predict(X_smiles, X_adducts)

	ccs_pred = np.array([i[0] for i in ccs_pred])

	print(X_ccs[0:10])
	print("---")
	print(ccs_pred[0:10])
	

	
	if args.o != "":
            Ofile_name = args.o
	else:
	    Ofile_name = None

        self.output_results(args.r, X_smiles, X_adducts, ccs_pred, Ofile_name)

	print("-----------Model stats-----------")
	self.output_global_stats(X_ccs, ccs_pred)
	



    def predict(self, args):
    # Useful for predicting unknown ccs values

        print("Starting prediction tool with the following args:" + str(args))
        if not path.isdir(args.m):
            raise IOError("Model directory cannot be found")
        if not path.isfile(path.join(args.m, "model.h5")):
            raise IOError("Model file is missing from directory")
        if not path.isfile(path.join(args.m, "adducts_encoder.json")):
            raise IOError("adduct_encoder.json is missing from the model directory")
        if not path.isfile(path.join(args.m, "smiles_encoder.json")):
            raise IOError("smiles_encoder.json is missing from the model directory")

        from DeepCCS.model import DeepCCS
        model = DeepCCS.DeepCCSModel()
        model.load_model_from_file(filename=path.join(args.m, "model.h5"),
                                   adduct_encoder_file=path.join(args.m, "adducts_encoder.json"),
                                   smiles_encoder_file=path.join(args.m, "smiles_encoder.json"))
        
        X_smiles, X_adducts = self.read_input_table(args.i)
        ccs_pred = model.predict(X_smiles, X_adducts)

	ccs_pred = np.array([i[0] for i in ccs_pred])

	if args.o != "":
	    Ofile_name = args.o
	else:
	    Ofile_name = None
        
	self.output_results(args.i, X_smiles, X_adducts, ccs_pred, Ofile_name)

        


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


    def read_reference_table(self, file_name):
    # Useful to read a reference table containing the ccs values corresponding to SMILES and adducts 
   
        if file_name[-4:] == ".csv":
            table = pd.read_csv(file_name, sep=",", header=0)
        elif file_name[-5:] == ".xlsx" or file_name[-4:] == ".xls":
            table = pd.read_excel(file_name, header=0)
        print(list(table.columns.values))
        if not all(i in list(table.columns.values) for i in ["SMILES", "Adducts", "CCS"]):
            raise ValueError("Supplied file must contain at leat 3 columns named 'SMILES', 'Adducts' and 'CCS'. "
                             "use the provided template if needed.")
        table = filter_data(table)
	ccs = np.array(table['CCS'])
        smiles = np.array(table['SMILES'])
        adducts = np.array(table['Adducts'])
        return smiles, adducts, ccs

    def output_results(self, Ifile_name, smiles, adducts, ccs_pred, Ofile_name):
	if Ifile_name[-4:] == ".csv":
            table = pd.read_csv(Ifile_name, sep=",", header=0)
        elif Ifile_name[-5:] == ".xlsx" or Ifile_name[-4:] == ".xls":
            table = pd.read_excel(Ifile_name, header=0)
        print(list(table.columns.values))

	if not all(i in list(table.columns.values) for i in ["SMILES", "Adducts"]):
            raise ValueError("Supplied file must contain at leat 2 columns named 'SMILES' and 'Adducts'. "
                             "use the provided template if needed.")

	out_df = table.assign(CCS_pred=pd.Series(np.zeros(len(table)), index=table.index))	
	for idx, row in enumerate(out_df.itertuples()):
	    for i, j in enumerate(smiles):
		if row.SMILES == j and row.Adducts == adducts[i]:
		    out_df.iloc[idx, -1] = ccs_pred[i]
	        elif row.SMILES != j and row.Adducts != adducts[i] and out_df.iloc[idx, -1] == 0:
		    out_df.iloc[idx, -1] = "-" 
	
	pd.options.display.max_colwidth = 2000	
	out_df_string = out_df.to_string(header=True)
	
	if Ofile_name == None:
            sys.stdout.write(out_df_string)
        else:
            f = open(Ofile_name, 'w')
            f.write(out_df.to_string(header=True).encode('utf-8'))
            f.close
	
	

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s.%(msecs)d %(levelname)s %(module)s - %(process)d - %(funcName)s: %(message)s")
    CommandLineInterface()
