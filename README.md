<img src="http://gdurl.com/PrrA" width=150 height=150 />

CCS prediction from SMILES using deep neural network.

## For the impatients

After installation, go to the DeepCCS/interface/ directory.

    python command_line_tool.py predict -i INPUT

- **INPUT** is the input file with at least a “SMILES” and an “Adducts” column

The default model and encoders files will be used. See the [predict](https://github.com/plpla/DeepCCS#predict) section bellow for more options.

## Installation

DeepCCS was tested and works under Python 3.6. We recommend to use [conda](https://conda.io/docs/user-guide/install/download.html).

Package required:
 * Numpy
 * Pandas
 * Scikit-learn
 * Tensorflow
 * Keras

To install, go to the `core` directory and simply perform the following command using a terminal:

    python setup.py install

## Functionalities

### Predict
Predict CCS using a SMILES and an adduct.

    DeepCCS predict -mp MODEL_DIR -ap ADDUCTS_ENCODER_DIR -sp SMILES_ENCODER_DIR -i INPUT_F -o OUTPUT_F

*Required args :*
- i : The input file, with at least a “SMILES” and a “Adducts” columns

*Optionnal args :*

- mp : the directory containing the model.h5 file (default="../saved_models/default/")
- ap : the directory containing the adducts_encoder.json file (default="../saved_models/default/")
- sp : the directory containing the smiles_encoder.json file (default="../saved_models/default/")
- o : is the desired name for the output file(ex: MyFile.csv), if none stdout will be use.


### Compare
Compare provided CCS values to the ones contained in every dataset used to train and test DeepCC (no predictions involved).

    DeepCCS compare -f H5_F -i REFERENCE_F -d S_DATASET1,S_DATASET2,... -o OUTPUT_P

*Required args :*
- i : The reference file, with at least “SMILES”, “Adducts” and “CCS” columns
- f : The hdf5 file containing all the source datasets

*Optionnal args :*
- d : The names of the source datasets (as a list without “ ”) to use for comparison, if none they are all considered.
    - Choices are : MetCCS_pos, MetCCS_neg, Agilent_pos, Agilent_neg,   Waters_pos, Waters_neg, PNL, McLean, CBM
- o : is the desired prefix for the output files (ex: compare_to_MetCCS_), because there is one output file per compared source dataset

### Evaluate
Perform CCS predictions and evaluate the model using measured values

    DeepCCS evaluate -mp MODEL_DIR -ap ADDUCTS_ENCODER_DIR -sp SMILES_ENCODER_DIR -i REFERENCE_F -o OUTPUT_F

*Required args :*
- i : Input reference file. Must contain at least “SMILES”, “Adducts” and “CCS” columns

*Optionnal args :*
- mp : Directory containing the model.h5 file (default="../saved_models/default/")
- ap : Directory containing the adducts_encoder.json file (default="../saved_models/default/")
- sp : Directory containing the smiles_encoder.json file (default="../saved_models/default/")
- o : Desired name for the output file (ex: MyFile.csv), if not specified stdout will be use.


### Train
Train a new model including your own measurements with or without the available datasets.

    DeepCCS train -f H5_F -ap ADDUCTS_ENCODER_DIR -sp SMILES_ENCODER_DIR -mtrain -pnnl -cbm -mclean -o OUTPUT_DIR -nd NEW_D1 -nepochs 100

*Required args :*
- f : The hdf5 file containing all the source datasets

*Optionnal args :*
- ap : the directory containing the adducts_encoder.json file. "d" will make the model train with the default
encoder. If  argument is not used a new encoder will be created a new encoder (default = None)
- sp : the directory containing the smiles_encoder.json file. "d" will make the model train with the default
encoder. If argument is not used, a new encoder will be created (default = None)
- mtrain : use the MetCCS_pos and MetCCS_neg datasets as training data (default = false)
- mtestA : use the Agilent_pos and Agilent_neg test datasets from MetCCS as training data (default = false)
- mtestW : use the Waters_pos and Waters_neg test datasets from MetCCS as training data (default = false)
- pnnl : use the PNNL dataset as training data (default = false)
- cbm : use the CBM2018 dataset as training data (default = false)
- mclean : use the McLean Lab dataset as training data (default = false)
- nd : New datasets to create the model. If multiple files, as a list seperated by "," (default = None)
- test: Proportion of each dataset that must be kept in the testing set (default: 0.2)
- o : Existing directory to ouput model and mappers (default = current directory)
- nepochs : Numbe of epoch to use for the model’s training (default = 150)

At least one dataset is required to train a new model. Datasets selected will be splited between the training
and testing set according to the `test` argument value except for `mtrain` which is always
completly in the training set.

### Additional notes
 * The `Adducts` column of the input file must contain adducts as: `M+H`, `M+Na`, `M-H` and `M-2H`.
 * The package includes a `DeepCCSModel` class that can be used directly in python without the command line tool.

## References
DeepCCS relies heavily on datasets that were previously published by others:

* Zhou Z, Shen X, Tu J, Zhu ZJ. Large-Scale Prediction of Collision Cross-Section Values for Metabolites in Ion
Mobility-Mass Spectrometry. Anal Chem. 2016 Nov 15;88(22):11084-11091. Epub 2016 Nov 1. PubMed PMID: 27768289.
* Zheng X, Aly NA, Zhou Y, Dupuis KT, Bilbao A, Paurus VL, Orton DJ, Wilson R, Payne SH, Smith RD, Baker ES. A structural
examination and collision cross section database for over 500 metabolites and xenobiotics using drift tube ion
mobility spectrometry. Chem Sci. 2017 Nov 1;8(11):7724-7736. doi: 10.1039/c7sc03464d. Epub 2017 Sep 28.
PubMed PMID: 29568436; PubMed Central PMCID: PMC5853271.
* May JC, Goodwin CR, Lareau NM, Leaptrot KL, Morris CB, Kurulugama RT, Mordehai A, Klein C, Barry W, Darland E, Overney G,
Imatani K, Stafford GC, Fjeldsted JC, McLean JA. Conformational ordering of biomolecules in the gas phase: nitrogen
collision cross sections measured on a prototype high resolution drift tube ion mobility-mass spectrometer.
Anal Chem. 2014 Feb 18;86(4):2107-16. doi: 10.1021/ac4038448. Epub 2014 Feb 4. PubMed PMID: 24446877; PubMed Central PMCID:
PMC3931330.
* Mollerup CB, Mardal M, Dalsgaard PW, Linnet K, Barron LP. Prediction of
collision cross section and retention time for broad scope screening in gradient
reversed-phase liquid chromatography-ion mobility-high resolution accurate mass
spectrometry. J Chromatogr A. 2018 Mar 23;1542:82-88. doi:
10.1016/j.chroma.2018.02.025. Epub 2018 Feb 15. PubMed PMID: 29472071.



