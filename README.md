# DeepCCS

<img src="https://drive.google.com/open?id=11IMVwofoQ_uGGL0b_y3qMiihVtIyoAc7", height="50"/>

CCS prediction from SMILES using deep neural network.

### For the impatients

After installation, go to the DeepCCS/interface/ directory.

    ./command_line_tool.py predict -i INPUT_F

- **INPUT_F** is the input file with at least a “SMILES” and a “Adducts” columns

The default modeland encoders files will be used.

### Information

DeepCCS is in alpha and under active development. You can still discover it yourself and available models should give
accurate predictions.

More information about installation and usage will come soon...

### References
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



