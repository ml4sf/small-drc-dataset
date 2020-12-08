# small-drc-dataset

## Getting started
* Datset1545.csv file contains a list of descriptors and diradical character data of 1545 molecules.
* Dataset508.csv file is a subset of Dataset1545.csv that contains compounds with nonzero diradical character only.
* The `pm6_optimized_geom` directory contains xyz files of PM6 optimized geometries.

## Computational Protocol
All calculations were carried out using an automated procedure. 
The initial geometries of the compounds are obtained from their corresponding 
SMILES representations using OpenBabel.<br>
The geometry optimization of the compounds was performed with the PM6 method. 
The vibrational frequencies analysis reveal that all optimized structures are in minimum.<br>
Vertical electronic excitations were obtained by the INDO/S method in combination with the CAS(2,2)
approach. The diradical character was estimated within the Yamaguchi’s approach at the PUHF/6-31G\*\* 
level. <br>
In addition, descriptors values were calculated by using the AlvaDesc software.<br>
All semi-empirical calculations were performed with the MOPAC 2016 program package. 
The PUHF results were obtained with the Gaussian 09 program.
The GNU parallel tool was extensively used to parallelize the computational procedures described above.
