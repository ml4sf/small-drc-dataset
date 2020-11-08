# small-drc-dataset
The Dataset contains the diradical character of 1566 optimized with PM6 compounds 
#Computational Protocol
All calculations were carried out using an automated procedure. The initial geometries of the compounds are obtained from their corresponding SMILES representations using OpenBabel8.
The geometry optimization of the compounds was performed with the PM69 method. 
The vibrational frequencies analysis reveal that all optimized structures are in minimum. 
Vertical electronic excitations were obtained by the INDO/S method in combination with the CAS(2,2)
approach.10 The diradical character was estimated within the Yamaguchi’s approach at the PUHF/6-31G\*\* 
level. In addition, descriptors values were calculated by using the AlvaDesc software.
All semi-empirical calculations were performed with the MOPAC 2016 program package. 
The PUHF results were obtained with the Gaussian 09 program.
The GNU parallel tool14 was extensively used to parallelize the computational procedures described above.
