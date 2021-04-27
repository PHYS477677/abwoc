# sample-data
Data which is used by the notebooks in the demos directory.

### sample-fits.zip
Zip file containing 9 .fits files. Used with visualization.ipynb. The 9 .fits files are images of the Sun on Aug 19, 2020 at 22:46 UTC across all wavelengths imaged by AIA, excluding 4500 Angstroms.

### script_testing_set_171_512_normalized.npy
A sample data set for running quick tests on the model training and evaluation functions. This file contains 12 total images (6 event and 6 null) in the 171 Angstrom wavelength, normalized and downsized to the format used for all of the abwoc CNNs. 
Used by model_train_eval.ipynb. 

### script_testing_set_171_512_normalized_labels.npy
The labels corresponding to the dataset described above. Images containing events are marked as '1' and images that do not contain events are marked as '0'.
Used by model_train_eval.ipynb.
