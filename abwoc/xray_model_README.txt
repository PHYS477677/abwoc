This folder contains files primarily for testing the performance of a model

The evaluate_model.py script will evaluate the loss and accuracy metrics for a given model, checkpoint, and dataset. This script requires 6 arguments that must be passed at runtime. They are:

  1. Dataset - path to the desired dataset, saved as a .npy file along with a matching .npy file containing label information
  2. Wavelength - the wavelength to use with the model, 'all' for the three wavelength CNN or '131' etc. for the single wavelength CNN
  3. Model Structure - the CNN structure, either '1layer', '2layer', or 'ctmodel' for our 1, 2, and 3 layer models respectively 
  4. Checkpoint Path - path to the directory of checkpoints, the checkpoint file names should end with the corresponding epoch, which will be standard if the model is trained using the `model_train.py` script  
  5. Start Epoch - integer specifying which saved epoch to load a checkpoint's weigths from. Enter -1 to evaluate the performance of the most recently saved model
  6. End Epoch - if it is desired to evaluate the performance of the model over a range of saved checkpoints, enter the final epoch here, if instead it is deisred to evaluate the performance of a single wavelength, leave this blank 
  
evaluate_model.py then loads the specificed dataset, creates a model with the desired structure, loads the saved model weights from a specified directory and outputs the loss and accuracy metrics in text form for a single epoch, or as a plot for a range of epochs.
  
An example of running this command is shown below: 
$ evaluate_model.py './validation_sets/true_validation_171_512_normalized.npy' '171' '1layer' './final_checkpoints/onewave_131_512_1800_1layer/' -1

This call will use the './validation_sets/true_validation_171_512_normalized.npy' dataset to evaluate the performance of our single layer, single wavelength CNN, using the weights from the most recent checkpoint in the './final_checkpoints/onewave_131_512_1800_1layer/' directory.
