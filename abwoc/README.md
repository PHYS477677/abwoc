# Packages and Functions within `abwoc`

## Data Science Pipeline 

The primary purpose of this python package is to facilitate different steps of the Data Science Pipeline for our project. The different portions of our pipeline that this package is used with are described below:

1. Data Loading - Done primarily with our `data_loading` sub-module
    1. Download AIA images based on desired wavelengths and events
    2. Downsize and normalize the images from their native 4096x4096 size, and save these downsized images
    3. Optionally create simulated images to test network structure
2. Neural Network Training - Done primarily with our `xray_model` module
    1. Cache images to machine
    2. Create a model using Keras
    3. Train a preexisting model and Save/Load model weights from checkpoints
3. Model Validation - Done primarily with our `xray_model` module
    1.  Test the performance of a trained model on various data sets

## Primary Included Modules 

`xray_model.py` &mdash; train or evaluate a model using the given dataset, model structure, and checkpoints

Functions Included:
    1. modlearn : This function will read in an existing model checkpoint, or create a new one, and train the model on a specified dataset for a given number of epochs.
    2. modeval : This function will evaluate the loss and accuracy metrics for a given model, checkpoint, and dataset.

`visualization.py` &mdash; display solar images at the desired resolution, and show events at different scales

### Sub-Directories and Sub-Modules
#### data_loading

`fetch_aia.py` &mdash; contains functions for searching and downloading AIA images based on wavelengths and event type

`genimage.py` &mdash; generates a simulated image of the Sun, optionally with an artificial event 

`reduce.py` &mdash; contains functions which reduce the size of images, and store those as compressed files

#### data_files

`event_df_main.csv` &mdash; a csv table containing the start times and end times for all space weather events reported by the NOAO in their space weather reports from the year 2010 through the year 2020.


