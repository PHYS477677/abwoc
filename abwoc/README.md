# Notebooks and scripts for Astros but Without Cheating

## Data Science Pipeline

1. Data Loading
    1. Download AIA images based on desired wavelengths and events
    2. Downsize and normalize the images from their native 4096x4096 size, and save these downsized images
    3. Optionally create simulated images to test network structure
2. Neural Network Training
    1. Cache images to machine
    2. Create a model using Keras
    3. Train a preexisting model and Save/Load model weights from checkpoints
3. Model Validation
    1.  Test the performance of a trained model on various data sets

## Files and Scripts

`xray_model.py` &mdash; train or evaluate a model using the given dataset, model structure, and checkpoints

Functions Included:
    1. modlearn : This function will read in an existing model checkpoint, or create a new one, and train the model on a specified dataset for a given number of epochs.
    2. modeval : This function will evaluate the loss and accuracy metrics for a given model, checkpoint, and dataset.

`visualization.py` &mdash; display solar images at the desired resolution, and show events at different scales

### Sub-Directories
#### data_loading

`fetch_aia.py` &mdash; contains functions for searching and downloading AIA images based on wavelengths and event type

`genimage.py` &mdash; generates a simulated image of the Sun, optionally with an artificial event 

`reduce.py` &mdash; contains functions which reduce the size of images, and store those as compressed files

#### data_files

`event_df_main.csv` &mdash; is a csv table containing the start times and end times for all space weather events reported by the NOAO in their space weather reports from the year 2010 through the year 2020.

# Packages and Functions within `abwoc`


