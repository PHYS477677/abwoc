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

`xray_model.py` -- train or evaluate a model using the given dataset, model structure, and checkpoints

`visualization.py` -- display solar images and show events at different scales

### data_loading

`fetch_aia.py` -- contains functions for searching and downloading AIA images based on wavelengths and event type

`genimage.py` -- generates a simulated image of the Sun, optionally with an artificial event 

`reduce.py` -- contains functions which reduce the size of images, and store those as compressed files
