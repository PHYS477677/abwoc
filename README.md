# Astros but Without the Cheating (abwoc)
## Project Overview
Despite a century and a half of research, solar flares are generally poorly understood. Science has an idea of how they form, however we are unable to precisely model these flares and prominences. Solar flares are generally classified by their peak wavelength (so a flare brightest in the x-ray we call an x-ray event), but they emit in a range of wavelengths, including the extreme ultraviolet. If different types of flares carry unique observational signatures in simple images of the Sun, this could hint at specific physical mechanisms underlying flares of varying types.

In this semester long project, we have used daily reports of solar events from the Space Weather Prediction Center (SWPC), and images of the Sun in EUV wavelengths to train a convolutional neural network to classify images on whether or not they contain an x-ray flare. 

## Solar Event Classifier 
```abwoc``` is used for downloading images of solar events in the desired wavelengths from AIA data. These images can then be used to train a CNN to identify to identify the desired solar event. Modules for testing the trained models are also included. 

## Installation
To install, use

```
pip install git+https://github.com/PHYS477677/abwoc
```

## Examples
Examples of usage of ```abwoc``` are provided in the 'demos' directory. Notebooks included are:
- visualization.ipynb &mdash; How to view the AIA images of the Sun
- model_train_eval.ipynb &mdash; How to train and evaluate CNN models on this dataset

## License
abwoc is released under the Apache license.

## Data used
The Joint Science Operations Center (JSOC) release of AIA data can be accessed at [jsoc.stanford.edu](http://jsoc.stanford.edu/). NOAA Space Weather Prediction Center (SWPC) Event Reports can be accessed at [swpc.noaa.gov](https://www.swpc.noaa.gov/products/solar-and-geophysical-event-reports).
