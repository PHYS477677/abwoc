# -*- coding: utf-8 -*-
"""data_reducer.py

This script is designed to be run within a Colab notebook.
This script reduces full size (4096x4096) .fits images, downscales them



Original file is located at
    https://colab.research.google.com/drive/1kBJ4YF6Nok36UcTkULBT2kBzDjiJkIUH
"""
# #Load the Drive helper and mount the google drive
# from google.colab import drive
# drive.mount('/content/drive')

# Install aiapy to colab disk
!pip install aiapy

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import os, os.path
import sys
import random
import glob
from astropy.io import fits
from aiapy.calibrate import register, update_pointing, normalize_exposure
import sunpy.map


# =============================================================================
#                                   Functions
# =============================================================================


def mean_pool(square_array, ratio):
    """
    Function to downsample a square array after applying a meanpool.

    Inputs
    ------
        square_array : Array to be downsampled. Must be a square array with
                        axis lengths that can be divisible by ratio
        ratio        : Downsampling ratio. i.e. a 1024x1024 array with a ratio
                       of 4 will be downsampled to 256x256

    Outputs
    -------
      Returns the downsampled array
    """
    # Dimensions of array
    alen_1 = np.size(square_array, 0)
    alen_2 = np.size(square_array, 1)
    # Confirming array is square
    if (alen_1 != alen_2):
        print("ERROR: ARRAY NOT SQUARE")
    else:
        return square_array.reshape(int(alen_1/ratio),
                                    int(ratio),
                                    int(alen_1/ratio),
                                    int(ratio)).mean(axis=(1, 3))


def convert_to_gz(fits_folder, gz_folder, down_sample_factor, date,
                  plot=False):
    """
    This script converts all fits files in a given folder to compressed
    .gz numpy arrays, after preforming a meanpool downsample.

    Inputs
    ------
        fits_folder       : string folder path where the fits files to be
                            downsampled are located
        gz_folder         : string folder path where the .gz files are to be
                            stored after downsampling
        down_sample_factor: int representing the dimension reduction size for
                            the downsample. (i.e. a down_sample_factor of 8
                            would turn 4096x4096 fits files to 512x512
                            compressed arrays.)
        date              : string showing the year and month during which the
                            .fits images were taken.
    """
    # Check how many files need to be run
    # Glob returns list of files with a certain file extension
    fits_list = glob.glob(fits_folder+'*.fits')
    num_fits = len(fits_list)

    # Glob returns list of files with a certain file extension
    gz_list = glob.glob(gz_folder+'*'+date+'*.gz')
    num_gz = len(gz_list)

    total_files = num_fits - num_gz #(total number of files remaining)
    print("%d fits files in "%num_fits+fits_folder)
    print("%d gz files in "%num_gz+gz_folder)
    if total_files < 1:
      print("All files converted, unless duplicate gz files exist.\n")
      return
    else:
      print("%d files remaining"%total_files)


    # Iterating over each file
    counter = 0
    for filename in fits_list:
      filename = os.path.basename(filename)
      if not os.path.exists(gz_folder+filename[:-5]+".gz"):

        img_map = sunpy.map.Map(fits_folder+filename)
        mean_data = mean_pool(img_map.data,down_sample_factor)

        if np.isnan(mean_data).any() or np.isinf(mean_data).any() or np.median(mean_data)<=1:
          print("bad image")
        else:
          np.savetxt(X=normalize_exp(mean_data, plot=plot),
                     fname=gz_folder+filename[:-5]+".gz")
          if counter%5 == 0:
            print("%.3f %% finished"%(100.0*counter/total_files))
          counter += 1

def normalize_exp(img_array, plot=False):
  """
  Normalizes image arrays to a range of 0 to 1, based on the normalization
  equation 1-exp(-image/factor), where the factor term is selected such that
  the image is not highly skewed towards values of ~0 or ~1.

  Inputs
  ------
    img_array : Numpy array which contains the pixel values for some image,
                which is to be normalized based on the 1-exp(-x) equations

  Outputs
  -------
    The normalized array ranging from 0 to 1
  """
  if plot:
    plt.imshow(img_array)
    plt.show()
  img_array[img_array<0] = 0

  norm_img = 1-np.exp(-img_array/100/np.median(img_array))
  #print(np.median(img_array))
  if plot:
    plt.imshow(norm_img,vmin=0,vmax=1)
    plt.colorbar()
  plt.show()
  return norm_img

##### Script
AIA_path = "/content/drive/Shareddrives/Phys 477 - Astro Project/AIA_files/"
event_folder = 'XRA_events/'
null_folder = 'XRA_nulls/'

wavelengths = ([131])
#date_paths = ["2015_01_","2015_02_","2015_03_","2015_04_","2015_05_","2015_06_",
#              "2015_07_","2015_08_","2015_09_","2015_10_","2015_11_","2015_12_"]
#date_paths = ["2015_07_","2015_08_","2015_09_","2015_10_","2015_11_"]
date_paths = ["2011_01_", "2014_01_", "2016_02_"]

for w in wavelengths:
  for date in date_paths:
    event_path = AIA_path + event_folder + date + str(w) +"/"
    null_path = AIA_path + null_folder + date + str(w) +"/"

    event_gz_path = AIA_path + event_folder + "gz_" + str(w) +"_512/"
    null_gz_path = AIA_path + null_folder + "gz_" + str(w) +"_512/"

    print("Converting path: " + event_folder[52:] + date + str(w) +"/ to gz_512")
    convert_to_gz(event_path, event_gz_path, 8, date[:-1], plot=False)
    print("Converting path: " + null_folder[52:] + date + str(w) +"/ to gz_512")
    convert_to_gz(null_path, null_gz_path, 8, date[:-1], plot=False)
print("\nAll Conversions Complete.")

