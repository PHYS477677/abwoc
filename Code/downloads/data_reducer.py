import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import sunpy.map
from sunpy.net import Fido, attrs as a
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
from aiapy.calibrate import register, update_pointing, normalize_exposure
import os, os.path
import pandas as pd
import random

def mean_pool(square_array,ratio):
  """
  Function to downsample a square array after applying a meanpool

  Inputs
  ------
    square_array : Array to be downsampled. Must be a square array with axes
                   lenghts that can be divisible by ratio
    ratio        : Downsampling ratio. i.e. a 1024x1024 array with a ratio of 4
                   will be downsampled to 256x256
  
  Outputs
  -------
    Returns the downsampled array
  """
  # Dimensions of array
  alen_1 = np.size(square_array,0)
  alen_2 = np.size(square_array,1)
  # Confirming array is square
  if (alen_1!=alen_2):
    print("ERROR: ARRAY NOT SQUARE")
  else:
    return square_array.reshape(int(alen_1/ratio), int(ratio), 
                                int(alen_1/ratio), int(ratio)).mean(axis=(1,3))

def convert_to_gz(fits_folder, down_sample_factor):
  """
  This script converts all fits files in a given folder to compressed 
  .gz numpy arrays, after preforming a meanpool downsample.

  Inputs
  ------
    fits_folder       : string folder path where the fits files to be 
                        downsampled are located
    down_sample_factor: int representing the dimension reduction size for the 
                        downsample. (i.e. a down_sample_factor of 8 would turn 
                        4096x4096 fits files to 512x512 compressed arrays )
  """
  # Iterating over each file
  for filename in os.listdir(fits_folder):
    if filename.endswith(".fits"): 
         #opening file
         fits_file = fits.open( fits_folder+filename , dtype=np.int16 )
         #saving downsampled file
         np.savetxt(X=mean_pool(np.array(fits_file[1].data),down_sample_factor),
                    fname=fits_folder+filename[:-5]+".gz")


# Primary script
fits_location = str(sys.argv[1])
down_sample = int(sys.argv[2])

convert_to_gz(fits_location,down_sample)

         

