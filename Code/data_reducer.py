"""
data_reducer.py.

This script is designed to be run within a pre-mounted Colab notebook.
This script reduces full size (4096x4096) .fits images, downscales them,
and then uploads them back into the google drive as .gz files.

To run within a colab script, first "import data_reducer", then execute
data_reducer.reduce_data(wavelengths, date_paths, normalize,
reduction_factor)

Parameters
----------
wavelengths : 1-D list of integers
    List of wavelengths over which to reduce the data.
    For example: [131, 171, 211]
date_paths : 1-D list of Strings
    List of years and months over which to reduce the data.
    Each string is given in the format "[year]_[month]_"
    For example: ["2012_03_", "2014_12_"]
normalize : Boolean, optional
    Determines whether to apply a 0-1 normalization to the reduced images.
    The default is True, which normalizes the images.
reduction_factor : Integer, optional
    Factor by which to reduce the original 4096 x 4096 images.
    For example: reduction_factor = 8 produces 512 x 512 images.
    The default is 8.
event : String, optional
    The type of event according to the event designations from the space
    weather reports. The default is "XRA". Possible values include "FLA".

Returns
-------
Success : Boolean
    Returns whether the function succeeded (True) or failed (False).

Original file is located at
    https://colab.research.google.com/drive/1kBJ4YF6Nok36UcTkULBT2kBzDjiJkIUH
"""

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import os.path
import glob
import sunpy.map

# =============================================================================
#                               Helper Functions
# =============================================================================


def mean_pool(square_array, ratio):
    """
    Downsample a square array after applying a meanpool.

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
                  normalize, plot=False):
    """
    Convert all fits files in a given folder to compressed
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

    # (total number of files remaining)
    total_files = num_fits - num_gz
    print("%d fits files in " % num_fits + fits_folder)
    print("%d gz files in " % num_gz+gz_folder)
    if total_files < 1:
        print("All files converted, unless duplicate gz files exist.\n")
        return
    else:
        print("%d files remaining" % total_files)

    # Iterating over each file
    counter = 0
    for filename in fits_list:
        filename = os.path.basename(filename)

        if not os.path.exists(gz_folder+filename[:-5]+".gz"):
            img_map = sunpy.map.Map(fits_folder+filename)
            mean_data = mean_pool(img_map.data, down_sample_factor)

            if np.isnan(mean_data).any() or np.isinf(mean_data).any() or \
                    np.median(mean_data) <= 1:
                print("bad image")
            else:
                np.savetxt(X=normalize_exp(mean_data, normalize, plot=plot),
                           fname=gz_folder+filename[:-5]+".gz")
                if counter % 5 == 0:
                    print("%.3f %% finished" % (100.0*counter/total_files))
                counter += 1


def normalize_exp(img_array, normalize, plot=False):
    """
    Normalize image arrays between 0 and 1 and optionally plot the images.

    Normalize image arrays based on the normalization equation
    1-exp(-image/factor), where the factor term is selected such that
    the image is not highly skewed towards values of ~0 or ~1.

    Inputs
    ------
      img_array : Numpy array which contains the pixel values for some image,
                  which is to be normalized based on the 1-exp(-x) equations
      normalize : Boolean, optional
        Determines whether to apply a 0-1 normalization to the reduced images.
        If true, then the images will be normalized.
      normalize : Boolean, optional
        Determines whether to plot the saved images using imshow.
        The default is False, which does not plot the images.

    Outputs
    -------
      The potentially normalized output image.
    """
    # Optionally plot the original images
    if plot:
        plt.imshow(img_array)
        plt.show()

    # Zero negative values
    img_array[img_array < 0] = 0

    # Optionally normalize the images
    if normalize:
        norm_img = 1-np.exp(-img_array/100/np.median(img_array))
        # Optionally plot the normalized images
        if plot:
            plt.imshow(norm_img, vmin=0, vmax=1)
            plt.colorbar()
            plt.show()
        return norm_img
    else:
        return img_array


# =============================================================================
#                                  Main Function
# =============================================================================
def reduce_data(wavelengths, date_paths, normalize=True,
                reduction_factor=8, event="XRA"):
    """
    Reduces full size .fits images and uploads them as .gz to google drive.

    Images are downscaled by a reduction factor, and then are uploaded back
        into the google drive as .gz files.

    Parameters
    ----------
    wavelengths : 1-D list of integers
        List of wavelengths over which to reduce the data.
        For example: [131, 171, 211]
    date_paths : 1-D list of Strings
        List of years and months over which to reduce the data.
        Each string is given in the format "[year]_[month]_"
        For example: ["2012_03_", "2014_12_"]
    normalize : Boolean, optional
        Determines whether to apply a 0-1 normalization to the reduced images.
        The default is True, which normalizes the images.
    reduction_factor : Integer, optional
        Factor by which to reduce the original 4096 x 4096 images.
        For example: reduction_factor = 8 produces 512 x 512 images.
        The default is 8.
    event : String, optional
        The type of event according to the event designations from the space
        weather reports. The default is "XRA". Possible values include "FLA".

    Returns
    -------
    Success : Boolean
        Returns whether the function succeeded (True) or failed (False).
    """
    # Check for bad arguments
    # Event check
    possible_events = ["XRA", "FLA"]
    event_is_valid = False
    for possible_event in possible_events:
        if event == possible_event:
            event_is_valid = True
            break
    if not event_is_valid:
        print("Event Type '" + event + "' is invalid")
        return False

    # Check for bad reduction factor
    if 4096 % reduction_factor:
        print("Reduction Factor '" + str() + "' is invalid for a 4096 x 4096"
              + " image")
        return False

    # Path variables
    AIA_path = "/content/drive/Shareddrives/Phys 477 - Astro Project/" \
               + "AIA_files/"
    event_folder = event + '_events/'
    null_folder = event + '_nulls/'
    normalize_path = ""
    if normalize:
        normalize_path += "_normalized"

    # Iterate over all combinations of dates and wavelengths
    for w in wavelengths:
        for date in date_paths:
            event_path = AIA_path + event_folder + date + str(w) + "/"
            null_path = AIA_path + null_folder + date + str(w) + "/"

            event_gz_path = AIA_path + event_folder + "gz_" + str(w) + "_" \
                + str(int(4096/reduction_factor)) + normalize_path + "/"

            null_gz_path = AIA_path + null_folder + "gz_" + str(w) + "_" \
                + str(int(4096/reduction_factor)) + normalize_path + "/"

            print("Converting event path: " + event_folder[52:] + date 
                  + str(w) + "/ to gz_" + str(w) + "_512" + normalize_path)
            convert_to_gz(event_path,
                          event_gz_path,
                          reduction_factor,
                          date[:-1],
                          normalize,
                          plot=False)

            print("Converting null path: " + null_folder[52:] + date
                  + str(w) + "/ to gz_" + str(w) + "_512" + normalize_path)
            convert_to_gz(null_path,
                          null_gz_path,
                          reduction_factor,
                          date[:-1],
                          normalize,
                          plot=False)

    print("\nAll Conversions Complete.")
    return True
