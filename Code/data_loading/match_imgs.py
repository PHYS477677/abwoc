# A script to read in a list of wavelengths, a integer number of images, an integer
# image dimension, an image source directory location, and an output directory, to save
# a compressed numpy array containing the image data to be passed into the CNN.

# Example Call: $match_imgs.py "../AIA_files/XRA_events/" [131,171,211] 10 "./test.npy"
#     This would create a 512x512x3x10 array from 10 512x512 observations in 131 171 and 211 angstroms
#     and save it as a .npy file

# Imports
import numpy as np
import os, random
import glob
import sys

# Arguments Given to Script
img_dir = sys.argv[1]
img_dim = int(sys.argv[2]) # Img dimension (img_dim = 512 for a 512x512 resolution image)
lambdas = np.array(sys.argv[3].strip('[]').split(',')).astype(np.int16) # Wavelengths to use
num_imgs = int(sys.argv[4]) # number of pictures to use
out_file = sys.argv[5] # Out directory

print(img_dir)
print(img_dim)
print(lambdas)
print(num_imgs)
print(out_file)

# Initial vars
num_lamb = np.size(lambdas)
count = 0

start_files = np.array(glob.glob(img_dir+"gz_"+str(lambdas[0])+"_"+str(img_dim)+"/*.gz"))
print(img_dir+"gz_"+str(lambdas[0])+"_"+str(img_dim)+"/*.gz")

# Iterate over all available images until composite array is built up
for i in np.arange(1,np.size(start_files)):
        
    # Initial exposure
    pot_file = start_files[i]
    print(pot_file)
    
    # Check that exposures exist
    exp_fail = False
    
    # Part of file name used to locate exposures taken at same time
    fstart = len(img_dir)+25
    file_id = pot_file[fstart:fstart+16]
    print(file_id)
    # Initializing array of different wavelength exposures
    img_array = np.empty([img_dim,img_dim,num_lamb])
    
    # Adding first wavelength data
    img_array[:,:,0] = np.loadtxt(pot_file,dtype=np.float32)

    # Adding additional wavelength data
    for lamb_i in np.arange(1,num_lamb):
        match_files = np.array(glob.glob(img_dir+"gz_"+str(lambdas[lamb_i])+"_"+str(img_dim)+"/*"+file_id+"*"))
        if np.size(match_files)>0:
            match_file = match_files[0]
            print(match_file)
            img_array[:,:,lamb_i] = np.loadtxt(match_file,dtype=np.float32)
        else:
            exp_fail = True
        
    # Check that data is good
    if np.isnan(img_array).any() or np.isinf(img_array).any() or exp_fail:
            print("problem img")
    else:
        if count == 0:
            comp_array = np.array([img_array,])
        else:
            comp_array = np.concatenate([comp_array,np.array([img_array,])],axis=0)
        count+=1
        
    # Check if we have found all the images we want
    if count >= num_imgs:
        break
    print(count)

np.save(out_file,comp_array)







