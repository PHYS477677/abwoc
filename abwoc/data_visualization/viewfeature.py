'''
viewfeature.py

Script to display a specific part of the Sun's surface in four
wavelengths: 94, 131, 171, and 211 Angstrom. Location of area of
interest (in helioprojective arcsec) must be known. viewfeature.py
(currently) must be in the same directory as findfile.py and reduce.py.
'''

import sunpy.map
import sunpy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sunpy.net import Fido, attrs as a
import astropy.units as u
from astropy.coordinates import SkyCoord
from matplotlib import ticker
from sunpy.visualization import axis_labels_from_ctype
import warnings
import findfile
import reduce

#Information about the date and time of the image
date = '2020_08_19'
time = '22_46'
downsample = 8 #factor by which to reduce the size of the image
imsize = int(4096/downsample)
title = 'B3.9 XRA event ' + date.replace('_','/') + ' ' + time.replace('_',':') \
        + '\nImage size ' + str(imsize) + 'x' + str(imsize)

#region of interest: a square of width xlength, height ylength, and
#   bottom left corner at x0,y0
xlength = 100 * u.arcsec
ylength = 100 * u.arcsec
x0 = 620 * u.arcsec
y0 = 140 * u.arcsec

#directory in which to find image .fits files
direc = 'AIAfiles'

#Find images and load them into Python
file1 = find_file.grab(direc,  '94', date, time, True)
file2 = find_file.grab(direc, '131', date, time, True)
file3 = find_file.grab(direc, '171', date, time, True)
file4 = find_file.grab(direc, '211', date, time, True)

map1 = sunpy.map.Map(file1)
map2 = sunpy.map.Map(file2)
map3 = sunpy.map.Map(file3)
map4 = sunpy.map.Map(file4)

smap = [map1,map2,map3,map4]

#Downsample image if specified
if downsample != 1:
    for i in range(4):
        smap[i] = reduce.reduce_map(smap[i], downsample)

#ignore warnings for convenience
warnings.filterwarnings("ignore")

#Create 2x2 subplot grid
fig, ax = plt.subplots(2,2, sharex=True, sharey=True, figsize=(7,7))
ax = ax.reshape(-1)

#Plot each wavelength
for i in range(4):
    bottom_left = SkyCoord(x0, y0, frame=smap[i].coordinate_frame)
    top_right = SkyCoord(x0 + length, y0 + length, frame = smap[i].coordinate_frame)

    submap = smap[i].submap(bottom_left, top_right=top_right)
    submap.plot(axes=ax[i], annotate=False)
    ax[i].set_title(toplot[i].wavelength)

    cb = plt.colorbar()
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()


#share axis labels over the whole figure
ylab = axis_labels_from_ctype(toplot[2].coordinate_system[1],
                                        toplot[2].spatial_units[1])
xlab = axis_labels_from_ctype(toplot[2].coordinate_system[0],
                                        toplot[2].spatial_units[0])
fig.text(0.5, 0.04, xlab, ha='center')
fig.text(0.01, 0.5, ylab, va='center', rotation='vertical')
plt.suptitle(title)

plt.show()
