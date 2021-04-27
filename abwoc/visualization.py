"""
visualization.py.

Collection of functions to display images of the Sun.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from sunpy.visualization import axis_labels_from_ctype
import astropy.units as u
from astropy.coordinates import SkyCoord
import warnings


def add_plots(smaps, fig, ax):
    """
    Add plots to a figure.

    Inputs:
        smaps: list of SunPy Map objects
        fig: matplotlib Figure object
        ax: flattened list of matplotlib Axis objects
    """
    # ignore warnings for convenience
    warnings.filterwarnings("ignore")

    for i in range(len(smaps)):
        smaps[i].plot(axes=ax[i], annotate=False)
        ax[i].set_title(smaps[i].wavelength)

        cb = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()
    return


def plots(smaps):
    """
    Plot an image of the Sun in multiple wavelengths, up to nine per figure.

    Inputs:
        smaps: list of SunPy Map objects (up to nine)

    Outputs:
        None.
    """
    nmap = len(smaps)

    if (nmap < 1 or nmap > 9):
        raise Exception('Please input a valid number of smaps.')

    nc = int(np.ceil(np.sqrt(nmap)))
    nr = nc - 1*(nmap <= (nc**2 - nc))

    fig, ax = plt.subplots(nr, nc, sharex=True, sharey=True,
                           figsize=(2*nc+2, 2*nr+1))
    if nmap == 1:
        ax = np.array([ax])
        fig.set_size_inches(5, 4.5)    # prevent text overlap

    ax = ax.reshape(-1)

    # hide extra axes
    for i in range(nmap, nr*nc):
        ax[i].axis('off')

    add_plots(smaps, fig, ax)

    # share axis labels over the whole figure
    ylab = axis_labels_from_ctype(smaps[0].coordinate_system[1],
                                  smaps[0].spatial_units[1])
    xlab = axis_labels_from_ctype(smaps[0].coordinate_system[0],
                                  smaps[0].spatial_units[0])
    fig.text(0.5, 0.04, xlab, ha='center')
    fig.text(0.01, 0.5, ylab, va='center', rotation='vertical')

    title = 'AIA ' + smaps[0].meta.get('date').replace('T', ' ')[:-3]
    plt.suptitle(title)

    plt.show()
    return


def plot_zoom(smaps, x0, y0, xlength, ylength):
    """
    Plot an image of the Sun zoomed in on a region of interest.

    Location of area (in helioprojective arcsec) must be known.

    Inputs:
        smaps: list of SunPy Map objects (up to nine)
        x0: bottom left corner of zoom area (helioprojective longitude
            - arcsec)
        y0: bottom left corner of zoom area (helioprojective lattitude
            - arcsec)
        xlength, ylength: width and height of zoom area (arcsec)

    Outputs:
        None.
    """
    x0 *= u.arcsec
    y0 *= u.arcsec
    xlength *= u.arcsec
    ylength *= u.arcsec

    nmap = len(smaps)
    submaps = [0]*nmap
    for i in range(nmap):
        bottom_left = SkyCoord(x0, y0, frame=smaps[i].coordinate_frame)
        top_right = SkyCoord(x0 + xlength, y0 + ylength,
                             frame=smaps[i].coordinate_frame)
        submaps[i] = smaps[i].submap(bottom_left, top_right=top_right)
    plots(submaps)
    return
