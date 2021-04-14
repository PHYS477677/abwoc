import glob
from astropy.time import Time
import astropy.units as u
from sunpy.net import Fido, attrs as a

def grab(dirPath, wavelength, date, time, download=False):
    """
    Simple method to return a specific AIA file that you've already downloaded, or download a file matching your search criteria if
    it doesn't exist in the specified directory.

    Inputs:
    dirpath: path to search for file (string, e.g. './AIAfiles')
    wavelength: wavelength in angstroms (string, e.g. '94')
    date: string, format yyyy_mm_dd
    time: string, format hh_mm[_ss] - second can be specified if known
    download: boolean, whether to download an image if it is not found

    Returns:
    A string containing the path to the file found
    """

    searchPath = dirPath + '/aia_lev1_' + wavelength + 'a_' + date + 't' + time + '*.fits'

    foundFiles = glob.glob(searchPath)
    

    if len(foundFiles)==0:
        if  not download: raise Exception('No files found that match input!')
        else:
            date = date.replace('_','-')
            time = time.replace('_',':')
            start = Time(date + 'T' + time)
            end = start + 59*u.second
            wv = int(wavelength)
            result = Fido.search(a.Time(start,end),
                                 a.Instrument.aia,
                                 a.Wavelength(wv*u.angstrom),
                                 a.Sample(1*u.minute))
            path = dirPath + '/{file}'
            downd = Fido.fetch(result, path=path)
            return downd
    
    elif len(foundFiles)>1:
        print('Multiple files match input! Returning the first file.')

    return foundFiles[0]

