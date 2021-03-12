import glob

def grab(dirPath, wavelength, date, time):
    """
    Simple method to return a specific AIA file that you've already downloaded

    Inputs:
    dirpath: path to search for file (string, e.g. './AIAfiles')
    wavelength: wavelength in angstroms (string, e.g. '94')
    date: string, format yyyy_mm_dd
    time: string, format hh_mm[_ss] - second can be specified if known

    Returns:
    A string containing the path to the file found
    """

    searchPath = dirPath + '/aia_lev1_' + wavelength + 'a_' + date + 't' + time + '*.fits'

    foundFiles = glob.glob(searchPath)

    try:
        file = foundFiles[0]
    except IndexError:
        raise Exception('No files found that match input!')

    if len(foundFiles)>1:
        warnings.warnarn('Multiple files match input! Returning the first file.')

    return file

