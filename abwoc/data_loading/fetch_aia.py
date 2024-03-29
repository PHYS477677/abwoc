"""
Fetches AIA images using Fido to be downloaded locally or uploaded to GDrive.

IS NOT STANDALONE, REQUIRES MODIFICATION TO BE USED (with GDrive IDs)
"""

import astropy.units as u
from astropy.time import Time
from sunpy.net import Fido, attrs as a
import numpy as np
import os
import pandas as pd
import random
import glob
import pydrive  # Google Drive API handler, must be installed "pip install PyDrive"
# Installing may require root access.
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


def get_images(times, local_path, up_path_id, sub_folder, lamb, year, month,
               dayi, drive, testing):
    """
    Download AIA images using the Fido search.

    Inputs
    ------
        times     : 2-D Array where each sub-array lists one hour and minute
        local_path: Directory where images are temporarily locally stored
        up_path_id: Google Drive directory ID (string) where files will be
                      uploaded
        sub_fold  : String of form 'yyyy_mm' - title of the subfolder
        lamb      : List of wavelengths to download images from
        year      : Year that images come from
        month     : Month that images come from
        day_s     : Day that events started during
        drive     : Google Drive object, to be used in upload()
        testing   : A boolean indicating whether this function is just being
                      tested. If true, this function will only print the found
                      images, it won't download them.

    Outputs
    -------
        No outputs are returned
    """
    for t in times:

        # Check to see if event time crosses over to the next day
        if t[0] > 23:
            t[0] = t[0] % 24
            day = dayi + 1
        else:
            day = dayi

    # Start times and end times to use for fido search
    start_time_fido = (str(year) + '-' + str(month).zfill(2) + '-'
                       + str(day).zfill(2) + 'T'
                       + str(t[0]).zfill(2) + ':' + str(t[1]).zfill(2) + ":00")

    end_time_fido = (str(year) + '-' + str(month).zfill(2) + '-'
                     + str(day).zfill(2) + 'T' + str(t[0]).zfill(2) + ':'
                     + str(t[1]).zfill(2) + ":59")

    # Used to select random image out of those taken during the minute
    # specified by 't'. Using the same seed for all wavelengths ensures that
    # each of the different wavelength images selected are as close together
    # (in time) as possible
    random_seed = random.random()

    for l in lamb:
        # All the observations during minute 't' in wavelength 'l'
        results = Fido.search(a.Time(start_time_fido, end_time_fido),
                              a.Instrument('AIA'),
                              a.Wavelength(wavemin=l*u.angstrom,
                                           wavemax=l*u.angstrom))

        # Check needed in case telescope was not running at time/has
        # no observations
        if np.size(results[0, :]) > 0:
            # Picks one of the images and downloades it
            f_select = int(np.size(results, 1)*random_seed)

            if testing:
                print(results[0, f_select])
            else:
                localpaths = Fido.fetch(results[0, f_select], path=local_path)
                upload(localpaths, up_path_id, l, sub_folder, drive)
                print("\nSet of localpaths Uploaded and Removed from local\n")


def upload(localpaths, up_path_id, wavelength, sub_fold, drive):
    """
    Upload local files to Google Drive. Deletes them from local storage.

    Inputs
    ------
        localpaths: List of paths for each downloaded image
        up_path_id: Google Drive directory ID (string) where files will be
                    uploaded e.g. XRA_events GDrive ID
        wavelength: Wavelength of images
        sub_fold  : String of form 'yyyy_mm' - title of the subfolder
        drive     : Google Drive object

    Outputs
    -------
        None.
    """
    # Name of subfolder (e.g. 2015_03_131)
    foldername = sub_fold+"_"+str(wavelength)
    team_drive_id = "8675309"

    # == Check if subfolder already exists ==
    folderlist = drive.ListFile({
        "q": "'"+up_path_id+"' in parents and mimeType='application/"
             + "vnd.google-apps.folder' and trashed=false",
        'supportsAllDrives': True,
        'driveId': team_drive_id,
        'includeItemsFromAllDrives': True,
        'corpora': 'drive'
    }).GetList()   # Get list of subfolders in event folder

    subfolder_exists = False
    if len(folderlist) > 0:  # Check if subfolder exists
        for folder in folderlist:
            if folder['title'] == foldername:
                # Store subfolder ID if subfolder exists
                subfolder_id = folder['id']
                subfolder_exists = True
                break

    if not subfolder_exists:  # If no subfolders exist for this event type
        # Create new subfolder (copy of above code, can be optimized)
        subfolder_file = drive.CreateFile({
            'title': foldername,
            'parents':  [{
                'kind': 'drive#fileLink',
                'teamDriveId': team_drive_id,
                'id': up_path_id
            }],
            'mimeType': "application/vnd.google-apps.folder"
        })
    subfolder_file.Upload(param={'supportsTeamDrives': True})  # Upload folder
    subfolder_id = subfolder_file['id']  # Get subfolder id

    for localpath in localpaths:  # For each file downloaded
        # Get filename
        pathhead, pathtail = os.path.split(localpath)
        # Upload the file to the subfolder
        fitsfile = drive.CreateFile({
            'title': pathtail,
            'parents':  [{
                'kind': 'drive#fileLink',
                'teamDriveId': team_drive_id,
                'id': subfolder_id
            }]
        })
        fitsfile.SetContentFile(localpath)
        fitsfile.Upload(param={'supportsTeamDrives': True})  # Upload
        os.remove(localpath)  # Delete from local storage


def get_null_times(df, n, year, month, day):
    """
    Return n random times, where events are not present, for a specified date.

    Inputs
    ------
        dataframe : pandas dataframe object listing all the relvent events to
                      consider
        n         : integer specifying the number of null images to return
        year      : integer specifying the year
        month     : integer specifying the month
        day       : integer specifying the day

    Outputs
    -------
        times     : an array which contains n specified hours/minutes where
                      there are no events occuring
    """
    # number of images selected so far
    count = 0
    times = np.array([])

    # iterate until found
    while count < n:

        # picking random times
        rand_hour = random.randrange(24)
        rand_min = random.randrange(60)

        # converting times to format that can be compared to event report
        rand_time = rand_hour*100+rand_min

        # check day of results
        num_overlap_1 = len((df[(df['start'] <= rand_time) &
                             (df['end'] >= rand_time) &
                             (df['year'] == year) &
                             (df['month'] == month) &
                             (df['day'] == day)]))

        # check previous day results
        num_overlap_2 = len((df[(df['start'] >= df['end']) &
                             (df['end'] >= rand_time) &
                             (df['year'] == year) &
                             (df['month'] == month) &
                             (df['day'] == day-1)]))

        # adding times
        if (num_overlap_1 == 0 and num_overlap_2 == 0):
            if count == 0:
                times = np.array([[int(rand_hour), int(rand_min)]])
            else:
                times = np.vstack((times, [int(rand_hour), int(rand_min)]))
            count += 1

    return times


def get_event_times(event_df, year, month, day):
    """
    Select one random time from each of the events in the input dataframe.

    Inputs
    ------
        event_df  : pandas dataframe object listing all the relevent events to
                      consider
        year      : integer specifying the year
        month     : integer specifying the month
        day       : integer specifying the day

    Outputs
    -------
        times     : an array which contains one specified hour/minute which
                     occurs during the duration of each event in event_df
    """
    times = np.array([])
    count = 0
    for index, event in event_df.iterrows():

        # duration of event in minutes
        t_diff = ((int(str(event.end).zfill(4)[0:2])
                   - int(str(event.start).zfill(4)[0:2])) % 24*60
                  + (int(str(event.end).zfill(4)[2:4])
                  - int(str(event.start).zfill(4)[2:4])))

        # selecting random time from event duration
        t_diff_selected = random.randrange(t_diff+1)
        t_selected = int(str(event.start).zfill(4)[0:2])*60 \
            + int(str(event.start).zfill(4)[2:4]) \
            + t_diff_selected

        # converting selected time to hours and minutes
        ts_hours = t_selected // 60
        ts_minutes = t_selected % 60

        # initialize or extend final array
        if count == 0:
            times = np.array([[ts_hours, ts_minutes]])
        else:
            times = np.vstack((times, [ts_hours, ts_minutes]))
        count += 1
    return times


def get_event_df(parent_df, event_type, year, month, day):
    """
    Produce event dataframe with events and corresponding start/end times.

    Returns a list of times corresponding to events that either started or
    ended during a given day (Note this can include events that started on
    day-1, but ended on day)

    Inputs
    ------
        parent_df : pandas dataframe containing all possible events to choose
                      from
        event_type: string identifying the type of event to choose from in
                      parent_df
        year      : int representing the year to choose events from
        month     : int representing the month to choose events from
        day       : int representing the day to choose events from

    Outputs
    -------
        A single pandas dataframe with all events from parent_df that fit the
        specifications given by the remaining inputs
    """
    # dataframe with events that started on day and ended on day
    df_new_1 = parent_df[(parent_df.event == event_type) &
                         (parent_df.year == year) &
                         (parent_df.month == month) &
                         (parent_df.day == day)]
    # dataframe with events that started on day-1 and ended on day
    df_new_2 = parent_df[(parent_df.event == event_type) &
                         (parent_df.year == year) &
                         (parent_df.month == month) &
                         (parent_df.day == day-1) &
                         (parent_df.start > parent_df.end)]

    # combined event dataframe
    return (pd.concat([df_new_1, df_new_2]))


def create_storage_dirs(event_path, null_path, sub_fold, lambdas):
    """
    Create storage directories for AIA files.

    Inputs
    ------
        event_path : path to where all event images for a given event type are
                       stored
        null_path  : path to where all null images for a given event type are
                       stored
        sub_fold   : folder name prefix for a specific time/date
        lambdas    : wavelengths for which images are being downloaded and
                       stored

    Outputs
    ------
        No outputs are returned
    """
    for wavelength in lambdas:
        # Full folder path name
        storage_path = event_path + sub_fold + "_" + str(wavelength)
        # Make folder if it does not exist
        if not os.path.isdir(storage_path):
            os.mkdir(storage_path)
    return


def grab(dirPath, wavelength, date, time, download=False):
    """
    Return a previously downloaded AIA file according to search criteria.

    Simple method to return a specific AIA file that you've already downloaded,
    or download a file matching your search criteria if it doesn't exist in the
    specified directory.

    Inputs:
    -----------
        dirpath: path to search for file (string, e.g. './AIAfiles')
        wavelength: wavelength in angstroms (string, e.g. '94')
        date: string, format yyyy_mm_dd
        time: string, format hh_mm[_ss] - second can be specified if known
        download: boolean, whether to download an image if it is not found

    Outputs:
    -----------
        A string containing the path to the file found
    """
    searchPath = dirPath + '/aia_lev1_' + wavelength + 'a_' + date + 't' \
        + time + '*.fits'

    foundFiles = glob.glob(searchPath)

    if len(foundFiles) == 0:
        if not download:
            raise Exception('No files found that match input!')
        else:
            date = date.replace('_', '-')
            time = time.replace('_', ':')
            start = Time(date + 'T' + time)
            end = start + 59*u.second
            wv = int(wavelength)
            result = Fido.search(a.Time(start, end),
                                 a.Instrument.aia,
                                 a.Wavelength(wv*u.angstrom),
                                 a.Sample(1*u.minute))
            path = dirPath + '/{file}'
            downd = Fido.fetch(result, path=path)
            return downd

    elif len(foundFiles) > 1:
        print('Multiple files match input! Returning the first file.')
    return foundFiles[0]
