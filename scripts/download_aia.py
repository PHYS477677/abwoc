'''
download_aia.py

TODO: Let script specify event csv file

In order to run download_aia.py there are a total of 7 arguments
that must be passed in at runtime. They are:

  1. Start year 
  2. End year
  3. Start month
  4. End month
  5. Start day
  6. End day
  7. Event type
  
linux_download_aia.py then searches the file 'event_df_main.csv' and
indentifies each space weather event  in event_df_main.csv that matches
the requirements specified by the 7 input arguments given at runtime.
It then downloads one AIA expsoure, as a fits file, for each located event,
which was taken during when the event was present. It also downloads an
additional image, one per located event, at a time where there was no
qualifying space weathe event present.  These images are then uploaded to
Google Drive and deleted from the local machine. 
  
As an example, in order to download one AIA image for each X-RAY event that
occured during the months of March and April in 2015, the python command
would look like:

$ linux_download_aia.py 2015 2016 3 5 1 32 XRA

When the script is first run, an internet browser window will open
requesting access from the application to Google Drive. Accept the access,
and the script will begin working. 

(Note that the upper bounds on the year/month/day are exclusive. So the
above script would not download any data from 2016 or the month of May or
the 32nd day of a month if it existed)
'''



# Basic imports
import os, os.path
import numpy as np
import pandas as pd

# Google Drive API imports
import pydrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# abwoc imports
from abwoc.data_loading.linux_download_aia import linux_download_aia


# Arguments given
year_min = int(sys.argv[1])
year_max = int(sys.argv[2])
month_min = int(sys.argv[3])
month_max = int(sys.argv[4])
day_min = int(sys.argv[5])
day_max = int(sys.argv[6])
event_type = str(sys.argv[7])

# Where the images are to be stored locally
local_path = './temp_AIA_files/'
if not os.path.isdir(local_path):
  os.mkdir(local_path)

# Dictionary of Google Drive folder IDs for each event type
path_id_dict = {
  'XRA_events': '1PwUaIaIXlWsCnpQ0ub86X1igdAFJCenf', 
  'XRA_nulls': '1HNvf0CWYWVVuv0zwMCVv1mU9hRwUoGcT', 
  'FLA_events': '1VuTnV6Q-0iOijzhNi3uyhMePPNb89q1N', 
  'FLA_nulls': '1-_n0qQarKgyfVXD6dkMJm5oDBiw6ATbv'
}
event_path_id = path_id_dict[event_type+"_events"]
null_path_id = path_id_dict[event_type+"_nulls"]

# Define time parameters
years = np.arange(year_min,year_max)
months = np.arange(month_min,month_max)
days = np.arange(day_min,day_max)

# Event database to select events from
df_main = pd.read_csv('./event_df_main.csv')

# Wavelengths to download
lambdas_used = [131,171,211]

# Connect to Google Drive
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

for y in years:
  for m in months:
    sub_fold = str(y)+"_"+str(m).zfill(2)
    #create_storage_dirs(local_event_path,local_null_path,sub_fold,lambdas_used)
    for d in days:
      # Create dataframe containing relevant events
      event_df = linux_download_aia.get_event_df(df_main,event_type,y,m,d)
      
      # Select one time for each event
      event_times = linux_download_aia.get_event_times(event_df, y, m, d)

      # Print selected times to screen
      print(event_times)

      # Select one null event time for each positive event
      num_events = np.size(event_times,0)
      null_times = linux_download_aia.get_null_times(event_df, num_events, y, m, d)

      # Print selected times to screen
      print(null_times)

      # Download images
      linux_download_aia.get_images(event_times,local_path,event_path_id,sub_fold,lambdas_used,y,m,d,drive,testing=False)
      linux_download_aia.get_images(null_times,local_path,null_path_id,sub_fold,lambdas_used,y,m,d,drive,testing=False)