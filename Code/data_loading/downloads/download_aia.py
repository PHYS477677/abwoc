import astropy.units as u
from astropy.coordinates import SkyCoord
import sunpy.map
from sunpy.net import Fido, attrs as a
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
from aiapy.calibrate import register, update_pointing, normalize_exposure
import os, os.path
import pandas as pd
import random
import sys


def get_images(times,down_path,lamb,year,month,dayi,testing):
  """
  Download AIA images using the Fido search.

  Inputs
  ------
    times     : 2-D Array where each sub-array lists one hour and minute
    down_path : Location where images are downloaded to
    lamb      : Wavelengths to download images from
    year      : Year that images come from
    month     : Month that images come from
    day_s     : Day that events started during
    testing   : A boolean indicating whether this function is just being tested.
                If true, this function will only print the found images, it won't
                download them. 
   
  Outputs
  -------
    No outputs are returned
  """

  for t in times:
   
    # Check to see if event time crosses over to the next day
    if t[0]>23:
        t[0] = t[0]%24
        day = dayi+1
    else:
        day = dayi

    #Start times and end times to use for fido search
    start_time_fido = (str(year)+'-'+str(month).zfill(2)+'-'+str(day).zfill(2)+'T'
                      +str(t[0]).zfill(2)+':'+str(t[1]).zfill(2)+":00" )
    

    end_time_fido = (str(year)+'-'+str(month).zfill(2)+'-'+str(day).zfill(2)+'T'
                    +str(t[0]).zfill(2)+':'+str(t[1]).zfill(2)+":59" )
    
    # Used to select random image out of those taken during the minute specified
    # by 't'. Using the same seed for all wavelengths ensures that each of the 
    # different wavelength images selected are as close together (in time) as possible
    random_seed = random.random()

    for l in lamb:
      # All the observations during minute 't' in wavelength 'l'
      results = Fido.search(a.Time(start_time_fido, end_time_fido), 
                        a.Instrument('AIA'),
                        a.Wavelength(wavemin=l*u.angstrom, wavemax=l*u.angstrom));

      # Check needed in case telescope was not running at time/has no observations
      if np.size(results[0,:])>0:
        # Picks one of the images and downloades it
        f_select = int(np.size(results,1)*random_seed)

        if testing:
            print(results[0,f_select])
        else:
            Fido.fetch(results[0,f_select],path=down_path+'_'+str(l)+'/')


def get_null_times(df, n, year, month, day):
  """
  returns n random times, where events are not present, for a specified date
   
  Inputs
  ------
    dataframe : pandas dataframe object listing all the relvent events to consider
    n         : integer specifying the number of null images to return
    year      : integer specifying the year
    month     : integer specifying the month
    day       : integer specifying the day

  Outputs
  -------
    times     : an array which contains n specified hours/minutes where there 
                are no events occuring
  """
  # number of images selected so far
  count = 0 
  times = np.array([])

  # iterate until found
  while count < n:

    # picking random times
    rand_hour = random.randrange(24)
    rand_min = random.randrange(60)

    # converting times to format that can be easily compared to event report
    rand_time = rand_hour*100+rand_min

    #check day of results
    num_overlap_1 = len((df[ (df['start'] <= rand_time) & 
                           (df['end'] >= rand_time) &
                           (df['year'] == year) &
                           (df['month'] == month) & 
                           (df['day'] == day) ] ))
    
    #check previous day results
    num_overlap_2 = len((df[ (df['start'] >= df['end']) & 
                             (df['end'] >= rand_time) &
                             (df['year'] == year) &
                             (df['month'] == month) &
                             (df['day'] == day-1)] ))
    
    #adding times
    if (num_overlap_1==0 and num_overlap_2==0):
      if count == 0:
        times = np.array([[int(rand_hour),int(rand_min)]])
      else:
        times = np.vstack((times, [int(rand_hour),int(rand_min)]))
      count+=1

  return times


def get_event_times(event_df, year, month, day):
  """
  Select one random time from each of the events in the input dataframe
   
  Inputs
  ------
    event_df  : pandas dataframe object listing all the relevent events to consider
    year      : integer specifying the year
    month     : integer specifying the month
    day       : integer specifying the day

  Outputs
  -------
    times     : an array which contains one specified hour/minute which occurs
                during the duration of each event in event_df
  """
  times = np.array([])
  count = 0
  for index, event in event_df.iterrows():

    #duration of event in minutes
    t_diff = ( (int(str(event.end).zfill(4)[0:2]) - int(str(event.start).zfill(4)[0:2]))%24*60 + 
               (int(str(event.end).zfill(4)[2:4]) - int(str(event.start).zfill(4)[2:4])) )
    
    #selecting random time from event duration
    t_diff_selected = random.randrange(t_diff+1)
    t_selected = int(str(event.start).zfill(4)[0:2])*60 + int(str(event.start).zfill(4)[2:4]) + t_diff_selected

    # converting selected time to hours and minutes
    ts_hours = t_selected // 60
    ts_minutes = t_selected % 60

    # initialize or extend final array
    if count==0:
      times = np.array([[ts_hours,ts_minutes]])
    else:
      times = np.vstack((times,[ts_hours,ts_minutes]))
    count+=1

  return times


def get_event_df(parent_df, event_type, year, month, day):
  """
  Returns a list of times corresponding to events that either started or ended
  during a given day (Note this can include events that started on day-1, but
  ended on day)

  Inputs
  ------
    parent_df : pandas dataframe containing all possible events to choose from
    event_type: string identifying the type of event to choose from in parent_df
    year      : int representing the year to choose events from
    month     : int representing the month to choose events from
    day       : int representing the day to choose events from

  Outputs
  -------
    A single pandas dataframe with all events from parent_df that fit the
    specifications given by the remaining inputs
  """
  # dataframe with events that started on day and ended on day
  df_new_1 = parent_df[ (parent_df.event==event_type) & 
                        (parent_df.year==year) & 
                        (parent_df.month==month) & 
                        (parent_df.day==day) ]
  # dataframe with events that started on day-1 and ended on day
  df_new_2 = parent_df[ (parent_df.event==event_type) & 
                        (parent_df.year==year) & 
                        (parent_df.month==month) & 
                        (parent_df.day==day-1) &
                        (parent_df.start > parent_df.end) ]

  #combined event dataframe
  return (pd.concat([df_new_1,df_new_2]))



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




def create_storage_dirs(event_path,null_path,sub_fold,lambdas):
  """
  Script to check if the directories for storing AIA images already exist,
  and makes them if not.

  Inputs
  ------
    event_path : path to where all event images for a given event type are stored
    null_path  : path to where all null images for a given event type are stored
    sub_fold   : folder name prefix for a specific time/date
    lambdas    : wavelenghts for which images are being downloaded and stored

  Outpus
  ------
    No outputs are returned
  """
  for wavelength in lambdas:
    # Full folder path name
    storage_path = event_path+sub_fold+"_"+str(wavelength)
    # Make folder if it does not exist
    if not os.path.isdir(storage_path):
      os.mkdir(storage_path)


# Primary Download Script Initial Variable Values

# Arguments given
year_min = int(sys.argv[1])
year_max = int(sys.argv[2])
month_min = int(sys.argv[3])
month_max = int(sys.argv[4])
day_min = int(sys.argv[5])
day_max = int(sys.argv[6])
event_type = str(sys.argv[7])

# Where the images are to be stored
event_path = '/content/drive/Shareddrives/Phys 477 - Astro Project/AIA_files/'+event_type+'_events/'
null_path = '/content/drive/Shareddrives/Phys 477 - Astro Project/AIA_files/'+event_type+'_nulls/'

years = np.arange(year_min,year_max)
months = np.arange(month_min,month_max)
days = np.arange(day_min,day_max)

# Event database to select events from
df_main = pd.read_csv('./event_df_main.csv')

# Wavelengths to download
lambdas_used = [131,171,211]

for y in years:
  for m in months:
    sub_fold = str(y)+"_"+str(m).zfill(2)
    #create_storage_dirs(event_path,null_path,sub_fold,lambdas_used)
    for d in days:
      # Create dataframe containing relevant events
      event_df = get_event_df(df_main,event_type,y,m,d)
      
      # Select one time for each event
      event_times = get_event_times(event_df, y, m, d)

      # Print selected times to screen
      print(event_times)

      # Select one null event time for each positive event
      num_events = np.size(event_times,0)
      null_times = get_null_times(event_df, num_events, y, m, d)

      # Print selected times to scree
      print(null_times)

      # Download images
      get_images(event_times,event_path+sub_fold,lambdas_used,y,m,d,testing=False)
      get_images(null_times,null_path+sub_fold,lambdas_used,y,m,d,testing=False)