This folder contains the files needed to download AIA images to our google drive from a linux machine

The primary script to run is linux_download_aia.py. In order to run linux_download_aia.py there are a total of 7 arguments that must be passed in at runtime. They are:

  1. Start year 
  2. End year
  3. Start month
  4. End month
  5. Start day
  6. End day
  7. Event type
  
linux_download_aia.py then searches the file 'event_df_main.csv' and indentifies each space weather event 
in event_df_main.csv that matches the requirements specified by the 7 input arguments given at runtime. 
It then downloads one AIA expsoure, as a fits file, for each located event, which was taken during when the event was present.
It also downloads an additional image, one per located event, at a time where there was no qualifying space weathe event present. 
These images are then uploaded to Google Drive and deleted from the local machine. 
  
As an example, in order to download one AIA image for each X-RAY event that occured during the months of March
and April in 2015, the python command would look like:

$ linux_download_aia.py 2015 2016 3 5 1 32 XRA

When the script is first run, an internet browser window will open requesting access from the application to Google Drive. Accept the access, and the script will begin working. 

(Note that the upper bounds on the year/month/day are exclusive. So the above script would not download any data from 2016 or the month of May or the 32nd day of a month if it existed)


