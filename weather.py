import os
import pandas as pd
from urllib.error import HTTPError

save_dir = "weather_data/"

def get_historic_weather_data(state: str, lat: float, lon: float, year: str, force=False):
    
    file_name = save_dir + f"{year}/{state}.csv"
    
    # if the file exists already, and we are not force collecting, return
    if os.path.exists(file_name) and not force:
        return
    
    # create pull API key
    attr = 'ghi,dni,solar_zenith_angle,air_temperature'
    api_key = "kFhYXYSimDJEC5tB0rZ6PIfXnm5fkrarNYNog2he"
    url = f'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day=false&interval=60&utc=false&full_name=Chris+Allum&email=ctallum@gmail.com&affiliation=Product+Insight&mailing_list=false&reason=research&api_key={api_key}&attributes={attr}'

    try:        
        df = pd.read_csv(url, skiprows=2)
        df = df.set_index(pd.date_range('1/1/{yr}'.format(yr=year), freq="60"+'Min', periods=int(525600/60)))
        
        # if the folder doesn't exit yet, create it
        folder_path = f"{save_dir}/{year}"
        os.makedirs(folder_path, exist_ok=True)
        
        # add the lat/lon at the top of the file 
        with open(file_name,'w') as f:
            f.write(f'Latitude,Longitude\n{lat},{lon}\n')
            
        df.to_csv(file_name, index=False, mode='a')
        
        print(f"Finished collecting {state} data")
    except HTTPError:
        print(f"[ERROR]: Failed to find data for {state}")