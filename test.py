# import os
# import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



# plt.show()

import os
import pandas as pd
from urllib.error import HTTPError

save_dir = "weather_data/"

def get_historic_weather_data(state: str, lat: float, lon: float, year: str, force=False):
    
    # file_name = save_dir + f"{year}/{state}.csv"
    file_name = f"fort myers/{year}.csv"
    
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
        
for year in range(2010,2021):
    get_historic_weather_data("Fort Myers", 26.64, -81.86, year)


# years = os.listdir("weather_data")

# ghi = np.zeros((11,24*3))

# for idx,year in enumerate(years):
#     file_name = f"weather_data/{year}/Arizona.csv"
    
#     df = pd.read_csv(file_name, skiprows=2)
    
#     df_ghi = df["Solar Zenith Angle"][7392:7464].astype(int)
    
#     print(df["Hour"][7392:7464])
    
        
#     ghi[idx,:] = df_ghi
    
# avg = np.max(ghi,axis = 0 )

# print(avg)

# plt.plot(avg, "-r")

# for row_idx, row in enumerate(ghi):
#     plt.plot(row, "--k", alpha=0.3)
    
# plt.title("Solar Radiation (GHI) from 11/5-11/7 in Phoenix, AZ from 2010-2020 ")
# plt.ylabel("GHI (W/m^2)")
# plt.xlabel("Hour")

# plt.legend(["Maximum"])
    
# plt.savefig("Phoenix Solar Radiation 2010-2020.png",dpi=300)
# # plt.show()
# with open("test.csv",'w') as f:
#     for val in avg:
#         f.write(f"{int(val)}\n")
        
        