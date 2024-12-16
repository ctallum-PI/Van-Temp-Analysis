from os import listdir
import pickle
import pandas as pd
from dataclasses import dataclass
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
from tqdm import tqdm
# from datetime import datetime

from weather import get_historic_weather_data
from common import ModelParams, WeatherData, PredictionInput, PredictionOutput



class Model:
    # model parameters
    def __init__(self, model_params: ModelParams):
        self.solar_absorptance = model_params.solar_absorptance
        self.heat_transfer_coefficient = model_params.heat_transfer_coefficient
        self.surface_area = 41.228
        self.thermal_capacity = model_params.thermal_capacity
        self.emissivity = model_params.emissivity
        self.year = model_params.year
        
        self.get_historic_data(2010,2020)
        
        self.weather_data_path = 'weather_data'
        
        self.threshold = model_params.threshold
        
        city_data = pd.read_csv('city locations.csv')
        self.states = city_data["State"].to_list()
        self.data_years = listdir(self.weather_data_path)
        
        self.Boltzmann_constant = 5.57e-8
        
        
    def predict_upper_bound(self,input: WeatherData, t0 = None) -> list[float]:
        """
        Primary math for calculating upper bound of van temperatures
        """        
        temps = input.temps
        ghi = input.GHI
        dni = input.DNI
        angle = input.angle
        
        angle = np.minimum(angle, 90)
        
        angle_rad = angle * (2*3.14/360)
        
        dhi = ghi - dni*np.cos(angle_rad)
                
        
        n_hours = len(temps)
        
        times = np.linspace(0, n_hours, n_hours*6, endpoint=False) # every 10 minutes
        dt = times[1] - times[0]
        dt_seconds = dt * 3600
        
        if t0 is None:
            cur_temp = temps[0]
        else:
            cur_temp = t0
        
        van_temp = []
                
        for t in times:
            # print(int(t))
            effective_sa = ghi[int(t)] * 8.274 + dhi[int(t)]*12.34*2 + dni[int(t)]*12.34*np.cos(1.57 - angle_rad[int(t)])
            # print(effective_sa * self.solar_absorptance)
            
            # Q_solar = self.solar_absorptance * ghi[int(t)] * self.surface_area
            Q_solar = effective_sa * self.solar_absorptance
            
            # print(Q_solar)
            
            
            Q_ambient = self.heat_transfer_coefficient * (temps[int(t)] - cur_temp) * self.surface_area
            Q_radiation = self.emissivity * self.Boltzmann_constant * ((temps[int(t)] + 273.15)**4-(cur_temp + 273.15)**4) * self.surface_area
            
            Q_total = Q_solar + Q_ambient + Q_radiation
            
            d_temperature = Q_total / self.thermal_capacity * dt_seconds
            
            cur_temp += d_temperature
            
            van_temp.append(cur_temp)

        
        
        return np.array(van_temp[::6])
        
        # return pred_temps
    
    def predict_lower_bound(self, input: WeatherData) -> list[float]:
        """
        Primary math for calculating the lower bound of van temperatures
        """
        return input.temps
        
    def predict_yearly_downtime(self, year: str, save_data = True) -> list[PredictionOutput]:
        
        # check hashed results
        for hash in os.listdir("hashed_model_results"):
            params = hash.split(".pkl")[0].split("-")
            if params[0] == year and float(params[2]) == self.solar_absorptance and float(params[3]) == self.heat_transfer_coefficient and float(params[4]) == self.thermal_capacity and float(params[5]) == self.emissivity:
                print("[INFO] Pickled results found, using those instead of re-calculating")
                with open(f"hashed_model_results/{hash}", 'rb') as f:
                    data = pickle.load(f)
                    
                    # same threshold
                    if float(params[1]) == self.threshold:
                    
                        self.save_downtime_csv(data, year)
                        return data
                    # not same threshold, so recalculate just that element
                    else:
                        for state_data_idx, state_data in enumerate(data):
                            n_critical_hours = (state_data.predicted_temps >= self.threshold).sum()
                            data[state_data_idx].n_critical_hours = n_critical_hours
                        
                        self.save_downtime_csv(data, year)
                        
                        return data
        
        n_states = len(self.states)
        n_years = len(self.data_years)
            
        if year == 'all':
            # array to track cumulative downtime across all data years
            pred_downtime_sum = np.zeros((n_states))
            pred_max_temp = np.zeros((n_states))
            
            pred_temps = []
            
            for year_idx, year in enumerate(self.data_years):
                output = self.predict_yearly_downtime(year, save_data = True)
                
                for idx, state_data in enumerate(output):
                    pred_downtime_sum[idx] += state_data.n_critical_hours
                    pred_max_temp[idx] = max(pred_max_temp[idx], state_data.max_van_temp)
                    
                    if year_idx == 0:
                        pred_temps.append(state_data.predicted_temps)
                    else:
                        pred_temps[idx] += state_data.predicted_temps
                    
                    
            pred_downtime_avg = pred_downtime_sum / n_years
            avg_temps = [sum_temps/n_years for sum_temps in pred_temps]
            
            # overwrite data and save
            for idx, state_data in enumerate(output):
                state_data.n_critical_hours = pred_downtime_avg[idx]
                state_data.max_van_temp = pred_max_temp[idx]
                state_data.predicted_temps = avg_temps[idx]
                
                
            self.save_downtime_csv(output, 'all')
            
            # self.hash_data("all",output)
            
            return output
            
        else:
            pred_downtime = []
            for state in tqdm(self.states):
                data = self.load_state_data(state, year)
                predicted_temps = self.predict_upper_bound(data.weather_data)
                
                n_critical_hours = (predicted_temps >= self.threshold).sum()
                
                pred_downtime.append(PredictionOutput(
                    state = data.state,
                    coordinates = data.coordinates,
                    predicted_temps = predicted_temps,
                    n_critical_hours = n_critical_hours,
                    max_van_temp = max(predicted_temps)
                ))
                
            if save_data:
                self.save_downtime_csv(pred_downtime, year)
                self.hash_data(year, pred_downtime)
        
            return pred_downtime
        
    def save_downtime_csv(self, pred_data:list[PredictionInput], year: str, path_base: str = "predicted_downtime") -> None:
        """
        Save predicted data as a CSV
        """
        df = pd.DataFrame()
        df["State"] = [data.state for data in pred_data]
        df["Latitude"] = [data.coordinates[0] for data in pred_data]
        df["Longitude"] = [data.coordinates[1] for data in pred_data]
        df["Max Temp"] = [int(data.max_van_temp) for data in pred_data]
        df["Critical Hours"] = [int(data.n_critical_hours) for data in pred_data]
        
        if year == "all":
            year_str = f"{self.data_years[0]}-{self.data_years[-1]}"
        else:
            year_str = year
        
        filename = f"csv_results/{year}-{self.threshold}-{self.solar_absorptance}-{self.heat_transfer_coefficient}-{self.thermal_capacity}-{self.emissivity}.csv"
        # filename =f"{path_base}_{year_str}.csv" 
        
        df.to_csv(filename, index=False,lineterminator='\n')
  
    def plot_downtime(self, pred_data_list: list[PredictionOutput], save_fig = False) -> None:
        """
        Plot a map of US states and downtime
        """
        year = self.year
        
        fig = plt.figure()

        ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.LambertConformal(), frameon=False)
        ax.patch.set_visible(False)
        ax.set_extent([-125, -66.5, 20, 50], ccrs.Geodetic())
        shapename = 'admin_1_states_provinces_lakes'
        states_shp = shpreader.natural_earth(resolution='110m', category='cultural', name=shapename)

        # setup up color data
        min_color = min(pred_data_list, key = lambda data: data.n_critical_hours).n_critical_hours
        max_color = max(pred_data_list, key = lambda data: data.n_critical_hours).n_critical_hours
        
        norm = Normalize(vmin = min_color, vmax=max_color)
        cmap = plt.cm.coolwarm


        def colorize_state(geometry):
            facecolor = 'black'
            for state_data in pred_data_list:
                latitude = state_data.coordinates[0]
                longitude = state_data.coordinates[1]
                if geometry.contains(sgeom.Point(longitude, latitude)):
                    color = list(cmap(norm(state_data.n_critical_hours)))
                    
                    facecolor = (color[0], color[1], color[2])            
            return {'facecolor': facecolor, 'edgecolor': 'black'}

        ax.add_geometries(
                shpreader.Reader(states_shp).geometries(),
                ccrs.PlateCarree(),
                styler=colorize_state)
        
        # get top 5 states
        pred_data_list.sort(key=lambda item: item.n_critical_hours, reverse=True)
        
        n_states = 5
                
        proxy_artists = []
        labels = []
        for state_data in pred_data_list[0:n_states]:
            color = list(cmap(norm(state_data.n_critical_hours)))[0:3]
            proxy_artists.append(mpatches.Rectangle((0,0), 1,1, facecolor=color))
            labels.append(f"{state_data.state}: {int(state_data.n_critical_hours)}hrs")

        ax.legend(proxy_artists, labels,
                loc='lower left', bbox_to_anchor=(0.025, -0.1), fancybox=True, title=f'States with most hours')
        
        if year == 'all':
            year_str = f"{self.data_years[0]}-{self.data_years[-1]}"
        else:
            year_str = year
        
        title_str = f"Number of Predicted Hours Yearly Exceeding {self.threshold}°C Using {year_str} Data"
        
        plt.title(title_str)
        
        if save_fig:
            hash = self.get_hash()
            file_name = f"results/downtime-{hash}.png"
            fig.savefig(file_name,dpi=300)
            
    def plot_max_temps(self, pred_data_list: list[PredictionOutput], save_fig = False) -> None:
        """
        Plot a map of US states and downtime
        """
        year = self.year
        
        fig = plt.figure()

        ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.LambertConformal(), frameon=False)
        ax.patch.set_visible(False)
        ax.set_extent([-125, -66.5, 20, 50], ccrs.Geodetic())
        shapename = 'admin_1_states_provinces_lakes'
        states_shp = shpreader.natural_earth(resolution='110m', category='cultural', name=shapename)

        # setup up color data
        min_color = min(pred_data_list, key = lambda data: data.max_van_temp).max_van_temp
        max_color = max(pred_data_list, key = lambda data: data.max_van_temp).max_van_temp

        norm = Normalize(vmin = min_color, vmax=max_color)
        cmap = plt.cm.coolwarm


        def colorize_state(geometry):
            facecolor = 'black'
            for state_data in pred_data_list:
                latitude = state_data.coordinates[0]
                longitude = state_data.coordinates[1]
                if geometry.contains(sgeom.Point(longitude, latitude)):
                    color = list(cmap(norm(state_data.max_van_temp)))
                    
                    facecolor = (color[0], color[1], color[2])            
            return {'facecolor': facecolor, 'edgecolor': 'black'}

        ax.add_geometries(
                shpreader.Reader(states_shp).geometries(),
                ccrs.PlateCarree(),
                styler=colorize_state)
        
        # get top 5 states
        pred_data_list.sort(key=lambda item: item.max_van_temp, reverse=True)
        
        n_states = 5
                
        proxy_artists = []
        labels = []
        for state_data in pred_data_list[0:n_states]:
            color = list(cmap(norm(state_data.max_van_temp)))[0:3]
            proxy_artists.append(mpatches.Rectangle((0,0), 1,1, facecolor=color))
            labels.append(f"{state_data.state}: {int(state_data.max_van_temp)}°C")

        ax.legend(proxy_artists, labels,
                loc='lower left', bbox_to_anchor=(0.025, -0.1), fancybox=True, title=f'Highest temperatures')
        
        if year == 'all':
            year_str = f"{self.data_years[0]}-{self.data_years[-1]}"
        else:
            year_str = year
        
        title_str = f"Maximum Temperature Predicted Across States Using {year_str} Data"
        
        plt.title(title_str)
        
        if save_fig:
            hash = self.get_hash()
            file_name = f"results/maximum-{hash}.png"
            fig.savefig(file_name,dpi=300)
            
    def load_state_data(self, state: str, year: str) -> PredictionInput:
        """
        Load historic weather file
        """
        file_path = os.path.join(self.weather_data_path, year, f"{state}.csv")
        
        # get first part of file name 
        df_top = pd.read_csv(file_path, nrows = 1)
        
        latitude = df_top['Latitude'][0]
        longitude = df_top["Longitude"][0]
        
        df = pd.read_csv(file_path, skiprows=2)
        
        # datetimes = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
        # datetime_list = datetimes.tolist()
        
        GHI = df['GHI'].astype('float')
        temps = df["Temperature"].astype('float')
        DNI = df["DNI"].astype('float')
        angle = df["Solar Zenith Angle"].astype('float')
        
        return PredictionInput(
            state = state,
            coordinates = (latitude, longitude),
            weather_data = WeatherData(
                GHI = GHI,
                DNI = DNI,
                temps = temps,
                angle = angle
            )
        )    
    
    def compare_to_known(self, path: str) -> list[float]:
        df = pd.read_csv(path)
        
        input = WeatherData(
            GHI = df["GHI"].astype(int),
            DNI = df["DNI"].astype(int),
            temps = df["Temperature"].astype(float),
            angle = df["Solar Zenith Angle"]
            
        )
        
        actual_temp = df["Van Temp"]
        datetimes = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
        
        ambient_temp = df["Temperature"]
        ghi = df["GHI"]
        
        dni = df["DNI"]
        angle = df["Solar Zenith Angle"]
        
        angle = np.minimum(angle, 90)
        
        angle_rad = angle * (2*3.14/360)
        
        dhi = ghi - dni*np.cos(angle_rad)
        
        effective_sa = ghi* 8.274 + dhi*12.34*2 + dni*12.34*np.cos(1.57 - angle_rad)
                
        
        pred_temp = self.predict_upper_bound(input, t0 = actual_temp[0])
        
        plt.plot(datetimes, actual_temp)
        plt.plot(datetimes, pred_temp)
        plt.plot(datetimes, ambient_temp)
        # plt.plot(datetimes, ghi * .01* 41.228)
        # plt.plot(datetimes, effective_sa* .01)
        
        # plt.plot(actual_temp)
        # plt.plot(pred_temp)
        # plt.plot(ambient_temp)
        
        plt.legend(["Recorded Van Temperature","Predicted Van Temperature", "Ambient Temp"], loc='upper left')
        
        plt.title("Predicted and Recorded Temperatures in Phoenix, AZ 11/5/24-11/7/24")
        plt.xlabel("Time")
        plt.ylabel("Temperature (°C)")
        
        # Format the x-axis to display dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%I:%M %p'))

        # Rotate the x-axis labels
        plt.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
        plt.savefig("AZ Comparison h=14.png", dpi=300)
        

        
        
    def get_historic_data(self, start_year, end_year):
        """
        Pull from DSRDB API to get weather data
        """
        # iterate through each year
        for year in range(start_year, end_year + 1):
            # find the city data 
            with open('city locations.csv','r') as f:
                df = pd.read_csv(f)
            # go through each state and save data for the year
            for idx, row in df.iterrows():
                state = row['State']
                latitude = row['Latitude']
                longitude = row['Longitude']
                get_historic_weather_data(state, latitude, longitude, year)

    def hash_data(self, year: str, data: list[PredictionOutput]) -> None:
        hash = f"{year}-{self.threshold}-{self.solar_absorptance}-{self.heat_transfer_coefficient}-{self.thermal_capacity}-{self.emissivity}"
        
        with open(f"hashed_model_results/{hash}.pkl", 'wb') as f:
            pickle.dump(data, f)
        
      
    def get_hash(self) -> str:
        hash = f"{self.year}-{self.threshold}-{self.solar_absorptance}-{self.heat_transfer_coefficient}-{self.thermal_capacity}-{self.emissivity}"
        
        return hash