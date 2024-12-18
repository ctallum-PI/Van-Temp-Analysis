from os import listdir
import pickle
import pandas as pd
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
        # make sure we get all the historic data ahead of time, will skip if 
        # data already exists
        self.weather_data_path = 'weather_data'
        self.get_historic_data(2010,2020)
        
        # get thermal body parameters
        self.solar_absorptance = model_params.solar_absorptance
        self.heat_transfer_coefficient = model_params.heat_transfer_coefficient
        self.thermal_capacity = model_params.thermal_capacity
        self.emissivity = model_params.emissivity
        self.year = model_params.year
        self.surface_area = 41.228
        self.Boltzmann_constant = 5.57e-8
        
        # set upper and lower operating temperature bounds 
        self.upper_threshold = model_params.threshold
        self.lower_threshold = -20
        
        # get a list of states
        city_data = pd.read_csv('city locations.csv')
        self.states = city_data["State"].to_list()
        
        # get a list of years that we have data for
        self.data_years = listdir(self.weather_data_path)
               
    def predict_upper_bound(self,input: WeatherData, t0 = None) -> list[float]:
        """
        Primary math for calculating upper bound of van temperatures
        """ 
        
        # unpack input parameters
        temps = input.temps
        ghi = input.GHI
        dni = input.DNI
        angle = input.angle
        
        # ignore solar angles that are beyond the horizon
        angle = np.minimum(angle, 90)
        
        # convert to radians from degrees
        angle_rad = angle * (2*3.14/360)
        
        # calculate diffused horizontal irradiance
        dhi = ghi - dni*np.cos(angle_rad)
                
        # assuming that each entry in input data is spaced 1hr apart
        n_hours = len(temps)
        
        times = np.linspace(0, n_hours, n_hours*6, endpoint=False) # every 10 minutes
        dt = times[1] - times[0]
        dt_seconds = dt * 3600
        
        # if we don't have a specified temp_0 input, we will synch it with ambient temperature to start
        if t0 is None:
            cur_temp = temps[0]
        else:
            cur_temp = t0
        
        # predicted van temperature array
        van_temp = []
                
        for t in times:
            
            # calculate the solar radiation based on surface area and rough approximation due to angled surfaces
            effective_sr = ghi[int(t)] * 8.274 + dhi[int(t)]*12.34*2 + dni[int(t)]*12.34*np.cos(1.57 - angle_rad[int(t)])
            # Q_solar = self.solar_absorptance * ghi[int(t)] * self.surface_area
            Q_solar = effective_sr * self.solar_absorptance

            # calculate the energy lost/gained due to convection with the air
            Q_ambient = self.heat_transfer_coefficient * (temps[int(t)] - cur_temp) * self.surface_area
            
            # calculate the energy lost/gained to solar radiation
            Q_radiation = self.emissivity * self.Boltzmann_constant * ((temps[int(t)] + 273.15)**4-(cur_temp + 273.15)**4) * self.surface_area
            
            # calculate the total net energy gain/loss every 10 minutes
            Q_total = Q_solar + Q_ambient + Q_radiation
            
            # calculate the change in temperature due to the change in energy
            d_temperature = Q_total / self.thermal_capacity * dt_seconds
            
            # update the current temperature
            cur_temp += d_temperature
            van_temp.append(cur_temp)

        return np.array(van_temp[::6]) # convert back form every 10 min to hrs

    def predict_lower_bound(self, input: WeatherData) -> list[float]:
        """
        Primary math for calculating the lower bound of van temperatures
        """
        
        # realistically, the van should never be getting cooler than ambient temperature
        return input.temps
        
    def run_historic_model(self, year_input: str = None) -> list[PredictionOutput]:
        # get year to calculate
        if year_input is None:
            year = self.year
        else:
            year = year_input 
            
        # check hashed results
        for hash in os.listdir("hashed_model_results"):
            params = hash.split(".pkl")[0].split("-")
            
            # check if the underlying model is the same
            if params[0] == year and float(params[1]) == self.solar_absorptance and float(params[2]) == self.heat_transfer_coefficient and float(params[3]) == self.thermal_capacity and float(params[4]) == self.emissivity:
                
                print("[INFO] Pickled results found, using those instead of re-calculating")
                with open(f"hashed_model_results/{hash}", 'rb') as f:
                    data: list[PredictionOutput] = pickle.load(f)
                    
                    # in case upper threshold is not the same, recalculate upper items
                    for state_data_idx, state_data in enumerate(data):
                        # calculate number of hours over-temp
                        n_hours_overtemp = (state_data.predicted_upper_temps >= self.upper_threshold).sum()
                        data[state_data_idx].n_hours_overtemp = n_hours_overtemp
                    
                    self.save_downtime_csv(data, year)
                    return data
        
        # if there is not a hashed pickle file, do all the calculations form scratch
        n_states = len(self.states)
        n_years = len(self.data_years)
        
            
        # if we are searching for an individual year
        if year != 'all':
            pred_output_list = []
            
            # iterate through each US state
            for state in tqdm(self.states):
                
                # get the weather data for the state
                data = self.load_state_data(state, year)
                
                # make predictions on temperature for the state
                pred_upper_temps = self.predict_upper_bound(data.weather_data)
                pred_lower_temps = self.predict_lower_bound(data.weather_data)
                
                # sum the hours each US state is above and below the thresholds
                n_hours_overtemp = (pred_upper_temps >= self.upper_threshold).sum()
                n_hours_undertemp = (pred_lower_temps <= self.lower_threshold).sum()
                
                pred_output_list.append(PredictionOutput(
                    state = data.state,
                    coordinates = data.coordinates,
                    predicted_upper_temps = pred_upper_temps,
                    predicted_lower_temps = pred_lower_temps,
                    n_hours_overtemp = n_hours_overtemp,
                    n_hours_undertemp = n_hours_undertemp,
                    max_van_temp = max(pred_upper_temps),
                    min_van_temp = min(pred_lower_temps),
                ))
                
            self.save_downtime_csv(pred_output_list, year)
            self.hash_data(year, pred_output_list)
        
            return pred_output_list
        
        if year == 'all':
            # if we are looking at all years, we will recursively call this function,
            # gather all the data, and then average it all

            # create a counter array with each slot allocated for each of the states
            pred_hours_overtemp = np.zeros((n_states))
            pred_hours_undertemp = np.zeros((n_states))
            pred_max_temp = np.zeros((n_states))
            pred_min_temp = np.zeros((n_states))
            
            pred_upper_temps = []
            pred_lower_temps = []
            
            # iterate through each of the years
            for year_idx, recursive_year in enumerate(self.data_years):
                
                # recursively call this function to get individual year data
                output = self.run_historic_model(recursive_year)
                
                # now go through each US State in a specific year
                for state_idx, state_data in enumerate(output):
                    # add number of hours outside of threshold
                    pred_hours_overtemp[state_idx] += state_data.n_hours_overtemp
                    pred_hours_undertemp[state_idx] += state_data.n_hours_undertemp
                    
                    # now keep looking for maximum and minimum temperatures for the year
                    pred_max_temp[state_idx] = max(pred_max_temp[state_idx], state_data.max_van_temp)
                    pred_min_temp[state_idx] = min(pred_min_temp[state_idx], state_data.min_van_temp)
                    
                    # if this is the first year, populate the pred_temp lists with the predicted 
                    # temps for each US State 
                    if year_idx == 0:
                        pred_upper_temps.append(state_data.predicted_upper_temps)
                        pred_lower_temps.append(state_data.predicted_lower_temps)
                    # for each other iteration, add new array to previous array
                    # dimensions should line up
                    else:
                        pred_upper_temps[state_idx] += state_data.predicted_upper_temps
                        pred_lower_temps[state_idx] += state_data.predicted_lower_temps
                    
            # now calculate averages for everything
            n_hours_overtemp_avg = pred_hours_overtemp / n_years
            n_hours_undertemp_avg = pred_hours_undertemp/ n_years
            
            pred_upper_temps_avg = [upper_temps / n_years for upper_temps in pred_upper_temps]
            pred_lower_temps_avg = [lower_temps / n_years for lower_temps in pred_lower_temps]
            
            # overwrite data and save using some info from last "output" year data
            all_year_output = []
            for state_idx in range(n_states):
                
                all_year_output.append(PredictionOutput(
                    state = output[state_idx].state,
                    coordinates = output[state_idx].coordinates,
                    predicted_upper_temps = pred_upper_temps_avg[state_idx],
                    predicted_lower_temps = pred_lower_temps_avg[state_idx],
                    n_hours_overtemp = n_hours_overtemp_avg[state_idx],
                    n_hours_undertemp = n_hours_undertemp_avg[state_idx], 
                    max_van_temp = pred_max_temp[state_idx],
                    min_van_temp = pred_min_temp[state_idx]
                ))
                
            self.save_downtime_csv(all_year_output, 'all')
            
            return all_year_output
        
    def save_downtime_csv(self, pred_data:list[PredictionOutput], year: str) -> None:
        """
        Save predicted data as a CSV
        """
        df = pd.DataFrame()
        df["State"] = [data.state for data in pred_data]
        df["Latitude"] = [data.coordinates[0] for data in pred_data]
        df["Longitude"] = [data.coordinates[1] for data in pred_data]
        df["Max Temp"] = [int(data.max_van_temp) for data in pred_data]
        df["Min Temp"] = [int(data.min_van_temp) for data in pred_data]
        df["Num Hours Overtemp"] = [int(data.n_hours_overtemp) for data in pred_data]
        df["Num hours Undertemp"] = [int(data.n_hours_undertemp) for data in pred_data]
        
        # same as hash, but manually to avoid issues with recursion in other functions
        filename = f"csv_results/{year}-{self.upper_threshold}-{self.solar_absorptance}-{self.heat_transfer_coefficient}-{self.thermal_capacity}-{self.emissivity}.csv"
        
        df.to_csv(filename, index=False,lineterminator='\n')
  
    def plot_hrs_overtemp(self, pred_data_list: list[PredictionOutput], save_fig = False) -> None:
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
        min_color = min(pred_data_list, key = lambda data: data.n_hours_overtemp).n_hours_overtemp
        max_color = max(pred_data_list, key = lambda data: data.n_hours_overtemp).n_hours_overtemp
        
        norm = Normalize(vmin = min_color, vmax=max_color)
        cmap = plt.cm.coolwarm

        def colorize_state(geometry):
            facecolor = 'black'
            for state_data in pred_data_list:
                latitude = state_data.coordinates[0]
                longitude = state_data.coordinates[1]
                if geometry.contains(sgeom.Point(longitude, latitude)):
                    color = list(cmap(norm(state_data.n_hours_overtemp)))
                    
                    facecolor = (color[0], color[1], color[2])            
            return {'facecolor': facecolor, 'edgecolor': 'black'}

        ax.add_geometries(
                shpreader.Reader(states_shp).geometries(),
                ccrs.PlateCarree(),
                styler=colorize_state)
        
        # get top 5 states
        pred_data_list.sort(key=lambda item: item.n_hours_overtemp, reverse=True)
        
        n_states = 5
                
        proxy_artists = []
        labels = []
        for state_data in pred_data_list[0:n_states]:
            color = list(cmap(norm(state_data.n_hours_overtemp)))[0:3]
            proxy_artists.append(mpatches.Rectangle((0,0), 1,1, facecolor=color))
            labels.append(f"{state_data.state}: {int(state_data.n_hours_overtemp)}hrs")

        ax.legend(proxy_artists, labels,
                loc='lower left', bbox_to_anchor=(0.025, -0.1), fancybox=True, title=f'States with most hours')
        
        if year == 'all':
            year_str = f"{self.data_years[0]}-{self.data_years[-1]}"
        else:
            year_str = year
        
        title_str = f"Number of Predicted Hours Yearly Exceeding {self.upper_threshold}°C Using {year_str} Data"
        
        plt.title(title_str)
        
        if save_fig:
            hash = self.get_hash()
            file_name = f"results/overtemp_hrs-{hash}-{self.upper_threshold}.png"
            fig.savefig(file_name,dpi=300)
        
    def plot_hrs_undertemp(self, pred_data_list: list[PredictionOutput], save_fig = False) -> None:
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
        min_color = min(pred_data_list, key = lambda data: data.n_hours_undertemp).n_hours_undertemp
        max_color = max(pred_data_list, key = lambda data: data.n_hours_undertemp).n_hours_undertemp
        
        norm = Normalize(vmin = min_color, vmax=max_color)
        cmap = plt.cm.coolwarm


        def colorize_state(geometry):
            facecolor = 'black'
            for state_data in pred_data_list:
                latitude = state_data.coordinates[0]
                longitude = state_data.coordinates[1]
                if geometry.contains(sgeom.Point(longitude, latitude)):
                    color = list(cmap(norm(state_data.n_hours_undertemp)))
                    
                    facecolor = (color[0], color[1], color[2])            
            return {'facecolor': facecolor, 'edgecolor': 'black'}

        ax.add_geometries(
                shpreader.Reader(states_shp).geometries(),
                ccrs.PlateCarree(),
                styler=colorize_state)
        
        # get top 5 states
        pred_data_list.sort(key=lambda item: item.n_hours_undertemp, reverse=True)
        
        n_states = 5
                
        proxy_artists = []
        labels = []
        for state_data in pred_data_list[0:n_states]:
            color = list(cmap(norm(state_data.n_hours_undertemp)))[0:3]
            proxy_artists.append(mpatches.Rectangle((0,0), 1,1, facecolor=color))
            labels.append(f"{state_data.state}: {int(state_data.n_hours_undertemp)}hrs")

        ax.legend(proxy_artists, labels,
                loc='lower left', bbox_to_anchor=(0.025, -0.1), fancybox=True, title=f'States with most hours')
        
        if year == 'all':
            year_str = f"{self.data_years[0]}-{self.data_years[-1]}"
        else:
            year_str = year
        
        title_str = f"Number of Predicted Hours Yearly Under {self.lower_threshold}°C Using {year_str} Data"
        
        plt.title(title_str)
        
        if save_fig:
            hash = self.get_hash()
            file_name = f"results/undertemp_hrs-{hash}-{self.lower_threshold}.png"
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
            
    def plot_min_temps(self, pred_data_list: list[PredictionOutput], save_fig: bool = False) -> None:
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
        min_color = min(pred_data_list, key = lambda data: data.min_van_temp).min_van_temp
        max_color = max(pred_data_list, key = lambda data: data.min_van_temp).min_van_temp

        norm = Normalize(vmin = min_color, vmax=max_color)
        cmap = plt.cm.coolwarm_r


        def colorize_state(geometry):
            facecolor = 'black'
            for state_data in pred_data_list:
                latitude = state_data.coordinates[0]
                longitude = state_data.coordinates[1]
                if geometry.contains(sgeom.Point(longitude, latitude)):
                    color = list(cmap(norm(state_data.min_van_temp)))
                    
                    facecolor = (color[0], color[1], color[2])            
            return {'facecolor': facecolor, 'edgecolor': 'black'}

        ax.add_geometries(
                shpreader.Reader(states_shp).geometries(),
                ccrs.PlateCarree(),
                styler=colorize_state)
        
        # get top 5 states
        pred_data_list.sort(key=lambda item: item.min_van_temp, reverse=False)
        
        n_states = 5
                
        proxy_artists = []
        labels = []
        for state_data in pred_data_list[0:n_states]:
            color = list(cmap(norm(state_data.min_van_temp)))[0:3]
            proxy_artists.append(mpatches.Rectangle((0,0), 1,1, facecolor=color))
            labels.append(f"{state_data.state}: {int(state_data.min_van_temp)}°C")

        ax.legend(proxy_artists, labels,
                loc='lower left', bbox_to_anchor=(0.025, -0.1), fancybox=True, title=f'Lowest temperatures')
        
        if year == 'all':
            year_str = f"{self.data_years[0]}-{self.data_years[-1]}"
        else:
            year_str = year
        
        title_str = f"Minimum Temperature Predicted Across States Using {year_str} Data"
        
        plt.title(title_str)
        
        if save_fig:
            hash = self.get_hash()
            file_name = f"results/minimum-{hash}.png"
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
    
    def compare_to_known(self, path: str) -> None:
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
        
        # some old calcuations for visualizing solar radiaiton
        # ghi = df["GHI"]
        # dni = df["DNI"]
        # angle = df["Solar Zenith Angle"]
        # angle = np.minimum(angle, 90)
        # angle_rad = angle * (2*3.14/360)
        # dhi = ghi - dni*np.cos(angle_rad)
        # effective_sa = ghi* 8.274 + dhi*12.34*2 + dni*12.34*np.cos(1.57 - angle_rad)
                
        pred_temp = self.predict_upper_bound(input, t0 = actual_temp[0])
        
        plt.plot(datetimes, actual_temp)
        plt.plot(datetimes, pred_temp)
        plt.plot(datetimes, ambient_temp)
        
        plt.legend(["Recorded Van Temperature","Predicted Van Temperature", "Ambient Temp"], loc='upper left')
        
        plt.title("Predicted and Recorded Temperatures in Phoenix, AZ 11/5/24-11/7/24")
        plt.xlabel("Time")
        plt.ylabel("Temperature (°C)")
        
        # Format the x-axis to display dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%I:%M %p'))

        # Rotate the x-axis labels
        plt.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
        # plt.savefig("AZ Comparison h=14.png", dpi=300)
        
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
                get_historic_weather_data(state, latitude, longitude, year, self.weather_data_path)

    def hash_data(self, year: str, data: list[PredictionOutput]) -> None:
        hash = f"{year}-{self.solar_absorptance}-{self.heat_transfer_coefficient}-{self.thermal_capacity}-{self.emissivity}"
        
        with open(f"hashed_model_results/{hash}.pkl", 'wb') as f:
            pickle.dump(data, f)
        
    def get_hash(self) -> str:
        hash = f"{self.year}-{self.solar_absorptance}-{self.heat_transfer_coefficient}-{self.thermal_capacity}-{self.emissivity}"
        
        return hash