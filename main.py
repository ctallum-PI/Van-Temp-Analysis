import matplotlib.pyplot as plt
from common import ModelParams
from model import Model

import pandas as pd
from datetime import timedelta

def main():
    
    model_params = ModelParams(
        year = 'all',
        threshold = 60,
        solar_absorptance = 0.8,
        emissivity = 0.5,
        heat_transfer_coefficient = 12,
        thermal_capacity = 1000000
    )
    
    model = Model(model_params)
    
    results = model.run_historic_model(force=True)

    model.plot_hrs_overtemp(results, save_fig=True)
    model.plot_hrs_undertemp(results, save_fig=True)
    model.plot_max_temps(results, save_fig=True)
    model.plot_min_temps(results, save_fig=True)
        
    # model.compare_to_known('validation/datasets/test.csv')
    
    # df = pd.read_csv('elastic.csv')
    
    # times = df["Time"]
    
    # times = pd.to_datetime(times) - timedelta(hours=5)
    # print(times)
    # temps = df["DMD Temp"].astype(float)
    
    # plt.plot(times, temps)
    # model.compare_to_known('validation/datasets/Phoenix Nov5-Nov7.csv')
    # model.compare_to_known('validation/datasets/Las Vegas Sept9.csv')
    # model.compare_to_known('validation/datasets/Fort Myers Aug1.csv')
        
    plt.show()
    

if __name__ == "__main__":
    main()