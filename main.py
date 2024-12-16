import matplotlib.pyplot as plt
from dataclasses import dataclass

from common import ModelParams

from model import Model

import numpy as np


def main():
    
    model_params = ModelParams(
        year = 'all',
        threshold = 65,
        solar_absorptance = 0.8,
        emissivity = 0.5,
        heat_transfer_coefficient = 14,
        thermal_capacity = 1000000
    )
    
    
    model = Model(model_params)
    
    # model.plot_downtime(model.predict_yearly_downtime(model_params.year), save_fig = False)
    # model.plot_max_temps(model.predict_yearly_downtime(model_params.year), save_fig = True)

        
    results = model.compare_to_known('validation/datasets/Phoenix Nov5-Nov7.csv')
    # results = model.compare_to_known('validation/datasets/Las Vegas Sept9.csv')
    # results = model.compare_to_known('validation/datasets/Fort Myers Aug1.csv')
    
    
    plt.show()
    



if __name__ == "__main__":
    main()