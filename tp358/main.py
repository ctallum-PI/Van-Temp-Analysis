import pickle
# from clean_tp358_data import ThermPro
from matplotlib import pyplot as plt
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

import numpy as np

@dataclass
class ThermPro:
    van_id: str
    probe_location: str
    times: list[datetime]
    temps: list[float]
    humidity: list[float]
    van_location: str = "Phoenix, Az"

with open("tp358/ThermProData.pkl",'rb') as f:
    ThermProData: list[ThermPro] = pickle.load(f)


max_temp = 0
for data in ThermProData:
    if data.probe_location == "ceiling":
        max_temp = max(max_temp, np.max(data.temps))
        
# print(f"Ceiling max: {max_temp}")
    
    
legend_titles = []
# plt.figure()

ceiling_temps = []

for dataset in ThermProData:
    
    if dataset.probe_location == "ceiling":
        ceiling_temps.append(np.array(dataset.temps))
        
# print(ceiling_temps)
ceiling_avg = ceiling_temps[0] + ceiling_temps[1] + ceiling_temps[2]

ceiling_avg /= 3

ceiling_avg = np.round(ceiling_avg,1)

with open("tp358/Phoenix Nov5-Nov7.csv") as f:
    df = pd.read_csv(f)


df["Van Temp"] = ceiling_avg

df.to_csv("tp358/Phoenix Nov5-Nov7.csv",index=False)

# print(df.head())