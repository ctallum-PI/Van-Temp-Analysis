import matplotlib.pyplot as plt
import numpy as np
import pickle
from datetime import datetime

from dataclasses import dataclass

sensor_data_sets = []

t_ranges = {0: (183, 255),
            1: (185, 257),
            2: (182, 254),
            3: (185, 257),
            4: (182, 254),
            5: (183, 255)}

van_ids = {
    0: "3977",
    1: "3977",
    2: "0949",
    3: "0949",
    4: "3915",
    5: "3915",
} 



plt.figure()

@dataclass
class ThermPro:
    van_id: str
    probe_location: str
    times: list[datetime]
    temps: list[float]
    humidity: list[float]
    van_location: str = "Phoenix, Az"
    
    
for idx in range(6):
    with open(f"tp358/data/data_{idx+1}.csv") as f:
        data = [a.strip().split(",") for a in f.readlines()[1:]]
            
        times = [datetime.strptime(row[0], "%Y-%m-%dT%H") for row in data]
        humidity = [float(row[1]) for row in data]
        temp = [float(row[2]) for row in data]
        
        t_range = t_ranges[idx]
                
        # print(f"Saving temp and humidity from sensor {idx+1} from times {times[t_range[0]]} to {times[t_range[1] -1]}")
        
        temp = temp[t_range[0]: t_range[1]]
        humidity = humidity[t_range[0]: t_range[1]]
        times = times[t_range[0]: t_range[1]]
        
        van_id = van_ids[idx]
        
        if idx % 2 == 0:
            probe_location = "ceiling"
        else:
            probe_location = "morpheus"
            
        sensor_data_sets.append(ThermPro(
            van_id = van_id,
            probe_location = probe_location,
            times = times,
            temps = temp,
            humidity = humidity
        ))
            
# print(sensor_data_sets)
            
with open("ThermProData.pkl", 'wb') as f:
    pickle.dump(sensor_data_sets, f)
        

        
# plt.show()

# plt.figure()
# plt.plot(temp[183:255] * (7/5) + 32)
# plt.show()
