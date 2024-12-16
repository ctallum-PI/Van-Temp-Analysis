import pandas as pd
import numpy as np

# Fort Myers

dni = np.zeros((8760,11))
angle = np.zeros((8760,11))
for idx,year in enumerate(range(2010,2021)):
    df = pd.read_csv(f"validation/raw_data/fort myers/{year}.csv", skiprows=2)
    
    dni[:,idx] = df["DNI"]
    angle[:,idx] = df["Solar Zenith Angle"]
    
# print(np.max(dni,1).shape)
# print(np.average(angle,1).shape)

max_dni = np.max(dni,1)
avg_angle = np.average(angle,1)

print(max_dni[5099:5104])
print(avg_angle[5099:5104])
    

# Las Vegas

dni = np.zeros((8760,11))
angle = np.zeros((8760,11))
for idx,year in enumerate(range(2010,2021)):
    df = pd.read_csv(f"weather_data/{year}/Nevada.csv", skiprows=2)
    
    dni[:,idx] = df["DNI"]
    angle[:,idx] = df["Solar Zenith Angle"]
    
# print(np.max(dni,1).shape)
# print(np.average(angle,1).shape)

max_dni = np.max(dni,1)
avg_angle = np.average(angle,1)

print(max_dni[6035:6040])
print(avg_angle[6035:6040])


print(df["Day"][6035], df["Hour"][6035])