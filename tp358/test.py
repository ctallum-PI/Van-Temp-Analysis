import pandas as pd

with open('Phoenix_weather.csv') as f:
    df = pd.read_csv(f)
    
temp_f = df["Temperature"]

temp_c = (temp_f - 32) * (5/9)



df["Temperature"] = temp_c.astype(int)

with open('Phoenix_weather_2.csv', 'w') as f: 
    df.to_csv(f, index=False,lineterminator='\n')