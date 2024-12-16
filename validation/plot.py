import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.dates as mdates
import datetime

df = pd.read_csv("datasets/Phoenix Nov5-Nov7.csv")

datetimes = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])

# Create side-by-side subplots
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharex=True)

ax1.plot(datetimes, df["Van Temp"], label="Van Temperature")
ax1.plot(datetimes, df["Temperature"], label="Ambient Temperature")
ax1.set_xlabel("Time")
ax1.set_ylabel("Temperature (°C)")

ax1.legend(["Recorded Van Temperature","Ambient Temperature"], loc ='upper left')
ax1.set_title("Temperature of Van in Phoenix, AZ 11/5/24-11/7/24")

# Format the x-axis to display dates
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%I:%M %p'))

# Rotate the x-axis labels
ax1.tick_params(axis='x', rotation=45)


ax2.plot(datetimes, df["GHI"], label="GHI")
ax2.set_ylabel("Solar Radiation (W/m^2)")

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%I:%M %p'))
ax2.tick_params(axis='x', rotation=45)
ax2.set_xlabel("Time")
ax2.set_title("Solar Radiation in Phoenix, AZ 11/5/24-11/7/24")
ax2.legend(["Global Horizontal Irradiance"])

plt.tight_layout()

# plt.show()

plt.savefig("Phoenix AZ 11-5.png",dpi = 300)


