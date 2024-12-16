import numpy as np
import matplotlib.pyplot as plt

# Constants
alpha = 0.6  # Absorption coefficient
A = 5.0  # Effective surface area (m²)
C_int = 2000000  # Thermal capacity of the car's interior (J/°C)
h_ambient = 40.0  # Heat transfer coefficient (W/°C)

# Time and data (hourly)
time = np.linspace(0, 24, 240)  # Time in hours (10-minute intervals)
GHI = np.maximum(1000 * np.sin((np.pi/12) * (time - 6)), 0)  # Approximate GHI curve
T_ambient = 20 + 5 * np.sin((np.pi/12) * (time - 8))  # Ambient temp peaks at 2 PM

# Initial conditions
T_int = [T_ambient[0]]  # Start at ambient temperature

# Simulation
dt = time[1] - time[0]  # Time step in hours
dt_seconds = dt * 3600  # Convert to seconds


for t in range(1, len(time)):
    Q_solar = alpha * GHI[t] * A
    Q_ambient = h_ambient * (T_ambient[t] - T_int[-1]) * A
    Q_radiation = 0.7 * (5.57*10**-8) * ((T_ambient[t] + 273.15)**4 - (T_int[-1] + 273.15)**4) * A 
    dT_int = (Q_solar + Q_ambient + Q_radiation) / C_int * dt_seconds
    T_int.append(T_int[-1] + dT_int)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(time, T_int, label='Internal Temperature')
plt.plot(time, T_ambient, label='Ambient Temperature', linestyle='--')
plt.plot(time, GHI / 100, label='Scaled GHI (x0.01)', linestyle=':')
plt.xlabel('Time (hours)')
plt.ylabel('Temperature (°C)')
plt.title('Vehicle Internal Temperature Over a Day')
plt.legend()
plt.grid()
plt.show()
