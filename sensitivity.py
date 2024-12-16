import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.figure()

heat_map = np.zeros((11,11))

for s_idx, s in enumerate(range(11)):
    for e, e_idx in enumerate(range(11)):
        solar = s/10
        emissivity = e/10
        file = f"csv_results/2010-65-{solar}-12-1000000-{emissivity}.csv"
        
        df = pd.read_csv(file)
        
        arizona_max = df["Max Temp"][3]
        
        max_all = np.max(df["Max Temp"])
        
        
        
        heat_map[s_idx, e_idx] = max_all
        
plt.imshow(heat_map, origin='lower')
plt.xlabel('Emissivity')
plt.ylabel("Solar Absorption")
plt.colorbar(orientation = 'vertical')  # Adds the colorbar with a label

num_ticks = heat_map.shape[0]  # Number of data points along each axis
plt.xticks(ticks=np.linspace(0, num_ticks-1, 6), labels=np.linspace(0, 1, 6).round(2))  # X-axis
plt.yticks(ticks=np.linspace(0, num_ticks-1, 6), labels=np.linspace(0, 1, 6).round(2))  # Y-axis
plt.title("Maximum predicted temperature of van using 2010 data")


plt.title('Maximum temperature of van using 2010 data', pad=30)  # Add padding below the title

fig = plt.gcf()  # Get the current figure
# Add caption below the title
plt.figtext(0.5, .9, 'Heat Transfer Coefficient = 12 W/m^2 K, Heat capacity = 1,000,000 J/K', 
            ha='center', fontsize=10,  transform=fig.transFigure)


# Show the plot
# plt.show()

# fig.savefig("Emissivity vs absorption.png",dpi = 300)

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

plt.figure()

heat_map = np.zeros((3,26))

solar_absorptance_array = np.linspace(0.7, 0.9, 3)
heat_transfer_coefficient_array = np.linspace(5,30,26)

for s_idx, solar_absorptance in enumerate(solar_absorptance_array):
    for h_idx, heat_transfer_coefficient in enumerate(heat_transfer_coefficient_array):
        file = f"csv_results/2010-65-{solar_absorptance}-{heat_transfer_coefficient}-1000000-{0.5}.csv"
        
        df = pd.read_csv(file)
        
        arizona_max = df["Max Temp"][3]
        
        max_all = np.max(df["Max Temp"])
        
        heat_map[s_idx, h_idx] = max_all
        
plt.imshow(heat_map, origin='lower')
plt.xlabel('Heat Transfer Coefficient')
plt.ylabel("Solar Absorption")
plt.colorbar(orientation = 'horizontal', pad= 0.2)  # Adds the colorbar with a label

num_ticks = heat_map.shape[0]  # Number of data points along each axis
plt.xticks(ticks=np.linspace(0, 25, 26), labels=np.linspace(5, 30, 26).astype(int))  # X-axis
plt.yticks(ticks=np.linspace(0, 2, 3), labels=np.linspace(0.7, 0.9, 3).round(2))  # Y-axis
plt.title("Maximum predicted temperature of van using 2010 data")


plt.title('Maximum temperature of van using 2010 data', pad=30)  # Add padding below the title

fig = plt.gcf()  # Get the current figure
# Add caption below the title
plt.figtext(0.5, .625, 'Solar Emissivity = 0.5, Heat capacity = 1,000,000 J/K', 
            ha='center', fontsize=10,  transform=fig.transFigure)

fig.set_size_inches((9, 4))

# Show the plot
plt.show()

# fig.savefig("heat transfer vs absorption.png",dpi = 300)
