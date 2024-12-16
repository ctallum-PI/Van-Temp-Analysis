import asyncio
from bleak import BleakScanner
from bleak.backends.device import BLEDevice
from tpy358 import query_tp357

import datetime


import csv

# Assuming the query_tp357 function is available to you
# from your_module import query_tp357

async def get_device_by_address(address: str) -> BLEDevice:
    """Find the thermostat device by its known address."""
    device = await BleakScanner.find_device_by_address(address)
    if device:
        print(f"Found thermostat device: {device.name} ({device.address})")
    else:
        print(f"No device found with address: {address}")
    return device

async def get_data_from_thermostat(device: BLEDevice, mode: str):
    """Query data from the thermostat using the provided mode."""
    if device:
        print(f"Querying thermostat data for mode: {mode}")
        data = await query_tp357(device, mode)
        # print("Data received:", data)
        return data
    else:
        print("No thermostat device found.")

async def main():
    # Replace this with the actual Bluetooth address of your thermostat device
    # device_address = "F5:4A:8F:65:F5:C0" # device 1
    device_address = "C5:0A:AC:50:3D:12" # device 2
    # device_address = "F4:FF:5B:68:96:57" # device 3
    # device_address = "D2:F0:CE:79:87:81" # device 4
    # device_address = "DB:72:A8:85:E5:BC" # device 5
    # device_address = "EC:C3:86:05:7D:35" # device 6
    
    # Get the thermostat device by its known address
    device = await get_device_by_address(device_address)
    
    start_date = datetime.datetime(2024, 11, 6, 0, 0)

    # Query data in 'day', 'week', or 'year' mode (choose the mode)
    data = await get_data_from_thermostat(device, mode='year')
    

    # print(data)

    keys = data[0].keys()

    with open('data_2.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)




# Run the main async function
asyncio.run(main())
