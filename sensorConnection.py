import asyncio
from bleak import BleakScanner
from bleak import BleakClient
import struct
import pandas as pd
import numpy as np


a_array=[]
g_array = []
def byteToIntConverter(x_bytes, y_bytes, z_bytes, ts_bytes, array):
    x = struct.unpack('>h', x_bytes)[0]
    y = struct.unpack('>h', y_bytes)[0]  
    z = struct.unpack('>h', z_bytes)[0] 
    ts = int.from_bytes(ts_bytes, byteorder='big',signed=False) 
    print(f"X Axis: {x}, Y Axis: {y}, Z  Axis: {z}, Time Stamp: {ts}")
    saveToCsv(x,y,z,ts,array)
    
def saveToCsv(x,y,z,ts, array):
    array.append([x,y,z,ts])
    
async def main():
    devices = await BleakScanner.discover()
    # for device in devices:
    #     print(device)
    ble_address = "FC:95:79:7E:F6:DB"    
    async with BleakClient(ble_address) as client:
        print("Connected to BLE device")
        print(client.is_connected)
        accelerometer_uuid = "07C80001-07C8-07C8-07C8-07C807C807C8"
        gyro_uuid = "07C80004-07C8-07C8-07C8-07C807C807C8"
        for _ in range(200):
            data = await client.read_gatt_char(accelerometer_uuid)
            g_data = await client.read_gatt_char(gyro_uuid)
            g_x_bytes = g_data[0:2]
            g_y_bytes = g_data[2:4]
            g_z_bytes = g_data[4:6]
            g_timestamp_bytes = g_data[6:9]
            byteToIntConverter(g_x_bytes, g_y_bytes, g_z_bytes, g_timestamp_bytes, g_array)
            # print('Byte array: ',data)
            x_bytes = data[0:2]
            y_bytes = data[2:4]
            z_bytes = data[4:6]
            timestamp_bytes = data[6:9]
            byteToIntConverter(x_bytes, y_bytes, z_bytes, timestamp_bytes, a_array)
        pd_df = pd.DataFrame(a_array, columns=['AccX','AccY','AccZ','Acc_Time Stamp'])
        pd_df.to_csv('AccelerometerData.csv',index=False)
        pd_df = pd.DataFrame(g_array, columns=['GyroX','GyroY','GyroZ','Gyro_Time Stamp'])
        pd_df.to_csv('GyroscopeData.csv',index=False)
        acc_df = pd.read_csv('AccelerometerData.csv')
        gyro_df = pd.read_csv('GyroscopeData.csv')
        combined_df = pd.concat([acc_df,gyro_df], ignore_index=True)
        combined_df.to_csv('AccGyroData.csv', index=False)
asyncio.run(main())