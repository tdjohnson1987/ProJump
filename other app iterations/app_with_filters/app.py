import streamlit as st
import asyncio
import pandas as pd
import numpy as np
import struct
import time
from bleak import BleakClient

from processing import process_sensor_data  # import your module!

# BLE UUIDs
ACCEL_UUID = "07C80001-07C8-07C8-07C8-07C807C807C8"
GYRO_UUID = "07C80004-07C8-07C8-07C8-07C807C807C8"
MAG_UUID = "07C80007-07C8-07C8-07C8-07C807C807C8"
BLE_ADDRESS = "FC:95:79:7E:F6:DB"

# Helper: Parse IMU bytes
def parse_sensor_data(data):
    x, y, z = struct.unpack(">hhh", data[0:6])
    ts = int.from_bytes(data[6:9], byteorder='big', signed=False)
    return x, y, z, ts

# Streamlit Layout
st.title("🏋️ Real-Time Jump Detection with IMU")
st.markdown("**Live BLE streaming and jump height estimation**")

start_btn = st.button("Start Stream")
stop_btn = st.button("Stop Stream")
status_placeholder = st.empty()
jump_placeholder = st.empty()

chart_placeholder = st.empty()

if "running" not in st.session_state:
    st.session_state.running = False

# Main async function to collect and process data
async def collect_data():
    async with BleakClient(BLE_ADDRESS) as client:
        st.success("Connected to IMU ✅")
        accel_data, gyro_data, mag_data = [], [], []

        while st.session_state.running:
            try:
                # Read BLE packets
                accel_raw, gyro_raw, mag_raw = await asyncio.gather(
                    client.read_gatt_char(ACCEL_UUID),
                    client.read_gatt_char(GYRO_UUID),
                    client.read_gatt_char(MAG_UUID)
                )

                ax, ay, az, ts = parse_sensor_data(accel_raw)
                gx, gy, gz, _ = parse_sensor_data(gyro_raw)
                mx, my, mz, _ = parse_sensor_data(mag_raw)

                accel_data.append([ts, ax, ay, az])
                gyro_data.append([ts, gx, gy, gz])
                mag_data.append([ts, mx, my, mz])

                # Create dataframe for processing every N samples
                if len(accel_data) >= 150:  # ~3 seconds at 50Hz
                    df = pd.DataFrame({
                        "time": np.array([d[0] for d in accel_data]) / 1000.0,
                        "AccX": [d[1] for d in accel_data],
                        "AccY": [d[2] for d in accel_data],
                        "AccZ": [d[3] for d in accel_data],
                        "GyroX": [d[1] for d in gyro_data],
                        "GyroY": [d[2] for d in gyro_data],
                        "GyroZ": [d[3] for d in gyro_data],
                        "MagX": [d[1] for d in mag_data],
                        "MagY": [d[2] for d in mag_data],
                        "MagZ": [d[3] for d in mag_data],
                    })

                    # Process IMU
                    results = process_sensor_data(df, fs=50, sensor_name="Live")
                    jumps = results["jumps"]

                    if len(jumps) > 0:
                        latest_jump = jumps[-1]
                        jump_placeholder.success(
                            f"🟢 Jump detected! Height: **{latest_jump['height_m']:.2f} m** "
                            f"(Flight time: {latest_jump['flight_time']:.2f}s)"
                        )
                    else:
                        jump_placeholder.info("No jump detected yet...")

                    accel_data, gyro_data, mag_data = [], [], []

                await asyncio.sleep(0.05)\
            
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
                st.session_state.running = False

# Start/Stop Logic
if start_btn:
    st.session_state.running = True
    asyncio.run(collect_data())


if stop_btn:
    st.session_state.running = False
    st.warning("Stream stopped.")
