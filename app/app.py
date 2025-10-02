import streamlit as st
import asyncio
from bleak import BleakClient
import struct
import pandas as pd
from datetime import datetime
import processing  # processing.py

# BLE UUIDs
ACCEL_UUID = "07C80001-07C8-07C8-07C8-07C807C807C8"
GYRO_UUID  = "07C80004-07C8-07C8-07C8-07C807C807C8"
BLE_ADDRESS = "FC:95:79:7E:F6:DB"

# Parse bytes
def parse_sensor_data(data):
    x, y, z = struct.unpack('>hhh', data[0:6])
    ts = int.from_bytes(data[6:9], byteorder='big', signed=False)
    return x, y, z, ts

# Realtime collection + processing
async def collect_and_process():
    st.write("Connecting to BLE device...")
    async with BleakClient(BLE_ADDRESS) as client:
        st.success("Connected!")

        # Placeholders
        col1, col2 = st.columns(2)
        acc_plot = col1.empty()
        gyro_plot = col2.empty()
        status = st.empty()

        accel_data, gyro_data, mag_data = [], [], []
        fs = 50  # assumed Hz

        while st.session_state.is_running:
            # Read sensor
            accel_bytes, gyro_bytes = await asyncio.gather(
                client.read_gatt_char(ACCEL_UUID),
                client.read_gatt_char(GYRO_UUID)
            )
            ax, ay, az, ts_a = parse_sensor_data(accel_bytes)
            gx, gy, gz, ts_g = parse_sensor_data(gyro_bytes)

            # Append
            now = datetime.now().timestamp()
            accel_data.append([now, ax/16384.0, ay/16384.0, az/16384.0])  # scale to g
            gyro_data.append([now, gx, gy, gz])  # deg/s
            # mag_data.append([...]) if available

            # Keep last N samples
            max_points = 512
            if len(accel_data) > max_points:
                accel_data = accel_data[-max_points:]
                gyro_data = gyro_data[-max_points:]

            # Convert to DataFrame
            df = pd.DataFrame(accel_data, columns=["time","AccX","AccY","AccZ"])
            df[["GyroX","GyroY","GyroZ"]] = pd.DataFrame([g[1:] for g in gyro_data])

            # Dummy mag if not streamed
            df[["MagX","MagY","MagZ"]] = 0.0

            # Run processing pipeline when we have enough data
            if len(df) > fs * 2:  # at least 2s of data
                acc_filt, vel_corr, stance, jumps, indices = processing.process_sensor_data(df, fs=fs)

                # Display jumps
                if jumps:
                    last_jump = jumps[-1]
                    status.success(
                        f"Jump detected! Height={last_jump['height']:.2f} m, Flight={last_jump['flight_time']:.2f}s"
                    )

            # Live plots
            acc_plot.line_chart(df.set_index("time")[["AccX","AccY","AccZ"]])
            gyro_plot.line_chart(df.set_index("time")[["GyroX","GyroY","GyroZ"]])

# Streamlit UI
st.title("Real-Time Jump Detection")
if "is_running" not in st.session_state:
    st.session_state.is_running = False

if st.button("Start"):
    st.session_state.is_running = True
    asyncio.run(collect_and_process())
if st.button("Stop"):
    st.session_state.is_running = False
