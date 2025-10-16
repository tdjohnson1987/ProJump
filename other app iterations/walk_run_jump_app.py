import streamlit as st
import asyncio
from bleak import BleakClient
import struct
import pandas as pd
import time
from datetime import datetime

# Define sensor UUIDs
ACCELEROMETER_UUID = "07C80001-07C8-07C8-07C8-07C807C807C8"
GYRO_UUID = "07C80004-07C8-07C8-07C8-07C807C807C8"
BLE_ADDRESS = "FC:95:79:7E:F6:DB"

# Helper function to parse byte data from the sensor
def parse_sensor_data(data):
    # Unpack the three axes (e.g., as signed 16-bit integers)
    x, y, z = struct.unpack('>hhh', data[0:6])
    # Unpack the timestamp (e.g., as an unsigned 24-bit integer)
    ts = int.from_bytes(data[6:9], byteorder='big', signed=False)
    return x, y, z, ts

# --- NEW: Activity Detection Function ---
def detect_activity(accel_data_list):
    if len(accel_data_list) < 10:
        return "Not enough data"
    
    # Get the latest AccZ value
    latest_accz = accel_data_list[-1][2] # AccZ is now at index 2
    
    # Simple rule for jumping: a large spike in AccZ
    if latest_accz > 2000:
        return f"Jump Detected {latest_accz}"
    
    # Analyze recent data for walking/running
    recent_accel_z = [item[2] for item in accel_data_list[-10:]] # AccZ is now at index 2
    max_accz = max(recent_accel_z)
    min_accz = min(recent_accel_z)
    
    amplitude = max_accz - min_accz

    # Rules for walking vs. running based on amplitude
    if amplitude > 1000:
        return f"Running {amplitude}"
    elif amplitude > 300:
        return f"Walking {amplitude}"
    else:
        return f"Still {amplitude}"

# The main function to collect and display sensor data
async def collect_and_plot_data():
    st.write("Connecting to BLE device...")
    try:
        async with BleakClient(BLE_ADDRESS) as client:
            st.write("Connected! Starting data collection.")
            
            # Create placeholders for the live graphs
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Live Accelerometer Data")
                accel_chart_placeholder = st.empty()
                accel_data_list = []
            
            with col2:
                st.subheader("Live Gyroscope Data")
                gyro_chart_placeholder = st.empty()
                gyro_data_list = []
            
            # --- NEW: Placeholder for activity status ---
            activity_placeholder = st.empty()

            # Infinite loop to collect data until the user stops it
            while st.session_state.is_running:
                # Use asyncio.gather to read data from both sensors simultaneously
                accel_data_bytes, gyro_data_bytes = await asyncio.gather(
                    client.read_gatt_char(ACCELEROMETER_UUID),
                    client.read_gatt_char(GYRO_UUID)
                )

                # Parse the data
                accel_x, accel_y, accel_z, accel_ts = parse_sensor_data(accel_data_bytes)
                gyro_x, gyro_y, gyro_z, gyro_ts = parse_sensor_data(gyro_data_bytes)

                # Append to data lists
                current_time = datetime.now()
                # --- NOTE: The lists now only store Time, X, Y, Z ---
                accel_data_list.append([current_time, accel_x, accel_y, accel_z])
                gyro_data_list.append([current_time, gyro_x, gyro_y, gyro_z])
                
                # Keep the data lists from growing too large (e.g., last 200 points)
                max_points = 200
                if len(accel_data_list) > max_points:
                    accel_data_list = accel_data_list[-max_points:]
                if len(gyro_data_list) > max_points:
                    gyro_data_list = gyro_data_list[-max_points:]
                
                # Convert lists to DataFrames
                accel_df = pd.DataFrame(accel_data_list, columns=['Time', 'AccX', 'AccY', 'AccZ'])
                gyro_df = pd.DataFrame(gyro_data_list, columns=['Time', 'GyroX', 'GyroY', 'GyroZ'])

                # --- NEW: Call the detection function and display the result ---
                activity = detect_activity(accel_data_list)
                activity_placeholder.info(f"Current Activity: {activity}")

                # Save to CSV
                accel_df.to_csv('AccelerometerData.csv',mode='a',index=False, header=False)
                gyro_df.to_csv('GyroscopeData.csv',mode='a', index=False,header=False)
                
                # Update the charts using their placeholders
                with col1:
                    accel_chart_placeholder.line_chart(accel_df.set_index('Time'))
                with col2:
                    gyro_chart_placeholder.line_chart(gyro_df.set_index('Time'))

                # Wait for a brief moment to control the update speed
                time.sleep(0.05)

    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        st.session_state.is_running = False
        st.write("Disconnected from BLE device.")

# --- Streamlit UI ---
st.title("Live Sensor Data Visualization")
st.write("Click 'Start Sensor' to connect and begin streaming data.")

# Initialize the state of the app if it's not already set
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

# Create the Start and Stop buttons
start_button = st.button("Start Sensor")
stop_button = st.button("Stop Sensor")

if start_button:
    st.session_state.is_running = True
    asyncio.run(collect_and_plot_data())
    
if stop_button:
    st.session_state.is_running = False

# Display a status message
if st.session_state.is_running:
    st.success("Sensor data streaming...")
else:
    st.info("Ready to connect. Press 'Start Sensor' to begin.")