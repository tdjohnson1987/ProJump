import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.integrate import cumulative_trapezoid 
from scipy.interpolate import interp1d
from scipy.ndimage import label
import imufusion

""" 
Full pipeline: 

    Raw IMU → 
    Madgwick (rotate to global frame, gravity subtracted) → 
    Acc mag (from global z-axis) → 
    Filtering (HPF → LPF) → 
    Velocity (by integration) → 
    Zero velocity update (ZUPT) → 
    Jump detection (flight time) → 
    Height estimation.

"""

# Filters 
def butter_highpass_filter(data, cutoff=0.1, fs=50, order=4):
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype="high", analog=False)
    return filtfilt(b, a, data)

def butter_lowpass_filter(data, cutoff=2, fs=50, order=4):
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)

# Madgwick 
def apply_madgwick(timestamp, gyro, accel, mag, fs):
    offset = imufusion.Offset(int(round(fs)))
    ahrs = imufusion.Ahrs()
    ahrs.settings = imufusion.Settings(
        imufusion.CONVENTION_NWU, 0.5, 2000, 10, 10, int(5 * fs)
    )

    dt = np.diff(timestamp, prepend=timestamp[0])
    euler_deg = np.empty((len(timestamp), 3))
    accel_global = np.empty_like(accel)

    for i in range(len(timestamp)):
        gyro[i] = offset.update(gyro[i])
        ahrs.update(gyro[i], accel[i], mag[i], dt[i])
        euler_deg[i] = ahrs.quaternion.to_euler()

        # Rotate accel to global frame
        accel_global[i] = ahrs.quaternion.rotate(accel[i])

    return euler_deg, accel_global

#  Zero velocity update - ZUPT 
def apply_zupt(accel_mag, velocity, time, stance_range=(0.90, 1.10)):
    stance = pd.Series(accel_mag).between(*stance_range).to_numpy()
    vel_corrected = velocity.copy()
    vel_corrected[stance] = 0.0

    not_stance = ~stance
    if not not_stance.any():
        vel_corrected[:] = 0.0
    else:
        interp_func = interp1d(
            time[not_stance],
            vel_corrected[not_stance],
            bounds_error=False,
            fill_value="extrapolate",
        )
        vel_corrected = interp_func(time)

    return vel_corrected, stance

#  Jump Detection 
from scipy.signal import find_peaks
from scipy.ndimage import label
import numpy as np

def detect_jumps_with_height(time, vel_signal, acc_filt, fs, 
                             pos_height=2, neg_height=2, distance=50, 
                             threshold=4, min_duration=0.2, max_duration=2.0,
                             sensor_name=""):
    """
    Detect jumps from velocity & acceleration signals.
    
    1. Peak detection (takeoffs & mid-jump).
    2. Use zero-crossing after positive peaks to refine landings.
    3. Identify flight epochs from low-acceleration mask.
    4. Calculate flight time & jump height.
    
    Returns:
        - jumps: list of dicts with start, end, flight_time, height
        - indices: dict with raw takeoffs, mid-jumps, landings
    """
    # 1. Peaks: negative = takeoffs, positive = mid-air/landing candidate
    peaks_positive, _ = find_peaks(vel_signal, height=pos_height, distance=distance)
    peaks_negative, _ = find_peaks(-vel_signal, height=neg_height, distance=distance)

    # 2. Zero-crossings after positive peaks → refine landings
    landing_indices = []
    for peak in peaks_positive:
        zero_crossings = np.where(np.diff(np.sign(vel_signal[peak:]))) [0]
        if len(zero_crossings) > 0:
            landing_idx = peak + zero_crossings[0]
            landing_indices.append(landing_idx)

    if sensor_name:
        print(f"[{sensor_name}] Detected {len(landing_indices)} landings and {len(peaks_negative)} takeoffs.")

    indices = {
        "takeoffs": peaks_negative,
        "mid-jump": peaks_positive,
        "landings": landing_indices,
    }

    # 3. Flight mask based on acceleration threshold
    min_samples = int(min_duration * fs)
    max_samples = int(max_duration * fs)
    flight_mask = np.abs(acc_filt) < threshold
    labels, num = label(flight_mask)

    jumps = []
    for i in range(1, num+1):
        idx = np.where(labels == i)[0]
        if min_samples <= len(idx) <= max_samples:
            start_idx = idx[0]
            # use first landing after start
            landing_after_start = [li for li in landing_indices if li > start_idx]
            if landing_after_start:
                end_idx = landing_after_start[0]
                flight_time = time[end_idx] - time[start_idx]
                height = (9.81 * flight_time**2) / 8
                jumps.append({
                    "start": start_idx,
                    "end": end_idx,
                    "flight_time": flight_time,
                    "height": height,
                })

    return jumps, indices

# Full Pipeline 
def process_sensor_data(df, fs=50):
    """
    df must have:
    ['time','AccX','AccY','AccZ','GyroX','GyroY','GyroZ','MagX','MagY','MagZ']
    """

    # Run Madgwick → rotate accel into global frame
    _, accel_global = apply_madgwick(
        df["time"].to_numpy(),
        df[["GyroX","GyroY","GyroZ"]].to_numpy(),
        df[["AccX","AccY","AccZ"]].to_numpy(),
        df[["MagX","MagY","MagZ"]].to_numpy(),
        fs,
    )

    # Acceleration magnitude
    acc_mag = np.linalg.norm(accel_global, axis=1)  

    # Filtering
    acc_filt = butter_highpass_filter(acc_mag, cutoff=0.1, fs=fs)
    acc_filt = butter_lowpass_filter(acc_filt, cutoff=2, fs=fs)

    # Velocity
    vel = cumulative_trapezoid(acc_filt, df["time"], initial=0)

    # ZUPT
    vel_corrected, stance = apply_zupt(acc_mag, vel, df["time"].to_numpy())

    # Jump detection
    jumps, indices  = detect_jumps_with_height(df["time"].to_numpy(), 
                                               vel_corrected, 
                                               acc_filt, 
                                               fs=fs, 
                                               sensor_name="Chest")

    return acc_filt, vel_corrected, stance, jumps, indices
