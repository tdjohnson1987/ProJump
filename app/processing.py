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

def detect_jumps_by_velocity_peaks(vel_signal, time, sensor_name="", 
                                   pos_height=2, neg_height=2, distance=50,
                                   min_duration=0.2, max_duration=2.0, fs=52,
                                   takeoff_offset=10):
    """
    Detect jumps based on velocity peaks:
        - Takeoff: a few indices after the positive (highest) peak
        - Landing: next negative peak (downward velocity)
    
    Parameters
    ----------
    vel_signal : array
        Filtered velocity signal (ZUPT-corrected)
    time : array
        Time vector (same length as vel_signal)
    sensor_name : str
        For printing/debugging (e.g. 'Chest' or 'Wrist') 
    pos_height, neg_height : float
        Peak thresholds (in m/s)
    distance : int
        Minimum distance between peaks (samples)
    min_duration, max_duration : float
        Flight duration bounds (seconds)
    fs : int
        Sampling frequency
    takeoff_offset : int
        Number of indices to shift after the positive peak for takeoff

    Returns
    -------
    jump_epochs : list of (takeoff_idx, landing_idx)
    """
    
    # Detect positive and negative peaks
    peaks_positive, _ = find_peaks(vel_signal, height=pos_height, distance=distance)
    peaks_negative, _ = find_peaks(-vel_signal, height=neg_height, distance=distance)

    if sensor_name:
        print(f"[{sensor_name}] Found {len(peaks_positive)} upward (+) peaks and {len(peaks_negative)} downward (-) peaks")


    jump_epochs = []

    # For each positive peak, shift by takeoff_offset then find landing
    for pos_peak in peaks_positive:
        takeoff_idx = pos_peak + takeoff_offset  # shift a few indices after the positive peak
        # Find the first landing index after the (shifted) takeoff
        landings_after = [p for p in peaks_negative if p > takeoff_idx]
        if len(landings_after) == 0:
            continue
        landing_idx = landings_after[0]

        # Compute flight time using the original 
        flight_time = time[landing_idx] - time[takeoff_idx]
        if min_duration <= flight_time <= max_duration:
            jump_epochs.append((takeoff_idx, landing_idx))

    print(f"[{sensor_name}] Detected {len(jump_epochs)} valid jumps.") 

    # Compute jump height from flight time
    jump_data = []

    for start, end in jump_epochs:
        flight_time = time[end] - time[start]
        height = 0.5 * 9.81 * (flight_time / 2) ** 2  # h = 0.5 * g * (t/2)^2
        jump_data.append({
            "takeoff_time": time[start],
            "landing_time": time[end],
            "flight_time": flight_time,
            "height": height
        })
    
    if sensor_name:
        print(f"[{sensor_name}] Computed heights for {len(jump_data)} jumps.")
    return jump_epochs, peaks_negative, peaks_positive, jump_data

# Full Pipeline 
def process_sensor_data(df, fs=50, sensor_name=""):
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

    # Jump detection and height estimation
    jump_data, pos_peaks, neg_peaks = detect_jumps_by_velocity_peaks(
    vel_corrected, df["time"].to_numpy(), sensor_name=sensor_name, fs=fs
)

    return {
        "acc_filt": acc_filt,
        "vel_corrected": vel_corrected,
        "stance": stance,
        "jumps": jump_data,
        "pos_peaks": pos_peaks,
        "neg_peaks": neg_peaks
    }