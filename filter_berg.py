import streamlit as st
import asyncio
from bleak import BleakClient
import struct
import pandas as pd
import math
from datetime import datetime, timedelta
from io import BytesIO
import matplotlib.pyplot as plt
from collections import deque
import time  # for perf_counter_ns()
import numpy as np
import imufusion


ACCELEROMETER_UUID = "07C80001-07C8-07C8-07C8-07C807C807C8"
GYRO_UUID          = "07C80004-07C8-07C8-07C8-07C807C807C8"
# BLE_ADDRESS = "FC:95:79:7E:F6:DB"
BLE_ADDRESS        = "df:be:c3:b8:75:a6"

DEFAULT_ACCEL = "Acceleration.csv"
DEFAULT_GYRO  = "Gyroscope.csv"

G_CONST = 9.80665  




from bleak import BleakClient, BleakError  # add BleakError

def _is_error_cancelled(err: Exception) -> bool:
    
    w = getattr(err, "winerror", None)
    if w == 1223:
        return True
    s = str(err)
    return ("1223" in s) or ("-2147023673" in s) or ("ERROR_CANCELLED" in s.upper())



# STREAMLIT SIDEBAR (TUNING)

st.sidebar.header("Jump Detector — Settings")

FREEFALL_G_THRESHOLD = st.sidebar.slider(
    "Free-fall threshold (g): count as in-air when |a| <",
    min_value=0.20, max_value=0.80, value=0.50, step=0.05,
    help="Lower = stricter (needs nearer to 0g). Higher = more sensitive."
)

DEBOUNCE_SAMPLES = st.sidebar.slider(
    "Debounce (samples below threshold to trigger)",
    min_value=1, max_value=5, value=2, step=1,
    help="Require this many consecutive samples in free-fall before counting."
)

BASELINE_SAMPLES = st.sidebar.slider(
    "Baseline calibration samples (to estimate 1g)",
    min_value=5, max_value=50, value=10, step=1,
    help="How many initial samples to average to estimate 1g magnitude."
)

SNAPSHOT_DELAY_S = st.sidebar.select_slider(
    "Snapshot delay after jump (s)",
    options=[1.0, 1.5, 2.0],
    value=1.5,
    help="Wait this long after detecting the jump before taking the snapshot."
)

SNAPSHOT_WINDOW_S = st.sidebar.select_slider(
    "Snapshot window width (s, centered)",
    options=[2.0, 3.0, 4.0, 5.0],
    value=2.0,
    help="Time window shown around the jump. Centered at mid-air if landing known."
)

st.sidebar.caption("Tip: stand still ~1s at start to calibrate 1g cleanly.")

st.sidebar.markdown("---")
st.sidebar.subheader("Sensor scales")
ACC_COUNTS_PER_G = st.sidebar.number_input(
    "Accelerometer counts per g",
    min_value=1000, max_value=40000, value=16384, step=256,
    help="Set this to your IMU's accel LSB/g (e.g., ±2g→16384, ±4g→8192, ±8g→4096)."
)
GYRO_COUNTS_PER_DPS = st.sidebar.number_input(
    "Gyro counts per °/s",
    min_value=10, max_value=1000, value=131, step=1,
    help="Set this to your gyro LSB/(deg/s) (e.g., ±250°/s→131, ±500→65.5)."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Madgwick (imufusion.Ahrs)")
MADGWICK_BETA = st.sidebar.slider(
    "Gain β (higher = more accel influence)",
    min_value=0.01, max_value=2.0, value=0.10, step=0.01
)
IMU_SAMPLERATE_HINT = st.sidebar.slider(
    "Sampling rate hint (Hz)",
    min_value=10, max_value=400, value=50, step=5,
    help="Used for internal recovery timing and bias filter. dt still comes from timestamps."
)
GYRO_RANGE_DPS = st.sidebar.select_slider(
    "Gyro range (±°/s)",
    options=[125, 250, 500, 1000, 2000],
    value=250
)
ACC_REJECTION = st.sidebar.slider(
    "Acceleration rejection",
    min_value=0.0, max_value=10.0, value=2.0, step=0.1,
    help="Higher rejects more accel transients from affecting orientation."
)
RECOVERY_TIME_S = st.sidebar.slider(
    "Recovery trigger time (s)",
    min_value=1.0, max_value=10.0, value=5.0, step=0.5,
    help="How long of good data before re-enabling rejected sensors."
)
USE_GYRO_BIAS = st.sidebar.checkbox(
    "Apply gyro bias correction",
    value=True,
    help="Estimates & removes slowly-varying gyro bias."
)


def ensure_session_state():
    d = st.session_state
    d.setdefault("is_running", False)
    d.setdefault("accel_path", None)
    d.setdefault("gyro_path", None)
    d.setdefault("jump_snaps", [])       # list of snapshots
    d.setdefault("jump_count", 0)
    d.setdefault("was_in_air", False)
    d.setdefault("freefall_counter", 0)
    d.setdefault("g_baseline", None)
    d.setdefault("baseline_ready", False)
    d.setdefault("baseline_buffer", deque(maxlen=BASELINE_SAMPLES))
    d.setdefault("pending_snaps", deque())  # queue of events awaiting snapshot
    d.setdefault("open_jumps", [])

    # Madgwick AHRS + gyro bias + timing
    if "ahrs" not in d:
        d.ahrs = imufusion.Ahrs()
    d.setdefault("gyro_offset", imufusion.Offset(IMU_SAMPLERATE_HINT))
    d.setdefault("last_time", None)

ensure_session_state()

def configure_ahrs():
    # Handle different imufusion APIs for convention and settings signature
    convention = getattr(imufusion, "CONVENTION_NWU", None)
    if convention is None:
        Convention = getattr(imufusion, "Convention", None)
        convention = getattr(Convention, "NorthWestUp", 0) if Convention else 0

    recovery_trigger = int(RECOVERY_TIME_S * IMU_SAMPLERATE_HINT)

    try:
        # Newer API: includes convention and gyroscope range
        st.session_state.ahrs.settings = imufusion.Settings(
            convention,
            float(MADGWICK_BETA),
            float(GYRO_RANGE_DPS),
            float(ACC_REJECTION),
            0.0,  # magnetic rejection off
            recovery_trigger
        )
    except TypeError:
        # Older API: (gain, acc_rejection, mag_rejection, recovery_trigger)
        st.session_state.ahrs.settings = imufusion.Settings(
            float(MADGWICK_BETA),
            float(ACC_REJECTION),
            0.0,
            recovery_trigger
        )

configure_ahrs()


def start_new_csv_run():
    run_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.session_state.accel_path = f"Accelerometer_{run_tag}.csv"
    st.session_state.gyro_path  = f"Gyroscope_{run_tag}.csv"

    pd.DataFrame(columns=['timestamp','acc_x','acc_y','acc_z']).to_csv(
        st.session_state.accel_path, index=False
    )
    pd.DataFrame(columns=['timestamp','gyro_x','gyro_y','gyro_z']).to_csv(
        st.session_state.gyro_path, index=False
    )


def parse_sensor_data(data: bytes):
    """
    Assumes payload: 6 bytes accel/gyro as >hhh (big-endian 16-bit signed) + 3 bytes timestamp (u24)
    Returns (x, y, z, ts)
    """
    x, y, z = struct.unpack('>hhh', data[0:6])
    ts = int.from_bytes(data[6:9], byteorder='big', signed=False)
    return x, y, z, ts


# QUATERNION HELPERS

def quat_conj(q):
    w, x, y, z = q
    return (w, -x, -y, -z)

def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw
    )

def rotate_vec_by_quat(v, q):
    """Rotate 3D vector v by quaternion q (w,x,y,z)."""
    vq = (0.0, v[0], v[1], v[2])
    qc = quat_conj(q)
    return quat_mul(quat_mul(q, vq), qc)[1:]  # drop w

def _quat_to_tuple(q):
    """Return (w,x,y,z) from imufusion quaternion or array-like."""
    try:
        return (float(q.w), float(q.x), float(q.y), float(q.z))
    except Exception:
        q = np.array(q, dtype=float).reshape(-1)
        if q.size >= 4:
            return (q[0], q[1], q[2], q[3])
        return (1.0, 0.0, 0.0, 0.0)

# SNAPSHOT (mark takeoff and landing)

def _nearest_idx_by_time(df: pd.DataFrame, target_time: datetime) -> int:
    return (df["Time"] - target_time).abs().idxmin()

def capture_jump_snapshot(
    accel_df: pd.DataFrame,
    takeoff_time: datetime,
    landing_time: datetime,
    height_m: float | None,
    window_s: float = 4.0
):
    if accel_df.empty or takeoff_time is None or landing_time is None:
        return

    center_time = takeoff_time + (landing_time - takeoff_time) / 2
    tof_ms = (landing_time - takeoff_time).total_seconds() * 1000.0

    half = window_s / 2.0
    t_start = center_time - timedelta(seconds=half)
    t_end   = center_time + timedelta(seconds=half)

    recent = accel_df[(accel_df["Time"] >= t_start) & (accel_df["Time"] <= t_end)]
    if recent.shape[0] < 20:
        recent = accel_df.tail(150)

    idx_take = _nearest_idx_by_time(recent, takeoff_time)
    take_row = recent.loc[idx_take]
    take_t = take_row["Time"]
    take_z = take_row["AccZ_vert"]

    idx_land = _nearest_idx_by_time(recent, landing_time)
    land_row = recent.loc[idx_land]
    land_t = land_row["Time"]
    land_z = land_row["AccZ_vert"]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    # Raw (counts) faint
    # ax.plot(recent["Time"], recent["AccX"], label="AccX raw", alpha=0.3, linewidth=1)
    # ax.plot(recent["Time"], recent["AccY"], label="AccY raw", alpha=0.3, linewidth=1)
    # ax.plot(recent["Time"], recent["AccZ"], label="AccZ raw", alpha=0.3, linewidth=1)
    # Linear (m/s^2) bold

    ax.plot(recent["Time"], recent["AccX_lin"], label="AccX linear (m/s²)", linewidth=1.8)
    ax.plot(recent["Time"], recent["AccY_lin"], label="AccY linear (m/s²)", linewidth=1.8)
    ax.plot(recent["Time"], recent["AccZ_lin"], label="AccZ linear (m/s²)", linewidth=1.8)
    ax.plot(recent["Time"], recent["AccZ_vert"], label="AccZ vertical (m/s²)", linewidth=1.8)

    ax.axvline(center_time, linestyle="--", linewidth=1)
    ax.axvline(take_t, linestyle="--", linewidth=1)
    ax.plot([take_t], [take_z], marker="o",markersize=4.0)
    ax.text(take_t, take_z, "  Takeoff", va="bottom", fontsize=6)

    ax.axvline(land_t, linestyle="--", linewidth=1)
    ax.plot([land_t], [land_z], marker="o",markersize=4.0)
    ax.text(land_t, land_z, "  Landing", va="bottom", fontsize=6)

    title_bits = [
        f"Jump snapshot @ {takeoff_time.strftime('%H:%M:%S.%f')[:-3]}",
        f"TOF ≈ {tof_ms:.0f} ms"
    ]
    if height_m is not None:
        title_bits.append(f"Height ≈ {height_m*100:.1f} cm")
    ax.set_title(" | ".join(title_bits))

    ax.set_xlabel("Time")
    ax.set_ylabel("Accel")
    # ZOOM
    x_lo = takeoff_time - timedelta(seconds=1)
    x_hi = landing_time + timedelta(seconds=1)
    ax.set_xlim(x_lo, x_hi)
    ax.legend(loc="upper left",markerscale=0.8, frameon=False)

    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    png_bytes = buf.getvalue()

    st.session_state.jump_snaps.append({
        "png": png_bytes,
        "when": center_time,
        "accz": float(take_z),
        "height_m": height_m,
        "tof_ms": tof_ms
    })
    if len(st.session_state.jump_snaps) > 20:
        st.session_state.jump_snaps = st.session_state.jump_snaps[-20:]


# JUMP DETECTION (freefall) — measured |a| in g

def _maybe_update_baseline(ax_g, ay_g, az_g):
    if st.session_state.baseline_ready:
        return
    mag_g = math.sqrt(ax_g*ax_g + ay_g*ay_g + az_g*az_g)
    st.session_state.baseline_buffer.append(mag_g)
    if len(st.session_state.baseline_buffer) == st.session_state.baseline_buffer.maxlen:
        st.session_state.g_baseline = sum(st.session_state.baseline_buffer) / len(st.session_state.baseline_buffer)
        st.session_state.baseline_ready = True

def detect_activity(measured_g_tuple, accel_data_list):
    ax_g, ay_g, az_g = measured_g_tuple

    if not st.session_state.baseline_ready:
        _maybe_update_baseline(ax_g, ay_g, az_g)
        return "Calibrating baseline...", False, None, None

    acc_mag_g = math.sqrt(ax_g*ax_g + ay_g*ay_g + az_g*az_g)
    freefall_threshold_g = FREEFALL_G_THRESHOLD * st.session_state.g_baseline

    if acc_mag_g < freefall_threshold_g:
        st.session_state.freefall_counter += 1
    else:
        st.session_state.freefall_counter = 0

    in_air = st.session_state.freefall_counter >= DEBOUNCE_SAMPLES

    if in_air:
        activity = "Jumping (in air)"
    else:
        # Use linear Z amplitude (m/s^2) to classify floor activity
        recent = accel_data_list[-10:] if len(accel_data_list) >= 10 else accel_data_list
        z_vals = [row[6] for row in recent]  # lin_az_mps2
        amp = (max(z_vals) - min(z_vals)) if z_vals else 0.0
        if amp > 8.0:        # ~0.8 g
            activity = "Active/Running"
        elif amp > 3.0:
            activity = "Light motion/Walking"
        else:
            activity = "Standing/Still"

    return activity, in_air, acc_mag_g, freefall_threshold_g



async def collect_and_plot_data():
    st.write("Connecting to BLE device...")
    try:
        async with BleakClient(BLE_ADDRESS) as client:
            st.success("Connected! Starting data collection.")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Live Accelerometer (raw + linear m/s², gravity removed)")
                
                accel_chart_placeholder = st.empty()
                accel_data_list = []
            with col2:
                st.subheader("Live Gyroscope (raw counts)")
                gyro_chart_placeholder = st.empty()
                gyro_data_list = []   # [Time, GyroX, GyroY, GyroZ] 

            activity_placeholder = st.empty()
            jump_placeholder     = st.empty()
            info_placeholder     = st.empty()

            # Reset detector & fusion state
            st.session_state.was_in_air = False
            st.session_state.freefall_counter = 0
            st.session_state.g_baseline = None
            st.session_state.baseline_ready = False
            st.session_state.baseline_buffer = deque(maxlen=BASELINE_SAMPLES)
            st.session_state.pending_snaps = deque()
            st.session_state.open_jumps = []
            st.session_state.last_time = None
            st.session_state.ahrs = imufusion.Ahrs()  # fresh instance
            configure_ahrs()
            st.session_state.gyro_offset = imufusion.Offset(IMU_SAMPLERATE_HINT)

            max_points = 300
            SLEEP_S = 0.05  # ~20 Hz

            while st.session_state.is_running:
                
                try:
                    accel_data_bytes, gyro_data_bytes = await asyncio.gather(
                        client.read_gatt_char(ACCELEROMETER_UUID),
                        client.read_gatt_char(GYRO_UUID)
                    )
                except (asyncio.CancelledError,) as e:
                    # Streamlit rerun or explicit task cancel
                    break
                except (BleakError, OSError) as e:
                    if _is_error_cancelled(e):
                        # user/device cancelled: exit loop gracefully
                        st.info("BLE operation cancelled. Stopping stream.")
                        break
                    else:
                        # real error: surface it
                        raise

           
                ax_c, ay_c, az_c, _ats = parse_sensor_data(accel_data_bytes)
                gx_c, gy_c, gz_c, _gts = parse_sensor_data(gyro_data_bytes)
                now = datetime.now()

                # Convert to physical units for fusion
                ax_g = ax_c / float(ACC_COUNTS_PER_G)
                ay_g = ay_c / float(ACC_COUNTS_PER_G)
                az_g = az_c / float(ACC_COUNTS_PER_G)

                gx_dps = gx_c / float(GYRO_COUNTS_PER_DPS)
                gy_dps = gy_c / float(GYRO_COUNTS_PER_DPS)
                gz_dps = gz_c / float(GYRO_COUNTS_PER_DPS)

                # Pack as numpy arrays (required by imufusion)
                gyro_vec = np.array([gx_dps, gy_dps, gz_dps], dtype=float)
                acc_vec  = np.array([ax_g,  ay_g,  az_g ], dtype=float)

                # Optional gyro bias correction (expects/returns numpy arrays)
                if USE_GYRO_BIAS:
                    gyro_vec = st.session_state.gyro_offset.update(gyro_vec)

                # Delta time
                if st.session_state.last_time is None:
                    dt = 1.0 / IMU_SAMPLERATE_HINT
                else:
                    dt = (now - st.session_state.last_time).total_seconds()
                    dt = max(1.0 / 400.0, min(dt, 1.0 / 5.0))  # clamp
                st.session_state.last_time = now

                # Madgwick update (no magnetometer) — expects numpy arrays
                st.session_state.ahrs.update_no_magnetometer(gyro_vec, acc_vec, dt)
                q = _quat_to_tuple(st.session_state.ahrs.quaternion)  # (w,x,y,z)

                # Gravity vector in body frame (unit g) from quaternion
                g_body = rotate_vec_by_quat((0.0, 0.0, -1.0), q)

                # Linear acceleration (m/s^2): (measured - gravity)*g
                lin_ax = (ax_g - g_body[0]) * G_CONST
                lin_ay = (ay_g - g_body[1]) * G_CONST
                lin_az = (az_g - g_body[2]) * G_CONST
                # world-vertical linear acc (m/s^2)
# Our rotate_vec_by_quat(v, q) applies q * v * q_conj (world->body).
# To rotate body->world, use q_conj.
                lin_body_g   = (lin_ax / G_CONST, lin_ay / G_CONST, lin_az / G_CONST)  # unit g
                lin_world_g  = rotate_vec_by_quat(lin_body_g, quat_conj(q))             # back to world
                lin_world_ms = tuple(v * G_CONST for v in lin_world_g)                  # m/s^2
                vert_acc_ms2 = lin_world_ms[2]  # Zup (NWU convention)


                # Append to buffers
                mag_g = math.sqrt(ax_g*ax_g + ay_g*ay_g + az_g*az_g)
                accel_data_list.append([now, ax_c, ay_c, az_c, lin_ax, lin_ay, lin_az, mag_g,vert_acc_ms2])
                gyro_data_list.append([now, gx_c, gy_c, gz_c])  # keep raw counts in chart/CSV

                if len(accel_data_list) > max_points:
                    accel_data_list = accel_data_list[-max_points:]
                if len(gyro_data_list) > max_points:
                    gyro_data_list = gyro_data_list[-max_points:]

                # DataFrames
                accel_df = pd.DataFrame(
                    accel_data_list,
                    columns=['Time','AccX','AccY','AccZ','AccX_lin','AccY_lin','AccZ_lin','Mag_g','AccZ_vert']
                )
                gyro_df  = pd.DataFrame(gyro_data_list,  columns=['Time','GyroX','GyroY','GyroZ'])

               
                activity, in_air, acc_mag_g, thr_g = detect_activity((ax_g, ay_g, az_g), accel_data_list)

                # Rising edge (takeoff)
                if in_air and not st.session_state.was_in_air:
                    st.session_state.jump_count += 1
                    event = {
                        "takeoff": now,
                        "takeoff_ns": time.perf_counter_ns(),
                        "landing": None,
                        "landing_ns": None,
                        "height_m": None,
                        "snap_done": False
                    }
                    st.session_state.open_jumps.append(event)
                    st.session_state.pending_snaps.append(event)

                # Falling edge (landing)
                if (not in_air) and st.session_state.was_in_air:
                    for ev in reversed(st.session_state.open_jumps):
                        if ev["landing"] is None:
                            ev["landing"] = now
                            ev["landing_ns"] = time.perf_counter_ns()

                            #time of flight computation
                            tof_s1=(ev["landing"]-ev["takeoff"]).total_seconds()

                            dt_ns = ev["landing_ns"] - ev["takeoff_ns"]
                            tof_s  = max(0.0, dt_ns / 1e9)
                            tof_ms = tof_s * 1000.0

                            #height computation 
                            if 160.0 <= tof_ms <= 2000.0:
                                ev["height_m"] = (G_CONST * (tof_s1 ** 2)) / 8.0
                            else:
                                ev["height_m"] = None
                            break

                st.session_state.was_in_air = in_air

                
                while st.session_state.pending_snaps:
                    ev = st.session_state.pending_snaps[0]
                    delay_ok = (now - ev["takeoff"]).total_seconds() >= SNAPSHOT_DELAY_S
                    landing_ok = ev["landing"] is not None
                    if delay_ok and landing_ok and not ev["snap_done"]:
                        capture_jump_snapshot(
                            accel_df,
                            takeoff_time=ev["takeoff"],
                            landing_time=ev["landing"],
                            height_m=ev["height_m"],
                            window_s=SNAPSHOT_WINDOW_S
                        )
                        ev["snap_done"] = True
                        st.session_state.pending_snaps.popleft()
                    else:
                        break

               
                activity_placeholder.info(f"Current Activity: {activity}")
                jump_placeholder.info(f"Jump Count: {st.session_state.jump_count}")

                if st.session_state.baseline_ready and acc_mag_g is not None:
                    g_est = st.session_state.g_baseline
                    info_placeholder.caption(
                        f"|a| measured: {acc_mag_g:.2f} g  •  1g baseline: {g_est:.2f} g  •  free-fall thr: {thr_g:.2f} g"
                    )
                else:
                    info_placeholder.caption("Calibrating 1g baseline… stand still briefly.")

                # Save CSV
                if st.session_state.accel_path:
                    pd.DataFrame([[now.isoformat(), ax_c, ay_c, az_c]],
                                 columns=['timestamp','acc_x','acc_y','acc_z']
                                 ).to_csv(st.session_state.accel_path, mode='a', header=False, index=False)
                if st.session_state.gyro_path:
                    pd.DataFrame([[now.isoformat(), gx_c, gy_c, gz_c]],
                                 columns=['timestamp','gyro_x','gyro_y','gyro_z']
                                 ).to_csv(st.session_state.gyro_path, mode='a', header=False, index=False)

                # Live charts
                with col1:
                    accel_chart_placeholder.line_chart(
                        accel_df.set_index('Time')[['AccX_lin','AccY_lin','AccZ_lin','AccZ_vert']]
                    )
                with col2:
                    gyro_chart_placeholder.line_chart(gyro_df.set_index('Time'))

                await asyncio.sleep(SLEEP_S)

    except Exception as e:
        if _is_error_cancelled(e):
            st.info("Operation cancelled by user/device (ERROR_CANCELLED).")
        else:
            st.error(f"An error occurred: {e}")
    finally:
        st.session_state.is_running = False
        st.info("Disconnected from BLE device.")


st.title("Live Sensor Data")
st.write("Press **Start Sensor** and stand still briefly for baseline calibration (1g).")

colA, colB, colC, colD = st.columns(4)
with colA:
    start_button = st.button("Start Sensor")
with colB:
    stop_button  = st.button("Stop Sensor")
with colC:
    new_csv_button = st.button("New CSV")
with colD:
    recal_button = st.button("Re-calibrate 1g / reset fusion")

if new_csv_button:
    start_new_csv_run()
    st.session_state.jump_snaps = []
    st.session_state.jump_count = 0
    st.success(f"Created new files:\n{st.session_state.accel_path}\n{st.session_state.gyro_path}")

if recal_button:
    # Reset baseline and fusion
    st.session_state.g_baseline = None
    st.session_state.baseline_ready = False
    st.session_state.baseline_buffer = deque(maxlen=BASELINE_SAMPLES)
    st.session_state.ahrs = imufusion.Ahrs()
    configure_ahrs()
    st.session_state.gyro_offset = imufusion.Offset(IMU_SAMPLERATE_HINT)
    st.session_state.last_time = None
    st.info("Baseline & Madgwick reset. Stand still for ~1s.")

if start_button:
    if not st.session_state.get('accel_path'):
        st.session_state.accel_path = DEFAULT_ACCEL
    if not st.session_state.get('gyro_path'):
        st.session_state.gyro_path = DEFAULT_GYRO
    st.session_state.is_running = True
    asyncio.run(collect_and_plot_data())

if stop_button:
    st.session_state.is_running = False

if st.session_state.is_running:
    st.success("Sensor data streaming…")
else:
    st.info("Ready. Press **Start Sensor** to begin.")

st.markdown("---")
st.subheader("Jump Snapshots")
if len(st.session_state.jump_snaps) == 0:
    st.caption("No jumps captured yet.")
else:
    for snap in st.session_state.jump_snaps:
        when = snap["when"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        st.image(snap["png"], caption=f"Snapshot centered @ {when}")
        details = []
        if snap.get("tof_ms") is not None:
            details.append(f"**TOF:** {snap['tof_ms']:.0f} ms")
        if snap.get("height_m") is not None:
            details.append(f"**Estimated height:** {snap['height_m']*100:.1f} cm")
        if details:
            st.markdown(" • ".join(details))
