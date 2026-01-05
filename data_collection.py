import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

# ===============================
# Kalman Filter (1D Velocity)
# ===============================
class Kalman1D:
    def __init__(self, q=0.05, r=4.0):
        self.x = None
        self.p = 1.0
        self.q = q
        self.r = r

    def update(self, z):
        if self.x is None:
            self.x = z
            return z
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x += k * (z - self.x)
        self.p *= (1 - k)
        return self.x


# ===============================
# Load CSVs
# ===============================
def load_data(files):
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["source"] = Path(f).name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# ===============================
# Basic Error Metrics
# ===============================
def compute_error(df):
    df["err"] = df["Radar_Doppler_Signed_kmh"] - df["GT_Speed_kmh"]
    return df


# ===============================
# Save 01: Raw + Error
# ===============================
def save_raw_with_error(df, outdir):
    path = outdir / "01_raw_with_error.csv"
    df.to_csv(path, index=False)
    print(f"[SAVE] {path}")


# ===============================
# Save 02: Per-Vehicle Error
# ===============================
def save_vehicle_error(df, outdir):
    grp = df.groupby("VehicleID")
    out = pd.DataFrame({
        "Samples": grp.size(),
        "Bias_kmh": grp["err"].mean(),
        "RMSE_kmh": grp["err"].apply(lambda x: np.sqrt(np.mean(x**2)))
    }).reset_index()

    path = outdir / "02_per_vehicle_error.csv"
    out.to_csv(path, index=False)
    print(f"[SAVE] {path}")
    return out


# ===============================
# Save 03: Distance-bin Error
# ===============================
def save_distance_error(df, outdir, bin_m=10.0):
    df["dist_bin"] = (df["Distance_m"] // bin_m) * bin_m
    grp = df.groupby("dist_bin")

    out = pd.DataFrame({
        "DistanceBin_m": grp["Distance_m"].mean(),
        "Bias_kmh": grp["err"].mean(),
        "RMSE_kmh": grp["err"].apply(lambda x: np.sqrt(np.mean(x**2))),
        "Samples": grp.size()
    }).reset_index(drop=True)

    path = outdir / "03_distance_bin_error.csv"
    out.to_csv(path, index=False)
    print(f"[SAVE] {path}")
    return out


# ===============================
# Kalman Filter
# ===============================
def apply_kalman_filter(df):
    df = df.sort_values("Time")
    filtered = []
    kf_map = {}

    for _, row in df.iterrows():
        vid = row["VehicleID"]
        if vid not in kf_map:
            kf_map[vid] = Kalman1D()

        v_filt = kf_map[vid].update(row["Radar_Doppler_Signed_kmh"])
        filtered.append(v_filt)

    df["RadarSpeed_KF_kmh"] = filtered
    df["err_kf"] = df["RadarSpeed_KF_kmh"] - df["GT_Speed_kmh"]
    return df


# ===============================
# Save 04: Kalman Timeseries
# ===============================
def save_kalman_timeseries(df, outdir):
    cols = [
        "Time", "VehicleID", "Distance_m",
        "GT_Speed_kmh",
        "Radar_Doppler_Signed_kmh",
        "RadarSpeed_KF_kmh",
        "err", "err_kf"
    ]
    out = df[cols]
    path = outdir / "04_kalman_filtered_timeseries.csv"
    out.to_csv(path, index=False)
    print(f"[SAVE] {path}")


# ===============================
# Visualization (선택)
# ===============================
def plot_kalman_effect(df):
    rmse_raw = np.sqrt(np.mean(df["err"]**2))
    rmse_kf  = np.sqrt(np.mean(df["err_kf"]**2))

    plt.figure(figsize=(5,4))
    plt.bar(["Raw Radar", "Kalman Filtered"], [rmse_raw, rmse_kf])
    plt.ylabel("RMSE (km/h)")
    plt.title("Kalman Filter Effect")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

    print(f"[Kalman] RMSE raw = {rmse_raw:.2f} km/h")
    print(f"[Kalman] RMSE KF  = {rmse_kf:.2f} km/h")


# ===============================
# Main
# ===============================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python radar_speed_analysis_all.py raw_log.csv")
        sys.exit(0)

    outdir = Path("analysis_output")
    outdir.mkdir(exist_ok=True)

    df = load_data(sys.argv[1:])
    df = compute_error(df)

    print(f"[INFO] Loaded {len(df)} samples")

    # 01 raw + error
    save_raw_with_error(df, outdir)

    # 02 vehicle-based
    save_vehicle_error(df, outdir)

    # 03 distance-based
    save_distance_error(df, outdir)

    # 04 kalman
    df = apply_kalman_filter(df)
    save_kalman_timeseries(df, outdir)

    # optional plot
    plot_kalman_effect(df)
