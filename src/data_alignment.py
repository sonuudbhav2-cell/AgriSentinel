import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure src directory is in Python path for clean imports
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from satellite_api import download_sentinel_bands, process_farm_indices
from weather_api import fetch_historical_weather


def aggregate_satellite_features(indices: dict) -> dict:
    """
    Computes statistical summaries (mean, std, min, max, 25th/75th percentiles) 
    from 2D satellite spatial index matrices (NDVI, SAVI, NDWI).
    """
    spatial_features = {}

    for name, matrix in indices.items():
        # Filter out non-finite numbers (NaNs/Infs) if present
        valid_pixels = matrix[np.isfinite(matrix)]

        if len(valid_pixels) == 0:
            raise ValueError(f"No valid pixels found for index: {name}")

        spatial_features[f"{name}_mean"] = float(np.mean(valid_pixels))
        spatial_features[f"{name}_std"] = float(np.std(valid_pixels))
        spatial_features[f"{name}_min"] = float(np.min(valid_pixels))
        spatial_features[f"{name}_max"] = float(np.max(valid_pixels))
        spatial_features[f"{name}_p25"] = float(np.percentile(valid_pixels, 25))
        spatial_features[f"{name}_p75"] = float(np.percentile(valid_pixels, 75))

    return spatial_features


def compute_rolling_weather_features(
    weather_df: pd.DataFrame, 
    capture_date: str
) -> dict:
    """
    Computes cumulative and rolling weather statistics leading up to the 
    satellite scene capture date (e.g., 3-day, 7-day, 14-day rainfall, temp, humidity).
    """
    weather_df["date"] = pd.to_datetime(weather_df["date"])
    target_dt = pd.to_datetime(capture_date)

    # Subset weather records on or before capture date
    past_weather = weather_df[weather_df["date"] <= target_dt].sort_values("date")

    if past_weather.empty:
        raise ValueError(f"No weather records found on or before capture date: {capture_date}")

    def get_last_n_days(df, days):
        cutoff = target_dt - pd.Timedelta(days=days - 1)
        return df[df["date"] >= cutoff]

    w_3d = get_last_n_days(past_weather, 3)
    w_7d = get_last_n_days(past_weather, 7)
    w_14d = get_last_n_days(past_weather, 14)

    weather_features = {
        "capture_date": capture_date,
        # Precipitation rolling sums
        "rain_3d_sum_mm": float(w_3d["precipitation_mm"].sum()),
        "rain_7d_sum_mm": float(w_7d["precipitation_mm"].sum()),
        "rain_14d_sum_mm": float(w_14d["precipitation_mm"].sum()),
        # Temperature statistics
        "temp_3d_mean_c": float(w_3d["temp_mean_c"].mean()),
        "temp_7d_mean_c": float(w_7d["temp_mean_c"].mean()),
        "temp_14d_mean_c": float(w_14d["temp_mean_c"].mean()),
        "temp_7d_max_c": float(w_7d["temp_max_c"].max()),
        # Relative humidity statistics
        "humidity_7d_mean_pct": float(w_7d["humidity_pct"].mean()),
        "humidity_14d_mean_pct": float(w_14d["humidity_pct"].mean()),
        # Wind speed
        "wind_7d_max_kmh": float(w_7d["wind_speed_max_kmh"].max())
    }

    return weather_features


def align_farm_data(
    lat: float, 
    lon: float, 
    capture_date: str = "2026-07-24", 
    data_dir: str = "data"
) -> pd.DataFrame:
    """
    Fuses spatial satellite features with temporal weather features for a given farm location.
    Saves and returns a single aligned Master Feature record.
    """
    out_path = Path(data_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    master_csv = out_path / f"master_dataset_{lat}_{lon}.csv"

    print(f"⚙️ Aligning spatial & temporal features for farm at ({lat}, {lon})...")

    # 1. Fetch & calculate spatial satellite matrices
    download_sentinel_bands(lat, lon, output_dir=data_dir)
    indices = process_farm_indices(lat, lon, data_dir=data_dir)
    spatial_feats = aggregate_satellite_features(indices)

    # 2. Fetch & calculate temporal weather features
    weather_df = fetch_historical_weather(lat, lon, days_back=30, output_dir=data_dir)
    weather_feats = compute_rolling_weather_features(weather_df, capture_date=capture_date)

    # 3. Fuse metadata + temporal weather + spatial satellite into a single record
    fused_record = {
        "latitude": lat,
        "longitude": lon,
        **weather_feats,
        **spatial_feats
    }

    df_master = pd.DataFrame([fused_record])

    # Save to CSV (append or create)
    if master_csv.exists():
        df_existing = pd.read_csv(master_csv)
        df_combined = pd.concat([df_existing, df_master]).drop_duplicates(
            subset=["latitude", "longitude", "capture_date"], keep="last"
        )
        df_combined.to_csv(master_csv, index=False)
    else:
        df_master.to_csv(master_csv, index=False)

    print(f"💾 Fused master record saved successfully to {master_csv.name}\n")
    return df_master


if __name__ == "__main__":
    test_lat, test_lon = 36.7783, -119.4179
    # Capture date corresponding to satellite pass seen in logs
    capture_dt = "2026-07-24"
    data_directory = "data" if Path("data").exists() else "../data"

    master_df = align_farm_data(test_lat, test_lon, capture_date=capture_dt, data_dir=data_directory)

    print("🚀 Fused Master Feature Vector:")
    print("=" * 45)
    print(master_df.T.to_string(header=False))