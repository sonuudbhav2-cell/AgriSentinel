import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


def fetch_historical_weather(
    lat: float, 
    lon: float, 
    days_back: int = 30, 
    output_dir: str = "data"
) -> pd.DataFrame:
    """
    Fetches daily historical weather data for the last N days using Open-Meteo API.
    Returns a cleaned Pandas DataFrame and saves it to CSV.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_file = out_path / f"weather_{lat}_{lon}_{days_back}d.csv"

    # Calculate date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)

    print(f"🌤️ Querying Open-Meteo API for ({lat}, {lon}) from {start_date} to {end_date}...")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "relative_humidity_2m_mean",
            "wind_speed_10m_max"
        ],
        "timezone": "auto"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    # Extract daily structured metrics
    daily_data = data.get("daily", {})
    df = pd.DataFrame({
        "date": pd.to_datetime(daily_data.get("time")),
        "temp_max_c": daily_data.get("temperature_2m_max"),
        "temp_min_c": daily_data.get("temperature_2m_min"),
        "temp_mean_c": daily_data.get("temperature_2m_mean"),
        "precipitation_mm": daily_data.get("precipitation_sum"),
        "humidity_pct": daily_data.get("relative_humidity_2m_mean"),
        "wind_speed_max_kmh": daily_data.get("wind_speed_10m_max")
    })

    # Save to local data folder
    df.to_csv(csv_file, index=False)
    print(f"💾 Weather dataset saved successfully to {csv_file.name}")
    
    return df


if __name__ == "__main__":
    # Test coordinates (same farm location)
    test_lat, test_lon = 36.7783, -119.4179
    data_directory = "data" if Path("data").exists() else "../data"

    print(f"Fetching 30-day weather history for farm at ({test_lat}, {test_lon})...")
    weather_df = fetch_historical_weather(test_lat, test_lon, days_back=30, output_dir=data_directory)
    
    print("\n📊 Weather Data Preview (First 5 Days):")
    print(weather_df.head())