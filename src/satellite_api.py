import rasterio
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pystac_client
import planetary_computer
import requests


def calculate_ndvi(nir_band: np.ndarray, red_band: np.ndarray) -> np.ndarray:
    """Calculates Normalized Difference Vegetation Index (NDVI) for overall plant health."""
    return (nir_band - red_band) / (nir_band + red_band + 1e-6)


def calculate_savi(nir_band: np.ndarray, red_band: np.ndarray, L: float = 0.5) -> np.ndarray:
    """Calculates Soil Adjusted Vegetation Index (SAVI) ignoring bare soil background."""
    return ((nir_band - red_band) / (nir_band + red_band + L)) * (1.0 + L)


def calculate_ndwi(nir_band: np.ndarray, swir_band: np.ndarray) -> np.ndarray:
    """Calculates Normalized Difference Water Index (NDWI) for leaf moisture / drought stress."""
    return (nir_band - swir_band) / (nir_band + swir_band + 1e-6)


def download_sentinel_bands(lat: float, lon: float, output_dir: str = "data"):
    """
    Queries Microsoft Planetary Computer STAC API for Sentinel-2 satellite tiles
    and downloads RED (B04), NIR (B08), and SWIR (B11) bands.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    nir_file = out_path / f"farm_{lat}_{lon}_NIR.tif"
    red_file = out_path / f"farm_{lat}_{lon}_RED.tif"
    swir_file = out_path / f"farm_{lat}_{lon}_SWIR.tif"

    # If all 3 files already exist locally, skip downloading
    if nir_file.exists() and red_file.exists() and swir_file.exists():
        print("✅ All satellite bands (NIR, RED, SWIR) are already present locally.")
        return

    print(f"📡 Querying Sentinel-2 catalog for coordinates ({lat}, {lon})...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    # Search for Sentinel-2 L2A images over the bounding point with low cloud cover
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        query={"eo:cloud_cover": {"lt": 15}},
        max_items=1,
    )

    items = list(search.item_collection())
    if not items:
        raise RuntimeError("No cloud-free Sentinel-2 imagery found for this location.")

    item = items[0]
    print(f"📅 Found satellite scene captured on: {item.datetime.strftime('%Y-%m-%d')}")

    # Map bands to our target local filenames
    band_map = {
        "B04": red_file,   # RED band
        "B08": nir_file,   # NIR band
        "B11": swir_file   # SWIR band (Short-Wave Infrared for water content)
    }

    for band_key, target_file in band_map.items():
        if not target_file.exists():
            print(f"⬇️ Downloading {band_key} ({target_file.name})...")
            asset_url = item.assets[band_key].href
            
            # Stream download to handle large geotiff files cleanly
            with requests.get(asset_url, stream=True) as r:
                r.raise_for_status()
                with open(target_file, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"💾 Saved {target_file.name}")


def process_farm_indices(lat: float, lon: float, data_dir: str = "data") -> dict:
    """
    Loads raw satellite band .tif files for a location and calculates all 3 vegetation indices:
    NDVI (Health), SAVI (Soil-Adjusted Health), and NDWI (Moisture/Drought).
    """
    base_path = Path(data_dir)
    nir_file = base_path / f"farm_{lat}_{lon}_NIR.tif"
    red_file = base_path / f"farm_{lat}_{lon}_RED.tif"
    swir_file = base_path / f"farm_{lat}_{lon}_SWIR.tif"

    # Read NIR and RED bands
    with rasterio.open(nir_file) as src:
        nir = src.read(1).astype('float32')

    with rasterio.open(red_file) as src:
        red = src.read(1).astype('float32')

    # Compute NDVI and SAVI
    indices = {
        "ndvi": calculate_ndvi(nir, red),
        "savi": calculate_savi(nir, red)
    }

    # Compute NDWI if SWIR band exists
    # Compute NDWI if SWIR band exists
    if swir_file.exists():
        with rasterio.open(swir_file) as src:
            # Resample SWIR band to match the exact resolution/shape of the NIR band
            swir = src.read(
                1,
                out_shape=nir.shape,
                resampling=Resampling.bilinear
            ).astype('float32')
            
        indices["ndwi"] = calculate_ndwi(nir, swir)
    else:
        print(f"⚠️ Warning: {swir_file.name} not found. Skipping NDWI calculation.")

    return indices


if __name__ == "__main__":
    # Test coordinates
    test_lat, test_lon = 36.7783, -119.4179
    data_directory = "data" if Path("data").exists() else "../data"

    # 1. Download missing bands (including SWIR)
    download_sentinel_bands(test_lat, test_lon, output_dir=data_directory)

    # 2. Process and compute all 3 indices
    print(f"\n⚙️ Processing satellite indices for farm at ({test_lat}, {test_lon})...")
    results = process_farm_indices(test_lat, test_lon, data_dir=data_directory)

    print("\n✅ Successfully calculated indices:")
    for key in results.keys():
        print(f"  - {key.upper()} matrix ready with shape {results[key].shape}")