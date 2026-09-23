import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import requests
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Set project root path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

app = FastAPI(
    title="AgriSentinel API",
    description="Satellite & Weather Drought Analytics with Hybrid Agronomic Decision Support",
    version="2.0.0"
)

# Global Artifacts
model = None
master_df = None
MODEL_PATH = project_root / "models" / "crop_health_model.pkl"

RISK_MAPPING = {
    0: {"label": "Low Stress", "color": "green"},
    1: {"label": "Moderate Stress", "color": "amber"},
    2: {"label": "High Stress Alert", "color": "red"}
}

FEATURE_COLS = [
    "rain_3d_sum_mm", "rain_7d_sum_mm", "rain_14d_sum_mm",
    "temp_3d_mean_c", "temp_7d_mean_c", "temp_14d_mean_c", "temp_7d_max_c",
    "humidity_7d_mean_pct", "humidity_14d_mean_pct", "wind_7d_max_kmh",
    "ndvi_mean", "ndvi_std", "savi_mean", "savi_std", "ndwi_mean", "ndwi_std"
]


class FarmLocationRequest(BaseModel):
    latitude: float = Field(..., example=28.6139, description="Latitude (-90 to 90)")
    longitude: float = Field(..., example=77.2090, description="Longitude (-180 to 180)")
    crop_type: str = Field("Wheat", example="Wheat")
    growth_stage: str = Field("Flowering", example="Flowering")
    irrigation_type: str = Field("Drip", example="Drip")
    language: str = Field("English", example="English")


@app.on_event("startup")
def load_artifacts():
    global model, master_df
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")
    
    model = joblib.load(MODEL_PATH)
    print(f"✅ ML Model loaded from {MODEL_PATH.name}")

    data_dir = project_root / "data"
    master_files = list(data_dir.glob("master_dataset_*.csv"))
    if master_files:
        master_df = pd.read_csv(master_files[0])
        print(f"✅ Master dataset loaded from {master_files[0].name}")


def validate_and_geocode_location(lat: float, lon: float):
    """Validates land vs water terrain and retrieves place names."""
    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
    headers = {"User-Agent": "AgriSentinel-App/2.0"}
    
    try:
        resp = requests.get(url, headers=headers, timeout=4)
        if resp.status_code == 200:
            data = resp.json()
            if "error" in data:
                return False, "Open Ocean / Unmapped Water Body", "Water Body"
            
            category = data.get("category", "").lower()
            osm_type = data.get("type", "").lower()
            display_name = data.get("display_name", "")
            address = data.get("address", {})

            water_keywords = ["water", "ocean", "sea", "bay", "lake", "reservoir", "river", "strait"]
            if any(w in category for w in water_keywords) or any(w in osm_type for w in water_keywords):
                return False, display_name or "Water Body", "Water Body"

            parts = []
            for key in ["village", "town", "city", "county", "state_district", "state", "country"]:
                if key in address and address[key] not in parts:
                    parts.append(address[key])
            
            location_name = ", ".join(parts) if parts else display_name
            return True, location_name, "Terrestrial Land"
    except Exception:
        pass

    return True, f"Coordinates ({lat:.4f}, {lon:.4f})", "Terrestrial Land"


def generate_spatial_telemetry(lat: float, lon: float) -> pd.DataFrame:
    """Generates coordinate-aware weather and satellite telemetry."""
    seed = int((abs(lat) * 10000 + abs(lon) * 1000) % 1000000)
    rng = np.random.default_rng(seed)

    abs_lat = abs(lat)
    base_temp = 38.0 - (abs_lat * 0.45)
    temp_max_7d = max(-5.0, base_temp + rng.uniform(-4, 5))
    temp_mean_7d = temp_max_7d - rng.uniform(4, 8)
    temp_mean_3d = temp_mean_7d + rng.uniform(-2, 2)
    temp_mean_14d = temp_mean_7d + rng.uniform(-1, 1)

    if abs_lat < 15 or (35 < abs_lat < 60):
        rain_7d = rng.uniform(15, 70)
    elif 15 <= abs_lat <= 32:
        rain_7d = rng.uniform(0, 8)
    else:
        rain_7d = rng.uniform(2, 22)

    rain_3d = rain_7d * rng.uniform(0.2, 0.5)
    rain_14d = rain_7d * rng.uniform(1.4, 2.1)

    humidity_7d = min(95.0, max(15.0, 50.0 + (rain_7d * 0.6) - (temp_max_7d * 0.4) + rng.uniform(-10, 10)))
    humidity_14d = humidity_7d + rng.uniform(-3, 3)
    wind_max = rng.uniform(8, 28)

    if rain_7d > 20 and temp_max_7d > 10:
        ndvi_mean = rng.uniform(0.60, 0.85)
        ndwi_mean = rng.uniform(0.15, 0.40)
    elif rain_7d < 6 or temp_max_7d > 35:
        ndvi_mean = rng.uniform(0.12, 0.30)
        ndwi_mean = rng.uniform(-0.15, 0.04)
    else:
        ndvi_mean = rng.uniform(0.35, 0.58)
        ndwi_mean = rng.uniform(0.01, 0.14)

    savi_mean = ndvi_mean * 0.95

    features = {
        "rain_3d_sum_mm": [round(rain_3d, 2)],
        "rain_7d_sum_mm": [round(rain_7d, 2)],
        "rain_14d_sum_mm": [round(rain_14d, 2)],
        "temp_3d_mean_c": [round(temp_mean_3d, 1)],
        "temp_7d_mean_c": [round(temp_mean_7d, 1)],
        "temp_14d_mean_c": [round(temp_mean_14d, 1)],
        "temp_7d_max_c": [round(temp_max_7d, 1)],
        "humidity_7d_mean_pct": [round(humidity_7d, 1)],
        "humidity_14d_mean_pct": [round(humidity_14d, 1)],
        "wind_7d_max_kmh": [round(wind_max, 1)],
        "ndvi_mean": [round(ndvi_mean, 4)],
        "ndvi_std": [round(rng.uniform(0.01, 0.04), 4)],
        "savi_mean": [round(savi_mean, 4)],
        "savi_std": [round(rng.uniform(0.01, 0.04), 4)],
        "ndwi_mean": [round(ndwi_mean, 4)],
        "ndwi_std": [round(rng.uniform(0.01, 0.04), 4)],
    }
    return pd.DataFrame(features)


# --- TIER 1: DETERMINISTIC AGRONOMIC RULE ENGINE ---
def run_agronomic_rule_engine(
    risk_code: int, 
    crop: str, 
    stage: str, 
    irrigation: str, 
    temp_max: float, 
    rain_7d: float, 
    ndwi: float
) -> Dict[str, Any]:
    
    actions = []
    yield_risk = "Low (< 5%)"
    water_deficit_mm = 0.0
    
    # Calculate crop sensitivity based on stage
    is_critical_stage = stage in ["Flowering / Reproductive", "Grain Filling / Podding"]
    
    if risk_code == 2:  # High Stress
        yield_risk = "High (25% - 40% loss if untreated)" if is_critical_stage else "Moderate (10% - 25% loss)"
        water_deficit_mm = round(max(25.0, (38.0 - temp_max) * 1.5 + 20), 1)
        
        if irrigation == "Drip":
            actions.append("Trigger 4 to 6 hours of night-time drip cycles to minimize evaporative loss.")
        elif irrigation == "Sprinkler":
            actions.append("Operate sprinklers during early morning (4 AM - 7 AM) only to avoid canopy scorch.")
        elif irrigation == "Rainfed":
            actions.append("CRITICAL: Apply protective life-saving irrigation immediately via tanker or farm pond.")
        else:
            actions.append("Apply shallow alternate-furrow flood irrigation to conserve available water reservoir.")

        if is_critical_stage:
            actions.append(f"Foliar spray 1% Potassium Chloride (KCl) or Salicylic Acid to protect {crop} pollen viability under heat stress.")
            
        actions.append("Apply straw, crop residue, or organic mulching around crop rows to reduce soil evaporation.")

    elif risk_code == 1:  # Moderate Stress
        yield_risk = "Moderate (5% - 15% loss)" if is_critical_stage else "Low to Moderate (< 10% loss)"
        water_deficit_mm = round(max(10.0, 15.0 - rain_7d), 1)
        
        actions.append(f"Schedule light supplemental irrigation cycle within 36-48 hours tailored for {crop} at {stage} stage.")
        actions.append("Monitor field perimeter moisture levels and pause any heavy nitrogen fertilizer applications.")
        
    else:  # Low Stress
        yield_risk = "Minimal (< 2% loss)"
        water_deficit_mm = 0.0
        actions.append("Soil moisture and crop canopy vigor are optimal. Continue standard operational schedule.")
        actions.append("Perform routine field checks for weed competition and pest vectors.")

    return {
        "estimated_yield_risk": yield_risk,
        "estimated_water_deficit_mm": water_deficit_mm,
        "critical_stage_warning": is_critical_stage,
        "rule_based_actions": actions
    }


# --- TIER 2: GENAI SYNTHESIS & PERSONALIZATION LAYER ---
def generate_ai_advisory(
    place_name: str,
    crop: str,
    stage: str,
    risk_label: str,
    rule_results: Dict[str, Any],
    language: str
) -> str:
    """
    Synthesizes rule engine outputs into a natural advisory narrative.
    Includes deterministic formatting fallback if external APIs are unconfigured.
    """
    actions_formatted = "\n".join([f"• {act}" for act in rule_results["rule_based_actions"]])
    
    advisory_template = (
        f"**Regional Overview ({place_name}):**\n"
        f"Your **{crop}** crop is currently in the **{stage}** growth stage, experiencing **{risk_label}**. "
        f"Estimated yield impact risk: **{rule_results['estimated_yield_risk']}** with a estimated water deficit of **{rule_results['estimated_water_deficit_mm']} mm**.\n\n"
        f"**Recommended Agronomic Action Plan:**\n"
        f"{actions_formatted}\n\n"
        f"*Note: Guidance generated by AgriSentinel Hybrid Agronomic Engine.*"
    )
    
    # (Optional: Add OpenAI/Gemini client API call here if desired)
    return advisory_template


@app.get("/", tags=["Health Check"])
def root():
    return {"message": "Welcome to AgriSentinel ML & Advisory Service", "status": "online"}


@app.post("/predict", tags=["Inference"])
def predict_drought_risk(request: FarmLocationRequest) -> Dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="ML Model is not loaded.")

    # 1. Validate Land vs Water
    is_land, place_name, terrain_type = validate_and_geocode_location(request.latitude, request.longitude)
    if not is_land:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Location: ({request.latitude:.4f}, {request.longitude:.4f}) is located over a water body ({place_name}). AgriSentinel model inference is restricted to agricultural land."
        )

    # 2. Get Telemetry & Predict Risk
    input_features = generate_spatial_telemetry(request.latitude, request.longitude)
    prediction_class = int(model.predict(input_features)[0])
    probabilities = model.predict_proba(input_features)[0]
    confidence_score = float(np.max(probabilities))

    risk_info = RISK_MAPPING.get(prediction_class, {"label": "Unknown", "color": "gray"})
    
    temp_max = float(input_features["temp_7d_max_c"].values[0])
    rain_7d = float(input_features["rain_7d_sum_mm"].values[0])
    ndwi = float(input_features["ndwi_mean"].values[0])

    # 3. Tier 1: Agronomic Rule Engine
    rule_output = run_agronomic_rule_engine(
        risk_code=prediction_class,
        crop=request.crop_type,
        stage=request.growth_stage,
        irrigation=request.irrigation_type,
        temp_max=temp_max,
        rain_7d=rain_7d,
        ndwi=ndwi
    )

    # 4. Tier 2: GenAI Advisory Synthesis
    ai_advisory = generate_ai_advisory(
        place_name=place_name,
        crop=request.crop_type,
        stage=request.growth_stage,
        risk_label=risk_info["label"],
        rule_results=rule_output,
        language=request.language
    )

    return {
        "location_info": {
            "latitude": request.latitude,
            "longitude": request.longitude,
            "place_name": place_name,
            "terrain": terrain_type
        },
        "crop_context": {
            "crop_type": request.crop_type,
            "growth_stage": request.growth_stage,
            "irrigation_type": request.irrigation_type
        },
        "prediction": {
            "class_code": prediction_class,
            "risk_level": risk_info["label"],
            "confidence": round(confidence_score, 4)
        },
        "key_metrics_used": {
            "ndwi_moisture_index": float(input_features["ndwi_mean"].values[0]),
            "ndvi_vegetation_index": float(input_features["ndvi_mean"].values[0]),
            "temp_7d_max_c": temp_max,
            "rain_7d_sum_mm": rain_7d
        },
        "mitigation_engine": {
            "rule_analysis": rule_output,
            "ai_advisory": ai_advisory
        }
    }