import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Set project root path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Initialize FastAPI App
app = FastAPI(
    title="AgriSentinel API",
    description="Real-time Satellite & Weather Analytics for Crop Drought Risk Assessment",
    version="1.0.0"
)

# Global variables for model and metadata
model = None
MODEL_PATH = project_root / "models" / "crop_health_model.pkl"

# Human-readable label mappings
RISK_MAPPING = {
    0: {"label": "Low Stress", "action": "Crops are healthy and well-hydrated. Continue normal operations."},
    1: {"label": "Moderate Stress", "action": "Moisture levels declining. Monitor field closely over the next 48 hours."},
    2: {"label": "High Stress Alert", "action": "CRITICAL: Severe moisture deficit or heat stress detected. Trigger irrigation immediately."}
}


# Request Schema using Pydantic
class FarmLocationRequest(BaseModel):
    latitude: float = Field(..., example=36.7783, description="Latitude coordinate of the farm field")
    longitude: float = Field(..., example=-119.4179, description="Longitude coordinate of the farm field")


@app.on_event("startup")
def load_ml_model():
    """Loads the serialized machine learning model on server startup."""
    global model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run 'python models/train.py' first.")
    
    model = joblib.load(MODEL_PATH)
    print(f"✅ ML Model successfully loaded from {MODEL_PATH.name}")


@app.get("/", tags=["Health Check"])
def root():
    return {"message": "Welcome to AgriSentinel ML Service", "status": "online"}


@app.get("/health", tags=["Health Check"])
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_file": MODEL_PATH.name
    }


@app.post("/predict", tags=["Inference"])
def predict_drought_risk(request: FarmLocationRequest) -> Dict[str, Any]:
    """
    Predicts crop drought stress risk for a specific farm location 
    by retrieving aligned weather and satellite features.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="ML Model is not loaded."
        )

    # Locate aligned dataset for the requested location
    data_dir = project_root / "data"
    master_files = list(data_dir.glob("master_dataset_*.csv"))

    if not master_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No aligned master feature data found. Please run data alignment pipeline first."
        )

    # Load recent aligned feature data
    master_df = pd.read_csv(master_files[0])
    
    # Feature columns used during model training
    feature_cols = [
        "rain_3d_sum_mm", "rain_7d_sum_mm", "rain_14d_sum_mm",
        "temp_3d_mean_c", "temp_7d_mean_c", "temp_14d_mean_c", "temp_7d_max_c",
        "humidity_7d_mean_pct", "humidity_14d_mean_pct", "wind_7d_max_kmh",
        "ndvi_mean", "ndvi_std", "savi_mean", "savi_std", "ndwi_mean", "ndwi_std"
    ]

    # Ensure all required features are present
    missing_cols = [c for c in feature_cols if c not in master_df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Master dataset is missing required feature columns: {missing_cols}"
        )

    # Extract single observation vector
    input_features = master_df[feature_cols].iloc[[0]]

    # Perform Model Inference
    prediction_class = int(model.predict(input_features)[0])
    probabilities = model.predict_proba(input_features)[0]
    confidence_score = float(np.max(probabilities))

    risk_info = RISK_MAPPING.get(prediction_class, {"label": "Unknown", "action": "N/A"})

    return {
        "coordinates": {
            "latitude": request.latitude,
            "longitude": request.longitude
        },
        "prediction": {
            "class_code": prediction_class,
            "risk_level": risk_info["label"],
            "confidence": round(confidence_score, 4),
            "recommended_action": risk_info["action"]
        },
        "key_metrics_used": {
            "ndwi_moisture_index": float(input_features["ndwi_mean"].values[0]),
            "ndvi_vegetation_index": float(input_features["ndvi_mean"].values[0]),
            "temp_7d_max_c": float(input_features["temp_7d_max_c"].values[0]),
            "rain_7d_sum_mm": float(input_features["rain_7d_sum_mm"].values[0])
        }
    }