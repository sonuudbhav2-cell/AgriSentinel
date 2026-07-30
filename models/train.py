import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

# Set path roots
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def create_drought_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a 3-class drought stress target based on NDWI moisture levels 
    and recent 7-day rainfall metrics:
      0 = Low Stress (Hydrated)
      1 = Moderate Stress (Monitoring)
      2 = High Stress (Action Required)
    """
    df = df.copy()

    conditions = [
        (df["ndwi_mean"] >= 0.05) & (df["rain_7d_sum_mm"] >= 2.0),  # Low stress
        (df["ndwi_mean"] < 0.05) & (df["ndwi_mean"] >= -0.05),     # Moderate stress
        (df["ndwi_mean"] < -0.05) | (df["temp_7d_max_c"] > 37.0)   # High stress
    ]
    choices = [0, 1, 2]

    df["drought_risk"] = np.select(conditions, choices, default=1)
    return df


def generate_historical_dataset(master_file: Path, num_samples: int = 200) -> pd.DataFrame:
    """
    Generates a rich, realistic multi-day synthetic dataset based on our 
    real master feature vector so the model has enough patterns to learn from.
    """
    base_df = pd.read_csv(master_file)
    base_row = base_df.iloc[0]

    np.random.seed(42)
    synthetic_rows = []

    for _ in range(num_samples):
        # Simulate realistic variations in weather and satellite readings
        temp_max = float(np.random.normal(base_row["temp_7d_max_c"], 4.0))
        temp_mean = temp_max - float(np.random.uniform(5.0, 10.0))
        rain_7d = float(np.random.exponential(scale=3.0) if np.random.rand() > 0.4 else 0.0)
        humidity = float(np.clip(np.random.normal(base_row["humidity_7d_mean_pct"], 10.0), 15.0, 90.0))

        # Satellite indices respond dynamically to weather
        ndvi_mean = float(np.clip(0.1 + (rain_7d * 0.02) - (temp_max * 0.003) + np.random.normal(0.2, 0.05), -0.2, 0.9))
        savi_mean = float(ndvi_mean * 1.3)
        ndwi_mean = float(np.clip((rain_7d * 0.03) - (temp_max * 0.005) + np.random.normal(0.0, 0.05), -0.5, 0.8))

        row = {
            "rain_3d_sum_mm": float(rain_7d * 0.4),
            "rain_7d_sum_mm": rain_7d,
            "rain_14d_sum_mm": float(rain_7d * 1.8),
            "temp_3d_mean_c": temp_mean,
            "temp_7d_mean_c": temp_mean,
            "temp_14d_mean_c": temp_mean,
            "temp_7d_max_c": temp_max,
            "humidity_7d_mean_pct": humidity,
            "humidity_14d_mean_pct": humidity,
            "wind_7d_max_kmh": float(np.random.normal(18.0, 5.0)),
            "ndvi_mean": ndvi_mean,
            "ndvi_std": float(np.random.uniform(0.05, 0.2)),
            "savi_mean": savi_mean,
            "savi_std": float(np.random.uniform(0.05, 0.25)),
            "ndwi_mean": ndwi_mean,
            "ndwi_std": float(np.random.uniform(0.05, 0.2))
        }
        synthetic_rows.append(row)

    df_dataset = pd.DataFrame(synthetic_rows)
    return create_drought_target(df_dataset)


def train_and_evaluate():
    data_dir = project_root / "data"
    master_files = list(data_dir.glob("master_dataset_*.csv"))

    if not master_files:
        raise FileNotFoundError("No master dataset CSV found. Run src/data_alignment.py first.")

    print(f"📥 Loading aligned farm data from {master_files[0].name}...")
    dataset = generate_historical_dataset(master_files[0], num_samples=250)

    # Separate input features (X) and target label (y)
    feature_cols = [col for col in dataset.columns if col != "drought_risk"]
    X = dataset[feature_cols]
    y = dataset["drought_risk"]

    # 80% Train, 20% Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"📊 Dataset split complete: {len(X_train)} training rows, {len(X_test)} testing rows.\n")

    # -------------------------------------------------------------
    # 1. Champion Model: Random Forest
    # -------------------------------------------------------------
    print("🌲 Training Champion Model: Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"   ► Random Forest Test Accuracy: {rf_acc * 100:.2f}%")

    # -------------------------------------------------------------
    # 2. Challenger Model: XGBoost
    # -------------------------------------------------------------
    print("⚡ Training Challenger Model: XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=4, 
        random_state=42, 
        eval_metric="mlogloss"
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_preds)
    print(f"   ► XGBoost Test Accuracy:       {xgb_acc * 100:.2f}%\n")

    # -------------------------------------------------------------
    # 3. Champion vs. Challenger Decision & Model Saving
    # -------------------------------------------------------------
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    saved_model_path = models_dir / "crop_health_model.pkl"

    if xgb_acc > rf_acc:
        winning_name = "XGBoost"
        winning_model = xgb_model
        winning_preds = xgb_preds
    else:
        winning_name = "Random Forest"
        winning_model = rf_model
        winning_preds = rf_preds

    print(f"🏆 WINNER: {winning_name} selected for deployment!")
    
    # Save model artifact
    joblib.dump(winning_model, saved_model_path)
    print(f"💾 Model saved to {saved_model_path.relative_to(project_root)}")

    # Detailed Evaluation Metrics of Winner
    print("\n📈 Winner Detailed Classification Report:")
    print("-" * 55)
    print(classification_report(y_test, winning_preds, target_names=["Low Stress", "Moderate Stress", "High Stress"]))

    print("🧩 Winner Confusion Matrix:")
    print(confusion_matrix(y_test, winning_preds))


if __name__ == "__main__":
    train_and_evaluate()