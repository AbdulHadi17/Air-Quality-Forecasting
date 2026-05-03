import os
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config_loader import get_config

# Global state to hold our model and mock data
app_state = {
    "model": None,
    "latest_X": None,
    "latest_coords": None,
    "target_scaler": None,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading XGBoost Model and extracting latest features...")
    config = get_config()
    models_dir = config["paths"]["models_dir"]
    processed_dir = config["paths"]["processed_dir"]
    
    # 1. Load the model
    try:
        app_state["model"] = joblib.load(os.path.join(models_dir, "xgb_baseline.pkl"))
        print("XGBoost loaded successfully.")
    except Exception as e:
        print(f"Failed to load XGBoost model: {e}")

    # 2. Load the target scaler
    try:
        app_state["target_scaler"] = joblib.load(os.path.join(models_dir, "target_scaler.pkl"))
        print("Target scaler loaded.")
    except Exception as e:
        print(f"Failed to load target scaler: {e}")

    # 3. Extract the latest 24h sequence for each station
    try:
        df = pd.read_parquet(os.path.join(processed_dir, "featured.parquet"))
        stations = pd.read_parquet(os.path.join(processed_dir, "stations_metadata.parquet"))
        scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))

        # Auto-detect features exactly like SequenceBuilder
        exclude = {"timestamp", "location_id", "target"}
        feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.float16]]

        # Clean and scale
        df_clean = df.dropna(subset=["target"]).copy()
        df_clean = df_clean.dropna(subset=feature_cols, how="all")
        df_clean[feature_cols] = df_clean[feature_cols].ffill().fillna(0)
        df_clean[feature_cols] = scaler.transform(df_clean[feature_cols])

        seq_length = config["forecasting"].get("sequence_length", 24)
        latest_X = []
        latest_coords = []

        # Build latest sequence per station
        for loc_id in sorted(df_clean["location_id"].unique()):
            loc_data = df_clean[df_clean["location_id"] == loc_id].sort_values("timestamp")
            if len(loc_data) >= seq_length:
                # We just take the last seq_length rows as the 'current' state
                tail = loc_data.tail(seq_length)
                latest_X.append(tail[feature_cols].values)

                # Find lat/lon
                meta = stations[stations["location_id"] == loc_id]
                if not meta.empty:
                    lat = meta.iloc[0]["latitude"]
                    lon = meta.iloc[0]["longitude"]
                    latest_coords.append([lat, lon])

        app_state["latest_X"] = np.array(latest_X, dtype=np.float32)
        app_state["latest_coords"] = latest_coords
        print(f"Extracted {len(latest_coords)} station sequences for prediction.")

    except Exception as e:
        print(f"Failed to build latest sequences: {e}")

    yield
    app_state.clear()

app = FastAPI(title="AQI Forecasting API", lifespan=lifespan)

class StationPredictionResponse(BaseModel):
    predictions: list[list[float]]  # list of [lat, lon, pm25_value]

@app.get("/")
def root():
    return {"status": "ok", "message": "AQI Forecasting API is running (XGBoost)"}

@app.get("/predict/recent", response_model=StationPredictionResponse)
def predict_recent():
    model = app_state.get("model")
    X_latest = app_state.get("latest_X")
    coords = app_state.get("latest_coords")
    target_scaler = app_state.get("target_scaler")
    
    if model is None or X_latest is None or len(X_latest) == 0:
        raise HTTPException(status_code=503, detail="Model or data not completely loaded")

    try:
        # XGBoost expects flattened sequences
        X_flat = X_latest.reshape(X_latest.shape[0], -1)
        preds = model.predict(X_flat).astype(np.float32)
        
        # Inverse transform
        if target_scaler is not None:
            preds = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            
        # Format as [[lat, lon, val], ...]
        response_preds = []
        for i, (lat, lon) in enumerate(coords):
            # Clip negative predictions to 0 for PM2.5
            val = max(0.0, float(preds[i]))
            response_preds.append([lat, lon, val])
            
        return {"predictions": response_preds}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
