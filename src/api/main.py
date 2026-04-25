import os
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.models.convlstm_model import ConvLSTMForecaster
from src.config_loader import get_config

# Global state to hold our model and mock data
app_state = {
    "model": None,
    "mock_X_test": None,
    "grid_lats": None,
    "grid_lons": None,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ConvLSTM Model and mock data...")
    config = get_config()
    
    # 1. Load the model
    try:
        model = ConvLSTMForecaster(config)
        model.load()
        app_state["model"] = model
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")

    # 2. Load the mock data (the test split)
    seq_dir = config["paths"]["sequences_dir"]
    try:
        app_state["mock_X_test"] = np.load(os.path.join(seq_dir, "convlstm_X_test.npy"))
        print(f"Loaded {len(app_state['mock_X_test'])} mock sequences.")
    except Exception as e:
        print(f"Failed to load mock sequences: {e}")

    # 3. Generate the grid (matching sequence_builder.py)
    bbox = config["openaq"]["bbox"]
    res = config["forecasting"].get("grid_resolution", 0.005)
    
    app_state["grid_lats"] = np.arange(bbox[1], bbox[3], res).tolist()
    app_state["grid_lons"] = np.arange(bbox[0], bbox[2], res).tolist()

    yield
    
    # Clean up
    app_state.clear()

app = FastAPI(title="AQI Forecasting API", lifespan=lifespan)

class GridPredictionResponse(BaseModel):
    grid_lats: list[float]
    grid_lons: list[float]
    predictions: list[list[float]]

@app.get("/")
def root():
    return {"status": "ok", "message": "AQI Forecasting API is running"}

@app.get("/predict/recent", response_model=GridPredictionResponse)
def predict_recent():
    model = app_state.get("model")
    X_test = app_state.get("mock_X_test")
    
    if model is None or getattr(model, "model", None) is None or X_test is None:
        raise HTTPException(status_code=503, detail="Model or data not completely loaded")
    
    if len(X_test) == 0:
        raise HTTPException(status_code=404, detail="No mock data available")

    # Pick the most recent (last) sequence to simulate "now"
    recent_seq = X_test[-1:] # Shape: (1, T, H, W, C)
    
    try:
        # Predict: returns (1, H, W)
        pred = model.predict(recent_seq, batch_size=1)
        pred_grid = pred[0].tolist() # (H, W) list
        
        return {
            "grid_lats": app_state["grid_lats"],
            "grid_lons": app_state["grid_lons"],
            "predictions": pred_grid
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
