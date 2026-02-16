
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import pandas as pd
import numpy as np
import os
from src.model import build_model
from src.preprocessing import Preprocessor
from utils.config import Config
from fastapi.middleware.cors import CORSMiddleware

# Initialize Config
config = Config("configs/config.yaml")

# Initialize App
app = FastAPI(title="Energy Load Forecaster API", version="1.0")

# CORS Configuration
origins = [
    "http://localhost:3000", # React Frontend
    "http://localhost:5173", # Vite Default
    "*", # Allow all for now (simplify dev)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model and Preprocessor
model = None
preprocessor = None

@app.on_event("startup")
async def load_artifacts():
    global model, preprocessor
    try:
        # Load Preprocessor
        preprocessor_path = os.path.join(config.get("output", "forecasts_dir"), "preprocessor.joblib")
        if os.path.exists(preprocessor_path):
             preprocessor = Preprocessor.load(preprocessor_path)
             print(f"Loaded preprocessor from {preprocessor_path}")
        else:
             print(f"Warning: Preprocessor not found at {preprocessor_path}. Predictions will fail.")

        # Load Model
        model_path = os.path.join(config.get("output", "forecasts_dir"), "best_model.pth")
        if os.path.exists(model_path):
            model = build_model(config)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            print(f"Loaded model from {model_path}")
        else:
            print(f"Warning: Model not found at {model_path}. Predictions will fail.")
            
    except Exception as e:
        print(f"Error loading artifacts: {e}")

class ForecastRequest(BaseModel):
    # Depending on input_length, we accept a list of features or similar.
    # For simplicity in this demo, we'll assume the frontend sends the last `input_length` hours of data
    # Or we can just trigger a forecast based on the *latest* available data in the dataset if no data provided.
    # Let's go with the latter for simplicity in this hackathon context, or accept a date.
    
    # Actually, to make it interactive, let's allow sending a sequence.
    # But constructing 168 hours of 5 features is hard for a user.
    # Let's make an endpoint that forecasts from the *end of the test set* for demonstration.
    pass

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None, "preprocessor_loaded": preprocessor is not None}

@app.get("/predict/demo")
def predict_demo():
    """
    Demo endpoint: Predicts the next 24 hours based on the held-out test set's last window.
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model or preprocessor not loaded.")
    
    try:
        # Load Raw Data locally for demo purposes
        # In a real app, this would be in a database
        from utils.data_loader import load_dataset
        raw_df = load_dataset(config.get("dataset", "raw_path"))
        
        # Taking the last window from the dataset to forecast into the "future" (relative to dataset)
        # We need input_length rows
        input_length = config.get("model", "input_length")
        
        # Use the preprocessor to transform this window
        # We need to grab enough data to ensure we have the window *before* the target
        # actually, fit_transform was done on train, transform on val/test. 
        # We can just use the tail of the df.
        
        input_df = raw_df.iloc[-input_length:].copy()
        
        # Preprocess
        # Note: transform expects a dataframe with target column present (even if ignored for X generation)
        # We need to handle this carefully. The preprocessor's `transform` method generates X and y.
        # We can reuse the internal methods if we want to be cleaner, or just use transform.
        
        # Let's rely on internal methods for a single prediction to avoid X/y array creation overhead/logic mismatch
        # But `transform` is the public API.
        # Let's just create a dummy target column if needed or rely on the fact that existing data has it.
        
        X, _ = preprocessor.transform(input_df)
        
        # X shape will be (1, input_length, features) if input_df has length input_length?
        # No, create_sequences does: range(len(df) - input_length - forecast_horizon + 1)
        # We want exactly ONE sequence.
        # So we need input_length rows.
        # create_sequences loop range: len=168, inp=168, hor=24. range(168 - 168 - 24 + 1) = range(-23) -> Empty.
        # Logic in preprocessor expects data to cover input + horizon to return a pair (X, y).
        # We want just inference.
        
        # We need to manually preprocess for inference since the class is designed for Training/Eval (returning X, y pairs).
        
        # Manual Inference Preprocessing:
        df_processed = preprocessor._handle_missing(input_df)
        df_processed = preprocessor._feature_engineering(df_processed)
        df_processed[preprocessor.processed_columns_] = preprocessor.scaler.transform(df_processed[preprocessor.processed_columns_])
        
        feature_cols = [preprocessor.target] + preprocessor.features
        input_tensor = torch.tensor(df_processed[feature_cols].values, dtype=torch.float32).unsqueeze(0) # (1, 168, features)
        
        with torch.no_grad():
            prediction_scaled = model(input_tensor).numpy()
            
        prediction_actual = preprocessor.inverse_transform_target(prediction_scaled).flatten()
        
        return {
            "forecast": prediction_actual.tolist(),
            "unit": "MW",
            "horizon": config.get("model", "forecast_horizon")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
