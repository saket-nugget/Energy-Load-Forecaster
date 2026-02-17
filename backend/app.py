
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
def predict_demo(days: int = 1):
    """
    Demo endpoint: Predicts the next N days (default 1) using autoregression.
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model or preprocessor not loaded.")
    
    try:
        from utils.data_loader import load_dataset
        raw_df = load_dataset(config.get("dataset", "raw_path"))
        
        input_length = config.get("model", "input_length")
        forecast_horizon = config.get("model", "forecast_horizon") # 24
        
        # We need the last `input_length` hours from the dataset to start
        current_window_df = raw_df.iloc[-input_length:].copy()
        
        all_forecasts = []
        
        threshold = config.get("anomaly_detection", "threshold") or 3.0
        
        # Autoregressive Loop
        for _ in range(days):
            # 1. Preprocess current window
            # Re-create preprocessing pipeline steps manually for inference
            # Ensure datetime is parsed
            current_window_df[preprocessor.datetime_col] = pd.to_datetime(current_window_df[preprocessor.datetime_col])
            
            # Feature Engineering
            df_processed = current_window_df.copy()
            df_processed["hour"] = df_processed[preprocessor.datetime_col].dt.hour
            df_processed["dayofweek"] = df_processed[preprocessor.datetime_col].dt.dayofweek
            df_processed["month"] = df_processed[preprocessor.datetime_col].dt.month
            df_processed["is_weekend"] = (df_processed[preprocessor.datetime_col].dt.dayofweek >= 5).astype(int)
            
            # Scale
            # Note: We must scale ONLY the columns that were scaled during training
            df_processed[preprocessor.processed_columns_] = preprocessor.scaler.transform(df_processed[preprocessor.processed_columns_])
            
            # Prepare Tensor
            feature_cols = [preprocessor.target] + preprocessor.features
            input_tensor = torch.tensor(df_processed[feature_cols].values, dtype=torch.float32).unsqueeze(0) # (1, 168, features)
            
            # 2. Predict next vector (24h)
            with torch.no_grad():
                prediction_scaled = model(input_tensor).numpy() # (1, 24)
            
            # Check for anomalies (Z-score based since data is scaled)
            # anomaly if |scaled_value| > threshold
            # prediction_scaled is (1, 24)
            is_anomaly = (np.abs(prediction_scaled) > threshold).astype(bool).flatten().tolist()
            
            # Inverse transform to get actual MW values
            prediction_actual = preprocessor.inverse_transform_target(prediction_scaled).flatten()
            
            # Append results
            # We associate anomalies with the forecast values
            # Structure: list of dicts or just parallel lists?
            # Let's keep it simple: flat list of values, flat list of booleans?
            # But we are extending `all_forecasts`.
            # Let's change `all_forecasts` to handle complex objects or just return separate list.
            # Returning separate list is easier for now to maintain compat.
            
            # But wait, predict_demo returns "forecast": list.
            # I'll create `all_anomalies` list.
            
            all_forecasts.extend(prediction_actual.tolist())
            if 'all_anomalies' not in locals():
                all_anomalies = []
            all_anomalies.extend(is_anomaly)
            
            # 3. Update Window for next iteration (Autoregression)
            # We need to append the *predicted* values to the window and slide it
            # We also need to generate the future TimeSteps for the features (hour, day, etc.)
            
            last_timestamp = current_window_df[preprocessor.datetime_col].iloc[-1]
            future_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=forecast_horizon, freq='h')
            
            # Create a dataframe for the new predictions
            new_rows = pd.DataFrame({
                preprocessor.datetime_col: future_timestamps,
                preprocessor.target: prediction_actual # We use the prediction as the "actual" history for the next step
            })
            
            # Append and slice to keep window size constant
            current_window_df = pd.concat([current_window_df, new_rows], ignore_index=True)
            current_window_df = current_window_df.iloc[-input_length:] 
            
        return {
            "forecast": all_forecasts,
            "anomalies": all_anomalies,
            "unit": "MW",
            "horizon_hours": len(all_forecasts),
            "days_requested": days
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
