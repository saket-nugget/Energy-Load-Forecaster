# src/evaluate.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

from src.model import TransformerForecaster
from src.preprocessing import preprocess_data
from utils.data_loader import load_data
from utils.logger import get_logger
from utils.config import load_config


def evaluate():
    # --- 1. Load Config ---
    config = load_config("configs/config.yaml")

    # --- 2. Logger ---
    logger = get_logger("evaluate")

    # --- 3. Load Test Data ---
    test_df, _ = load_data(config["data"]["test_path"], None)
    X_test, y_test = preprocess_data(test_df, config)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # --- 4. Load Model ---
    model = TransformerForecaster(
        input_dim=config["model"]["input_dim"],
        model_dim=config["model"]["model_dim"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        output_dim=config["model"]["output_dim"],
        dropout=config["model"]["dropout"]
    )
    model.load_state_dict(torch.load("outputs/forecasts/best_model.pth"))
    model.eval()

    # --- 5. Run Predictions ---
    with torch.no_grad():
        predictions = model(X_test).numpy()
        y_true = y_test.numpy()

    # --- 6. Metrics ---
    mae = np.mean(np.abs(y_true - predictions))
    rmse = np.sqrt(np.mean((y_true - predictions) ** 2))
    mape = np.mean(np.abs((y_true - predictions) / (y_true + 1e-8))) * 100

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }

    logger.info(f"📊 Evaluation Metrics: {metrics}")

    # --- 7. Save Results ---
    os.makedirs("outputs/evaluation", exist_ok=True)
    pd.DataFrame(predictions, columns=["forecast"]).to_csv("outputs/evaluation/forecasts.csv", index=False)
    pd.DataFrame([metrics]).to_csv("outputs/evaluation/metrics.csv", index=False)

    logger.info("✅ Forecasts and metrics saved in outputs/evaluation/")


if __name__ == "__main__":
    evaluate()
