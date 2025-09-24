# src/evaluate.py
import torch
import numpy as np
import pandas as pd
import os

# use build_model factory from src.model
from src.model import build_model
from src.preprocessing import Preprocessor
from utils.data_loader import load_data
from utils.logger import get_logger
from utils.config import Config


def evaluate():
    config = Config("configs/config.yaml")

    logger = get_logger("evaluate")

    # NOTE: load_data signature / usage will be aligned in a later step
    test_path = config.get("dataset","raw_path")
    test_df, _ = load_data(test_path, None)
    X_test, y_test = Preprocessor.preprocess(test_df, config)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Build the model using shared factory
    model = build_model(config)

    model.load_state_dict(torch.load("outputs/forecasts/best_model.pth"))
    model.eval()

    with torch.no_grad():
        predictions = model(X_test).numpy()
        y_true = y_test.numpy()

    mae = np.mean(np.abs(y_true - predictions))
    rmse = np.sqrt(np.mean((y_true - predictions) ** 2))
    mape = np.mean(np.abs((y_true - predictions) / (y_true + 1e-8))) * 100

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }

    logger.info(f"📊 Evaluation Metrics: {metrics}")

    os.makedirs("outputs/evaluation", exist_ok=True)
    pd.DataFrame(predictions, columns=["forecast"]).to_csv("outputs/evaluation/forecasts.csv", index=False)
    pd.DataFrame([metrics]).to_csv("outputs/evaluation/metrics.csv", index=False)

    logger.info("✅ Forecasts and metrics saved in outputs/evaluation/")


if __name__ == "__main__":
    evaluate()
