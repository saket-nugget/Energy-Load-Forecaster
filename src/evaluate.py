# src/evaluate.py

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from utils.config import Config
from utils.logger import get_logger
from utils.data_loader import load_dataset
from src.model import build_model
from src.preprocessing import Preprocessor

def run_evaluation(config):
    """
    Runs the model evaluation pipeline.
    """
    logger = get_logger("evaluate")

    logger.info("1. Loading test data and preprocessor...")
    raw_df = load_dataset(config.get("dataset", "raw_path"))
    
    train_size = int(len(raw_df) * 0.7)
    val_size = int(len(raw_df) * 0.15)
    test_df = raw_df[train_size + val_size:]
    
    preprocessor_path = os.path.join(config.get("output", "forecasts_dir"), "preprocessor.joblib")
    preprocessor = Preprocessor.load(preprocessor_path)
    X_test, y_test = preprocessor.transform(test_df)
    logger.info(f"Test data loaded and processed. Shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    logger.info("2. Creating PyTorch DataLoader...")
    batch_size = config.get("training", "batch_size")
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info("3. Loading model and evaluating...")
    model = build_model(config)
    # ... (rest of the file is identical) ...
    model_path = os.path.join(config.get("output", "forecasts_dir"), "best_model.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    predictions, actuals = [], []
# src/evaluate.py

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from utils.config import Config
from utils.logger import get_logger
from utils.data_loader import load_dataset
from src.model import build_model
from src.preprocessing import Preprocessor

def run_evaluation(config):
    """
    Runs the model evaluation pipeline.
    """
    logger = get_logger("evaluate")

    logger.info("1. Loading test data and preprocessor...")
    raw_df = load_dataset(config.get("dataset", "raw_path"))
    
    train_size = int(len(raw_df) * 0.7)
    val_size = int(len(raw_df) * 0.15)
    test_df = raw_df[train_size + val_size:]
    
    preprocessor_path = os.path.join(config.get("output", "forecasts_dir"), "preprocessor.joblib")
    preprocessor = Preprocessor.load(preprocessor_path)
    X_test, y_test = preprocessor.transform(test_df)
    logger.info(f"Test data loaded and processed. Shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    logger.info("2. Creating PyTorch DataLoader...")
    batch_size = config.get("training", "batch_size")
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info("3. Loading model and evaluating...")
    model = build_model(config)
    # ... (rest of the file is identical) ...
    model_path = os.path.join(config.get("output", "forecasts_dir"), "best_model.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predictions.append(outputs.numpy())
            actuals.append(y_batch.numpy())

    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)

    # Inverse transform to get original scale (MW)
    predictions = preprocessor.inverse_transform_target(predictions)
    actuals = preprocessor.inverse_transform_target(actuals)

    mae = np.mean(np.abs(actuals - predictions))
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100
    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
    logger.info(f"Evaluation Metrics: {metrics}")

    eval_dir = "outputs/evaluation"
    os.makedirs(eval_dir, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(os.path.join(eval_dir, "metrics.csv"), index=False)
    pd.DataFrame({"actuals": actuals.flatten(), "predictions": predictions.flatten()}).to_csv(os.path.join(eval_dir, "predictions.csv"), index=False)
    
    logger.info("✅ Evaluation complete.")
    return actuals, predictions

if __name__ == "__main__":
    config = Config("configs/config.yaml")
    run_evaluation(config)