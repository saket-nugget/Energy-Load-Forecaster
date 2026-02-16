# src/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from utils.config import Config
from utils.logger import get_logger
from utils.data_loader import load_dataset
from src.model import build_model
from src.preprocessing import Preprocessor

def run_training(config):
    """
    Runs the model training pipeline.
    """
    logger = get_logger("train")
    
    logger.info("1. Loading and splitting data for training...")
    raw_path = config.get("dataset", "raw_path")
    if not os.path.exists(raw_path):
        logger.error(f"Dataset not found at {raw_path}. Please run 'python src/download_data.py' first.")
        raise FileNotFoundError(f"Dataset not found at {raw_path}")

    raw_df = load_dataset(raw_path)
    
    train_size = int(len(raw_df) * 0.7)
    val_size = int(len(raw_df) * 0.15)
    train_df = raw_df[:train_size]
    val_df = raw_df[train_size : train_size + val_size]
    logger.info(f"Data split - Train: {train_df.shape}, Val: {val_df.shape}")

    logger.info("2. Preprocessing data and creating sequences...")
    preprocessor = Preprocessor(config)
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_val, y_val = preprocessor.transform(val_df)
    
    logger.info(f"Sequence shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"Sequence shapes - X_val: {X_val.shape}, y_val: {y_val.shape}")
    
    preprocessor_path = os.path.join(config.get("output", "forecasts_dir"), "preprocessor.joblib")
    preprocessor.save(preprocessor_path)

    logger.info("3. Creating PyTorch DataLoaders...")
    batch_size = config.get("training", "batch_size")
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info("4. Building and training model...")
    model = build_model(config)
    # ... (rest of the file is identical) ...
    training_cfg = config.get("training")
    output_dir = config.get("output", "forecasts_dir")
    os.makedirs(output_dir, exist_ok=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=training_cfg["learning_rate"])

    best_val_loss = float("inf")
    for epoch in range(training_cfg["epochs"]):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        logger.info(f"Epoch {epoch+1}/{training_cfg['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            logger.info(f"✅ New best model saved with validation loss: {best_val_loss:.4f}")

    logger.info("🏁 Training complete.")

if __name__ == "__main__":
    config = Config("configs/config.yaml")
    run_training(config)