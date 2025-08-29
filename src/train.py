import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml
import os

from src.model import TransformerForecaster
from src.preprocessing import preprocess_data
from utils.data_loader import load_data
from utils.logger import get_logger
from utils.config import load_config


def train():
    config = load_config("configs/config.yaml")

    logger = get_logger("train")

    train_df, val_df = load_data(config["data"]["train_path"],
                                 config["data"]["val_path"])
    X_train, y_train = preprocess_data(train_df, config)
    X_val, y_val = preprocess_data(val_df, config)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    model = TransformerForecaster(
        input_dim=config["model"]["input_dim"],
        model_dim=config["model"]["model_dim"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        output_dim=config["model"]["output_dim"],
        dropout=config["model"]["dropout"]
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    best_val_loss = float("inf")
    for epoch in range(config["training"]["epochs"]):
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

        logger.info(f"Epoch {epoch+1}/{config['training']['epochs']} - "
                    f"Train Loss: {train_loss/len(train_loader):.4f} "
                    f"Val Loss: {val_loss/len(val_loader):.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("outputs/forecasts", exist_ok=True)
            torch.save(model.state_dict(), "outputs/forecasts/best_model.pth")
            logger.info("✅ Best model saved!")

if __name__ == "__main__":
    train()
