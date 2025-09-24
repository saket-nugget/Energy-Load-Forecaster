# src/model.py

import torch
import torch.nn as nn
from utils.logger import get_logger

logger = get_logger(__name__)

# --- The TransformerForecaster class remains completely unchanged ---
class TransformerForecaster(nn.Module):
    def __init__(self, num_input_features, input_length, forecast_horizon, hidden_size, num_layers, dropout=0.1, num_heads=4):
        super(TransformerForecaster, self).__init__()
        self.input_projection = nn.Linear(num_input_features, hidden_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, input_length, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(hidden_size, forecast_horizon)
        logger.info(f"Initialized TransformerForecaster model with {num_input_features} input features.")
    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.pos_embedding
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)
        x = self.global_pool(x).squeeze(-1)
        out = self.fc_out(x)
        return out
# --- End of unchanged section ---


def build_model(config):
    model_config = config.get("model")
    dataset_config = config.get("dataset")

    num_input_features = 1 + len(dataset_config["features"])

    model = TransformerForecaster(
        num_input_features=num_input_features,
        input_length=model_config["input_length"],
        forecast_horizon=model_config["forecast_horizon"],
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
        # THIS IS THE ONLY CHANGE IN THIS FILE:
        num_heads=model_config["num_heads"] # Read from config instead of being hardcoded
    )

    logger.info("PyTorch TransformerForecaster model built successfully")
    return model