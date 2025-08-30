import torch
import torch.nn as nn
from utils.logger import get_logger

logger = get_logger(__name__)


class TransformerForecaster(nn.Module):
    def __init__(self, input_length, forecast_horizon, hidden_size, num_layers, dropout=0.1, num_heads=4):
        super(TransformerForecaster, self).__init__()

        # Input projection (maps 1 feature → hidden_size)
        self.input_projection = nn.Linear(1, hidden_size)

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, input_length, hidden_size))

        # Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True  # keeps (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # equivalent to GlobalAveragePooling1D

        # Output layer
        self.fc_out = nn.Linear(hidden_size, forecast_horizon)

        logger.info("Initialized TransformerForecaster model")

    def forward(self, x):
        # x shape: (batch, seq_len, 1)
        x = self.input_projection(x)  # -> (batch, seq_len, hidden_size)

        # add positional embedding
        x = x + self.pos_embedding

        # transformer encoding
        x = self.transformer_encoder(x)

        # global pooling: (batch, hidden_size, seq_len) → (batch, hidden_size, 1) → (batch, hidden_size)
        x = x.permute(0, 2, 1)  # (batch, hidden_size, seq_len)
        x = self.global_pool(x).squeeze(-1)

        # forecast
        out = self.fc_out(x)  # (batch, forecast_horizon)
        return out


def build_model(config):
    model_config = config["model"]

    model = TransformerForecaster(
        input_length=model_config["input_length"],
        forecast_horizon=model_config["forecast_horizon"],
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
        num_heads=4
    )

    logger.info("PyTorch TransformerForecaster model built successfully")
    return model
