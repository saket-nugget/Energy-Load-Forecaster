import os
import pandas as pd

from utils.config import load_config
from utils.data_loader import load_data
from utils.logger import get_logger

from src.preprocessing import preprocess_data
from src.model import TransformerForecaster
from src.train import train_model
from src.evaluate import evaluate_model
from src.anomaly_detection import detect_anomalies


def main():
    config = load_config("configs/config.yaml")
    logger = get_logger("Energy-Load-Forecaster", log_path="outputs/logs/run.log")

    logger.info("Pipeline started.")

    train_path = "data/train.csv"
    test_path = "data/test.csv"

    logger.info("Loading data...")
    train_df = load_data(train_path)
    test_df = load_data(test_path)

    logger.info("Preprocessing data...")
    X_train, y_train, scaler = preprocess_data(train_df, config)
    X_test, y_test, _ = preprocess_data(test_df, config, scaler=scaler)

    logger.info("Training model...")
    model = TransformerForecaster(
        input_dim=X_train.shape[-1],
        d_model=config["model"]["d_model"],
        nhead=config["model"]["nhead"],
        num_encoder_layers=config["model"]["num_layers"],
        dim_feedforward=config["model"]["ff_dim"],
        dropout=config["model"]["dropout"],
        output_dim=1
    )

    train_model(model, X_train, y_train, config)

    logger.info("Evaluating model...")
    predictions, metrics = evaluate_model(model, X_test, y_test, config)

    logger.info(f"Evaluation metrics: {metrics}")

    logger.info("Running anomaly detection...")
    anomalies = detect_anomalies(y_test, predictions)

    pd.DataFrame({"Actual": y_test.flatten(), "Predicted": predictions.flatten()}).to_csv(
        "outputs/forecasts/predictions.csv", index=False
    )
    anomalies.to_csv("outputs/anomalies/anomalies.csv", index=False)

    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    os.makedirs("outputs/forecasts", exist_ok=True)
    os.makedirs("outputs/anomalies", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    main()
