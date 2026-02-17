# src/anomaly_detection.py

import os
import numpy as np
import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)

def detect_anomalies(actuals, predictions, config):
    """
    Detects anomalies based on the difference between actuals and predictions.
    The detection method is controlled by the configuration.
    """
    anomaly_cfg = config.get("anomaly_detection")
    output_path = os.path.join(config.get("output", "anomalies_dir"), "anomalies.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    actuals = np.array(actuals).flatten()
    predictions = np.array(predictions).flatten()
    errors = np.abs(actuals - predictions)

    # Use the method from the config file to determine the threshold
    method = anomaly_cfg.get("method", "zscore") # Default to zscore if not specified
    if method == "zscore":
        threshold_val = anomaly_cfg.get("threshold", 3.0)
        threshold = np.mean(errors) + threshold_val * np.std(errors)
        logger.info(f"Using z-score method with a threshold of {threshold_val} std deviations.")
    # You could add other methods here like "iqr", "static", etc.
    # elif method == "iqr":
    #     ...
    else:
        logger.warning(f"Unknown anomaly detection method '{method}'. Defaulting to z-score.")
        threshold = np.mean(errors) + 3.0 * np.std(errors)

    anomalies = errors > threshold

    results = pd.DataFrame({
        "Actual": actuals,
        "Predicted": predictions,
        "Error": errors,
        "Anomaly": anomalies
    })

    results.to_csv(output_path, index=False)
    logger.info(f"Detected {np.sum(anomalies)} anomalies. Results saved to {output_path}")

    return results