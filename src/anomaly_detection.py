# src/anomaly_detection.py

import os
import numpy as np
import pandas as pd

def detect_anomalies(actuals, predictions, save_path="outputs/anomalies/anomalies.csv"):
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    actuals = np.array(actuals).flatten()
    predictions = np.array(predictions).flatten()

    errors = np.abs(actuals - predictions)
    threshold = np.mean(errors) + 3 * np.std(errors)

    anomalies = errors > threshold

    results = pd.DataFrame({
        "Actual": actuals,
        "Predicted": predictions,
        "Error": errors,
        "Anomaly": anomalies
    })

    results.to_csv(save_path, index=False)
    print(f"[INFO] Anomalies saved to {save_path}")

    return results
