# src/preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.dataset_cfg = config.get("dataset")
        self.model_cfg = config.get("model")
        
        self.features = self.dataset_cfg["features"]
        self.target = self.dataset_cfg["target"]
        self.datetime_col = self.dataset_cfg["datetime_col"]
        
        self.input_length = self.model_cfg["input_length"]
        self.forecast_horizon = self.model_cfg["forecast_horizon"]
        
        self.scaler = StandardScaler()
        self.processed_columns_ = None

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.fillna(method="ffill").fillna(method="bfill")
        return df

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy[self.datetime_col] = pd.to_datetime(df_copy[self.datetime_col])
        df_copy["hour"] = df_copy[self.datetime_col].dt.hour
        df_copy["dayofweek"] = df_copy[self.datetime_col].dt.dayofweek
        df_copy["month"] = df_copy[self.datetime_col].dt.month
        df_copy["is_weekend"] = (df_copy[self.datetime_col].dt.dayofweek >= 5).astype(int)
        logger.info("Added time-based features.")
        return df_copy

    def _create_sequences(self, df: pd.DataFrame):
        X, y = [], []
        feature_cols = [self.target] + self.features
        
        for i in range(len(df) - self.input_length - self.forecast_horizon + 1):
            X.append(df[feature_cols].iloc[i : i + self.input_length].values)
            y.append(df[self.target].iloc[i + self.input_length : i + self.input_length + self.forecast_horizon].values)
        
        logger.info(f"Created {len(X)} sequences.")
        return np.array(X), np.array(y)

    def fit_transform(self, df: pd.DataFrame):
        logger.info("Fitting and transforming data (train set)...")
        df_processed = self._handle_missing(df)
        df_processed = self._feature_engineering(df_processed)

        self.processed_columns_ = [self.target] + self.features
        
        df_processed[self.processed_columns_] = self.scaler.fit_transform(df_processed[self.processed_columns_])
        logger.info("Scaler has been fitted and data transformed.")
        
        X, y = self._create_sequences(df_processed)
        return X, y

    def transform(self, df: pd.DataFrame):
        logger.info("Transforming data (validation/test set)...")
        if self.processed_columns_ is None:
            raise RuntimeError("The preprocessor has not been fitted yet. Call 'fit_transform' first.")
            
        df_processed = self._handle_missing(df)
        df_processed = self._feature_engineering(df_processed)

        df_processed[self.processed_columns_] = self.scaler.transform(df_processed[self.processed_columns_])
        logger.info("Data has been scaled using the existing scaler.")
        
        X, y = self._create_sequences(df_processed)
        return X, y

    def save(self, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self, file_path)
        logger.info(f"Preprocessor saved to {file_path}")

    @staticmethod
    def load(file_path: str):
        logger.info(f"Loading preprocessor from {file_path}")
        return joblib.load(file_path)