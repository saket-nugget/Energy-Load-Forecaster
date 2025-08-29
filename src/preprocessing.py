import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils.logger import get_logger

logger = get_logger(__name__)

class Preprocessor:
    def __init__(self, features=None, target=None):
        self.features = features
        self.target = target
        self.scaler = StandardScaler()

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.fillna(method="ffill").fillna(method="bfill")
        logger.info("Handled missing values")
        return df

    def feature_engineering(self, df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df["hour"] = df[datetime_col].dt.hour
        df["dayofweek"] = df[datetime_col].dt.dayofweek
        df["month"] = df[datetime_col].dt.month
        df["is_weekend"] = df[datetime_col].dt.dayofweek >= 5
        logger.info("Added time-based features")
        return df

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.features:
            self.features = [col for col in df.columns if col != self.target]

        df[self.features] = self.scaler.fit_transform(df[self.features])
        logger.info("Scaled features")
        return df

    def preprocess(self, df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
        df = self.handle_missing(df)
        df = self.feature_engineering(df, datetime_col)
        df = self.scale_features(df)
        return df
