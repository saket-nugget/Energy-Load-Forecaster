import pandas as pd
import os
from utils.logger import get_logger
from utils.config import load_config

logger = get_logger(__name__)

def load_dataset(file_path: str, datetime_col: str = None) -> pd.DataFrame:
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    if datetime_col and datetime_col in df.columns:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        logger.info(f"Loaded dataset with datetime parsing: {file_path}")
    else:
        logger.info(f"Loaded dataset without datetime parsing: {file_path}")

    return df


def merge_weather_data(df: pd.DataFrame, weather_df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    if weather_df is not None and datetime_col in df.columns and datetime_col in weather_df.columns:
        df = df.merge(weather_df, on=datetime_col, how="left")
        logger.info("Merged weather data with main dataset")
    else:
        logger.warning("Weather data not merged (missing datetime_col or weather_df)")
    return df


def load_data(config_path: str = "configs/config.yaml", weather_df: pd.DataFrame = None):
    config = load_config(config_path)
    dataset_cfg = config["dataset"]

    train_df = load_dataset(dataset_cfg["train_path"], dataset_cfg.get("datetime_col"))
    test_df = load_dataset(dataset_cfg["test_path"], dataset_cfg.get("datetime_col"))

    train_df = merge_weather_data(train_df, weather_df, dataset_cfg.get("datetime_col"))
    test_df = merge_weather_data(test_df, weather_df, dataset_cfg.get("datetime_col"))

    logger.info("Datasets loaded successfully")
    return train_df, test_df
