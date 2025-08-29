import pandas as pd
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger(__name__)

def load_dataset(path: str, datetime_col: str = None) -> pd.DataFrame:
    try:
        if datetime_col:
            df = pd.read_csv(path, parse_dates=[datetime_col])
        else:
            df = pd.read_csv(path)

        logger.info(f"Loaded dataset from {path} with shape {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error loading dataset from {path}: {str(e)}")
        raise

def load_data(config_path: str = "configs/config.yaml"):

    config = load_config(config_path)
    dataset_cfg = config["dataset"]

    train_path = dataset_cfg["train_path"]
    test_path = dataset_cfg["test_path"]
    datetime_col = dataset_cfg.get("datetime_col")

    train_df = load_dataset(train_path, datetime_col)
    test_df = load_dataset(test_path, datetime_col)

    return train_df, test_df
