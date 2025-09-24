# utils/data_loader.py

import pandas as pd
import os
from utils.logger import get_logger

logger = get_logger(__name__)

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads a dataset from a CSV file.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    logger.info(f"Loaded dataset from {file_path} with shape {df.shape}")
    
    return df