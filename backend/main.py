# main.py

import os
from utils.config import Config
from utils.logger import get_logger
from src.train import run_training
from src.evaluate import run_evaluation
from src.anomaly_detection import detect_anomalies

def main():
    """
    Main pipeline orchestrator.
    """
    config = Config("configs/config.yaml")
    logger = get_logger("main_pipeline")
    
    logger.info("="*50)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("="*50)
    run_training(config)
    
    logger.info("="*50)
    logger.info("STARTING EVALUATION PIPELINE")
    logger.info("="*50)
    actuals, predictions = run_evaluation(config)

    logger.info("="*50)
    logger.info("STARTING ANOMALY DETECTION")
    logger.info("="*50)
    
    # THIS IS THE ONLY CHANGE IN THIS FILE:
    # Pass the entire config object to the function
    detect_anomalies(actuals, predictions, config)

    logger.info("FULL PIPELINE FINISHED SUCCESSFULLY!")

if __name__ == "__main__":
    main()