"""
Model training script for fraud detection system
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.fraud_detection.data.ingestion import DataIngestion
from src.fraud_detection.data.preprocessing import DataPreprocessor
from src.fraud_detection.pipelines.training_pipeline import TrainingPipeline
from src.fraud_detection.utils.logger import logger
from src.fraud_detection.utils.helpers import save_object, save_json

def main():
    parser = argparse.ArgumentParser(description='Train fraud detection model')
    parser.add_argument('--data_path', type=str, default='data/raw/creditcard.csv',
                       help='Path to the credit card dataset')
    parser.add_argument('--config_path', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test size ratio')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--save_artifacts', action='store_true',
                       help='Save trained artifacts')
    
    args = parser.parse_args()
    
    logger.info("Starting training pipeline...")
    
    # Load data
    ingestion = DataIngestion(args.data_path)
    df = ingestion.load_csv()
    logger.info(f"Loaded data with shape: {df.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = ingestion.split_data(
        df, target_col='Class', 
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Initialize training pipeline
    pipeline = TrainingPipeline(args.config_path)
    
    # Run full pipeline
    evaluation_results, best_model = pipeline.run_full_pipeline(
        X_train, y_train, X_test, y_test
    )
    
    # Save artifacts
    if args.save_artifacts:
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = artifacts_dir / "trained_models" / "best_model.pkl"
        model_path.parent.mkdir(exist_ok=True)
        save_object(best_model, model_path)
        
        # Save scaler if exists
        if hasattr(pipeline, 'scaler'):
            scaler_path = artifacts_dir / "scalers" / "amount_scaler.pkl"
            scaler_path.parent.mkdir(exist_ok=True)
            save_object(pipeline.scaler, scaler_path)
        
        # Save evaluation results
        results_path = artifacts_dir / "thresholds" / "evaluation_results.json"
        results_path.parent.mkdir(exist_ok=True)
        save_json(evaluation_results, results_path)
        
        # Save performance metrics
        metrics = {
            model_name: result['metrics']
            for model_name, result in evaluation_results.items()
        }
        metrics_path = artifacts_dir / "thresholds" / "performance_metrics.json"
        save_json(metrics, metrics_path)
        
        logger.info("Saved all artifacts")
    
    logger.info("Training completed successfully")
    
    return evaluation_results

if __name__ == "__main__":
    main()