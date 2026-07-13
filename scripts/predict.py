"""
Prediction script for fraud detection
"""

import sys
import os
import argparse
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.inference.predictor import FraudPredictor
from src.fraud_detection.utils.logger import logger

def main():
    parser = argparse.ArgumentParser(description='Predict fraud for transactions')
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSON file with transactions')
    parser.add_argument('--output', type=str, default='predictions.json',
                       help='Output file for predictions')
    parser.add_argument('--batch', action='store_true',
                       help='Process as batch')
    
    args = parser.parse_args()
    
    # Load input data
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    # Initialize predictor
    predictor = FraudPredictor()
    
    if args.batch:
        logger.info(f"Processing batch of {len(data)} transactions")
        results = predictor.predict_batch(data)
    else:
        logger.info("Processing single transaction")
        results = predictor.predict(data)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Predictions saved to {args.output}")
    
    # Print summary
    if args.batch:
        total = len(results)
        fraud_count = sum(1 for r in results if r.get('is_fraud', False))
        print(f"\nSummary:")
        print(f"Total transactions: {total}")
        print(f"Fraud detected: {fraud_count}")
        print(f"Fraud rate: {fraud_count/total*100:.2f}%")

if __name__ == "__main__":
    main()