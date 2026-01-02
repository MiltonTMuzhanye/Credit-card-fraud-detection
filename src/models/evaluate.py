import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, fbeta_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, cost_matrix=None):
        # Default cost matrix: FN cost = 100, FP cost = 10
        self.cost_matrix = cost_matrix or {'fn': 100, 'fp': 10, 'tp': 0, 'tn': 0}
        
    def calculate_business_cost(self, y_true, y_pred):
        """Calculate total business cost"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        total_cost = (
            self.cost_matrix['fn'] * fn +
            self.cost_matrix['fp'] * fp +
            self.cost_matrix['tp'] * tp +
            self.cost_matrix['tn'] * tn
        )
        
        return total_cost, {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
    
    def find_optimal_threshold(self, y_true, y_proba):
        """Find optimal threshold based on business cost"""
        thresholds = np.arange(0.1, 0.9, 0.05)
        costs = []
        
        for threshold in thresholds:
            y_pred = (y_proba > threshold).astype(int)
            cost, _ = self.calculate_business_cost(y_true, y_pred)
            costs.append(cost)
        
        optimal_idx = np.argmin(costs)
        return thresholds[optimal_idx], costs[optimal_idx]
    
    def evaluate_model(self, model, X_test, y_test, threshold=0.5):
        """Comprehensive model evaluation"""
        # Get predictions
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba > threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_proba),
            'average_precision': average_precision_score(y_test, y_proba),
            'f2_score': fbeta_score(y_test, y_pred, beta=2, pos_label=1),
            'precision': precision_score(y_test, y_pred, pos_label=1),
            'recall': recall_score(y_test, y_pred, pos_label=1),
        }
        
        # Calculate business cost
        total_cost, confusion = self.calculate_business_cost(y_test, y_pred)
        metrics['business_cost'] = total_cost
        metrics['confusion_matrix'] = confusion
        
        # Find optimal threshold
        optimal_threshold, optimal_cost = self.find_optimal_threshold(y_test, y_proba)
        metrics['optimal_threshold'] = optimal_threshold
        metrics['optimal_cost'] = optimal_cost
        
        # Classification report
        metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        
        return metrics, y_proba, y_pred
    
    def plot_metrics(self, metrics, y_test, y_proba, save_path=None):
        """Create evaluation plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        axes[0, 0].plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
        axes[0, 0].plot([0, 1], [0, 1], 'k--')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        axes[0, 1].plot(recall, precision)
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title(f'Precision-Recall Curve (AP = {metrics["average_precision"]:.3f})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, (y_proba > 0.5).astype(int))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        axes[1, 0].set_title('Confusion Matrix (threshold=0.5)')
        
        # Cost vs Threshold
        thresholds = np.arange(0.1, 0.9, 0.05)
        costs = []
        for threshold in thresholds:
            y_pred = (y_proba > threshold).astype(int)
            cost, _ = self.calculate_business_cost(y_test, y_pred)
            costs.append(cost)
        
        axes[1, 1].plot(thresholds, costs, 'b-')
        axes[1, 1].axvline(metrics['optimal_threshold'], color='r', linestyle='--', 
                          label=f'Optimal: {metrics["optimal_threshold"]:.2f}')
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Business Cost')
        axes[1, 1].set_title('Cost vs Decision Threshold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig