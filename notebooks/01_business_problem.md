# Fraud Detection System - Business Problem

## 1. Problem Statement
Credit card fraud detection with extreme class imbalance (0.17% fraud). 
The goal is to minimize financial losses while maintaining customer experience.

## 2. Business Objectives
- **Primary**: Detect fraudulent transactions with high recall
- **Secondary**: Minimize false positives to avoid customer friction
- **Tertiary**: Provide explainable predictions for compliance

## 3. Success Metrics
- **Recall**: >90% (catch fraud)
- **Precision**: >80% (minimize false alarms)
- **AUC-ROC**: >0.95
- **Business Cost**: Minimize $Cost = 100*FN + 10*FP

## 4. Data Characteristics
- 284,807 transactions
- 31 features (28 PCA components + Time + Amount + Class)
- Highly imbalanced (492 frauds vs 284,315 legitimate)

## 5. Cost Matrix
|                  | Predicted Fraud | Predicted Legitimate |
|------------------|-----------------|----------------------|
| **Actual Fraud** | $0 (TN)         | $100 (FN)            |
| **Actual Legit** | $10 (FP)        | $0 (TN)              |

## 6. Deployment Requirements
- Real-time prediction (< 100ms)
- 99.9% uptime
- Model retraining weekly
- Audit trail for compliance