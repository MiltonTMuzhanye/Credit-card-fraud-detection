# Credit_card_fraud_detection

This project applies machine learning models to detect fraudulent credit card transactions using an imbalanced dataset. The objective is to identify fraudulent transactions as accurately as possible while handling the class imbalance effectively.

## Dataset
  Source: https://www.kaggle.com/mlg-ulb/creditcardfraud
  
## Project Workflow

  ### Exploratory Data Analysis (EDA)
   - Class distribution visualization
   - Transaction amount patterns by fraud status
   - Scatter plots of Time vs Amount by class
   - Correlation analysis to identify top features most linked with fraud
   - 
  ### Preprocessing
   - Drop unnecessary columns (Time)
   - Standardize Amount using StandardScaler
   - Use a ColumnTransformer for clean preprocessing pipelines
   - Stratified train-test split to preserve class distribution
   - 
  ### Modeling
    Trained and evaluated three classifiers:
       - Logistic Regression 
       - Random Forest Classifier
       - XGBoost Classifier
       
  ### Evaluation Metrics
    Each model was evaluated using:
     - ROC AUC Score
     - Classification Report (Precision, Recall, F1 score)
     - Confusion Matrix
     - Precision-Recall Curve
     - Cross-Validation ROC AUC Scores
     
  ## Results Summary

    | Model              | ROC AUC | Precision | Recall  |
    |-------------------|---------|-----------|---------|
    | LogisticRegression| 0.971   | 0.059     | 0.918   |
    | XGBoost           | 0.967   | 0.894     | 0.857   |
    | RandomForest      | 0.958   | 0.961     | 0.745   |

Best model (based on ROC AUC): LogisticRegression

## Visualizations
  - Class distribution bar chart
  - Transaction amount boxplot & histogram
  - Correlation plot
  - Precision-Recall Curve
  - Feature Importance

## Conclusion
  This project successfully applied machine learning to detect credit card fraud, with Logistic Regression achieving the best ROC AUC score (0.971)
