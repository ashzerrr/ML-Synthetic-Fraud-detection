# ML-Synthetic-Fraud-detection

Project Overview

This project aims to detect fraudulent transactions using machine learning techniques. It evaluates and compares the performance of Random Forest and XGBoost models on a highly imbalanced dataset. Key evaluation metrics such as F1-score, ROC-AUC, PR-AUC, precision, and recall are used to assess the effectiveness of fraud detection.

Dataset

The dataset includes transaction records with the following key features:

Transaction details (e.g., transaction_id, amount, time_since_last_transaction)
Merchant information (e.g., merchant_fraud_rate, merchant_city_encoded, merchant_state_encoded)
User behavior (e.g., use_chip_encoded, fraud_count_x, mean_transaction_x)


Top 10 Features Used for Model Training

merchant_fraud_rate – Indicates the historical fraud rate of the merchant
fraud_count_x – Number of past fraudulent transactions associated with the user
zip_encoded – Encoded ZIP code for location-based risk assessment
use_chip_encoded – Whether a chip was used in the transaction (higher security)
mean_transaction_x – Average transaction amount for the user
transaction_id – Unique identifier for the transaction
mcc_encoded – Encoded merchant category code for business classification
merchant_city_encoded – Encoded merchant city for fraud pattern analysis
merchant_state_encoded – Encoded merchant state for geographic fraud trends
time_since_last_transaction – Time elapsed since the last transaction, identifying anomalies
Model Performance & Evaluation

Random Forest
Accuracy: 99.97%
F1-Score (Fraud Class): 0.89
ROC-AUC: 0.9872
PR-AUC: 0.9190


XGBoost
Accuracy: 95.78%
F1-Score (Fraud Class): 0.07
ROC-AUC: 0.9983
PR-AUC: 0.6867


Key Findings:
Random Forest has a balanced performance with high fraud precision (0.93) and recall (0.86).
XGBoost has perfect recall (1.00) but very low precision (0.03), leading to excessive false positives.
ROC-AUC is high for both models, but PR-AUC highlights that Random Forest is better suited for fraud detection in this dataset.
