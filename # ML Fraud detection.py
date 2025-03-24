# ML Fraud detection

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.metrics import ConfusionMatrixDisplay

# Load the data
transactions_df = pd.read_csv("transactions_data.csv")
cards_df = pd.read_csv("cards_data.csv")
users_df = pd.read_csv("users_data.csv")

with open("mcc_codes.json", "r") as f:
    mcc_codes_df = json.load(f)
with open("train_fraud_labels.json", "r") as f:
    train_fraud_labels_df = json.load(f)

#Clean Transactions Data

# Convert the dictionary to a DataFrame
train_fraud_labels_df = pd.DataFrame(list(train_fraud_labels_df["target"].items()), columns=['transaction_id', 'Fraud label'])

# Convert 'transaction_id' to int for proper merging
train_fraud_labels_df['transaction_id'] = train_fraud_labels_df['transaction_id'].astype(int)

# Merge the transactions dataframe with the fraud labels dataframe
merged_df = pd.merge(transactions_df, train_fraud_labels_df, how='left', left_on='id', right_on='transaction_id')

#Some transactions have missing fraud labels
# Separate the labeled and unlabeled transactions
labeled_transactions = merged_df.dropna(subset=['Fraud label'])
unlabeled_transactions = merged_df[merged_df['Fraud label'].isna()]

labeled_merged_df = labeled_transactions.copy()


#Binary encode the 'Fraud label' column
labeled_merged_df['Fraud label'] = labeled_merged_df['Fraud label'].apply(lambda x: 1 if x == 'Yes' else (0 if x == 'No' else -1))

# label encoding for errors
label_encoder = LabelEncoder()
labeled_merged_df['errors_encoded'] = label_encoder.fit_transform(labeled_merged_df['errors'])

#label encoding for mcc_code
labeled_merged_df['mcc_encoded'] = label_encoder.fit_transform(labeled_merged_df['mcc'])

#label encoding for zip
labeled_merged_df['zip_encoded'] = label_encoder.fit_transform(labeled_merged_df['zip'])

#label encoding for state
labeled_merged_df['merchant_state_encoded'] = label_encoder.fit_transform(labeled_merged_df['merchant_state'])

#label encoding for city
labeled_merged_df['merchant_city_encoded'] = label_encoder.fit_transform(labeled_merged_df['merchant_city'])

#label encoding for merchant_id
labeled_merged_df['merchant_id_encoded'] = label_encoder.fit_transform(labeled_merged_df['merchant_id'])

#label encoding use_chip
labeled_merged_df['use_chip_encoded'] = label_encoder.fit_transform(labeled_merged_df['use_chip'])

# convert amount to float
labeled_merged_df['amount'] = labeled_merged_df['amount'].replace(r'[\$,]', '', regex=True).astype(float)

# convert date to datetime format
labeled_merged_df['date'] = pd.to_datetime(labeled_merged_df['date']) 
labeled_merged_df['year'] = labeled_merged_df['date'].dt.year
labeled_merged_df['month'] = labeled_merged_df['date'].dt.month
labeled_merged_df['day'] = labeled_merged_df['date'].dt.day
labeled_merged_df['hour'] = labeled_merged_df['date'].dt.hour
labeled_merged_df['minute'] = labeled_merged_df['date'].dt.minute
labeled_merged_df['second'] = labeled_merged_df['date'].dt.second


labeled_merged_df_encode_mapping = labeled_merged_df[['use_chip', 'use_chip_encoded', 'merchant_id', 'merchant_id_encoded',
                                                       'merchant_city', 'merchant_city_encoded', 'merchant_state', 'merchant_state_encoded', 
                                                       'zip', 'zip_encoded', 'mcc', 'mcc_encoded', 'errors', 'errors_encoded']]

# Drop the columns that have been encoded
labeled_merged_df = labeled_merged_df.drop(['use_chip', 'merchant_id', 'merchant_city', 'merchant_state', 'zip', 'mcc', 'errors'], axis=1)


# Cleaning for Users_data

#convert to numeric and remove dollar sign on continous variables

users_df['yearly_income'] = users_df['yearly_income'].replace(r'[\$,]', '', regex=True).astype('float')
users_df['per_capita_income'] = users_df['per_capita_income'].replace(r'[\$,]', '', regex=True).astype('float')
users_df['total_debt'] = users_df['total_debt'].replace(r'[\$,]', '', regex=True).astype('float')


#label encoding for gender

users_df['gender_encoded'] = label_encoder.fit_transform(users_df['gender'])

# drop unused columns 
users_df=users_df.drop(columns=['birth_year','birth_month','address','gender',])



# cleaning for Cards

# label encoding for card_number
cards_df['card_number_encoded'] = label_encoder.fit_transform(cards_df['card_number'])
# label encoding for card brand
cards_df['card_brand_encoded'] = label_encoder.fit_transform(cards_df['card_brand'])

# label encoding for card_type
cards_df['card_type_encoded'] = label_encoder.fit_transform(cards_df['card_type'])

#label encoding for cvv
cards_df['cvv_encoded']=label_encoder.fit_transform(cards_df['cvv'])

#label encoding for has_chip
cards_df['has_chip_encoded']=label_encoder.fit_transform(cards_df['has_chip'])

#label encoding for card_on_dark_web
cards_df['card_on_dark_web_encoded']=label_encoder.fit_transform(cards_df['card_on_dark_web'])

# convert credit limit in float and remove $

cards_df['credit_limit'] = cards_df['credit_limit'].replace(r'[\$,]', '', regex=True).astype('float')

# Convert the 'expires' column to datetime and extract year and month 
cards_df['expires_year'] = pd.to_datetime(cards_df['expires'],format='%m/%Y').dt.year
cards_df['expires_month'] = pd.to_datetime(cards_df['expires'],format='%m/%Y').dt.month

#convert 'acc_open_year to datetime and extract year and month
cards_df['acct_open_year'] = pd.to_datetime(cards_df['acct_open_date'],format='%m/%Y').dt.year
cards_df['acct_open_month'] = pd.to_datetime(cards_df['acct_open_date'],format='%m/%Y').dt.month

#drop columns
cards_df=cards_df.drop(columns=['card_on_dark_web','has_chip', 'cvv','card_type','card_number','card_brand','expires','acct_open_date'])


# Merge the dataframes
labeled_merged_df = labeled_merged_df.merge(users_df, how='left', left_on='client_id', right_on='id')


labeled_merged_df = labeled_merged_df.merge(cards_df, how='left', left_on='card_id', right_on='id')


# Drop the columns that are not needed
labeled_merged_df = labeled_merged_df.drop(['id_x', 'id_y', 'client_id_x','client_id_y','current_age','id'], axis=1)

print(labeled_merged_df.info())

#Data visualization 

# class imbalance

plt.figure(figsize=(6, 4))
sns.countplot(x="Fraud label", data=labeled_merged_df, palette="coolwarm")
plt.title("Fraud vs. Non-Fraud Transactions")
plt.show()

# Distribution of transaction amounts
plt.figure(figsize=(8, 5))
sns.histplot(labeled_merged_df['amount'], bins=50, kde=True)
plt.title("Distribution of Transaction Amounts")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.show()

#Heat map of feature correlation
plt.figure(figsize=(28, 20))
sns.heatmap(labeled_merged_df.corr(), cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Transaction frequency by hour
plt.figure(figsize=(10, 5))
sns.countplot(x="hour", data=labeled_merged_df, hue="Fraud label", palette="coolwarm")
plt.title("Transaction Frequency by Hour (Fraud vs. Non-Fraud)")
plt.xlabel("Hour of the Day")
plt.ylabel("Transaction Count")
plt.show()

# Feature Engineering

# Flag transactions in the top 5% as 'high_amount'
threshold = labeled_merged_df['amount'].quantile(0.95)
labeled_merged_df['high_amount'] = (labeled_merged_df['amount'] > threshold).astype(int)

# Calculate the time since the last transaction
labeled_merged_df = labeled_merged_df.sort_values(by=['date'])
labeled_merged_df['time_since_last_transaction'] = labeled_merged_df['date'].diff().dt.total_seconds().fillna(0)

# Card activity count

card_activity = labeled_merged_df.groupby('card_id').agg({
    'amount': ['mean', 'std', 'max'],
    'Fraud label': 'sum'  # Total frauds associated with the card
})
card_activity.columns = ['mean_transaction', 'std_transaction', 'max_transaction', 'fraud_count']
labeled_merged_df = labeled_merged_df.merge(card_activity, on="card_id", how="left")

#Merchant fraud likelihood

merchant_fraud_rate = labeled_merged_df.groupby('merchant_id_encoded')['Fraud label'].mean().fillna(0)
labeled_merged_df['merchant_fraud_rate'] = labeled_merged_df['merchant_id_encoded'].map(merchant_fraud_rate)


print(labeled_merged_df.info()) 

#model implementation

# Define the top 10 most important features (based on previous feature importance results and HeatMap)
top_10_features = [
    "merchant_fraud_rate", "fraud_count", "zip_encoded", "use_chip_encoded",
    "mean_transaction", "transaction_id", "mcc_encoded", "merchant_city_encoded",
    "merchant_state_encoded", "time_since_last_transaction"
]

# Filter dataset to include only the selected features
X_selected = labeled_merged_df[top_10_features]
y = labeled_merged_df['Fraud label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to oversample the minority class in the training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Random Forest with class weights
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

# XGBoost with class weights
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
                    use_label_encoder=False, eval_metric="logloss", random_state=42)

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Random Forest Cross-Validation
rf_scores = cross_val_score(rf, X_train_smote, y_train_smote, cv=cv, scoring='accuracy')
print(f"Random Forest - Mean Accuracy: {np.mean(rf_scores):.4f}, Std Dev: {np.std(rf_scores):.4f}")

# XGBoost Cross-Validation
xgb_scores = cross_val_score(xgb, X_train_smote, y_train_smote, cv=cv, scoring='accuracy')
print(f"XGBoost - Mean Accuracy: {np.mean(xgb_scores):.4f}, Std Dev: {np.std(xgb_scores):.4f}")

# Train Models
rf.fit(X_train_smote, y_train_smote)
xgb.fit(X_train_smote, y_train_smote)

# Predictions
rf_preds = rf.predict(X_test)
xgb_preds = xgb.predict(X_test)

# Print Accuracy & Classification Report
print("\nRandom Forest Test Results:")
print(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
print(classification_report(y_test, rf_preds))

print("\nXGBoost Test Results:")
print(f"Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")
print(classification_report(y_test, xgb_preds))

# Calculate ROC-AUC
rf_probs = rf.predict_proba(X_test)[:, 1]
xgb_probs = xgb.predict_proba(X_test)[:, 1]
rf_roc_auc = roc_auc_score(y_test, rf_probs)
xgb_roc_auc = roc_auc_score(y_test, xgb_probs)
print(f"Random Forest ROC-AUC: {rf_roc_auc:.4f}")
print(f"XGBoost ROC-AUC: {xgb_roc_auc:.4f}")

# Calculate PR-AUC
rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_probs)
rf_pr_auc = auc(rf_recall, rf_precision)
xgb_precision, xgb_recall, _ = precision_recall_curve(y_test, xgb_probs)
xgb_pr_auc = auc(xgb_recall, xgb_precision)
print(f"Random Forest PR-AUC: {rf_pr_auc:.4f}")
print(f"XGBoost PR-AUC: {xgb_pr_auc:.4f}")

# Plot Feature Importances for Random Forest
rf_importances = pd.DataFrame({'Feature': top_10_features, 'Importance': rf.feature_importances_})
rf_importances = rf_importances.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=rf_importances['Importance'], y=rf_importances['Feature'])
plt.title("Random Forest - Feature Importance")
plt.show()

# Plot Feature Importances for XGBoost
xgb_importances = pd.DataFrame({'Feature': top_10_features, 'Importance': xgb.feature_importances_})
xgb_importances = xgb_importances.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=xgb_importances['Importance'], y=xgb_importances['Feature'])
plt.title("XGBoost - Feature Importance")
plt.show()

# Plot ROC Curve for Random Forest and XGBoost
plt.figure(figsize=(10, 6))
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {rf_roc_auc:.4f})")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {xgb_roc_auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# Plot Precision-Recall Curve for Random Forest and XGBoost
plt.figure(figsize=(10, 6))
plt.plot(rf_recall, rf_precision, label=f"Random Forest (PR-AUC = {rf_pr_auc:.4f})")
plt.plot(xgb_recall, xgb_precision, label=f"XGBoost (PR-AUC = {xgb_pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend()
plt.show()

# Plot ROC Curve for Random Forest and XGBoost
plt.figure(figsize=(10, 6))
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {rf_roc_auc:.4f})")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {xgb_roc_auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# Plot Precision-Recall Curve for Random Forest and XGBoost
plt.figure(figsize=(10, 6))
plt.plot(rf_recall, rf_precision, label=f"Random Forest (PR-AUC = {rf_pr_auc:.4f})")
plt.plot(xgb_recall, xgb_precision, label=f"XGBoost (PR-AUC = {xgb_pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend()
plt.show()


# Random Forest Confusion Matrix
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay(confusion_matrix(y_test, rf_preds), display_labels=["Not Fraud", "Fraud"]).plot(cmap="Blues")
plt.title("Random Forest - Confusion Matrix")
plt.show()

# XGBoost Confusion Matrix
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay(confusion_matrix(y_test, xgb_preds), display_labels=["Not Fraud", "Fraud"]).plot(cmap="Oranges")
plt.title("XGBoost - Confusion Matrix")
plt.show()

# Combine Feature Importance from Both Models
feature_importances = pd.DataFrame({
    'Feature': top_10_features,
    'Random Forest Importance': rf.feature_importances_,
    'XGBoost Importance': xgb.feature_importances_
}).melt(id_vars='Feature', var_name='Model', value_name='Importance')

# Plot Combined Feature Importance
plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importances, x='Importance', y='Feature', hue='Model', palette="viridis")
plt.title("Feature Importance Comparison")
plt.show()
