# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE  # For handling class imbalance
from sklearn.preprocessing import StandardScaler

# 1. Load Dataset
credit_card_data = pd.read_csv('/content/credit_data.csv')

# 2. Exploratory Data Analysis
print("First 5 rows of the dataset:")
print(credit_card_data.head())
print("\nDataset Information:")
print(credit_card_data.info())

# Check for missing values
print("\nMissing Values:")
print(credit_card_data.isnull().sum())

# Distribution of Legit and Fraudulent Transactions
class_counts = credit_card_data['Class'].value_counts()
print("\nClass Distribution:")
print(class_counts)

# Visualizing Class Distribution
sns.countplot(x='Class', data=credit_card_data)
plt.title("Transaction Class Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = credit_card_data.corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap")
plt.show()

# 3. Data Preprocessing
# Fix missing values in the Class column
print("\nMissing values in Class column before cleaning:")
print(credit_card_data['Class'].isnull().sum())

# Drop rows with missing target values
credit_card_data = credit_card_data.dropna(subset=['Class'])

# Separating the features and target variable
X = credit_card_data.drop(columns='Class', axis=1)
Y = credit_card_data['Class']

# Standardize the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, Y_resampled = smote.fit_resample(X_scaled, Y)
print("\nClass Distribution After SMOTE:")
print(pd.Series(Y_resampled).value_counts())

# 4. Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X_resampled, Y_resampled, test_size=0.2, stratify=Y_resampled, random_state=42
)

print(f"\nShape of Training Data: {X_train.shape}")
print(f"Shape of Testing Data: {X_test.shape}")

# 5. Model Training
model = LogisticRegression(random_state=42)
model.fit(X_train, Y_train)

# 6. Model Evaluation
# Training Accuracy
train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_predictions)
print("\nTraining Accuracy:", train_accuracy)

# Testing Accuracy
test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, test_predictions)
print("Testing Accuracy:", test_accuracy)

# Detailed Evaluation
print("\nClassification Report on Test Data:")
print(classification_report(Y_test, test_predictions))

# Confusion Matrix
conf_matrix = confusion_matrix(Y_test, test_predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()

# ROC-AUC Score and Curve
roc_auc = roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1])
print("ROC-AUC Score:", roc_auc)

fpr, tpr, thresholds = roc_curve(Y_test, model.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()