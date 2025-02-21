import pandas as pd
from sklearn.dummy import DummyClassifier
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Read the preprocessed data

# Construct the full path to the CSV file
csv_file_path_pre_processed = Path.cwd().parent / 'preprocessed_data_1.csv'

# Read the CSV file using pandas
data = pd.read_csv(csv_file_path_pre_processed)

# Split the data into input values, X, and output value, y
X = data.drop(columns=['increase_stock'])
y = data['increase_stock']

# Split Data into Train & Test Sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Create a Dummy Classifier (always predicts the majority class)
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)

# Make predictions
y_dummy_pred = dummy_clf.predict(X_test)

# Evaluate performance
dummy_accuracy = accuracy_score(y_test, y_dummy_pred)
dummy_f1 = f1_score(y_test, y_dummy_pred, zero_division=1)  # Avoid division errors
dummy_roc_auc = roc_auc_score(y_test, y_dummy_pred)

# Print Results
print("Dummy Classifier Performance (Majority Class Strategy):")
print(f"Accuracy: {dummy_accuracy:.4f}")
print(f"F1-Score: {dummy_f1:.4f}")
print(f"ROC-AUC Score: {dummy_roc_auc:.4f}")