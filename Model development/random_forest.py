import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

#  Load dataset
data = pd.read_csv("preprocessed_data_1.csv")

# Separate features (X) and target (y)
X = data.drop(columns=['increase_stock'])
y = data['increase_stock']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Decide amount of folds and do stratified K-fold
k_folds = 5
kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42) # shuffle ensures the data is randomized before splitting.

# Defines the hyperparameters to test in GridSearchCV
param_grid = {
  'n_estimators': [100, 300],      # Number of trees (more for stability)
  'max_depth': [15, 20],           # Maximum depth of each tree
  'min_samples_split': [5, 10],    # Minimum samples required to split a node (more to increase recall)
  'min_samples_leaf': [2, 4],      # Minimum samples per leaf node (Larger generelize better)
  'max_features': ['sqrt'],        # Each tree only uses square root of total features per split
  'class_weight': ['balanced']     # Adjusts weights to handle imbalanced data
}

# Initialize the tandom forest classifier
rf_model = RandomForestClassifier(random_state=42)

# Grid search with cross-validation to find the best hyperparameters
grid_search = GridSearchCV(
    estimator = rf_model,           # Random Forest as the base model
    param_grid = param_grid,        # Tests all combinations of hyperparameters
    cv = kfold,                     # Stratified K-Fold cross-validation (5 folds)
    scoring = 'f1',                 # Optimizes the F1-score (useful for imbalanced datasets)
    n_jobs = -1,                    # Use all available CPU cores
    verbose = 1                     # Displays progress updates
)

# Train model with grid_search
grid_search.fit(X_train, y_train)

# Get best hyperparameters & best F1 score
best_params = grid_search.best_params_
best_f1 = grid_search.best_score_

# Print best hyperparameters and F1 Score
print("\nBest Hyperparameters Found:")
print(best_params)
print("\nBest F1 Score from Cross-Validation:")
print(best_f1)

# Train final model with best hyperparameters
best_rf = RandomForestClassifier(**best_params, random_state=42)
best_rf.fit(X_train, y_train)

# Make predictions on test data
y_pred = best_rf.predict(X_test)

# Evaluate model performance
print("Classification Report:\n")
print(classification_report(y_test, y_pred)) # Confusion Matrix: Top: TP, FP & Bottom: FN, TN
print("\n Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# Evaluating final model performance

f1 = f1_score(y_test, y_pred)                 # Measures the balance between precision & recall (good for imbalanced data)
accuracy = accuracy_score(y_test, y_pred)     # Overall correctness of predictions
precision = precision_score(y_test, y_pred)   # Proportion of positive predictions that were actually correct
recall = recall_score(y_test, y_pred)         # Proportion of actual positive cases correctly identified
roc_auc = roc_auc_score(y_test, y_pred)       # Measures model’s ability to distinguish between classes (higher = better)

# Print results
print("\nSummary of Model Performance:")
print(f"- F1 Score: {f1:.4f} ")
print(f"- Accuracy: {accuracy:.4f} ")
print(f"- Precision: {precision:.4f} ")
print(f"- Recall: {recall:.4f} ")
print(f"- AUC-ROC: {roc_auc:.4f} ")