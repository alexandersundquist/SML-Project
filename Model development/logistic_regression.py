import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score, accuracy_score, precision_score, recall_score

# Load Preprocessed data
file_path = "preprocessed_data_1.csv"
df = pd.read_csv(file_path)

# Drop increase stock
X = df.drop(columns=['increase_stock'])
y = df['increase_stock']

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Stratified k-fold
k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# Hyperparameter for gridsearch
param_grid = [
    {'C': [0.1, 0.5, 1, 5, 10, 20], 'penalty': ['l1'], 'solver': ['saga'], 'max_iter': [500]},
    {'C': [0.1, 0.5, 1, 5, 10, 20], 'penalty': ['l2'], 'solver': ['liblinear'],  'max_iter': [500]}
]

# Logistic regression model
logistic_model = LogisticRegression(random_state=42)

# Grid search cross-validation
grid_search = GridSearchCV(
    logistic_model, param_grid, cv=skf, scoring='f1', n_jobs=-1, verbose=1
)

grid_search.fit(X_train, y_train)

# Print the best parameters and f1 score
best_params = grid_search.best_params_
best_f1 = grid_search.best_score_

print("\n Best Hyperparameters Found:")
print(best_params)
print(f"Best F1 Score from Cross-Validation: {best_f1:.4f}")

# Train the final Model with best parameters
final_model = LogisticRegression(**best_params, random_state=42)
final_model.fit(X_train, y_train)

y_prob_test = final_model.predict_proba(X_test)[:, 1]

# Threshold tuning
thresholds = np.arange(0.4, 0.6, 0.8)
f1_scores = []

for threshold in thresholds:
    y_pred = (y_prob_test >= threshold).astype(int)
    f1_scores.append(f1_score(y_test, y_pred))

best_threshold = thresholds[np.argmax(f1_scores)]
best_f1_final = max(f1_scores)

print(f"\n Best Threshold Found: {best_threshold:.2f}")
print(f"Best F1-Score at Best Threshold: {best_f1_final:.4f}")

# Best threshold
y_pred_final = (y_prob_test >= best_threshold).astype(int)

# Evaluate final model
final_accuracy = accuracy_score(y_test, y_pred_final)
final_precision = precision_score(y_test, y_pred_final)
final_recall = recall_score(y_test, y_pred_final)
final_f1 = f1_score(y_test, y_pred_final)
final_roc_auc = roc_auc_score(y_test, y_prob_test)

print("\n Final Model Evaluation:")
print(f"Accuracy: {final_accuracy:.4f}")
print(f"Precision: {final_precision:.4f}")
print(f"Recall: {final_recall:.4f}")
print(f"F1-Score: {final_f1:.4f}")
print(f"ROC-AUC Score: {final_roc_auc:.4f}")

# Classification report
print("\n Classification Report:\n")
print(classification_report(y_test, y_pred_final))