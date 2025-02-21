import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_predict, StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import defaultdict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Load data
bikedata = pd.read_csv("training_data_vt2025.csv")


# Normalization scaler
n_scaler = MinMaxScaler()
bikedata["humidity"] = n_scaler.fit_transform(bikedata[["humidity"]])
bikedata["cloudcover"] = n_scaler.fit_transform(bikedata[["cloudcover"]])

# Standardization
s_scaler = StandardScaler()
bikedata["temp"] = s_scaler.fit_transform(bikedata[["temp"]])
bikedata["dew"] = s_scaler.fit_transform(bikedata[["dew"]])
bikedata["windspeed"] = s_scaler.fit_transform(bikedata[["windspeed"]])

# Binary transformation
bikedata['precip'] = bikedata['precip'].apply(lambda x: 1 if x > 0 else 0)
bikedata['snow'] = bikedata['snowdepth'].apply(lambda x: 1 if x > 0 else 0)
bikedata['visibility'] = bikedata['visibility'].apply(lambda x: 1 if x >= 16 else 0)

# Sine & Cosine encoding
bikedata['hour_sin'] = np.sin(2 * np.pi * bikedata['hour_of_day'] / 24)
bikedata['hour_cos'] = np.cos(2 * np.pi * bikedata['hour_of_day'] / 24)
bikedata['day_sin'] = np.sin(2 * np.pi * bikedata['day_of_week'] / 7)
bikedata['day_cos'] = np.cos(2 * np.pi * bikedata['day_of_week'] / 7)
bikedata['month_sin'] = np.sin(2 * np.pi * bikedata['month'] / 12)
bikedata['month_cos'] = np.cos(2 * np.pi * bikedata['month'] / 12)

#bikedata = pd.read_csv("fully_processed_data1.csv")

X = bikedata.drop(columns=['increase_stock', 'hour_of_day', 'day_of_week', 'month',  'snowdepth' , 'cloudcover', 'windspeed', 'dew'])
y = bikedata['increase_stock']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Encode the target variable
mapping = {'low_bike_demand': 0, 'high_bike_demand': 1}
y_train_encoded = [mapping[label] for label in y_train]
y_test_encoded = [mapping[label] for label in y_test]
'''
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
'''


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
k_values = list(range(1, 50))

parameter_grid = {
    'n_neighbors' : (k_values),
    'weights' : ['uniform', 'distance'],
    'metric' : ['minkowski'],
    'p' : [1, 2, 3,4, 5]
}

KNN_model = KNeighborsClassifier()

grid_search = GridSearchCV(KNN_model, parameter_grid, cv = kf, scoring = 'f1', verbose = 1)

grid_search.fit(X_train, y_train_encoded)

best_parameters = grid_search.best_params_
best_f1 = grid_search.best_score_

print("\nBest Hyperparameters Found:")
print(best_parameters)
print("\n Best F1 Score from Cross-Validation:")
print(best_f1)



# Train final model with best hyperparameters
Final_KNN_model = KNeighborsClassifier(**best_parameters)
Final_KNN_model.fit(X_train, y_train_encoded)

y_pred = Final_KNN_model.predict(X_test)

# Evaluate model performance
print("Classification Report:\n")
print(classification_report(y_test_encoded, y_pred))
print("\n Confusion Matrix:\n")
print(confusion_matrix(y_test_encoded, y_pred))