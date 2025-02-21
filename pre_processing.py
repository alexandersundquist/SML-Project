# Import packages
import pandas as pd
import numpy as np

# Import data
data = pd.read_csv('training_data_vt2025.csv')

def cyclical_encoding(df, column, period):
    df[column + '_sin'] = np.round(np.sin(2 * np.pi * df[column] / period), 6)
    df[column + '_cos'] = np.round(np.cos(2 * np.pi * df[column] / period), 6)
    df.drop(columns=[column], inplace=True)  # Remove the original column
    return df

def pre_processing(data):
    # Make copy of dataset
    data_processed = data.copy()

    # Create new summertime feature
    data_processed['is_summer'] = ((data_processed['month'] >= 3) & (data_processed['month'] <= 11)).astype(int)

    # Normalize calendar data using cosine encoding
    data_processed = cyclical_encoding(data_processed, 'day_of_week', 7)
    data_processed = cyclical_encoding(data_processed, 'hour_of_day', 24)
    data_processed = cyclical_encoding(data_processed, 'month', 12) 

    # Give target feature numerical values
    data_processed['increase_stock'] = data_processed['increase_stock'].replace({'high_bike_demand': 1, 'low_bike_demand': 0})

    # Create binary category of features
    data_processed['is_raining'] = (data_processed['precip'] != 0).astype(int)
    data_processed['is_snowing'] = (data_processed['snowdepth'] != 0).astype(int)
    data_processed['is_visible'] = (data_processed['visibility'] != 16).astype(int)

    # Drop columns
    data_processed = data_processed.drop(columns=['holiday', 'snow', 'snowdepth', 'precip', 'visibility', 'summertime'])

    return data_processed

new_data= pre_processing(data)

# Save the processed data
new_data.to_csv("preprocessed_data_1.csv", index=False)

print("Preprocessing complete. File saved as preprocessed_data_1.csv")