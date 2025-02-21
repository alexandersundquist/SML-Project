# Import packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms

#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('png')
from IPython.core.pylabtools import figsize
figsize(10, 6) # Width and hight
#plt.style.use('seaborn-white')

# Import training data
data = pd.read_csv('training_data_vt2025.csv')

print(data.head())

print(data.info())

print(data.describe().T)

# Plot distributions of each feature
data.hist(figsize=(12, 8), bins=30, edgecolor="black")
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

# Plot boxplots for each feature
plt.figure(figsize=(12, 6))
data.plot(kind="box", subplots=True, layout=(5, 4), figsize=(14, 8), sharex=False, sharey=False)
plt.suptitle("Box Plots for Outlier Detection", fontsize=16)
plt.show()

# Plot correlation heatmap
df_encoded = data.copy()
df_encoded["increase_stock"] = df_encoded["increase_stock"].astype('category').cat.codes  # Encode target as numeric
corr = df_encoded.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()


# Plot time features against increase_stock

# Convert target variable to string if needed
data['increase_stock'] = data['increase_stock'].astype(str)

# Define categorical features
categorical_features = ['month', 'hour_of_day', 'day_of_week']

# Create a figure with subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # Adjust size as needed

# Loop through features and plot each on a separate subplot
for ax, feature in zip(axes, categorical_features):
    data.groupby(feature)['increase_stock'].value_counts().unstack().plot(kind='bar', ax=ax)
    ax.set_title(f'{feature} vs. Increase Stock')
    ax.set_xlabel(feature)
    ax.set_ylabel('Count')
    ax.legend(title='Increase Stock Category')

# Adjust layout and display the plots
plt.tight_layout()
plt.show()


# Plot holiday and weekday against increase_stock

# Define categorical features
categorical_features = ['holiday', 'weekday']

# Create a figure with subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # Adjust size as needed

# Loop through features and plot each on a separate subplot
for ax, feature in zip(axes, categorical_features):
    data.groupby(feature)['increase_stock'].value_counts().unstack().plot(kind='bar', ax=ax)
    ax.set_title(f'{feature} vs. Increase Stock')
    ax.set_xlabel(feature)
    ax.set_ylabel('Count')
    ax.legend(title='Increase Stock Category')

# Adjust layout and display the plots
plt.tight_layout()
plt.show()


# Calculate percentage distribution of increase_stock for holidays
holiday_counts = data.groupby("holiday")["increase_stock"].value_counts(normalize=True) * 100

# Calculate percentage distribution of increase_stock for weekdays
weekday_counts = data.groupby("weekday")["increase_stock"].value_counts(normalize=True) * 100

# Display the results
print("Percentage Distribution of Increase Stock for Holidays:")
print(holiday_counts, "\n")

print("Percentage Distribution of Increase Stock for Weekdays:")
print(weekday_counts)


# Plot numerical features against increase_stock

# Define the numerical features to compare against increase_stock
numeric_features = ['temp', 'precip', 'snowdepth', 'humidity', 'windspeed', 'cloudcover', 'visibility']

# Set plot style
sns.set_style("whitegrid")

# Create subplots for better visualization
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 10))
axes = axes.flatten()

# Plot each numeric feature against increase_stock
for i, feature in enumerate(numeric_features):
    if i < len(axes):  # Avoid index errors if fewer than 8 features
        sns.violinplot(x="increase_stock", y=feature, data=data, ax=axes[i])
        axes[i].set_title(f"{feature} vs. Increase Stock")
        axes[i].set_xlabel("Increase Stock")
        axes[i].set_ylabel(feature)

# Remove empty subplot (since we have 7 plots but a 2x4 grid)
fig.delaxes(axes[-1])

# Adjust layout
plt.tight_layout()
plt.show()