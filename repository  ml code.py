import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset
data = pd.read_csv(r"D:\gihub\Tele-customer.csv")

# Print all column names
print("Column Names in CSV File:")
print(list(data.columns))

# Automatically detect binary columns (having exactly 2 unique non-null values)
binary_columns = [col for col in data.columns if data[col].nunique(dropna=True) == 2 and col != 'Churn']

# Print detected binary columns
print("\nAutomatically Detected Binary Columns:")
print(binary_columns)

# Ensure 'Churn' is included for classification
if 'Churn' not in data.columns:
    raise ValueError("'Churn' column not found in dataset!")

# Select binary columns + Churn, and drop missing values
data = data[binary_columns + ['Churn']].dropna()

# Convert binary categorical columns to 0/1
for col in binary_columns:
    unique_vals = sorted(data[col].unique())
    data[col] = data[col].apply(lambda x: 1 if x == unique_vals[-1] else 0)

# Encode 'Churn' target to 0/1
data['Churn'] = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Inputs and outputs
X = data[binary_columns].values
y_true = data['Churn'].values

# --- McCulloch-Pitts Neuron Implementation ---
def mp_neuron(inputs, weights, threshold):
    summation = np.dot(inputs, weights)
    return 1 if summation >= threshold else 0

# Hardcoded weights and threshold (can be tuned manually)
weights = [1] * len(binary_columns)  # Default: weight 1 for each feature
threshold = int(np.ceil(len(binary_columns) / 1))  # Simple threshold: half of total inputs

# Predict using MP Neuron
predictions = [mp_neuron(x, weights, threshold) for x in X]

# Evaluate accuracy
accuracy = np.mean(predictions == y_true)
print(f"\nAccuracy of McCulloch-Pitts Neuron on Customer Churn Data: {accuracy * 100:.2f}%")

# Show confusion matrix and classification report
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, predictions))

print("\nClassification Report:")
print(classification_report(y_true, predictions))
