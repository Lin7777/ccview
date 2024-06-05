import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load new crew information
new_file_path = './data/new_cc_data.xlsx'
new_data = pd.read_excel(new_file_path)

# Preprocess the new data

features_new = new_data.drop(columns=['工号', '姓名', '总排名', '总排名百分比'])

# Load and apply LabelEncoders
label_encoders = joblib.load('label_encoders.pkl')
for column, le in label_encoders.items():
    if column in features_new.columns:
        features_new[column] = le.transform(features_new[column])

# Standardize numerical features (use the same scaler as training data)
scaler = joblib.load('scaler.pkl')
features_new_scaled = scaler.transform(features_new)

# Load trained models
model_names = ['logistic_regression', 'mlp', 'random_forest', 'decision_tree', 'knn', 'svm']
predictions = {}
for model_name in model_names:
    model = joblib.load(f'{model_name}_model.pkl')
    predictions[model_name] = model.predict(features_new_scaled)

# Add prediction results to the new data
for model_name, prediction in predictions.items():
    new_data[f'Risk_Prediction_{model_name}'] = prediction

# Save the new data with prediction results to an Excel file
output_file_path = './data/predicted_cc_data.xlsx'  # Replace with your desired file path
new_data.to_excel(output_file_path, index=False)

print("Prediction results have been saved to:", output_file_path)
