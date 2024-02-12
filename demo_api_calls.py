import pandas as pd
import requests
from ucimlrepo import fetch_ucirepo

# Fetch the Bank Marketing dataset
bank_marketing = fetch_ucirepo(id=222)
X = bank_marketing.data.features  # Features DataFrame
y = bank_marketing.data.targets  # Targets Series

# Prepare the target Series and convert 'yes'/'no' to 1/0
target_column = "y"
y = y[target_column].map({'yes': 1, 'no': 0})

# Concatenate features and target for simplicity in this demo
df = pd.concat([X, y], axis=1)

# Save a portion of the dataset for training
train_df = df.sample(frac=0.8, random_state=42)
train_df.to_csv("train_dataset.csv", index=False)

# Save the remaining portion for prediction
predict_df = df.drop(train_df.index)
predict_samples = predict_df.sample(n=5, random_state=42)  # Select 5 samples for prediction

# API Calls
BASE_URL = "http://localhost:8000"

# Upload dataset (simulated by sending the file path as part of the request)
with open("train_dataset.csv", "rb") as file:
    upload_response = requests.post(f"{BASE_URL}/upload/", files={"file": file})
print("Upload Response:", upload_response.json())

# Train model
with open("train_dataset.csv", "rb") as file:
    train_response = requests.post(f"{BASE_URL}/train/", files={"file": file})
print("Train Response:", train_response.json())

# Preparing data for the prediction request without using column names
data_for_prediction = predict_samples.drop(columns=[target_column]).apply(lambda row: row.where(pd.notnull(row), None).tolist(), axis=1).tolist()

# Prepare the prediction request without feature names
predict_request = {
    "data": [{"features": row} for row in data_for_prediction]
}
print("Predict Request:", predict_request)

# Make prediction
predict_response = requests.post(f"{BASE_URL}/predict/", json=predict_request)
print("Predict Response:", predict_response.json())
