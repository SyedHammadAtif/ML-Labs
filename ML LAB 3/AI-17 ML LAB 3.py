import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
file_path = "/mnt/data/ML Lab3 dataset (1).csv"
df = pd.read_csv(file_path)

# Drop unnecessary column
df = df.drop(columns=["Order_ID"])

# One-hot encode categorical features
categorical_features = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)  # Avoid dummy variable trap

# Fill missing values in Courier_Experience_yrs with median
df["Courier_Experience_yrs"].fillna(df["Courier_Experience_yrs"].median(), inplace=True)

# Split features and target
X = df.drop(columns=["Delivery_Time_min"])
y = df["Delivery_Time_min"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

# Print evaluation metrics
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
