import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# Load dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target

df.to_csv('california_housing.csv', index=False)

# Task 1: Multiple Regression with different train-test splits
X = df.drop(columns=['Target'])
y = df['Target']

splits = [(0.7, 0.3), (0.8, 0.2)]
random_states = [42, 100]

for split in splits:
    for state in random_states:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split[1], random_state=state)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"Train-Test Split: {split}, Random State: {state}")
        print(f"R2 Score: {r2_score(y_test, y_pred):.4f}, RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}\n")

# Task 2: Subset of predictors
selected_features = ['MedInc', 'HouseAge']  # Selecting subset of features
X_subset = df[selected_features]

X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.3, random_state=42)
model_subset = LinearRegression()
model_subset.fit(X_train, y_train)
y_pred_subset = model_subset.predict(X_test)
print("Subset Features R2 Score:", r2_score(y_test, y_pred_subset))

# Task 3: Polynomial Regression on one predictor
degree_list = [2, 3, 4]
predictor = 'MedInc'
X_poly = df[[predictor]]

for degree in degree_list:
    poly = PolynomialFeatures(degree=degree)
    X_poly_transformed = poly.fit_transform(X_poly)
    X_train, X_test, y_train, y_test = train_test_split(X_poly_transformed, y, test_size=0.3, random_state=42)
    model_poly = LinearRegression()
    model_poly.fit(X_train, y_train)
    y_pred_poly = model_poly.predict(X_test)
    print(f"Degree {degree} Polynomial R2 Score: {r2_score(y_test, y_pred_poly):.4f}")

    # Plot
    plt.scatter(X_poly, y, color='blue', label='Actual Data')
    sorted_X = np.sort(X_poly.values, axis=0)
    plt.plot(sorted_X, model_poly.predict(poly.transform(sorted_X)), color='red', label=f'Degree {degree}')
    plt.xlabel(predictor)
    plt.ylabel('Target')
    plt.title(f'Polynomial Regression (Degree {degree})')
    plt.legend()
    plt.show()
