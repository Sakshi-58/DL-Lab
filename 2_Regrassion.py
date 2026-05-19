# Implement Simple and Multiple Linear Regression to predict continuous variables. 
# a. Perform data preprocessing (handle missing values, feature scaling). 
# b. Fit a Simple Linear Regression model on a dataset (e.g., predicting house prices). 
# c. Extend to Multiple Linear Regression with multiple features. 
# d. Evaluate models using MSE, RMSE, and R² Score. 
# e. Visualize the regression line and predictions. 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("insurance.csv")

df.head()

df.isnull().sum()

df.describe()
df.fillna(df.mean(numeric_only=True), inplace=True)

y = df['charges']
X = df.drop('charges', axis=1)

X = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Select only 'age' column after encoding
age_index = list(X.columns).index('age')
X_simple = X_scaled[:, age_index].reshape(-1, 1)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

simple_model = LinearRegression()
simple_model.fit(X_train_s, y_train_s)

y_pred_s = simple_model.predict(X_test_s)

print("Simple Linear Regression")
print("MSE:", mean_squared_error(y_test_s, y_pred_s))
print("R2 Score:", r2_score(y_test_s, y_pred_s))

# Visualization
plt.scatter(X_test_s, y_test_s)
plt.plot(X_test_s, y_pred_s, color='red')
plt.xlabel("Age (scaled)")
plt.ylabel("Insurance Charges")
plt.title("Simple Linear Regression")
plt.show()

# Train-test split
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

multi_model = LinearRegression()
multi_model.fit(X_train_m, y_train_m)

y_pred_m = multi_model.predict(X_test_m)

print("\nMultiple Linear Regression")
print("MSE:", mean_squared_error(y_test_m, y_pred_m))
print("R2 Score:", r2_score(y_test_m, y_pred_m))

# Visualization with regression (ideal) line
plt.scatter(y_test_m, y_pred_m, alpha=0.6)

# Regression / ideal line (Perfect prediction line)
min_val = min(y_test_m.min(), y_pred_m.min())
max_val = max(y_test_m.max(), y_pred_m.max())

plt.plot([min_val, max_val], [min_val, max_val], color='red')

plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Multiple Linear Regression")
plt.show()


# Visualization
plt.scatter(y_test_m, y_pred_m)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Multiple Linear Regression")
plt.show()






















# Implement Simple and Multiple Linear Regression
# With Data Preprocessing, Evaluation and Visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("insurance.csv")

print("First 5 Rows:")
print(df.head())

# =========================================================
# DATA PREPROCESSING
# =========================================================

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing numerical values with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Separate independent and dependent variables
y = df['charges']
X = df.drop('charges', axis=1)

# Convert categorical columns into numerical
X = pd.get_dummies(X, drop_first=True)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================================================
# SIMPLE LINEAR REGRESSION
# =========================================================

# Use only AGE feature
age_index = list(X.columns).index('age')
X_simple = X_scaled[:, age_index].reshape(-1, 1)

# Train-test split
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

# Create model
simple_model = LinearRegression()

# Train model
simple_model.fit(X_train_s, y_train_s)

# Predictions
y_pred_s = simple_model.predict(X_test_s)

# =========================================================
# EVALUATION - SIMPLE LINEAR REGRESSION
# =========================================================

mse_s = mean_squared_error(y_test_s, y_pred_s)
rmse_s = np.sqrt(mse_s)
r2_s = r2_score(y_test_s, y_pred_s)

print("\n===== Simple Linear Regression =====")
print("MSE  :", mse_s)
print("RMSE :", rmse_s)
print("R2 Score :", r2_s)

# =========================================================
# VISUALIZATION - SIMPLE LINEAR REGRESSION
# =========================================================

# Sort values for smooth regression line
sorted_indices = X_test_s[:, 0].argsort()

plt.figure(figsize=(8, 5))

# Scatter plot
plt.scatter(
    X_test_s,
    y_test_s,
    color='blue',
    label='Actual Data'
)

# Regression line
plt.plot(
    X_test_s[sorted_indices],
    y_pred_s[sorted_indices],
    color='red',
    linewidth=2,
    label='Regression Line'
)

plt.xlabel("Age (Scaled)")
plt.ylabel("Insurance Charges")
plt.title("Simple Linear Regression")
plt.legend()
plt.grid(True)
plt.show()

# =========================================================
# MULTIPLE LINEAR REGRESSION
# =========================================================

# Train-test split
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Create model
multi_model = LinearRegression()

# Train model
multi_model.fit(X_train_m, y_train_m)

# Predictions
y_pred_m = multi_model.predict(X_test_m)

# =========================================================
# EVALUATION - MULTIPLE LINEAR REGRESSION
# =========================================================

mse_m = mean_squared_error(y_test_m, y_pred_m)
rmse_m = np.sqrt(mse_m)
r2_m = r2_score(y_test_m, y_pred_m)

print("\n===== Multiple Linear Regression =====")
print("MSE  :", mse_m)
print("RMSE :", rmse_m)
print("R2 Score :", r2_m)

# =========================================================
# VISUALIZATION 1 - ACTUAL VS PREDICTED
# =========================================================

plt.figure(figsize=(8, 5))

# Scatter plot
plt.scatter(
    y_test_m,
    y_pred_m,
    color='green',
    alpha=0.6
)

# Ideal prediction line
min_val = min(y_test_m.min(), y_pred_m.min())
max_val = max(y_test_m.max(), y_pred_m.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    color='red',
    linewidth=2
)

plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Multiple Linear Regression")
plt.grid(True)
plt.show()

# =========================================================
# VISUALIZATION 2 - RESIDUAL PLOT
# =========================================================

# Residuals
residuals = y_test_m - y_pred_m

plt.figure(figsize=(8, 5))

plt.scatter(
    y_pred_m,
    residuals,
    color='purple',
    alpha=0.6
)

# Horizontal reference line
plt.axhline(y=0, color='red', linestyle='--')

plt.xlabel("Predicted Charges")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True)
plt.show()

# =========================================================
# VISUALIZATION 3 - DISTRIBUTION OF ERRORS
# =========================================================

plt.figure(figsize=(8, 5))

plt.hist(
    residuals,
    bins=20
)

plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Distribution of Errors")
plt.grid(True)
plt.show()