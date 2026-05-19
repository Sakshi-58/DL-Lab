# 11. Implement a basic RNN for handling sequential data. 
# a. Build an RNN for a time-series prediction task. 
# b. Train on sequential data (e.g., stock prices). 
# c. Evaluate the model using MSE or RMSE for regression tasks. 
# d. Visualize predictions vs actual values over time. 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout

print("TensorFlow version:", tf.__version__)

np.random.seed(42)

days = 1000
time = np.arange(days)

price = (
    100
    + 0.05 * time
    + 15 * np.sin(2 * np.pi * time / 50)
    + 8  * np.sin(2 * np.pi * time / 200)
    + np.random.normal(0, 2, days)
)

df = pd.DataFrame({'Day': time, 'Close': price})
print("Shape:", df.shape)
print(df.head())

plt.figure(figsize=(14, 4))
plt.plot(df['Day'], df['Close'], color='steelblue', linewidth=1)
plt.title('Synthetic Stock Price Data (1000 Days)')
plt.xlabel('Day')
plt.ylabel('Price ($)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Normalize to [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Close']])

# Create sequences
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

SEQ_LEN = 30  # use last 30 days to predict next day

X, y = create_sequences(scaled_data, SEQ_LEN)
print("X shape:", X.shape)  # (samples, 30, 1)
print("y shape:", y.shape)

# Train/Test split (80/20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Train samples:", len(X_train))
print("Test  samples:", len(X_test))

model = Sequential([
    SimpleRNN(64, activation='tanh', return_sequences=True,
              input_shape=(SEQ_LEN, 1)),
    Dropout(0.2),

    SimpleRNN(32, activation='tanh', return_sequences=False),
    Dropout(0.2),

    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Plot loss curve
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'],     label='Train Loss', color='steelblue')
plt.plot(history.history['val_loss'], label='Val Loss',   color='tomato')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

y_pred_scaled = model.predict(X_test)

# Inverse transform to original price scale
y_pred_actual = scaler.inverse_transform(y_pred_scaled)
y_test_actual = scaler.inverse_transform(y_test)

mse  = mean_squared_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mse)

ss_res = np.sum((y_test_actual - y_pred_actual) ** 2)
ss_tot = np.sum((y_test_actual - np.mean(y_test_actual)) ** 2)
r2     = 1 - (ss_res / ss_tot)

print("=" * 35)
print(f"  MSE  : {mse:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  R²   : {r2:.4f}")
print("=" * 35)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Full timeline
all_prices = scaler.inverse_transform(scaled_data)
axes[0].plot(all_prices, color='lightgray', linewidth=1, label='All Data')
axes[0].axvline(x=split + SEQ_LEN, color='orange',
                linestyle='--', linewidth=1.5, label='Train/Test Split')
test_start = split + SEQ_LEN
axes[0].plot(range(test_start, test_start + len(y_pred_actual)),
             y_pred_actual, color='tomato', linewidth=1.5, label='Predicted')
axes[0].set_title('Stock Price — Full Timeline with Predictions')
axes[0].set_xlabel('Day')
axes[0].set_ylabel('Price ($)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Zoomed test set
axes[1].plot(y_test_actual, color='steelblue', linewidth=1.5, label='Actual Price')
axes[1].plot(y_pred_actual, color='tomato',    linewidth=1.5,
             linestyle='--', label='Predicted Price')
axes[1].set_title(f'Test Set — Actual vs Predicted   (RMSE = {rmse:.2f})')
axes[1].set_xlabel('Test Day')
axes[1].set_ylabel('Price ($)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

errors = y_test_actual.flatten() - y_pred_actual.flatten()

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(errors, color='mediumpurple', linewidth=1)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.title('Prediction Error Over Time')
plt.xlabel('Test Day')
plt.ylabel('Error')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(errors, bins=30, color='mediumpurple', edgecolor='white', alpha=0.8)
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.title('Error Distribution')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Mean Error : {errors.mean():.4f}")
print(f"Std  Error : {errors.std():.4f}")