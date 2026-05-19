#  Implementation of RNN model for Stock Price Prediction 

# Import libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Generate stock-like data using sine wave
np.random.seed(0)

time = np.arange(0, 200)
prices = np.sin(0.05 * time) + np.random.normal(0, 0.1, 200)

prices = prices.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(prices)

def create_dataset(data, time_step=10):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:i+time_step, 0])
        Y.append(data[i+time_step, 0])
    return np.array(X), np.array(Y)

time_step = 10
X, y = create_dataset(scaled_data, time_step)

X = X.reshape(X.shape[0], X.shape[1], 1)

train_size = int(len(X) * 0.8)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


model = Sequential()
model.add(SimpleRNN(50, activation='tanh', input_shape=(time_step, 1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=20, batch_size=16)

test_pred = model.predict(X_test)

test_pred = scaler.inverse_transform(test_pred)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))

rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))
mae = mean_absolute_error(y_test_actual, test_pred)

print("RMSE:", rmse)
print("MAE:", mae)


plt.plot(y_test_actual, label="Actual")
plt.plot(test_pred, label="Predicted")
plt.legend()
plt.title("RNN Stock Prediction (Dummy Data)")
plt.show()



























import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout

# ── INSTALL & LOAD REAL STOCK DATA ───────────────────────────
# !pip install yfinance

# Download Apple stock data (2020 to 2024)
df = yf.download('AAPL', start='2020-01-01', end='2024-01-01')
print(df.head())
print("Shape:", df.shape)

# Use only 'Close' price
prices = df[['Close']].values
print("Prices shape:", prices.shape)

# Plot raw stock data
plt.figure(figsize=(12, 4))
plt.plot(df.index, prices, color='blue')
plt.title("Apple (AAPL) Stock Price 2020-2024")
plt.xlabel("Date")
plt.ylabel("Close Price (USD)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ── a. PRE-PROCESSING ─────────────────────────────────────────

# Normalize 0-1
scaler      = MinMaxScaler()
scaled_data = scaler.fit_transform(prices)

# Create sequences
def create_dataset(data, time_step=60):            # 60 days lookback
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:i+time_step, 0])
        Y.append(data[i+time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60                                     # Look back 60 days
X, y      = create_dataset(scaled_data, time_step)

# Reshape for RNN: (samples, timesteps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Train-test split (80-20)
train_size = int(len(X) * 0.8)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Train samples : {X_train.shape[0]}")
print(f"Test samples  : {X_test.shape[0]}")

# ── b. BUILD & TRAIN RNN MODEL ────────────────────────────────
model = Sequential([
    SimpleRNN(50, activation='tanh',
              return_sequences=True,               # Stack two RNN layers
              input_shape=(time_step, 1)),
    Dropout(0.2),
    SimpleRNN(50, activation='tanh'),
    Dropout(0.2),
    Dense(1)
])

model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# ── TRAINING LOSS PLOT ────────────────────────────────────────
plt.figure(figsize=(8, 3))
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ── c. PREDICTIONS ────────────────────────────────────────────
test_pred  = model.predict(X_test)
train_pred = model.predict(X_train)

# Inverse transform back to USD
test_pred      = scaler.inverse_transform(test_pred)
train_pred     = scaler.inverse_transform(train_pred)
y_test_actual  = scaler.inverse_transform(y_test.reshape(-1, 1))
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))

# ── d. EVALUATE METRICS ───────────────────────────────────────
mse  = mean_squared_error(y_test_actual, test_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test_actual, test_pred)

print("\n── Model Evaluation ──")
print(f"  MSE  : {mse:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")

# ── e. VISUALIZATION ──────────────────────────────────────────

# Train vs Test predictions side by side
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(y_train_actual, label='Actual',    color='blue')
plt.plot(train_pred,     label='Predicted', color='orange')
plt.title("Train: Actual vs Predicted")
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(y_test_actual, label='Actual',    color='blue')
plt.plot(test_pred,     label='Predicted', color='red')
plt.title("Test: Actual vs Predicted")
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)

plt.suptitle("RNN - AAPL Stock Price Prediction")
plt.tight_layout()
plt.show()

# Full timeline overlay plot
full_pred = np.empty_like(prices)
full_pred[:] = np.nan
full_pred[train_size + time_step + 1 :
          train_size + time_step + 1 + len(test_pred)] = test_pred

plt.figure(figsize=(14, 5))
plt.plot(df.index, prices,     label='Actual Price', color='blue',  alpha=0.6)
plt.plot(df.index, full_pred,  label='Predicted',    color='red',   linewidth=2)
plt.title("AAPL Full Timeline - RNN Stock Prediction")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()