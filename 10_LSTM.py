#   Using LSTM for prediction of future weather of cities in Python  

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# !kaggle datasets download -d sudalairajkumar/daily-temperature-of-major-cities
# !unzip -o daily-temperature-of-major-cities.zip

df = pd.read_csv("city_temperature.csv")

df = df[df['City'] == 'Cairo']

# Clean data properly
df = df[['AvgTemperature']]

# Remove all invalid values
df = df[df['AvgTemperature'] > -50]   # keeps only real temperatures

# Remove missing values
df = df.dropna()

data = df.values

scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(data)

X, y = [], []
time_step = 10

for i in range(len(data) - time_step - 1):
    X.append(data[i:(i + time_step), 0])
    y.append(data[i + time_step, 0])

X, y = np.array(X), np.array(y)

# Reshape for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

model = Sequential()
model.add(LSTM(50, input_shape=(time_step, 1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, y, epochs=5, batch_size=16)

pred = model.predict(X)

pred = scaler.inverse_transform(pred)
actual = scaler.inverse_transform(y.reshape(-1,1))

plt.figure(figsize=(10,5))

plt.plot(actual, label="Actual")
plt.plot(pred, label="Predicted")

plt.title("Weather Prediction using LSTM")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.legend()

plt.show()

future_steps = 5
last_data = data[-time_step:]

future = []

for i in range(future_steps):
    input_data = last_data.reshape(1, time_step, 1)
    pred_val = model.predict(input_data)

    future.append(pred_val[0][0])
    last_data = np.append(last_data[1:], pred_val)

future = scaler.inverse_transform(np.array(future).reshape(-1,1))

print("Future Temperatures:", future)

















#https://www.kaggle.com/code/anshuls235/studying-india-s-aqi/input





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ── LOAD & FILTER DATA ────────────────────────────────────────
df = pd.read_csv("/content/city_temperature.csv")

# Filter for one city
df = df[df['City'] == 'Cairo']
df = df[['AvgTemperature']]

# Clean invalid temperatures
df = df[df['AvgTemperature'] > -50]
df = df.dropna()
df = df.reset_index(drop=True)

print("Data shape:", df.shape)
print(df.head())

# Plot raw temperature data
plt.figure(figsize=(12, 4))
plt.plot(df['AvgTemperature'].values, color='blue')
plt.title("Cairo - Raw Temperature Data")
plt.xlabel("Days")
plt.ylabel("Temperature (°F)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ── a. PRE-PROCESSING ─────────────────────────────────────────
data   = df.values
scaler = MinMaxScaler(feature_range=(0, 1))
data   = scaler.fit_transform(data)

# Create sequences
def create_dataset(data, time_step=30):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:i+time_step, 0])
        Y.append(data[i+time_step, 0])
    return np.array(X), np.array(Y)

time_step = 30                                     # Fixed: 30 days lookback
X, y      = create_dataset(data, time_step)

# Reshape for LSTM: (samples, timesteps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Train-test split (80-20)                         # Fixed: was missing
train_size = int(len(X) * 0.8)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Train samples : {X_train.shape[0]}")
print(f"Test samples  : {X_test.shape[0]}")

# ── b. BUILD & TRAIN LSTM MODEL ───────────────────────────────
model = Sequential([
    LSTM(50, return_sequences=True,                # Fixed: stacked LSTM
         input_shape=(time_step, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),              # Fixed: added validation
    verbose=1
)

# ── LOSS PLOT ─────────────────────────────────────────────────  # Fixed: was missing
plt.figure(figsize=(8, 3))
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ── c. PREDICTIONS & METRICS ──────────────────────────────────
train_pred = model.predict(X_train)
test_pred  = model.predict(X_test)

# Inverse transform
train_pred     = scaler.inverse_transform(train_pred)
test_pred      = scaler.inverse_transform(test_pred)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual  = scaler.inverse_transform(y_test.reshape(-1, 1))

# Metrics                                          # Fixed: was missing
mse  = mean_squared_error(y_test_actual, test_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test_actual, test_pred)

print("\n── Model Evaluation ──")
print(f"  MSE  : {mse:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")

# ── d. VISUALIZATION ──────────────────────────────────────────

# Train vs Test side by side
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(y_train_actual, label='Actual',    color='blue')
plt.plot(train_pred,     label='Predicted', color='orange')
plt.title("Train: Actual vs Predicted")
plt.xlabel("Days")
plt.ylabel("Temperature (°F)")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(y_test_actual, label='Actual',    color='blue')
plt.plot(test_pred,     label='Predicted', color='red')
plt.title("Test: Actual vs Predicted")
plt.xlabel("Days")
plt.ylabel("Temperature (°F)")
plt.legend()
plt.grid(True)

plt.suptitle("LSTM Weather Prediction - Cairo")
plt.tight_layout()
plt.show()

# ── e. FUTURE PREDICTION ──────────────────────────────────────
future_steps = 30                                  # Predict next 30 days
last_data    = data[-time_step:]
future_preds = []

for i in range(future_steps):
    input_seq = last_data.reshape(1, time_step, 1)
    pred_val  = model.predict(input_seq, verbose=0)
    future_preds.append(pred_val[0][0])
    last_data = np.append(last_data[1:], pred_val)  # Slide window

# Inverse transform future predictions
future_preds = scaler.inverse_transform(
    np.array(future_preds).reshape(-1, 1))

print("\n── Future 30-Day Temperature Forecast ──")
for i, temp in enumerate(future_preds):
    print(f"  Day {i+1:2} : {temp[0]:.2f} °F")

# Plot future predictions                          # Fixed: was only printed
plt.figure(figsize=(12, 4))
plt.plot(range(len(y_test_actual)),
         y_test_actual,   label='Actual (Test)',   color='blue')
plt.plot(range(len(y_test_actual)),
         test_pred,       label='Predicted (Test)',color='orange')
plt.plot(range(len(y_test_actual),
               len(y_test_actual) + future_steps),
         future_preds,    label='Future Forecast', color='red',
         linestyle='--',  linewidth=2)
plt.title("LSTM - Cairo Weather Forecast (Next 30 Days)")
plt.xlabel("Days")
plt.ylabel("Temperature (°F)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
