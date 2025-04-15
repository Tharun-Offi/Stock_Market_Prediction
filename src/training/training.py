import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout

# Enable GPU Acceleration in TensorFlow
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU detected and enabled for TensorFlow")
else:
    print("No GPU detected. Running on CPU.")

os.makedirs('model', exist_ok=True)

# Load dataset
print("Loading dataset...")
data = pd.read_csv('NIFTY50_all.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(f"Dataset loaded. Shape: {data.shape}")

# Feature Engineering
def create_features(data, target_column='Close', window_size=10):
    features, target = [], []
    for i in range(len(data) - window_size):
        features.append(data.iloc[i:i + window_size].values)
        target.append(data.iloc[i + window_size][target_column])
    return np.array(features, dtype=np.float32), np.array(target, dtype=np.float32)

window_size = 10
print("Creating features...")
features, target = create_features(data)
print(f"Features shape: {features.shape}, Target shape: {target.shape}")

# Normalize features
scaler = MinMaxScaler()
features = features.reshape(features.shape[0], -1)
features = scaler.fit_transform(features)
features = features.reshape(features.shape[0], window_size, -1)
print("Feature normalization completed.")

# Save scaler
joblib.dump(scaler, 'model/scaler.pkl')
print("Scaler saved.")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)
print(f"Train-Test Split: X_train: {X_train.shape}, X_test: {X_test.shape}")

# Train ML Models (GPU-Compatible)
models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, n_jobs=-1),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, tree_method='hist', device='cpu'),
    'SVM': SVR(),
}

predictions = {}
mae_scores = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    predictions[name] = model.predict(X_test.reshape(X_test.shape[0], -1))
    mae_scores[name] = mean_absolute_error(y_test, predictions[name])
    print(f'{name} MAE: {mae_scores[name]}')
    joblib.dump(model, f'model/{name.lower()}_model.pkl')
    print(f"{name} model saved.")

# Train Hybrid CNN-LSTM Model
print("Building CNN-LSTM model...")
cnn_lstm_model = Sequential([
    Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(window_size, features.shape[2])),
    MaxPooling1D(pool_size=2),
    LSTM(128, activation='relu', return_sequences=True),
    LSTM(64, activation='relu'),
    Dropout(0.2),
    Dense(1)
])
cnn_lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse', jit_compile=True)
print("CNN-LSTM model compiled.")

history = cnn_lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1, validation_data=(X_test, y_test))
print("CNN-LSTM model training completed.")

# Save loss values
loss_values = history.history['loss']
val_loss_values = history.history['val_loss']
print(f'Final Training Loss: {loss_values[-1]}')
print(f'Final Validation Loss: {val_loss_values[-1]}')

cnn_lstm_model.save('model/cnn_lstm_model.h5')
print("CNN-LSTM model saved.")

# Predictions from CNN-LSTM
cnn_lstm_pred = cnn_lstm_model.predict(X_test).flatten()
print("CNN-LSTM predictions completed.")

# Train Ridge Regression for Stacking
print("Training Ridge Regression Stacking Model...")
stacking_features = np.column_stack([predictions['SVM'], predictions['RandomForest'], predictions['XGBoost'], cnn_lstm_pred])
stacking_model = Ridge()
stacking_model.fit(stacking_features, y_test)
ensemble_pred = stacking_model.predict(stacking_features)
print("Stacking model trained.")

ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
print(f'Ensemble Model MAE: {ensemble_mae}')

# Save Stacking Model
joblib.dump(stacking_model, 'model/stacking_model.pkl')
print("Stacking model saved.")

# Plot results
# plt.figure(figsize=(12, 6))
# plt.plot(y_test, label='Actual', color='blue')
# plt.plot(ensemble_pred, label='Ensemble Prediction', color='red', linestyle='dashed')
# plt.xlabel('Time')
# plt.ylabel('Stock Price')
# plt.title('Stock Price Prediction: Actual vs. Predicted')
# plt.legend()
# plt.show()

print("Model training completed and ready for real-time predictions.")
