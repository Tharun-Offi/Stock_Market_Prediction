import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import yfinance as yf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained models
svm_model = joblib.load('model/svm_model.pkl')
rf_model = joblib.load('model/randomforest_model.pkl')
xgb_model = joblib.load('model/xgboost_model.pkl')
stacking_model = joblib.load('model/stacking_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Load CNN-LSTM model with custom loss
cnn_lstm_model = load_model('model/cnn_lstm_model.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

window_size = 10  # Same as used during training
required_days = 50  # Features required for scaler

def predict_today_price(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    hist = stock.history(period="3mo")  # Fetch last 6 months to ensure sufficient data

    if len(hist) < window_size:
        return {"error": f"Not enough data available for prediction (got {len(hist)} days, need {window_size})."}

    last_features = hist['Close'].values[-required_days:]
    if len(last_features) < required_days:
        last_features = np.pad(last_features, (required_days - len(last_features), 0), mode='edge')

    last_features = last_features.reshape(1, -1)
    last_features = scaler.transform(last_features).reshape(1, window_size, -1)

    svm_pred = svm_model.predict(last_features.reshape(1, -1))[0]
    rf_pred = rf_model.predict(last_features.reshape(1, -1))[0]
    xgb_pred = xgb_model.predict(last_features.reshape(1, -1))[0]
    cnn_lstm_pred = cnn_lstm_model.predict(last_features).flatten()[0]

    stacking_features = np.column_stack([[svm_pred], [rf_pred], [xgb_pred], [cnn_lstm_pred]])
    final_pred = stacking_model.predict(stacking_features)[0]

    return {"stock": stock_symbol, "predicted_price": round(float(final_pred), 2)}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    stock_symbol = request.args.get('symbol', 'AAPL')
    result = predict_today_price(stock_symbol)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)