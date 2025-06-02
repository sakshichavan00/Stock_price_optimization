import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense

# ===== Alpha Vantage Setup =====
api_key = 'GLVZ9GJN4IW7GRUB'
symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']  # Multiple stocks for analysis

# ===== EOQ Calculation =====
def calculate_eoq(D, S, H):
    return math.sqrt((2 * D * S) / H)

# ===== Data Preparation =====
def prepare_data(data):
    data = data.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    })[['Open','High','Low','Close','Volume']][::-1].reset_index()
    return data

# ===== LSTM Forecasting =====
def lstm_forecast(data):
    close_vals = data['Close'].values.reshape(-1,1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_vals)

    X_all, y_all = [], []
    for i in range(60, len(scaled)):
        X_all.append(scaled[i-60:i,0])
        y_all.append(scaled[i,0])
    X_all, y_all = np.array(X_all), np.array(y_all)
    X_all = X_all.reshape((X_all.shape[0], X_all.shape[1], 1))

    split = int(0.8 * len(X_all))
    X_train, X_test = X_all[:split], X_all[split:]
    y_train, y_test = y_all[:split], y_all[split:]

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60,1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    pred_scaled = model.predict(X_test)
    pred = scaler.inverse_transform(pred_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1,1))
    return actual, pred

# ===== Multi-Stock Processing =====
ts = TimeSeries(key=api_key, output_format='pandas')
results = {}

for symbol in symbols:
    print(f"\n=== Processing: {symbol} ===")
    try:
        data, _ = ts.get_daily(symbol=symbol, outputsize='full')
        data = prepare_data(data)
        actual, pred = lstm_forecast(data)

        rmse = math.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        results[symbol] = {'RMSE': rmse, 'MAE': mae}

        # Plot
        plt.figure()
        plt.plot(actual, label='Actual')
        plt.plot(pred, label='Predicted')
        plt.title(f"{symbol} - LSTM Forecast")
        plt.xlabel("Test Day")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error processing {symbol}: {e}")

# ===== Summary Report =====
print("\n==== Forecasting Performance Summary ====")
for sym, res in results.items():
    print(f"{sym}: RMSE = {res['RMSE']:.4f}, MAE = {res['MAE']:.4f}")
