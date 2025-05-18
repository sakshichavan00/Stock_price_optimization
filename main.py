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
import xgboost as xgb
import optuna

# ===== Alpha Vantage Setup =====
api_key = 'GLVZ9GJN4IW7GRUB'  # Replace with your real key
symbol = 'AAPL'

print("Downloading data from Alpha Vantage...")
ts = TimeSeries(key=api_key, output_format='pandas')
data, meta = ts.get_daily(symbol=symbol, outputsize='full')
data = data.rename(columns={
    '1. open': 'Open',
    '2. high': 'High',
    '3. low': 'Low',
    '4. close': 'Close',
    '5. volume': 'Volume'
})[['Open','High','Low','Close','Volume']][::-1].reset_index()
print(data.head(), '\n')

# ===== EOQ Calculation =====
def calculate_eoq(D, S, H):
    return math.sqrt((2 * D * S) / H)

# example parameters
D = 10000   # annual “demand”
S = 2       # per-trade cost
H = 0.05    # per‐unit holding cost
eoq = calculate_eoq(D, S, H)
print(f"EOQ Trade Size: {eoq:.2f} shares\n")

# ===== Prepare LSTM Dataset =====
close_vals = data['Close'].values.reshape(-1,1)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(close_vals)

# build sequences of 60 days
X_all, y_all = [], []
for i in range(60, len(scaled)):
    X_all.append(scaled[i-60:i,0])
    y_all.append(scaled[i,0])
X_all, y_all = np.array(X_all), np.array(y_all)
X_all = X_all.reshape((X_all.shape[0], X_all.shape[1], 1))

# train/test split (80/20)
split = int(0.8 * len(X_all))
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]

# ===== LSTM Forecasting =====
print("Training LSTM model...")
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60,1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
print("LSTM training complete.\n")

# predict & invert scale
pred_scaled = model.predict(X_test)
pred = scaler.inverse_transform(pred_scaled)
actual = scaler.inverse_transform(y_test.reshape(-1,1))

# metrics
lstm_rmse = math.sqrt(mean_squared_error(actual, pred))
lstm_mae  = mean_absolute_error(actual, pred)
print(f"LSTM   →  RMSE: {lstm_rmse:.4f}, MAE: {lstm_mae:.4f}\n")

# plot forecast vs actual
plt.figure()
plt.plot(actual, label='Actual Price')
plt.plot(pred,   label='LSTM Predicted')
plt.title("LSTM Forecast vs Actual")
plt.xlabel("Test Day")
plt.ylabel("Price (USD)")
plt.legend()
plt.tight_layout()
plt.show()

# ===== XGBoost Features =====
data['MA10']       = data['Close'].rolling(10).mean()
data['Volatility'] = data['Close'].rolling(10).std()
data.dropna(inplace=True)

X_feat = data[['MA10','Volatility']]
y_feat = data['Close'].shift(-1).dropna()
X_feat = X_feat[:-1]  # align

X_tr, X_te, y_tr, y_te = train_test_split(X_feat, y_feat, shuffle=False, test_size=0.2)
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_tr, y_tr)

# importance plot
xgb.plot_importance(xgb_model)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()

# XGBoost metrics
xgb_pred = xgb_model.predict(X_te)
xgb_rmse = math.sqrt(mean_squared_error(y_te, xgb_pred))
xgb_mae  = mean_absolute_error(y_te, xgb_pred)
print(f"XGBoost→  RMSE: {xgb_rmse:.4f}, MAE: {xgb_mae:.4f}\n")

# ===== Simple Signal Backtest (LSTM‐based) =====
# align test DataFrame
test_df = data.iloc[-len(pred):].copy()
test_df['LSTM_Pred'] = pred.flatten()
test_df['Signal']    = np.where(test_df['LSTM_Pred'] > test_df['Close'], 1, 0)
test_df['Market_Ret']   = test_df['Close'].pct_change()
test_df['Strat_Ret']    = test_df['Signal'].shift(1) * test_df['Market_Ret']
test_df.dropna(inplace=True)

# cumulative returns
test_df['Cum_Market'] = (1 + test_df['Market_Ret']).cumprod()
test_df['Cum_Strat']  = (1 + test_df['Strat_Ret']).cumprod()

# Sharpe Ratio (annualized)
sharpe = (test_df['Strat_Ret'].mean() / test_df['Strat_Ret'].std()) * np.sqrt(252)
# Max Drawdown
rolling_max = test_df['Cum_Strat'].cummax()
drawdown   = test_df['Cum_Strat'] / rolling_max - 1
max_dd     = drawdown.min()

print(f"Strategy → Sharpe Ratio: {sharpe:.4f}, Max Drawdown: {max_dd:.4%}\n")

# plot portfolio curves
plt.figure()
plt.plot(test_df['Cum_Market'], label='Buy & Hold')
plt.plot(test_df['Cum_Strat'],  label='LSTM Strategy')
plt.title("Portfolio Value Over Time")
plt.xlabel("Test Day")
plt.ylabel("Cumulative Return")
plt.legend()
plt.tight_layout()
plt.show()

# ===== Transaction & Holding Cost Comparison =====
# count trades (changes in position)
trades = int(test_df['Signal'].diff().fillna(test_df['Signal']).abs().sum())
fixed_size = 100  # baseline lot
cost_fixed = trades * (S + H * fixed_size)
cost_eoq   = trades * (S + H * eoq)
print(f"Trades Executed: {trades}")
print(f"Total Cost (Fixed {fixed_size}‐share lots): ${cost_fixed:.2f}")
print(f"Total Cost (EOQ {eoq:.1f}‐share lots):   ${cost_eoq:.2f}")
print(f"Cost Savings: ${cost_fixed - cost_eoq:.2f}\n")

# bar chart
plt.figure()
plt.bar(['Fixed Lot','EOQ Lot'], [cost_fixed, cost_eoq])
plt.title("Transaction + Holding Cost Comparison")
plt.ylabel("Total Cost (USD)")
plt.tight_layout()
plt.show()

# ===== Bayesian Optimization of EOQ Parameters =====
def objective(trial):
    H_t = trial.suggest_float("H", 0.01, 0.1)
    S_t = trial.suggest_int("S", 1, 10)
    eoq_t = calculate_eoq(D, S_t, H_t)
    # simple proxy: minimize total per‐trade cost
    return eoq_t * H_t + S_t

print("Running Bayesian Optimization...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20, show_progress_bar=True)

print("Best EOQ Params:", study.best_params, "\n")

# plot optimization history
trials = study.trials_dataframe(attrs=('number','value'))
plt.figure()
plt.plot(trials['number'], trials['value'], marker='o')
plt.title("Optuna Optimization History")
plt.xlabel("Trial")
plt.ylabel("Objective Value")
plt.tight_layout()
plt.show()
