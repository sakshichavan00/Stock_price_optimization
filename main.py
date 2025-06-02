# ===== Imports =====
import pandas as pd
import numpy as np
import yfinance as yf
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
import datetime

# ===== Configuration =====
selected_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
random_symbols = ['IBM', 'INTC', 'GE', 'BA', 'QCOM']
start_date = '2018-01-01'
end_date = datetime.datetime.today().strftime('%Y-%m-%d')

# ===== EOQ Function =====
def calculate_eoq(D, S, H):
    return math.sqrt((2 * D * S) / H)

# ===== Download Stock Data =====
def download_stock_data(symbol):
    try:
        df = yf.download(symbol, start=start_date, end=end_date)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().reset_index()
        return df
    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
        return None

# ===== LSTM + Strategy + EOQ Backtest =====
def train_lstm_and_backtest(data, D=10000, S=2, H=0.05, plot=False):
    if data is None or data.empty:
        return None

    eoq = calculate_eoq(D, S, H)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i, 0])
        y.append(scaled[i, 0])

    if not X: return None

    X = np.array(X).reshape(-1, 60, 1)
    y = np.array(y)
    split = int(0.8 * len(X))
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)

    pred_scaled = model.predict(X_test)
    pred = scaler.inverse_transform(pred_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    test_df = data.iloc[-len(pred):].copy()
    test_df['LSTM_Pred'] = pred.flatten()
    test_df['Signal'] = (test_df['LSTM_Pred'] > test_df['Close']).astype(int)
    test_df['Market_Ret'] = test_df['Close'].pct_change()
    test_df['Strat_Ret'] = test_df['Signal'].shift(1) * test_df['Market_Ret']
    test_df.dropna(inplace=True)

    test_df['Cum_Market'] = (1 + test_df['Market_Ret']).cumprod()
    test_df['Cum_Strat'] = (1 + test_df['Strat_Ret']).cumprod()

    sharpe = (test_df['Strat_Ret'].mean() / test_df['Strat_Ret'].std()) * np.sqrt(252)
    max_dd = (test_df['Cum_Strat'] / test_df['Cum_Strat'].cummax() - 1).min()
    trades = int(test_df['Signal'].diff().fillna(test_df['Signal']).abs().sum())
    fixed_lot = 100
    cost_fixed = trades * (S + H * fixed_lot)
    cost_eoq = trades * (S + H * eoq)

    if plot:
        plt.figure()
        plt.plot(actual, label='Actual')
        plt.plot(pred, label='LSTM Prediction')
        plt.title("LSTM Forecast")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.plot(test_df['Cum_Market'], label='Buy & Hold')
        plt.plot(test_df['Cum_Strat'], label='Strategy')
        plt.title("Cumulative Returns")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.bar(['Fixed Lot', 'EOQ Lot'], [cost_fixed, cost_eoq])
        plt.title("Transaction + Holding Cost")
        plt.tight_layout()
        plt.show()

    return {
        'sharpe': sharpe,
        'drawdown': max_dd,
        'cum_return': test_df['Cum_Strat'].iloc[-1],
        'cost_fixed': cost_fixed,
        'cost_eoq': cost_eoq,
        'symbol': data['Date'].iloc[0].strftime('%Y-%m-%d'),
        'start_date': data['Date'].min().strftime('%Y-%m-%d'),
        'end_date': data['Date'].max().strftime('%Y-%m-%d')
    }

# ===== Portfolio Runner =====
def run_portfolio(symbols):
    results = []
    for sym in symbols:
        print(f"\nProcessing {sym}...")
        data = download_stock_data(sym)
        result = train_lstm_and_backtest(data)
        if result:
            result['symbol'] = sym
            print(f"{sym}: Return {result['cum_return']:.2f}, Sharpe {result['sharpe']:.2f}")
            results.append(result)
        else:
            print(f"Skipping {sym}")
    return pd.DataFrame(results)

# ===== Summary Printer =====
def print_summary(df, label):
    if df.empty:
        print(f"\nNo data for {label}")
        return
    print(f"\n{label} Summary:")
    print(f"Avg Sharpe: {df['sharpe'].mean():.2f}")
    print(f"Avg Return: {df['cum_return'].mean():.2f}")
    print(f"Avg Savings: ${df['cost_fixed'].mean() - df['cost_eoq'].mean():.2f}")

# ===== Main Execution =====
if __name__ == "__main__":
    print(">>> Multi-Stock EOQ Strategy Started <<<")

    print("\nRunning selected portfolio...")
    selected_df = run_portfolio(selected_symbols)

    print("\nRunning random portfolio...")
    random_df = run_portfolio(random_symbols)

    print_summary(selected_df, "Selected Portfolio")
    print_summary(random_df, "Random Portfolio")

    all_df = pd.concat([selected_df, random_df])
    if not all_df.empty:
        best_stock = all_df.loc[all_df['cum_return'].idxmax()]
        print(f"\nBest Stock: {best_stock['symbol']} | Return: {best_stock['cum_return']:.2f}, Sharpe: {best_stock['sharpe']:.2f}")
        best_data = download_stock_data(best_stock['symbol'])
        _ = train_lstm_and_bac
