import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.commodities import Commodities
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from xgboost import XGBRegressor
import os


results = {}
stock_stats = []


# api_key = 'GLVZ9GJN4IW7GRUB'
api_key = "K757OWEW19L34ML9"
# symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']  # Multiple stocks for analysis
ts = TimeSeries(key=api_key, output_format="pandas")
fx = ForeignExchange(key=api_key, output_format="pandas")

assets = [
    {"symbol": "AAPL", "type": "stock"},
    {"symbol": "MSFT", "type": "stock"},
    {"symbol": "TSLA", "type": "stock"},
    {"symbol": "AMZN", "type": "stock"},
    {"symbol": "EURUSD", "type": "forex", "from_symbol": "EUR", "to_symbol": "USD"},
    {"symbol": "USDJPY", "type": "forex", "from_symbol": "USD", "to_symbol": "JPY"},
    # {'symbol': 'XAUUSD', 'type': 'commodity'},  # Gold
    # {'symbol': 'XAGUSD', 'type': 'commodity'},  # Silver
    {"symbol": "COPPER", "type": "commodity", "api_method": "get_copper"},
    {"symbol": "NATURAL_GAS", "type": "commodity", "api_method": "get_natural_gas"},
]


def prepare_data(data):
    columns = data.columns
    print(f"Preparing data with columns: {columns}")  # Debugging statement

    # Case 1: Stock or forex style
    if "1. open" in columns:
        data = data.rename(
            columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. volume": "volume",
            }
        )[["open", "high", "low", "close", "volume"]]

    # Case 2: Commodity style (likely just a 'value' column)
    elif "value" in columns:
        data = data.rename(columns={"value": "close"})
        data["open"] = data["close"]
        data["high"] = data["close"]
        data["low"] = data["close"]
        data["volume"] = 0
        data = data[["open", "high", "low", "close", "volume"]]

    else:
        print(
            f"Warning: Unknown data format for columns: {columns}. Attempting fallback."
        )
        if "close" not in columns:
            data["close"] = data.iloc[:, -1]  # Use last column as close if possible
        data["open"] = data["close"]
        data["high"] = data["close"]
        data["low"] = data["close"]
        data["volume"] = 0
        data = data[["open", "high", "low", "close", "volume"]]

    # Ensure 'close' column exists
    if "close" not in data.columns:
        raise ValueError("Processed data does not contain 'close' column.")

    print(f"Processed data preview:\n{data.head()}\n")  # Debugging statement

    return data[::-1].reset_index(drop=True)  # sort oldest to newest


def load_or_fetch_data(asset, api_key, directory="asset_data"):
    import os
    from alpha_vantage.timeseries import TimeSeries
    from alpha_vantage.foreignexchange import ForeignExchange
    from alpha_vantage.commodities import Commodities

    os.makedirs(directory, exist_ok=True)
    symbol = asset["symbol"]
    filepath = os.path.join(directory, f"{symbol}.csv")

    if os.path.exists(filepath):
        print(f"Loading cached data for {symbol}")
        df = pd.read_csv(filepath)
        df = prepare_data(df)
    else:
        print(f"Fetching data for {symbol} from Alpha Vantage...")

        if asset["type"] == "stock":
            ts = TimeSeries(key=api_key, output_format="pandas")
            df, _ = ts.get_daily(symbol=symbol, outputsize="full")

        elif asset["type"] == "forex":
            fx = ForeignExchange(key=api_key, output_format="pandas")
            df, _ = fx.get_currency_exchange_daily(
                from_symbol=asset["from_symbol"],
                to_symbol=asset["to_symbol"],
                outputsize="full",
            )
            df["5. volume"] = 0

        elif asset["type"] == "commodity":
            cm = Commodities(key=api_key, output_format="pandas")
            # Dynamically call the correct commodity method
            method_name = asset["api_method"]  # e.g., 'get_copper', 'get_gold'
            if hasattr(cm, method_name):
                func = getattr(cm, method_name)
                df, _ = func(interval="monthly")
                df["5. volume"] = 0
            else:
                raise ValueError(f"Commodities API has no method: {method_name}")

        df.to_csv(filepath)
        print(f"Saved {symbol} data to {filepath}")
        df = prepare_data(df)

    return df


# data = load_or_fetch_data(symbols[4], api_key)
data_dict = {}
for asset in assets:
    df = load_or_fetch_data(asset, api_key)
    data_dict[asset["symbol"]] = df

df.head()  # Display the first few rows of the data


def add_technical_indicators(df):
    if "close" not in df.columns:
        raise KeyError("The 'close' column is missing in the DataFrame.")

    df["MA5"] = df["close"].rolling(window=5).mean()
    df["MA10"] = df["close"].rolling(window=10).mean()
    df["MA20"] = df["close"].rolling(window=20).mean()
    df["Return_5"] = df["close"].pct_change(periods=5)
    df["Volatility_20"] = df["close"].rolling(window=20).std()
    df["RSI"] = compute_rsi(df["close"], 14)
    df["MACD"] = compute_macd(df["close"])
    df = df.dropna()
    return df


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(series, slow=26, fast=12):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def plot_xgboost_feature_importance(model, feature_names, symbol):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(feature_names)), importances[indices], align="center")
    plt.xticks(
        range(len(feature_names)), [feature_names[i] for i in indices], rotation=45
    )
    plt.title(f"{symbol} - XGBoost Feature Importances")
    plt.xlabel("Features")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.show()


def select_features(df, target_col="close", symbol=""):
    df = df.select_dtypes(include=[np.number])

    # target: future return (use percent change)
    y = df[target_col].pct_change().shift(-1).dropna()
    df = df.iloc[:-1].reset_index(drop=True)  # Align features to y

    X = df.drop(columns=[target_col])
    model = XGBRegressor()
    model.fit(X, y)

    # plot_xgboost_feature_importance(model, X.columns, symbol)

    importances = model.feature_importances_
    feature_importance_df = (
        pd.DataFrame({"Feature": X.columns, "Importance": importances})
        .sort_values(by="Feature")
        .reset_index(drop=True)
    )

    print("\nFull Feature Importance Ranking:")
    print(feature_importance_df)

    top_features = X.columns[np.argsort(model.feature_importances_)][-5:]
    X_selected = X[top_features].copy().reset_index(drop=True)
    y_target = (
        df[target_col].iloc[1:].reset_index(drop=True)
    )  # align with prediction target

    y_target.name = target_col  # ✅ set the name to 'Close' (important for merging)

    return X_selected, y_target


def lstm_forecast_multivariate(data, target="close"):
    assert isinstance(target, str), "Target must be a string"
    assert target in data.columns, "Target column not found in input data"

    # Separate features and target
    feature_cols = [col for col in data.columns if col != target]
    full_data = data[feature_cols + [target]]  # Ensures correct column order

    # Scale data
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(full_data)
    target_index = full_data.columns.get_loc(target)

    # Create LSTM input windows
    X_all, y_all = [], []
    for i in range(60, len(scaled)):
        X_all.append(scaled[i - 60 : i, :-1])  # input: all features except target
        y_all.append(scaled[i, target_index])  # output: target

    X_all, y_all = np.array(X_all), np.array(y_all)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, shuffle=False
    )

    # LSTM model
    model = Sequential(
        [
            LSTM(
                64, return_sequences=True, input_shape=(X_all.shape[1], X_all.shape[2])
            ),
            LSTM(32),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Predict
    pred_scaled = model.predict(X_test)
    y_test = y_test.reshape(-1, 1)

    # Only reverse scale the target column
    pred_full = np.zeros((len(pred_scaled), full_data.shape[1]))
    actual_full = np.zeros_like(pred_full)
    pred_full[:, target_index] = pred_scaled[:, 0]
    actual_full[:, target_index] = y_test[:, 0]

    pred = scaler.inverse_transform(pred_full)[:, target_index]
    actual = scaler.inverse_transform(actual_full)[:, target_index]

    return actual, pred


def classify_stocks_by_return(stock_stats_df):
    # Sort by cumulative return descending
    sorted_df = stock_stats_df.sort_values(
        by="cumulative_return", ascending=False
    ).reset_index(drop=True)
    n = len(sorted_df)
    a_cutoff = int(0.2 * n)
    b_cutoff = int(0.5 * n)

    categories = [
        "A" if i < a_cutoff else "B" if i < b_cutoff else "C" for i in range(n)
    ]
    sorted_df["ABC"] = categories
    return sorted_df


def calculate_eoq(D, S, H):
    return math.sqrt((2 * D * S) / H) if H > 0 else 50


def simulate_portfolio(assets_data, forecasts, abc_map, variant_config):
    cash = 100000
    holdings = {symbol: 0 for symbol in assets_data}
    total_cost = 0
    portfolio_values = []

    S = variant_config.get("S", 0)
    H = variant_config.get("H", 0)
    use_eoq = variant_config.get("use_eoq", False)
    use_abc = variant_config.get("use_abc", False)
    tuned = variant_config.get("tuned", False)

    for t in range(len(next(iter(assets_data.values()))["actual"])):
        daily_value = cash
        for symbol, data in assets_data.items():
            price = data["actual"][t]
            pred = data["pred"][t]
            signal = pred - price
            abc = abc_map.get(symbol, "B")
            D = abs(pred - price) * 100  # estimated "demand" from forecast
            threshold = 0.02 if use_abc and abc == "C" else 0.01

            if abs(signal) < threshold:
                continue

            # Determine trade size
            if use_eoq:
                H_adj = H * (1.2 if abc == "A" else 0.8 if abc == "C" else 1)
                size = int(calculate_eoq(D, S, H_adj))
            else:
                size = 80 if abc == "A" else 50 if abc == "B" else 30

            if signal > 0 and cash >= size * price + S:  # Buy
                holdings[symbol] += size
                cash -= size * price + S
                total_cost += S
            elif signal < 0 and holdings[symbol] >= size:  # Sell
                holdings[symbol] -= size
                cash += size * price - S
                total_cost += S

            daily_value += holdings[symbol] * price

        portfolio_values.append(daily_value)

    return portfolio_values, total_cost


def compute_portfolio_metrics(values, forecast_rmse):
    values = np.array(values)
    returns = np.diff(values) / values[:-1]
    cumulative_return = (values[-1] - values[0]) / values[0]
    sharpe = np.mean(returns) / np.std(returns) if np.std(returns) else 0
    drawdown = np.max(np.maximum.accumulate(values) - values)
    max_drawdown = drawdown / np.max(np.maximum.accumulate(values))
    return {
        "Cumulative Return": cumulative_return,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
        "Forecast RMSE": forecast_rmse,
    }


def run_portfolio_experiments(data_dict, abc_map):
    results = []
    configs = {
        "LSTM Only": {
            "use_abc": False,
            "use_eoq": False,
            "S": 0,
            "H": 0,
            "tuned": False,
        },
        "+ ABC": {"use_abc": True, "use_eoq": False, "S": 0, "H": 0, "tuned": False},
        "+ EOQ": {"use_abc": True, "use_eoq": True, "S": 5, "H": 0.01, "tuned": False},
        "Full System": {
            "use_abc": True,
            "use_eoq": True,
            "S": 3,
            "H": 0.005,
            "tuned": True,
        },
    }

    # Collect forecasts and actuals for all assets
    assets_data = {}
    total_rmse = 0

    for symbol, df in data_dict.items():
        data = add_technical_indicators(df)
        X_sel, y_target = select_features(data, symbol=symbol)
        merged = pd.concat([X_sel, y_target], axis=1).dropna().reset_index(drop=True)
        actual, pred = lstm_forecast_multivariate(merged, target="close")
        rmse = math.sqrt(mean_squared_error(actual, pred))
        total_rmse += rmse
        assets_data[symbol] = {"actual": actual, "pred": pred}

    avg_rmse = total_rmse / len(assets_data)

    for name, config in configs.items():
        portfolio_vals, total_cost = simulate_portfolio(
            assets_data, forecasts=None, abc_map=abc_map, variant_config=config
        )
        metrics = compute_portfolio_metrics(portfolio_vals, avg_rmse)
        metrics["Model Variant"] = name
        metrics["Total Transaction Cost"] = total_cost
        results.append(metrics)

    return pd.DataFrame(results)


# Build stock_stats for ABC classification
stock_stats = []
assets_data = {}
total_rmse = 0

for symbol, df in data_dict.items():
    try:
        data = add_technical_indicators(df)
        X_selected, y_target = select_features(data, symbol=symbol)
        merged = (
            pd.concat([X_selected, y_target], axis=1).dropna().reset_index(drop=True)
        )
        actual, pred = lstm_forecast_multivariate(merged, target="close")

        # Populate stock_stats
        initial_price = actual[0]
        final_price = actual[-1]
        cumulative_return = (final_price - initial_price) / initial_price
        volatility = np.std(np.diff(actual) / actual[:-1])
        stock_stats.append(
            {
                "symbol": symbol,
                "cumulative_return": cumulative_return,
                "volatility": volatility,
            }
        )

        # Populate assets_data for portfolio experiments
        rmse = math.sqrt(mean_squared_error(actual, pred))
        total_rmse += rmse
        assets_data[symbol] = {"actual": actual, "pred": pred}

    except Exception as e:
        print(f"Skipping {symbol} due to error: {e}")

# Classify using ABC
stock_stats_df = pd.DataFrame(stock_stats)
classified_df = classify_stocks_by_return(stock_stats_df)
abc_map = dict(zip(classified_df["symbol"], classified_df["ABC"]))

# Run portfolio experiments
final_results_df = run_portfolio_experiments(assets_data, abc_map)
final_results_df.to_csv("portfolio_backtest_summary.csv", index=False)
print(final_results_df)
