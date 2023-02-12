
import pandas as pd
import numpy as np


def download_data(symbol, period, interval):
    valid_periods = ["1d", "5d", "1mo", "3mo",
                     "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    valid_intervals = ["1m", "2m", "5m", "15m", "30m",
                       "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]

    if period not in valid_periods:
        print("Invalid period, please choose from: {}".format(valid_periods))
        return
    if interval not in valid_intervals:
        print("Invalid interval, please choose from: {}".format(valid_intervals))
        return

    # Check if the interval and period combination is valid
    if interval == "1m" and period != "1d" or period != "5d":
        print("Interval 1m is only available for period 1d.")
        return
    elif interval != "1m" and period == "1d" and period != "5d":
        print("For data intervals less than 1d, the period must be less than or equal to 60 days.")
        return

    stock_data = yf.download(symbol, period=period, interval=interval)
    return stock_data


def calculate_technical_indicators(df):
    # Calculate moving average
    df['sma'] = df['Close'].rolling(window=24).mean()

    # Calculate Bollinger Bands
    df['std'] = df['Close'].rolling(window=24).std()
    df['upper_band'] = df['sma'] + (df['std'] * 2)
    df['lower_band'] = df['sma'] - (df['std'] * 2)

    # Calculate RSI
    df['delta'] = df['Close'].diff()
    gain = df['delta'].where(df['delta'] > 0, 0)
    loss = -df['delta'].where(df['delta'] < 0, 0)
    avg_gain = gain.rolling(window=24).mean()
    avg_loss = loss.rolling(window=24).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    return df


def preprocessing(stock_data):
    df = stock_data
    df.reset_index(inplace=True)
    df['ds'] = pd.to_datetime(df['Datetime'], utc=True)
    df['ds'] = df['ds'].dt.tz_convert(None)
    df[['y']] = df[['Close']]
    df = calculate_technical_indicators(df)
    df = df[["Close", "sma", "upper_band", "lower_band", "rsi"]].reset_index()
    return df
