
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


def preprocess_data(data, lookback=1):
    # Scale the data between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    # Split the data into input (X) and output (y)
    X = []
    y = []
    for i in range(lookback, data.shape[0]):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)

    # Reshape the input data to be [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler


def build_model(X):
    # Initialize the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True,
              input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def predict_price(ticker, interval, lookback=1):
    # Get the stock price data
    data = get_data(ticker, interval)

    # Preprocess the data
    X, y, scaler = preprocess_data(data, lookback)

    # Build and fit the LSTM model
    model = build_model(X)
    model.fit(X, y, epochs=100, batch_size=32)

    # Use the model to predict the next closing price
    future_price = model.predict(X[-1].reshape(1, lookback, 1))

    # Inverse transform the prediction back to the original scale
    future_price = scaler.inverse_transform(future_price)

    return future_price[0][0]


# Example usage:
ticker = "AAPL"
interval = "1h"
