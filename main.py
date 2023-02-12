
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as plt

from models.lstm_model import lstm_predictor
from models.arima_model import arima_predictor
from models.prophet_model import prophet_predictor
from data_processing import download_data, preprocessing

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def ensemble_models(df):
    # Prophet Model
    y_pred_prophet = prophet_predictor(df)
    # ARIMA Model
    y_pred_arima = arima_predictor(df)
    # LSTM Model
    y_pred_lstm = lstm_predictor(df)
    # Model Ensemble
    yhat = (yhat_prophet + yhat_arima + yhat_lstm) / 3
    return yhat, y_true, y_pred_prophet, y_pred_lstm, y_pred_arima


def calculate_errors(y_true, y_pred_prophet, y_pred_lstm, y_pred_arima):
    error_dict = {}
    # caluclate errors (mae, mse and r2_score) for all models and store in a dict
    error_dict["prophet_mae"] = mean_absolute_error(y_true, y_pred_prophet)
    error_dict["lstm_mae"] = mean_absolute_error(y_true, y_pred_lstm)
    error_dict["arima_mae"] = mean_absolute_error(y_true, y_pred_arima)
    error_dict["prophet_mse"] = mean_squared_error(y_true, y_pred_prophet)
    error_dict["lstm_mse"] = mean_squared_error(y_true, y_pred_lstm)
    error_dict["arima_mse"] = mean_squared_error(y_true, y_pred_arima)
    error_dict["prophet_r2"] = r2_score(y_true, y_pred_prophet)
    error_dict["lstm_r2"] = r2_score(y_true, y_pred_lstm)
    error_dict["arima_r2"] = r2_score(y_true, y_pred_arima)
    return error_dict


def main():
    # download stock data
    stock_data = download_data(symbol, period, interval)
    # preprocess the data
    df = preprocessing(stock_data)
    # predict price using all models and ensemble the
    yhat, y_true, y_pred_prophet, y_pred_lstm, y_pred_arima = ensemble_models(
        df)
    error_dict = calculate_errors(
        y_true, y_pred_prophet, y_pred_lstm, y_pred_arima)


if __name__ == '__main__':
    main()
