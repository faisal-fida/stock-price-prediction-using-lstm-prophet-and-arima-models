
from prophet import Prophet
from sklearn.model_selection import train_test_split


def prophet_predictor(df):
    train, test = train_test_split(data, test_size=0.2, random_state=0)
    train.shape, test.shape

    model = Prophet()
    model.fit(train)

    future = model.make_future_dataframe(periods=15, freq="H")
    forecast = model.predict(future)

    y_true = test['y']
    y_pred_prophet = forecast[['yhat']]
    y_pred_prophet = y_pred_prophet.iloc[-84:]

    future = model.make_future_dataframe(periods=1, freq="H")
    forecast = model.predict(future)
    estimated_closing_price_hour = forecast.tail(1)["yhat"].values[0]

    future = model.make_future_dataframe(periods=1, freq="5min")
    forecast = model.predict(future)
    estimated_closing_price_5min = forecast.tail(1)["yhat"].values[0]

    future = model.make_future_dataframe(periods=1, freq="10min")
    forecast = model.predict(future)
    estimated_closing_price_10min = forecast.tail(1)["yhat"].values[0]

    future = model.make_future_dataframe(periods=1, freq="15min")
    forecast = model.predict(future)
    estimated_closing_price_15min = forecast.tail(1)["yhat"].values[0]

    future = model.make_future_dataframe(periods=1, freq="30min")
    forecast = model.predict(future)
    estimated_closing_price_30min = forecast.tail(1)["yhat"].values[0]

    return estimated_closing_price_hour, estimated_closing_price_5min, estimated_closing_price_10min, estimated_closing_price_15min, estimated_closing_price_30min
