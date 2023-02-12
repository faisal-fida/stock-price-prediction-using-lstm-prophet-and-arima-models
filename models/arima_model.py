from statsmodels.tsa.arima.model import ARIMA


def arima_predictor(symbol, next_year=False, next_month=False, next_day=False, next_hour=False, next_5min=False, next_10min=False, next_15min=False, next_30min=False):
    # Download the data
    # data = yf.download(symbol, interval='1d')
    # data.dropna(inplace=True)

    # Calculate the number of periods for each prediction type
    periods = {
        "next_year": 365,
        "next_month": 30,
        "next_day": 1,
        "next_hour": 1 / 24,
        "next_5min": 1 / 24 / 12,
        "next_10min": 1 / 24 / 6,
        "next_15min": 1 / 24 / 4,
        "next_30min": 1 / 24 / 2
    }

    # Create a dictionary to store the prediction results
    predictions = {}

    # Check which prediction types the user wants to calculate
    if next_year:
        model = ARIMA(data["Close"], order=(5, 1, 0))
        model_fit = model.fit(disp=0)
        future = model_fit.forecast(steps=periods["next_year"])
        predictions["next_year"] = future[0][-1]
    if next_month:
        model = ARIMA(data["Close"], order=(5, 1, 0))
        model_fit = model.fit(disp=0)
        future = model_fit.forecast(steps=periods["next_month"])
        predictions["next_month"] = future[0][-1]
    if next_day:
        model = ARIMA(data["Close"], order=(5, 1, 0))
        model_fit = model.fit(disp=0)
        future = model_fit.forecast(steps=periods["next_day"])
        predictions["next_day"] = future[0][-1]
    if next_hour:
        model = ARIMA(data["Close"], order=(5, 1, 0))
        model_fit = model.fit(disp=0)
        future = model_fit.forecast(steps=periods["next_hour"])
        predictions["next_hour"] = future[0][-1]
    if next_5min:
        model = ARIMA(data["Close"], order=(5, 1, 0))
        model_fit = model.fit(disp=0)
        future = model_fit.forecast(steps=periods["next_5min"])
        predictions["next_5min"] = future[0][-1]
    if next_10min:
        model = ARIMA(data["Close"], order=(5, 1, 0))
        model_fit = model.fit(disp=0)
