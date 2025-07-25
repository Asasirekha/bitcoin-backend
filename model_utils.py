# model_utils.py
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime, timedelta

def get_model():
    today = datetime.today().strftime('%Y-%m-%d')
    df = yf.download('BTC-USD', start='2017-01-01', end=today)
    df = df[['Close']].dropna()
    df['Days'] = np.arange(len(df)).reshape(-1, 1)
    
    X = df[['Days']]
    y = df['Close']
    
    model = LinearRegression()
    model.fit(X, y)
    return model, len(df)

def predict_next_days(n_days=30):
    model, last_day_index = get_model()
    future_days = np.arange(last_day_index, last_day_index + n_days).reshape(-1, 1)
    future_prices = model.predict(future_days)

    start_date = datetime.today()
    future_dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(n_days)]

    results = [{'date': d, 'price': round(p, 2)} for d, p in zip(future_dates, future_prices)]
    return results
