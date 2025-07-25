from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import numpy as np

app = Flask(__name__)
CORS(app, origins=["https://bitcoin-frontend-1bxbtket3-adapa-sasi-rekhas-projects.vercel.app/"])

@app.route('/')
def home():
    return "🚀 Flask Backend for Bitcoin Price Prediction is running."

@app.route('/predict')
def predict_price():
    start_date_str = request.args.get('start_date', datetime.today().strftime('%Y-%m-%d'))
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    except ValueError:
        return jsonify({'error': 'Invalid start_date format. Use YYYY-MM-DD'}), 400

    end_date = datetime.today()

    # Load historical Bitcoin data
    btc_data = yf.download("BTC-USD", start="2020-01-01", end=end_date.strftime('%Y-%m-%d'), interval="1d")

    if btc_data.empty:
        return jsonify({'error': 'Failed to fetch Bitcoin data'}), 500

    # Reset index to get 'Date' column
    btc_data = btc_data.reset_index()

    # Use numeric representation of date for regression
    btc_data['DateOrdinal'] = pd.to_datetime(btc_data['Date']).map(datetime.toordinal)
    X = btc_data[['DateOrdinal']]
    y = btc_data[['Close']]

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Predict next 7 days from the selected start date
    future_dates = [start_date + timedelta(days=i) for i in range(7)]
    future_date_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    predicted_prices = model.predict(future_date_ordinals)

    # Convert to INR (approximate conversion rate)
    usd_to_inr = 83.5

    predictions = [
        {
            "date": d.strftime('%Y-%m-%d'),
            "price_usd": round(p.item(), 3),  # ✅ FIX: Convert NumPy value to Python float
            "price_inr": round(p.item() * usd_to_inr, 3)
        }
        for d, p in zip(future_dates, predicted_prices)
    ]

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
