import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load saved model
model = load_model('lstm_stock_model.h5')

# Streamlit UI
st.title("📈 Stock Price Predictor")
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", "AAPL")

if st.button("Predict"):
    # Fetch latest data
    data = yf.download(ticker, period="1y")
    close_prices = data['Close'].values.reshape(-1, 1)
    
    # Normalize
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(close_prices)
    
    # Create sequences (last 60 days to predict tomorrow)
    last_60_days = scaled_prices[-60:].reshape(1, 60, 1)
    
    # Predict
    predicted_price_scaled = model.predict(last_60_days)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)
    
    # Plot
    st.subheader(f"Predicted Price for {ticker}: ${predicted_price[0][0]:.2f}")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data.index, data['Close'], label="Historical Price")
    ax.axhline(y=predicted_price[0][0], color='r', linestyle='--', label="Predicted Price")
    ax.set_title(f"{ticker} Stock Price Prediction")
    ax.legend()
    st.pyplot(fig)