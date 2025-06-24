import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ====================== STEP 1: Data Collection ======================
def fetch_data(ticker="AAPL", start_date="2015-01-01", end_date="2024-12-31"):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.to_csv(f"{ticker}_stock_data.csv")
    return data

# ====================== STEP 2: Preprocessing ======================
def preprocess_data(data):
    # Use only 'Close' price
    close_prices = data['Close'].values.reshape(-1, 1)
    
    # Normalize data (0 to 1)
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(close_prices)
    
    # Create time-series sequences
    def create_sequences(data, seq_length=60):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)
    
    X, y = create_sequences(scaled_prices)
    
    # Train-test split (80-20)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, X_test, y_train, y_test, scaler

# ====================== STEP 3: LSTM Model ======================
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ====================== STEP 4: Training & Evaluation ======================
def train_and_evaluate(model, X_train, y_train, X_test, y_test, scaler):
    # Train
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    
    # Predict
    y_pred = model.predict(X_test)
    y_pred_actual = scaler.inverse_transform(y_pred)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual, label="Actual Price", color='blue')
    plt.plot(y_pred_actual, label="Predicted Price", color='red')
    plt.title("Stock Price Prediction (LSTM)")
    plt.legend()
    plt.savefig('prediction_plot.png')  # Save for Streamlit
    plt.show()
    
    return model, history

# ====================== MAIN EXECUTION ======================
if __name__ == "__main__":
    # Step 1: Fetch data
    data = fetch_data()
    
    # Step 2: Preprocess
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
    
    # Step 3: Build model
    model = build_lstm_model((X_train.shape[1], 1))
    
    # Step 4: Train & evaluate
    model, history = train_and_evaluate(model, X_train, y_train, X_test, y_test, scaler)
    
    # Save model for deployment
    model.save('lstm_stock_model.h5')
    print("Model saved as 'lstm_stock_model.h5'")