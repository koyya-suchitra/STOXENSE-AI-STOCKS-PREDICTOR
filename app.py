# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
from utils import fetch_stock_data, preprocess_data

# Streamlit page config
st.set_page_config(page_title="📈 LSTM Stock Predictor", layout="wide")
st.title("📊 STOXENSE AI: Stock Price Trend Predictor with LSTM")

# Input box for ticker
ticker = st.text_input("Enter Stock Ticker (e.g., TSLA, INFY.NS, AAPL)", value="TSLA")

# Main prediction block
if st.button("Predict") or ticker:
    with st.spinner("Fetching and processing data..."):
        df = fetch_stock_data(ticker)

        # Check if data was returned
        if df.empty:
            st.error(f"❌ No data found for ticker '{ticker}'. Please check the symbol.")
            st.stop()

        try:
            X, y, scaler, scaled = preprocess_data(df)
        except ValueError as e:
            st.error(f"⚠️ Data preprocessing error: {e}")
            st.stop()

        if X.shape[0] < 10:
            st.error("⚠️ Not enough data to train the model. Try a different ticker.")
            st.stop()

        # Show technical indicators
        st.subheader(f"📉 {ticker} Stock Closing Price vs 50-Day Moving Average")
        st.line_chart(df[['Close', 'MA50']])
        st.subheader("📊 Relative Strength Index (RSI) - Momentum Indicator")
        st.line_chart(df[['RSI']])

        # LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        st.info("🧠 Training LSTM model... (10 epochs)")
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)

        # Prediction
        predicted = model.predict(X)

        # Inverse transform only the 'Close' column
        predicted_close = scaler.inverse_transform(
            np.hstack([predicted, np.zeros((predicted.shape[0], scaled.shape[1] - 1))])
        )[:, 0]

        actual_close = scaler.inverse_transform(
            np.hstack([y.reshape(-1, 1), np.zeros((y.shape[0], scaled.shape[1] - 1))])
        )[:, 0]

        # 📊 Plot predictions
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(actual_close, label="📈 Actual Close")
        ax.plot(predicted_close, label="🤖 Predicted Close")
        ax.set_title(f"LSTM Stock Prediction for {ticker}")
        ax.legend()
        st.pyplot(fig)

        # Save model
        os.makedirs("model", exist_ok=True)
        model_path = f"model/{ticker}_lstm_model.h5"
        model.save(model_path)
        st.success(f"✅ Model saved to `{model_path}`")
