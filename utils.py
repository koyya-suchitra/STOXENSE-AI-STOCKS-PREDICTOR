# utils.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import datetime
import os

def fetch_stock_data(ticker, force_live=False):
    today = datetime.date.today().strftime('%Y-%m-%d')
    file_path = f"data/{ticker}.csv"

    if not force_live and os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    else:
        try:
            import yfinance as yf
            df = yf.download(ticker, start="2021-01-01", end=today, auto_adjust=True)
            os.makedirs("data", exist_ok=True)
            df.to_csv(file_path)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to fetch data for {ticker}: {str(e)}")

    # Drop multi-index if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Close']].copy()
    df['MA50'] = df['Close'].rolling(window=50).mean()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)
    return df


def preprocess_data(df, look_back=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i - look_back:i])
        y.append(scaled[i, 0])  # Close price

    return np.array(X), np.array(y), scaler
