# utils.py
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import datetime

def fetch_stock_data(ticker):
    import datetime
    import yfinance as yf
    today = datetime.date.today().strftime('%Y-%m-%d')
    df = yf.download(ticker, start="2018-01-01", end=today, auto_adjust=True)

    # 👇 FIX: Drop multi-index if present
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

def preprocess_data(df):
    df = df.dropna()
    numeric_df = df.select_dtypes(include=['number'])

    if numeric_df.empty:
        raise ValueError("Dataframe has no numeric columns after filtering. Check the input data.")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(numeric_df)

    X = []
    y = []
    window_size = 50

    for i in range(window_size, len(scaled)):
        X.append(scaled[i - window_size:i])
        y.append(scaled[i, 0])  # Predict 'Close'

    return np.array(X), np.array(y), scaler, scaled


