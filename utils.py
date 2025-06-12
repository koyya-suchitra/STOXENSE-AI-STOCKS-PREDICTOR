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
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(numeric_df)

    # Convert to numpy array and create X, y as per your logic
    X, y = [], []

    for i in range(60, len(scaled)):
        X.append(scaled[i - 60:i])
        y.append(scaled[i, 0])  # Assuming the target is the first column like 'Close'

    return np.array(X), np.array(y), scaler

