# 📊 STOXENSE AI: Stock Price Trend Predictor with LSTM

Welcome to **STOXENSE AI**, an AI-powered stock trend prediction dashboard. This project uses **LSTM (Long Short-Term Memory)** models to forecast stock price movements based on historical prices and technical indicators like **Moving Average (MA50)** and **Relative Strength Index (RSI)**.

The app is built with **Streamlit**, features live or cached data support using `.csv` files, and is deployable to **Streamlit Cloud**.

---

## 🚀 Features

- 📈 Predicts stock price trends using past data
- 🤖 Deep learning with LSTM networks (Keras + TensorFlow)
- 📊 Visual indicators: Closing Price, MA50, and RSI
- 🧠 Trains the model on-the-fly (10 epochs)
- 💾 Saves trained models for reuse
- 🔍 Supports both live data fetch and offline `.csv` fallback
- 🌐 Deployable on Streamlit Cloud (without needing live API access)

---

## 📦 Tech Stack

- **Python**
- **TensorFlow / Keras** – LSTM model
- **scikit-learn** – Normalization
- **yfinance** – Historical stock data
- **Pandas**, **NumPy** – Data wrangling
- **Matplotlib** – Charting
- **Streamlit** – UI and deployment

---

## 📁 Project Structure
<pre> <code> stockprice/ ├── app.py # Main Streamlit dashboard ├── stock_lstm.ipynb # Jupyter notebook for model dev/testing ├── utils.py # Stock data fetch & preprocessing (with CSV fallback) ├── requirements.txt # Python dependencies ├── model/ # Saved LSTM models (.h5) ├── data/ # Cached stock data in CSV format ├── .gitignore # Exclude venv and model files └── venv/ # Virtual environment (ignored) </code> </pre>
---
# 📌 How it Works
### Data Fetching
App tries to load stock data from data/{ticker}.csv.
If not found and running locally, it uses yfinance to fetch fresh data and caches it as a .csv.
### Preprocessing
Calculates 50-day moving average (MA50) and RSI.
Normalizes the data using MinMaxScaler.
Generates input sequences for LSTM.
### Model Training
A 2-layer LSTM model is trained for 10 epochs.
Saves model to model/{ticker}_lstm_model.h5.
### Prediction
Predicts and plots closing prices.
Displays Actual vs Predicted graph.

---
## 💡 Example Stock Tickers You Can Use

You can input the following company ticker symbols in the app to fetch and predict stock price trends.

| Company     | Ticker   |
|-------------|----------|
| Tesla       | TSLA     |
| Apple       | AAPL     |
| Infosys     | INFY.NS  |
| TCS         | TCS.NS   |
| Google      | GOOGL    |
| Microsoft   | MSFT     |

> ℹ️ **Note**: For Indian stocks listed on NSE, append `.NS` to the ticker symbol.  
> Example: `INFY.NS` for Infosys, `TCS.NS` for Tata Consultancy Services.
>  
---

## 🤖 Credits
Created by Suchitra Koyya as a project to demonstrate LSTM-based stock prediction using deep learning and Streamlit.

---
## ⚙️ Setup Instructions (Local)

### 1. Clone the repository

```bash
git clone https://github.com/koyya-suchitra/STOXENSE-AI-STOCKS-PREDICTOR.git
cd Stock-Price-Predictor
```

### 2. Create a virtual environment
```bash
python -m venv venv
```
### 3. Activate the virtual environment
Windows:
```bash
venv\Scripts\activate
```
macOS/Linux:
```bash
source venv/bin/activate
```
### 4. Install dependencies
```bash
pip install -r requirements.txt
```
### 5. Run the Streamlit app
```bash
streamlit run app.py
```
