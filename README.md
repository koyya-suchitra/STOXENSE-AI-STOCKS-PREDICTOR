# 📊 STOXENSE AI: Stock Price Trend Predictor with LSTM

Welcome to **STOXENSE AI**, an AI-powered interactive dashboard that predicts stock price trends of different companies using a Long Short-Term Memory (LSTM) deep learning model. It includes visual analysis with technical indicators like **50-day Moving Average (MA50)** and **Relative Strength Index (RSI)**.

---

## 🚀 Features

- 📈 Predict stock price trends using historical data
- 🤖 LSTM-based AI model built with TensorFlow/Keras
- 📊 Visualizes **Closing Price**, **MA50**, and **RSI**
- 📤 Option to save trained models for future use
- 🔎 Choose any stock ticker (e.g., `TSLA`, `AAPL`, `INFY.NS`)
- 🌐 Live deployment using **Streamlit Cloud**

---

## 📦 Tech Stack

- **Python**, **Pandas**, **NumPy**
- **TensorFlow / Keras** for LSTM modeling
- **Scikit-learn** for normalization
- **Matplotlib** for plotting
- **yfinance** for live stock data
- **Streamlit** for web UI

---

## 📁 Project Structure

stockprice/
├── app.py # Streamlit frontend app
├── stock_lstm.ipynb # Jupyter notebook for dev/testing
├── utils.py # Data processing functions
├── requirements.txt # Dependencies
├── .gitignore # Git ignore rules
├── model/ # Saved LSTM models
└── venv/ # Virtual environment (ignored by Git)

---

## ⚙️ Setup Instructions (Local)

### 1. Clone the repository

```bash
git clone https://github.com/koyya-suchitra/Stock-Price-Predictor.git
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
