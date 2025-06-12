# 📊 STOX AI: Stock Price Trend Predictor with LSTM

Welcome to **STOX AI**, an AI-powered interactive dashboard that predicts stock price trends using a Long Short-Term Memory (LSTM) deep learning model. It includes visual analysis with technical indicators like **50-day Moving Average (MA50)** and **Relative Strength Index (RSI)**.

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

1. **Clone the repository:**

```bash
git clone https://github.com/koyya-suchitra/Stock-Price-Predictor.git
cd stockprice
Create a virtual environment:

bash
Copy
Edit
python -m venv venv
Activate the virtual environment:

Windows:

bash
Copy
Edit
venv\Scripts\activate
macOS/Linux:

bash
Copy
Edit
source venv/bin/activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py