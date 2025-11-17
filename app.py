import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf

st.set_page_config(layout="wide", page_title="Financial Sector LSTM Dashboard")


st.title("📈 Indian Financial Sector: Index & LSTM Forecast")
st.markdown("A dashboard to visualize a custom market-cap-weighted financial index and forecast it using an LSTM model.")

# --- Tickers and Market Caps (Constants) ---
TICKERS = [
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS",
    "AXISBANK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "HDFCLIFE.NS", "SBILIFE.NS", "ICICIPRULI.NS"
]

MARKET_CAPS = {
    "HDFCBANK.NS": 1100000,
    "ICICIBANK.NS": 800000,
    "SBIN.NS": 700000,
    "KOTAKBANK.NS": 350000,
    "AXISBANK.NS": 300000,
    "BAJFINANCE.NS": 450000,
    "BAJAJFINSV.NS": 200000,
    "HDFCLIFE.NS": 140000,
    "SBILIFE.NS": 110000,
    "ICICIPRULI.NS": 90000
}

# --- 1. Data Loading and Index Creation (Cached) ---
@st.cache_data(ttl="1d")  
def load_data(tickers, market_caps):
    """
    Downloads stock data, cleans it, and computes the weighted sector index.
    """
    with st.status("Downloading financial data from yfinance..."):
        raw = yf.download(tickers, start="2015-01-01", end="2025-01-01")["Close"]
        
        # Drop columns with no data
        raw = raw.dropna(axis=1, how="all")
        
        st.write(f"Tickers successfully downloaded: {', '.join(raw.columns.tolist())}")

        
        filtered_caps = {t: market_caps[t] for t in raw.columns if t in market_caps}
        
        weights = np.array(list(filtered_caps.values()))
        weights = weights / weights.sum()  # normalize to sum = 1
        
       
        weights_df = pd.DataFrame({
            "Ticker": filtered_caps.keys(),
            "Weight": weights
        }).sort_values(by="Weight", ascending=False).reset_index(drop=True)

       
        st.write("Calculating market-cap weighted index...")
        sector_index = (raw[filtered_caps.keys()] * weights).sum(axis=1)
        sector_index.name = "Index"
        
    return raw, sector_index, weights_df

# --- 2. LSTM Model Training and Prediction (Cached) ---
def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

@st.cache_data
def train_and_predict(data_series, window_size=60):
    """
    Preprocesses data, builds, trains, and predicts using an LSTM model.
    Caches the results.
    """
    
    tf.random.set_seed(42)
    np.random.seed(42)
    
    status_placeholder = st.empty()
    status_placeholder.info(f"Training an LSTM model with a {window_size}-day lookback window. This may take a moment on the first run...")
    
    
    with st.spinner("Scaling and preparing data..."):
        df = data_series.to_frame()
        scaler = MinMaxScaler(feature_range=(0, 1))
        df["Scaled"] = scaler.fit_transform(df[["Index"]])
        
        values = df["Scaled"].values
        X, y = create_sequences(values, window_size)

        # Split into train and test
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

       
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Get dates for the test set
        test_dates = df.index[window_size + train_size:]

    
    with st.spinner("Building LSTM model architecture..."):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mean_squared_error")

    
    with st.spinner("Training model (this runs only once)..."):
        history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

   
    with st.spinner("Generating predictions..."):
        pred = model.predict(X_test)

    
    with st.spinner("Inverse scaling results..."):
        predicted_prices = scaler.inverse_transform(pred)
        actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    status_placeholder.success("Model training and prediction complete!")
    
    return actual_prices, predicted_prices, history.history, test_dates


raw_data, sector_index, weights_df = load_data(TICKERS, MARKET_CAPS)


tab1, tab2, tab3 = st.tabs(["Sector Index Dashboard", "LSTM Prediction", "Data Explorer"])


with tab1:
    st.header("Financial Sector Market Cap Weighted Index")
    
    
    col1, col2, col3 = st.columns(3)
    latest_value = sector_index.iloc[-1]
    prev_value = sector_index.iloc[-2]
    delta = latest_value - prev_value
    col1.metric("Latest Index Value", f"{latest_value:.2f}", f"{delta:.2f} (from prev. day)")
    col2.metric("Max Value", f"{sector_index.max():.2f}")
    col3.metric("Min Value", f"{sector_index.min():.2f}")

   
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=sector_index.index, y=sector_index.values, 
                             name='Sector Index', line=dict(color='royalblue', width=2)))
    fig1.update_layout(
        title="Financial Sector Market Cap Weighted Index (2015-2025)",
        xaxis_title="Date",
        yaxis_title="Index Value (Weighted Price)",
        hovermode="x unified"
    )
    st.plotly_chart(fig1, use_container_width=True)


    st.subheader("Index Components & Weights")
    st.dataframe(weights_df, use_container_width=True, hide_index=True)


with tab2:
    st.header("LSTM Model Prediction")
    

    WINDOW = 60
    actual, predicted, history, test_dates = train_and_predict(sector_index, WINDOW)


    st.subheader("Actual vs. Predicted Trend")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=test_dates, y=actual.flatten(), name='Actual Trend',
                             line=dict(color='blue', width=2)))
    fig2.add_trace(go.Scatter(x=test_dates, y=predicted.flatten(), name='Predicted Trend',
                             line=dict(color='red', width=2, dash='dot')))
    fig2.update_layout(
        title="Financial Sector Index Prediction (Test Set)",
        xaxis_title="Date",
        yaxis_title="Sector Index Value",
        hovermode="x unified"
    )
    st.plotly_chart(fig2, use_container_width=True)


    with st.expander("View Model Training Loss"):
        loss_df = pd.DataFrame(history)
        loss_df.index = loss_df.index + 1
        
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(y=loss_df['loss'], name='Training Loss',
                                 line=dict(color='green')))
        fig3.update_layout(
            title="Model Loss Over Epochs",
            xaxis_title="Epoch",
            yaxis_title="Mean Squared Error Loss"
        )
        st.plotly_chart(fig3, use_container_width=True)


with tab3:
    st.header("Data Explorer")
    
    st.subheader("Calculated Sector Index")
    st.dataframe(sector_index)
    
    st.subheader("Component Stock Prices (Close)")
    st.dataframe(raw_data)