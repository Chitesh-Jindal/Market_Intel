
# ============================================================
# MARKET INTEL
# Indian Stock Intelligence & Analysis Platform
# ============================================================


# SECTION 1: IMPORTS AND CONFIGURATION

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from ta.momentum import RSIIndicator


# Application configuration
st.set_page_config(
    page_title="Market Intel",
    page_icon="📈",
    layout="wide"
)

APP_TITLE = "Market Intel"
APP_SUBTITLE = "Indian Stock Intelligence"

DEFAULT_STOCK = "RELIANCE.NS"
DATA_PERIOD = "2y"
DATA_INTERVAL = "1d"


# ============================================================
# SECTION 2: DATA COLLECTION
# ============================================================

@st.cache_data(ttl=3600)
def load_stock_data(stock):

    stock = stock.strip().upper()

    if not stock:
        raise ValueError("Please enter a stock symbol.")

    data = yf.download(
        stock,
        period=DATA_PERIOD,
        interval=DATA_INTERVAL,
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if data.empty:
        raise ValueError(
            "No stock data found. Check the stock symbol."
        )

    # Handle MultiIndex columns returned by yfinance
    if isinstance(data.columns, pd.MultiIndex):

        expected_columns = {
            "Open", "High", "Low", "Close", "Volume"
        }

        for level in range(data.columns.nlevels):

            level_values = set(
                data.columns.get_level_values(level)
            )

            if expected_columns.issubset(level_values):
                data.columns = data.columns.get_level_values(level)
                break

        else:
            raise ValueError(
                "Unexpected data format returned by yfinance."
            )

    return data


# ============================================================
# SECTION 3: DATA PROCESSING
# ============================================================

def process_stock_data(data):

    data = data.copy()

    # Sort observations chronologically
    data = data.sort_index()

    # Remove duplicate dates
    data = data[~data.index.duplicated(keep="last")]

    # Convert numerical columns to numeric types
    numeric_columns = [
        "Open", "High", "Low", "Close", "Volume"
    ]

    for column in numeric_columns:

        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    # Closing price is essential for our analysis
    if "Close" not in data.columns:
        raise ValueError("Closing price data is unavailable.")

    # Remove observations with missing closing prices
    data = data.dropna(subset=["Close"])

    # Remove invalid closing prices
    data = data[data["Close"] > 0]

    if len(data) < 60:
        raise ValueError(
            "Insufficient historical data for analysis."
        )

    # Calculate daily returns
    data["Daily_Return (%)"] = (
        data["Close"].pct_change() * 100
    )

    return data


# ============================================================
# SECTION 4: TECHNICAL INDICATORS
# ============================================================

def calculate_indicators(data):

    data = data.copy()

    # Simple Moving Averages
    data["SMA_10"] = (
        data["Close"].rolling(window=10).mean()
    )

    data["SMA_20"] = (
        data["Close"].rolling(window=20).mean()
    )

    # Exponential Moving Average
    data["EMA_20"] = (
        data["Close"].ewm(
            span=20,
            adjust=False,
            min_periods=20
        ).mean()
    )

    # Relative Strength Index
    rsi_indicator = RSIIndicator(
        close=data["Close"],
        window=14
    )

    data["RSI_14"] = rsi_indicator.rsi()

    return data


# ============================================================
# TEMPORARY MAIN EXECUTION
# ============================================================

def main():

    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    st.divider()

    stock = st.text_input(
        "Enter Indian stock symbol",
        DEFAULT_STOCK
    )

    if st.button("Analyze Stock", type="primary"):

        try:

            with st.spinner("Fetching and analyzing stock data..."):

                # Section 2
                data = load_stock_data(stock)

                # Section 3
                data = process_stock_data(data)

                # Section 4
                data = calculate_indicators(data)

            st.success("Stock data successfully analyzed!")

            # Latest stock information
            latest = data.iloc[-1]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Latest Closing Price",
                f"₹{latest['Close']:.2f}"
            )

            col2.metric(
                "Daily Return",
                f"{latest['Daily_Return (%)']:.2f}%"
            )

            col3.metric(
                "RSI (14)",
                f"{latest['RSI_14']:.2f}"
            )

            st.divider()

            # Historical price analysis
            st.subheader("Historical Price Analysis")

            st.line_chart(
                data[["Close", "SMA_10", "SMA_20", "EMA_20"]]
            )

            # RSI analysis
            st.subheader("Relative Strength Index")

            st.line_chart(data["RSI_14"])

            # Recent historical data
            st.subheader("Recent Historical Data")

            st.dataframe(
                data.tail(10).round(2),
                use_container_width=True
            )

        except Exception as error:

            st.error(f"Analysis failed: {error}")


if __name__ == "__main__":
    main()
