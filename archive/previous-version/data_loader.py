
import yfinance as yf
import pandas as pd


def load_stock_data(stock, period="2y"):

    # Download stock data
    data = yf.download(stock, period=period)

    # Handle empty data
    if data.empty:
        return None

    # Handle MultiIndex columns returned by yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Calculate daily returns
    data["Daily_Return (%)"] = data["Close"].pct_change() * 100

    return data

