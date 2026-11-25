import pandas as pd
import yfinance as yf


def load_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download and validate daily OHLCV data from Yahoo Finance."""
    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data is None or data.empty:
        raise ValueError(
            f"No market data was returned for {ticker}. Check the ticker or try again."
        )

    # yfinance may return MultiIndex columns even for a single ticker.
    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0)
        level1 = data.columns.get_level_values(1)
        if "Close" in level0:
            data.columns = level0
        elif "Close" in level1:
            data.columns = level1
        else:
            data.columns = ["_".join(map(str, c)) for c in data.columns]

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Downloaded data is missing columns: {missing}")

    data = data[required].copy()
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    data = data.loc[~data.index.duplicated(keep="last")].sort_index()
    data["Volume"] = data["Volume"].fillna(0)

    if len(data) < 250:
        raise ValueError(
            f"Only {len(data)} trading sessions were downloaded. "
            "Use a longer period (3y or 5y) for model training."
        )
    return data
