import numpy as np
import pandas as pd


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    previous_close = close.shift(1)
    parts = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    )
    return parts.max(axis=1)


def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate indicators with pandas/numpy (no external TA package required)."""
    df = data.copy()
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    # Trend and moving averages
    df["SMA_10"] = close.rolling(10).mean()
    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()
    df["EMA_20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA_50"] = close.ewm(span=50, adjust=False).mean()
    df["SMA_Spread"] = df["SMA_10"] - df["SMA_20"]

    # RSI (Wilder-style exponential smoothing)
    change = close.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df.loc[(avg_loss == 0) & (avg_gain > 0), "RSI"] = 100
    df.loc[(avg_gain == 0) & (avg_loss > 0), "RSI"] = 0

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger Bands
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_Middle"] = bb_mid
    df["BB_Upper"] = bb_mid + 2 * bb_std
    df["BB_Lower"] = bb_mid - 2 * bb_std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / bb_mid.replace(0, np.nan)
    bb_range = (df["BB_Upper"] - df["BB_Lower"]).replace(0, np.nan)
    df["BB_Percent_B"] = (close - df["BB_Lower"]) / bb_range

    # Average True Range and ADX
    tr = _true_range(high, low, close)
    df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    atr_safe = df["ATR"].replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr_safe
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr_safe
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["ADX"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    # Stochastic oscillator
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    stoch_range = (high14 - low14).replace(0, np.nan)
    df["Stochastic"] = 100 * (close - low14) / stoch_range
    df["Stochastic_Signal"] = df["Stochastic"].rolling(3).mean()

    # Commodity Channel Index
    typical = (high + low + close) / 3
    typical_mean = typical.rolling(20).mean()
    mean_deviation = typical.rolling(20).apply(
        lambda values: np.mean(np.abs(values - np.mean(values))), raw=True
    )
    df["CCI"] = (typical - typical_mean) / (0.015 * mean_deviation.replace(0, np.nan))

    # On-balance volume
    direction = np.sign(close.diff()).fillna(0)
    df["OBV"] = (direction * volume).cumsum()
    df["OBV_Change"] = df["OBV"].diff(5) / volume.rolling(20).sum().replace(0, np.nan)

    return df
