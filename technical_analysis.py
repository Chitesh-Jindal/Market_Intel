
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands


def calculate_technical_indicators(data):

    # Simple Moving Averages
    data["SMA_10"] = data["Close"].rolling(window=10).mean()
    data["SMA_20"] = data["Close"].rolling(window=20).mean()
    data["SMA_Spread"] = data["SMA_10"] - data["SMA_20"]

    # Exponential Moving Average
    data["EMA_20"] = data["Close"].ewm(
        span=20, adjust=False
    ).mean()

    # Relative Strength Index
    rsi_indicator = RSIIndicator(
        close=data["Close"], window=14
    )
    data["RSI"] = rsi_indicator.rsi()

    # Moving Average Convergence Divergence
    macd_indicator = MACD(close=data["Close"])

    data["MACD"] = macd_indicator.macd()
    data["MACD_Signal"] = macd_indicator.macd_signal()

    # Bollinger Bands
    bb_indicator = BollingerBands(
        close=data["Close"],
        window=20,
        window_dev=2
    )

    data["BB_Upper"] = bb_indicator.bollinger_hband()
    data["BB_Middle"] = bb_indicator.bollinger_mavg()
    data["BB_Lower"] = bb_indicator.bollinger_lband()

    return data


def get_technical_signals(data):

    latest = data.iloc[-1]

    # Trend signal
    if (
        latest["SMA_10"] > latest["SMA_20"]
        and latest["EMA_20"] > latest["SMA_20"]
    ):
        trend_signal = "Positive"

    elif (
        latest["SMA_10"] < latest["SMA_20"]
        and latest["EMA_20"] < latest["SMA_20"]
    ):
        trend_signal = "Negative"

    else:
        trend_signal = "Mixed"

    # Momentum signal
    if latest["RSI"] > 70:
        momentum_signal = "Strong / Overbought"

    elif latest["RSI"] < 30:
        momentum_signal = "Weak / Oversold"

    elif latest["MACD"] > latest["MACD_Signal"]:
        momentum_signal = "Positive"

    else:
        momentum_signal = "Negative"

    # Volatility signal
    if latest["Close"] > latest["BB_Upper"]:
        volatility_signal = "High / Above Upper Band"

    elif latest["Close"] < latest["BB_Lower"]:
        volatility_signal = "High / Below Lower Band"

    else:
        volatility_signal = "Normal"

    return {
        "Trend": trend_signal,
        "Momentum": momentum_signal,
        "Volatility": volatility_signal
    }

