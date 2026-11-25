
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)

from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange

# XGBoost is optional until installed
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# --------------------------------------------------
# 1. FEATURE DEFINITIONS
# --------------------------------------------------

EXISTING_FEATURES = [
    "Close_vs_SMA20",
    "SMA_Spread_Relative",
    "Close_vs_EMA20",
    "MACD_Hist_Relative",
    "BB_Percent_B",
    "BB_Width_Relative",
    "Volume_Relative"
]

NEW_FEATURES = [
    "Return_1D",
    "Return_2D",
    "Return_5D",
    "Return_10D",
    "Return_20D",
    "Volatility_5D",
    "Volatility_10D",
    "Volatility_20D",
    "RSI",
    "ROC_10",
    "ADX",
    "ATR_Relative",
    "Volume_Change",
    "Volume_ZScore",
    "Daily_Range",
    "Open_Close_Return"
]

ALL_FEATURES = EXISTING_FEATURES + NEW_FEATURES


# --------------------------------------------------
# 2. FEATURE ENGINEERING
# --------------------------------------------------

def prepare_ml_dataset(data):

    data = data.copy()

    # Relative volume
    data["Volume_SMA20"] = (
        data["Volume"].rolling(20).mean()
    )

    # Existing seven features
    data["Close_vs_SMA20"] = (
        data["Close"] / data["SMA_20"] - 1
    )

    data["SMA_Spread_Relative"] = (
        data["SMA_Spread"] / data["Close"]
    )

    data["Close_vs_EMA20"] = (
        data["Close"] / data["EMA_20"] - 1
    )

    data["MACD_Hist_Relative"] = (
        (data["MACD"] - data["MACD_Signal"])
        / data["Close"]
    )

    bb_range = (
        data["BB_Upper"] - data["BB_Lower"]
    ).replace(0, np.nan)

    data["BB_Percent_B"] = (
        (data["Close"] - data["BB_Lower"])
        / bb_range
    )

    data["BB_Width_Relative"] = (
        bb_range / data["BB_Middle"]
    )

    data["Volume_Relative"] = (
        data["Volume"] / data["Volume_SMA20"]
    )

    # New price momentum features
    for period in [1, 2, 5, 10, 20]:

        data[f"Return_{period}D"] = (
            data["Close"].pct_change(period)
        )

    # Rolling volatility
    daily_returns = data["Close"].pct_change()

    for period in [5, 10, 20]:

        data[f"Volatility_{period}D"] = (
            daily_returns.rolling(period).std()
        )

    # Momentum indicators
    data["RSI"] = RSIIndicator(
        close=data["Close"],
        window=14
    ).rsi()

    data["ROC_10"] = ROCIndicator(
        close=data["Close"],
        window=10
    ).roc()

    # Trend strength
    data["ADX"] = ADXIndicator(
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        window=14
    ).adx()

    # Average True Range relative to price
    atr = AverageTrueRange(
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        window=14
    ).average_true_range()

    data["ATR_Relative"] = (
        atr / data["Close"]
    )

    # Volume behaviour
    data["Volume_Change"] = (
        data["Volume"].pct_change()
    )

    volume_std = data["Volume"].rolling(20).std()

    data["Volume_ZScore"] = (
        (data["Volume"] - data["Volume_SMA20"])
        / volume_std.replace(0, np.nan)
    )

    # Daily price range
    data["Daily_Range"] = (
        (data["High"] - data["Low"])
        / data["Close"]
    )

    # Intraday price movement
    data["Open_Close_Return"] = (
        data["Close"] / data["Open"] - 1
    )

    # Next-day return and target
    data["Next_Day_Return (%)"] = (
        data["Close"].shift(-1)
        / data["Close"] - 1
    ) * 100

    data["Target"] = (
        data["Next_Day_Return (%)"] > 0
    ).astype(int)

    # Remove rows with unavailable features or target
    valid_data = data.dropna(
        subset=ALL_FEATURES + ["Next_Day_Return (%)"]
    ).copy()

    X = valid_data[ALL_FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )

    valid_indices = X.dropna().index

    X = X.loc[valid_indices]
    y = valid_data.loc[valid_indices, "Target"]

    return X, y, valid_data.loc[valid_indices]


# --------------------------------------------------
# 3. CHRONOLOGICAL DATA SPLIT
# --------------------------------------------------

def split_dataset(X, y):

    n = len(X)

    train_end = int(n * 0.64)
    validation_end = int(n * 0.80)

    # Purge boundary observations because each label
    # uses the following trading day's closing price.
    X_train = X.iloc[:train_end - 1]
    y_train = y.iloc[:train_end - 1]

    X_val = X.iloc[train_end:validation_end - 1]
    y_val = y.iloc[train_end:validation_end - 1]

    X_test = X.iloc[validation_end:]
    y_test = y.iloc[validation_end:]

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test
    }


# --------------------------------------------------
# 4. DEFINE MACHINE LEARNING MODELS
# --------------------------------------------------

def get_models():

    models = {

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced"
            ))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=3,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=10,
            random_state=42
        ),

        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=7,
            l2_regularization=1.0,
            random_state=42
        )
    }

    if XGBOOST_AVAILABLE:

        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.03,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )

    return models


# --------------------------------------------------
# 5. TRAIN AND COMPARE MODELS
# --------------------------------------------------

def train_models(X_train, y_train, X_val, X_test):

    models = get_models()

    trained_models = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        val_prob = model.predict_proba(X_val)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]

        trained_models[name] = {
            "model": model,
            "val_prob": val_prob,
            "test_prob": test_prob
        }

    return trained_models
