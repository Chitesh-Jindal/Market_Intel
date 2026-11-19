
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# Existing seven ML features
FEATURE_COLUMNS = [
    "Close_vs_SMA20",
    "SMA_Spread_Relative",
    "Close_vs_EMA20",
    "MACD_Hist_Relative",
    "BB_Percent_B",
    "BB_Width_Relative",
    "Volume_Relative"
]


def prepare_ml_dataset(data):

    data = data.copy()

    # Feature engineering
    data["Volume_SMA20"] = data["Volume"].rolling(window=20).mean()

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
        (data["MACD"] - data["MACD_Signal"]) / data["Close"]
    )

    bb_range = (
        data["BB_Upper"] - data["BB_Lower"]
    ).replace(0, float("nan"))

    data["BB_Percent_B"] = (
        (data["Close"] - data["BB_Lower"]) / bb_range
    )

    data["BB_Width_Relative"] = (
        (data["BB_Upper"] - data["BB_Lower"])
        / data["BB_Middle"]
    )

    data["Volume_Relative"] = (
        data["Volume"]
        / data["Volume_SMA20"].replace(0, float("nan"))
    )

    # Next-day return
    data["Next_Day_Return (%)"] = (
        data["Close"].shift(-1) / data["Close"] - 1
    ) * 100

    # Remove rows without a next-day return
    data = data.dropna(subset=["Next_Day_Return (%)"])

    # Binary classification target
    data["Target"] = (
        data["Next_Day_Return (%)"] > 0
    ).astype(int)

    # Select existing seven features
    ml_data = data[FEATURE_COLUMNS].dropna()

    X = ml_data
    y = data.loc[ml_data.index, "Target"]

    return X, y, data


def train_model(X, y):

    # Chronological split: 64% training, 16% validation, 20% testing
    train_end = int(len(X) * 0.64)
    validation_end = int(len(X) * 0.80)

    X_train = X.iloc[:train_end]
    X_val = X.iloc[train_end:validation_end]
    X_test = X.iloc[validation_end:]

    y_train = y.iloc[:train_end]
    y_val = y.iloc[train_end:validation_end]
    y_test = y.iloc[validation_end:]

    # Feature scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Feature drift diagnostics
    feature_names = X_train.columns

    train_scaled_df = pd.DataFrame(
        X_train_scaled,
        columns=feature_names,
        index=X_train.index
    )

    val_scaled_df = pd.DataFrame(
        X_val_scaled,
        columns=feature_names,
        index=X_val.index
    )

    drift_data = pd.DataFrame({
        "Training Mean": train_scaled_df.mean(),
        "Validation Mean": val_scaled_df.mean(),
        "Validation Min": val_scaled_df.min(),
        "Validation Max": val_scaled_df.max()
    })

    drift_data["Validation Mean Shift"] = (
        drift_data["Validation Mean"]
        - drift_data["Training Mean"]
    )

    drift_data["Absolute Mean Shift"] = (
        drift_data["Validation Mean Shift"].abs()
    )

    drift_data["Outside +/-3 SD"] = (
        (drift_data["Validation Min"] < -3)
        | (drift_data["Validation Max"] > 3)
    )

    drift_data = drift_data.sort_values(
        "Absolute Mean Shift",
        ascending=False
    )

    # Logistic Regression
    model = LogisticRegression(max_iter=1000)

    model.fit(X_train_scaled, y_train)

    # Feature coefficient diagnostics
    coefficient_data = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_[0]
    })

    coefficient_data["Absolute Coefficient"] = (
        coefficient_data["Coefficient"].abs()
    )

    coefficient_data = coefficient_data.sort_values(
        "Absolute Coefficient",
        ascending=False
    )

    # Average feature contributions
    contribution_data = pd.DataFrame({
        "Feature": feature_names,
        "Training Contribution": (
            X_train_scaled.mean(axis=0) * model.coef_[0]
        ),
        "Validation Contribution": (
            X_val_scaled.mean(axis=0) * model.coef_[0]
        )
    })

    contribution_data["Contribution Shift"] = (
        contribution_data["Validation Contribution"]
        - contribution_data["Training Contribution"]
    )

    contribution_data["Absolute Shift"] = (
        contribution_data["Contribution Shift"].abs()
    )

    contribution_data = contribution_data.sort_values(
        "Absolute Shift",
        ascending=False
    )

    # Predicted probabilities
    y_train_prob = model.predict_proba(X_train_scaled)[:, 1]
    y_val_prob = model.predict_proba(X_val_scaled)[:, 1]
    y_test_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Return everything required by the application and evaluation module
    return {
        "model": model,
        "scaler": scaler,

        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,

        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,

        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,

        "y_train_prob": y_train_prob,
        "y_val_prob": y_val_prob,
        "y_test_prob": y_test_prob,

        "drift_data": drift_data,
        "coefficient_data": coefficient_data,
        "contribution_data": contribution_data,

        "model_intercept": model.intercept_[0]
    }

