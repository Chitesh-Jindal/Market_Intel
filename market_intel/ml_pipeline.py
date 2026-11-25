"""Leakage-aware one-day stock-direction modelling and rolling validation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

FEATURES = [
    "Return_1D", "Return_2D", "Return_5D", "Return_10D", "Return_20D",
    "Volatility_5D", "Volatility_10D", "Volatility_20D",
    "Close_vs_SMA20", "Close_vs_SMA50", "Close_vs_EMA20",
    "SMA_Spread_Relative", "EMA_Spread_Relative",
    "RSI", "MACD_Hist_Relative", "BB_Percent_B", "BB_Width",
    "ADX", "ATR_Relative", "Stochastic", "Stochastic_Signal",
    "CCI", "OBV_Change", "Volume_Relative", "Volume_Change",
    "Volume_ZScore", "Daily_Range", "Open_Close_Return",
    "Distance_From_20D_High", "Distance_From_20D_Low",
]

# A compact ablation is kept as a candidate, not assumed to be superior.
COMPACT_FEATURES = [
    "Return_1D", "Return_5D", "Return_20D", "Volatility_20D",
    "Close_vs_SMA20", "Close_vs_SMA50", "RSI", "MACD_Hist_Relative",
    "BB_Percent_B", "ADX", "ATR_Relative", "Volume_Relative",
]

FEATURE_SETS = {
    "Compact 12": COMPACT_FEATURES,
    "Full 30": FEATURES,
}


def build_ml_dataset(indicators: pd.DataFrame):
    """Features use data through close(t); target is close(t) to close(t+1)."""
    df = indicators.copy()
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)
    daily_return = close.pct_change()

    for days in [1, 2, 5, 10, 20]:
        df[f"Return_{days}D"] = close.pct_change(days)
    for days in [5, 10, 20]:
        df[f"Volatility_{days}D"] = daily_return.rolling(days).std()

    df["Close_vs_SMA20"] = close / df["SMA_20"] - 1
    df["Close_vs_SMA50"] = close / df["SMA_50"] - 1
    df["Close_vs_EMA20"] = close / df["EMA_20"] - 1
    df["SMA_Spread_Relative"] = df["SMA_Spread"] / close
    df["EMA_Spread_Relative"] = (df["EMA_20"] - df["EMA_50"]) / close
    df["MACD_Hist_Relative"] = df["MACD_Hist"] / close
    df["ATR_Relative"] = df["ATR"] / close

    volume_mean = volume.rolling(20).mean()
    volume_std = volume.rolling(20).std().replace(0, np.nan)
    df["Volume_Relative"] = volume / volume_mean.replace(0, np.nan)
    df["Volume_Change"] = volume.pct_change()
    df["Volume_ZScore"] = (volume - volume_mean) / volume_std
    df["Daily_Range"] = (df["High"] - df["Low"]) / close
    df["Open_Close_Return"] = close / df["Open"] - 1

    high20 = df["High"].rolling(20).max()
    low20 = df["Low"].rolling(20).min()
    df["Distance_From_20D_High"] = close / high20 - 1
    df["Distance_From_20D_Low"] = close / low20 - 1

    # Forward columns are backtesting labels only; never included in FEATURES.
    df["Next_Day_Return"] = close.shift(-1) / close - 1
    df["Next_Session_Open"] = df["Open"].shift(-1)
    df["Next_Session_Close"] = df["Close"].shift(-1)
    df["Target"] = (df["Next_Day_Return"] > 0).astype(int)

    feature_frame = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    usable_feature_rows = feature_frame.dropna()
    if usable_feature_rows.empty:
        raise ValueError("Not enough data to calculate all technical features.")
    latest_features = usable_feature_rows.iloc[[-1]].copy()

    valid = (
        feature_frame.notna().all(axis=1)
        & df["Next_Day_Return"].notna()
        & df["Next_Session_Open"].notna()
        & df["Next_Session_Close"].notna()
        & (df["Next_Session_Open"] > 0)
        & (df["Next_Session_Close"] > 0)
    )
    X = feature_frame.loc[valid].astype(float)
    y = df.loc[valid, "Target"].astype(int)
    labelled = df.loc[valid].copy()

    if len(X) < 250:
        raise ValueError(
            f"Only {len(X)} usable labelled rows remain. Select at least 3–5 years of data."
        )
    return X, y, labelled, latest_features


def _candidate_models():
    """Fast, fixed, conservative configurations; model choice happens on validation."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=0.5, max_iter=2000, class_weight="balanced",
                solver="lbfgs", random_state=42
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=4, min_samples_leaf=12,
            max_features="sqrt", class_weight="balanced_subsample",
            random_state=42, n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=90, learning_rate=0.03, max_depth=1,
            min_samples_leaf=12, random_state=42,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.03, max_leaf_nodes=7,
            min_samples_leaf=20, l2_regularization=2.0,
            early_stopping=False, random_state=42,
        ),
    }


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    gap: int = 1,
):
    """Chronological 60/20/20 split with a one-row purge at both boundaries."""
    n = len(X)
    train_end = int(n * train_fraction)
    validation_end = int(n * (train_fraction + validation_fraction))

    if train_end < 150 or validation_end - train_end < 30 or n - validation_end < 30:
        raise ValueError("Not enough data for train/validation/test. Select 5 years of data.")

    split = {
        "X_train": X.iloc[:train_end-gap],
        "y_train": y.iloc[:train_end-gap],
        "X_val": X.iloc[train_end:validation_end-gap],
        "y_val": y.iloc[train_end:validation_end-gap],
        "X_test": X.iloc[validation_end:],
        "y_test": y.iloc[validation_end:],
    }
    for name in ["y_train", "y_val", "y_test"]:
        if split[name].nunique() < 2:
            raise ValueError(f"The {name[2:]} period contains only one class.")
    return split


def choose_threshold(y_true, probabilities, objective: str = "macro_f1"):
    """Tune a threshold on validation data only."""
    if objective not in {"macro_f1", "accuracy"}:
        raise ValueError("objective must be 'macro_f1' or 'accuracy'")
    best_threshold, best_score = 0.50, -1.0

    for threshold in np.arange(0.40, 0.601, 0.025):
        prediction = (np.asarray(probabilities) >= threshold).astype(int)
        if objective == "accuracy":
            score = accuracy_score(y_true, prediction)
        else:
            score = f1_score(y_true, prediction, average="macro", zero_division=0)

        if score > best_score + 1e-12 or (
            np.isclose(score, best_score)
            and abs(threshold - 0.50) < abs(best_threshold - 0.50)
        ):
            best_threshold, best_score = float(threshold), float(score)

    return best_threshold, float(best_score)


def _select_on_validation(X_train, y_train, X_val, y_val):
    """Compare 4 models × 2 feature sets; never inspect test data."""
    rows = []
    candidates = {}

    for feature_set_name, feature_columns in FEATURE_SETS.items():
        for model_name, estimator in _candidate_models().items():
            model = clone(estimator)
            model.fit(X_train[feature_columns], y_train)
            val_prob = model.predict_proba(X_val[feature_columns])[:, 1]
            threshold, val_f1 = choose_threshold(y_val, val_prob, "macro_f1")
            val_pred = (val_prob >= threshold).astype(int)

            key = (feature_set_name, model_name)
            candidates[key] = {
                "estimator": model,
                "features": feature_columns,
                "feature_set_name": feature_set_name,
                "model_name": model_name,
                "threshold": threshold,
                "validation_macro_f1": val_f1,
            }
            rows.append({
                "Feature Set": feature_set_name,
                "Model": model_name,
                "Validation Macro F1": val_f1,
                "Validation Accuracy": accuracy_score(y_val, val_pred),
                "Threshold": threshold,
            })

    # Stable ordering preserves deterministic tie-breaking by the declared
    # feature-set/model order; validation accuracy is reported, not used as
    # a second selection target.
    table = pd.DataFrame(rows).sort_values(
        "Validation Macro F1", ascending=False, kind="stable"
    ).reset_index(drop=True)

    best_row = table.iloc[0]
    best_key = (best_row["Feature Set"], best_row["Model"])
    return candidates[best_key], table


def train_and_evaluate(X: pd.DataFrame, y: pd.Series):
    """Select on validation, test once, then refit selected model for current signal."""
    split = chronological_split(X, y)
    selected, validation_table = _select_on_validation(
        split["X_train"], split["y_train"], split["X_val"], split["y_val"]
    )

    selected_features = list(selected["features"])
    threshold = selected["threshold"]

    X_train_val = pd.concat([
        split["X_train"][selected_features],
        split["X_val"][selected_features],
    ])
    y_train_val = pd.concat([split["y_train"], split["y_val"]])

    final_model = clone(selected["estimator"])
    final_model.fit(X_train_val, y_train_val)
    test_probabilities = final_model.predict_proba(
        split["X_test"][selected_features]
    )[:, 1]

    latest_model = clone(selected["estimator"])
    latest_model.fit(X[selected_features], y)

    return {
        "model": final_model,
        "model_name": selected["model_name"],
        "feature_set_name": selected["feature_set_name"],
        "selected_features": selected_features,
        "threshold": threshold,
        "validation_results": validation_table,
        "X_test": split["X_test"][selected_features],
        "y_test": split["y_test"],
        "test_probabilities": test_probabilities,
        "y_train": y_train_val,
        "latest_model": latest_model,
        "split": split,
    }


def walk_forward_evaluate(X: pd.DataFrame, y: pd.Series, folds: int = 4):
    """
    Repeated expanding-window evaluation.

    Each fold uses an earlier training window, a later validation window for
    model/threshold selection, and a subsequent untouched test window.
    """
    if folds != 4:
        raise ValueError("This release uses four fixed expanding-window folds.")

    n = len(X)
    fold_rows = []

    for fold in range(folds):
        train_end = int(n * (0.50 + 0.10 * fold))
        validation_end = int(n * (0.60 + 0.10 * fold))
        test_end = int(n * (0.70 + 0.10 * fold))

        X_train, y_train = X.iloc[:train_end-1], y.iloc[:train_end-1]
        X_val, y_val = X.iloc[train_end:validation_end-1], y.iloc[train_end:validation_end-1]
        X_test, y_test = X.iloc[validation_end:test_end], y.iloc[validation_end:test_end]

        if min(len(X_train), len(X_val), len(X_test)) < 30:
            continue
        if min(y_train.nunique(), y_val.nunique(), y_test.nunique()) < 2:
            continue

        selected, _ = _select_on_validation(X_train, y_train, X_val, y_val)
        feature_columns = list(selected["features"])
        model = clone(selected["estimator"])
        X_fit = pd.concat([X_train[feature_columns], X_val[feature_columns]])
        y_fit = pd.concat([y_train, y_val])
        model.fit(X_fit, y_fit)

        probabilities = model.predict_proba(X_test[feature_columns])[:, 1]
        threshold = selected["threshold"]
        predictions = (probabilities >= threshold).astype(int)

        majority_class = int(y_fit.mode().iloc[0])
        baseline_predictions = np.full(len(y_test), majority_class, dtype=int)
        accuracy = accuracy_score(y_test, predictions)
        baseline_accuracy = accuracy_score(y_test, baseline_predictions)

        fold_rows.append({
            "Fold": fold + 1,
            "Test Start": X_test.index.min().strftime("%Y-%m-%d"),
            "Test End": X_test.index.max().strftime("%Y-%m-%d"),
            "Training Rows": len(X_train),
            "Validation Rows": len(X_val),
            "Test Rows": len(X_test),
            "Selected Model": selected["model_name"],
            "Feature Set": selected["feature_set_name"],
            "Threshold": threshold,
            "Validation Macro F1": selected["validation_macro_f1"],
            "Test Accuracy": accuracy,
            "Baseline Accuracy": baseline_accuracy,
            "Accuracy Edge (pp)": (accuracy - baseline_accuracy) * 100,
            "Macro F1": f1_score(y_test, predictions, average="macro", zero_division=0),
            "Balanced Accuracy": balanced_accuracy_score(y_test, predictions),
            "ROC-AUC": roc_auc_score(y_test, probabilities),
        })

    if not fold_rows:
        raise ValueError("Could not form four valid walk-forward folds for this dataset.")

    folds_df = pd.DataFrame(fold_rows)
    summary = {
        "Folds": int(len(folds_df)),
        "Mean Test Accuracy": float(folds_df["Test Accuracy"].mean()),
        "Mean Baseline Accuracy": float(folds_df["Baseline Accuracy"].mean()),
        "Mean Accuracy Edge (pp)": float(folds_df["Accuracy Edge (pp)"].mean()),
        "Mean Macro F1": float(folds_df["Macro F1"].mean()),
        "Mean Balanced Accuracy": float(folds_df["Balanced Accuracy"].mean()),
        "Mean ROC-AUC": float(folds_df["ROC-AUC"].mean()),
        "Folds Beating Baseline": int((folds_df["Accuracy Edge (pp)"] > 0).sum()),
    }
    return {"fold_results": folds_df, "summary": summary}
