"""Synthetic smoke test. This validates software integration, not stock accuracy."""
import numpy as np
import pandas as pd

from market_intel.technical_analysis import calculate_technical_indicators
from market_intel.ml_pipeline import build_ml_dataset, train_and_evaluate, walk_forward_evaluate
from market_intel.evaluation import evaluate_test, build_feature_importance
from market_intel.backtesting import run_backtest
from threadpoolctl import threadpool_limits


def make_synthetic_ohlcv(rows=1250, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=rows)
    daily_returns = np.zeros(rows)
    noise = rng.normal(0.0, 0.012, rows)
    for i in range(1, rows):
        daily_returns[i] = 0.10 * daily_returns[i - 1] + noise[i]
    close = 100.0 * np.exp(np.cumsum(daily_returns))
    open_price = close * np.exp(rng.normal(0.0, 0.004, rows))
    high = np.maximum(open_price, close) * (1 + rng.uniform(0.001, 0.012, rows))
    low = np.minimum(open_price, close) * (1 - rng.uniform(0.001, 0.012, rows))
    volume = rng.integers(100_000, 1_000_000, rows)
    return pd.DataFrame({
        "Open": open_price, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    }, index=dates)


def main():
    with threadpool_limits(limits=1):
        _run_checks()


def _run_checks():
    raw = make_synthetic_ohlcv()
    indicators = calculate_technical_indicators(raw)
    X, y, labelled, latest_features = build_ml_dataset(indicators)

    assert not X.isna().any().any()
    assert np.isfinite(X.to_numpy()).all()
    assert len(latest_features) == 1

    modelling = train_and_evaluate(X, y)
    metrics, baseline, confusion, predictions = evaluate_test(
        modelling["y_test"], modelling["test_probabilities"],
        modelling["threshold"], modelling["y_train"],
    )
    importance = build_feature_importance(
        modelling["model"], modelling["X_test"], modelling["y_test"],
        modelling["selected_features"],
    )
    test_rows = labelled.loc[modelling["X_test"].index]
    equity, backtest_metrics, counts = run_backtest(
        test_rows, modelling["test_probabilities"], buy_threshold=0.60
    )
    walk_forward = walk_forward_evaluate(X, y, folds=4)

    assert len(predictions) == len(modelling["y_test"])
    assert confusion.shape == (2, 2)
    assert len(importance) > 0
    assert set(equity.columns) == {"ML Signal", "Buy & Hold", "Momentum"}
    assert len(backtest_metrics) == 3
    assert sum(counts.values()) == len(test_rows)
    assert walk_forward["summary"]["Folds"] == 4
    assert len(modelling["selected_features"]) in (12, 30)

    print("PASS: features, 8-way validation selection, holdout, walk-forward, importance and backtest")
    print(f"Rows: {len(X)} | Selected model: {modelling['model_name']} | Feature set: {modelling['feature_set_name']}")
    print(f"Holdout accuracy: {metrics['Accuracy']:.3f} | Baseline: {baseline['Baseline Accuracy']:.3f}")
    print(f"Walk-forward accuracy: {walk_forward['summary']['Mean Test Accuracy']:.3f} | "
          f"Baseline: {walk_forward['summary']['Mean Baseline Accuracy']:.3f}")
    print("Synthetic metrics are software checks only, not evidence of market predictive performance.")


if __name__ == "__main__":
    main()
