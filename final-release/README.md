# Market Intel — Validated 1-Day Edition

Market Intel is a Streamlit research application for Indian equities. It analyses historical OHLCV data, calculates technical features, compares classification models, estimates next-trading-session direction, evaluates performance against chronological baselines, and backtests a cautious long-only signal.

## Run locally (macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Select an NSE ticker, choose 5 years of history, and click **Run Market Intel**.

## Project files

- `app.py` — Streamlit dashboard and five application tabs
- `data_loader.py` — Yahoo Finance daily OHLCV downloader
- `technical_analysis.py` — pandas/numpy technical indicators
- `ml_pipeline.py` — feature engineering, model/feature-set comparison, one-day prediction and four-window walk-forward validation
- `evaluation.py` — holdout metrics, baseline and permutation feature importance
- `backtesting.py` — next-session-open-to-close ML, Buy & Hold and Momentum backtests
- `validate_project.py` — synthetic software integration test (not an accuracy test)
- `benchmark_validation.py` — reproduce the five-stock benchmark from a local `benchmark_data.zip`
- `benchmark_holdout_results.csv` — holdout results from the supplied benchmark
- `benchmark_walk_forward_results.csv` — four walk-forward folds for each stock
- `benchmark_backtest_results.csv` — backtest results for each stock
- `AUDIT_RESULTS.md` — detailed experimental results and limitations
- `LAB_EVALUATION_GUIDE.md` — project explanation and viva preparation

## Prediction target

For each completed session t, features use information available through close(t). The target is:

`Close(t+1) / Close(t) - 1`

Class 1 means the next session's close-to-close return is positive. Class 0 means it is zero or negative. The latest completed feature row has no known outcome yet and is used only to generate the current estimate.

## Modelling workflow

1. Calculate lagged returns, rolling volatility, moving-average distances, RSI, MACD, Bollinger Bands, ADX, ATR, Stochastic, CCI, OBV and volume features.
2. Compare four classifiers across a compact 12-feature set and a full 30-feature set:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - HistGradientBoosting
3. Fit each candidate on chronological training data, select the model and threshold on a later validation block using Macro F1, and never use the holdout test block for selection.
4. Refit the selected configuration on training + validation and evaluate on the final chronological holdout.
5. Refit the selected configuration on all known labelled observations for the latest live estimate.
6. Run four expanding-window walk-forward periods. Each fold uses a separate later validation block for model/threshold selection and a subsequent test block.

The fixed model configurations replace the earlier broad exhaustive grid search. This lowers runtime and avoids giving a large tuning sweep the appearance of creating predictive signal where the multi-stock evidence does not support it.

## Prediction and action labels

The app displays the predicted direction and estimated probability. BUY/SELL labels are enabled only when the single holdout beats its majority baseline, the mean walk-forward accuracy exceeds its mean baseline, at least 3 of 4 walk-forward windows beat their own baselines, and mean balanced accuracy and ROC-AUC are each at least 0.52. Otherwise, the application displays `HOLD — INSUFFICIENT HISTORICAL EDGE`.

These are conservative application guardrails, not formal statistical significance thresholds or calibrated guarantees. A probability is not a promise of a future return.

## Backtesting

The signal is formed after close(t), so the strategy enters at the next session's open and exits at that session's close. This is an executable next-session intraday return segment; it is not the same as the close-to-close prediction target. ML and Momentum one-session trades deduct 0.10% on entry and exit. Buy & Hold deducts entry and final liquidation costs. Win rate counts invested sessions, and exposure is reported separately.

## Checks

Run:

```bash
python validate_project.py
```

This exercises feature engineering, model selection, the chronological holdout, walk-forward evaluation, feature importance and backtesting with deterministic synthetic data. It validates code integration only; it does not establish financial predictive performance.

## Important limitations

The supplied five-stock benchmark found little average advantage over majority-class baselines and near-chance average discrimination. See `AUDIT_RESULTS.md`. Daily market returns are noisy and market relationships change over time. This is an educational research prototype, not financial advice.
