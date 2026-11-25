# Market Intel — Five-Stock Validation Audit

## Executive conclusion

The latest supplied `benchmark_data.zip` was tested through the project's feature-engineering, model-selection, evaluation, and backtesting code. The benchmark contains five NSE tickers, each with 1,236 daily OHLCV rows from **2021-11-25 through 2026-11-25**. No OHLCV fields were missing and no duplicate dates were found.

**The one-day classifier does not demonstrate a reliable, stock-independent predictive edge.** Across 20 expanding-window test periods (four per stock), mean accuracy was **50.92%**, compared with a mean training-majority baseline of **50.67%**. Only **10 of 20** test periods beat their own baseline. Mean Macro F1 was **0.452**, balanced accuracy **0.506**, and ROC-AUC **0.506**. These are close to chance-level results.

The final release therefore focuses on a fast, transparent pipeline and a conservative HOLD gate. It does not claim that the model is profitable or reliably predictive.

## Data and test design

- Source: user-supplied `benchmark_data.zip` containing Yahoo Finance-style daily OHLCV CSVs.
- Tickers: `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`, `INFY.NS`, `ICICIBANK.NS`.
- Raw data per ticker: 1,236 rows; dates 2021-11-25 to 2026-11-25.
- Missing OHLCV values: zero for all five files.
- Duplicate dates: zero for all five files.
- Usable labelled rows after indicator warm-up and removing the final row without a known next-session label: 1,184 per stock.
- Target: `Close(t+1) / Close(t) - 1 > 0`.
- Features use only information available through close(t). Forward next-session OHLC fields are retained for backtesting and never included as model features.

### Single chronological holdout

The single holdout uses 60% training, 20% validation, and 20% test, with one observation purged at each split boundary. The test window for all five stocks was **2025-12-05 through 2026-11-23**.

| Stock | Selected model | Feature set | Holdout accuracy | Baseline accuracy | Difference | Macro F1 | Balanced accuracy | ROC-AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|
| HDFC Bank | HistGradientBoosting | Full 30 | 52.74% | 42.19% | +10.55 pp | 0.491 | 0.498 | 0.452 |
| ICICI Bank | HistGradientBoosting | Compact 12 | 53.59% | 50.21% | +3.38 pp | 0.527 | 0.535 | 0.529 |
| Infosys | Logistic Regression | Compact 12 | 48.10% | 43.46% | +4.64 pp | 0.481 | 0.492 | 0.483 |
| Reliance | Gradient Boosting | Compact 12 | 54.43% | 49.37% | +5.06 pp | 0.541 | 0.546 | 0.539 |
| TCS | HistGradientBoosting | Compact 12 | 54.43% | 54.43% | 0.00 pp | 0.469 | 0.515 | 0.521 |

A few single holdouts show accuracy above baseline, but HDFC and Infosys have ROC-AUC below 0.50, and TCS only ties its baseline. These single windows are not sufficient evidence of robust skill.

### Four expanding-window walk-forward periods

For each ticker, model and feature-set selection is done on an earlier validation block. The selected model is refit on train + validation and evaluated on the following untouched test block. The four test blocks are:

1. 2024-12-20 to 2025-06-13
2. 2025-06-16 to 2025-12-04
3. 2025-12-05 to 2026-06-02
4. 2026-06-03 to 2026-11-23

Each test block contains 118 or 119 observations. The majority baseline is recalculated from that fold's training + validation data.

| Stock | Mean test accuracy | Mean baseline | Difference | Macro F1 | Balanced accuracy | ROC-AUC | Folds beating baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| HDFC Bank | 49.36% | 47.03% | +2.32 pp | 0.430 | 0.493 | 0.470 | 2/4 |
| ICICI Bank | 52.34% | 50.21% | +2.13 pp | 0.484 | 0.528 | 0.502 | 2/4 |
| Infosys | 53.16% | 49.36% | +3.80 pp | 0.454 | 0.514 | 0.519 | 3/4 |
| Reliance | 51.26% | 50.42% | +0.84 pp | 0.477 | 0.529 | 0.541 | 2/4 |
| TCS | 48.50% | 56.33% | -7.83 pp | 0.416 | 0.467 | 0.499 | 1/4 |

### Aggregate across all 20 ticker-fold test windows

| Metric | Result |
|---|---:|
| Mean test accuracy | 50.92% |
| Mean majority-baseline accuracy | 50.67% |
| Mean accuracy difference | +0.25 percentage points |
| Mean Macro F1 | 0.452 |
| Mean balanced accuracy | 0.506 |
| Mean ROC-AUC | 0.506 |
| Test windows beating baseline | 10 / 20 |

The small average accuracy difference and near-0.50 discrimination metrics indicate that the model's predictive edge is not established.

## Experiments that did not provide a dependable improvement

These experiments used validation-only model/threshold selection and then evaluated on chronological test windows.

| Experiment | Mean accuracy | Mean baseline | Mean difference | Folds above baseline |
|---|---:|---:|---:|---:|
| Macro-F1 selection, 4 models × compact/full feature sets | 50.92% | 50.67% | +0.25 pp | 10/20 |
| Accuracy-based threshold/model selection | 50.21% | 50.67% | -0.46 pp | 7/20 |
| Four-model probability ensemble, full features | 50.00% | 50.67% | -0.68 pp | 12/20 |
| Four-model probability ensemble, compact features | 50.00% | 50.67% | -0.68 pp | 11/20 |
| Full 30-feature set only, broader six-configuration comparison | 51.09% | 50.67% | +0.42 pp | 11/20 |
| Compact 12-feature set only, same six-configuration comparison | 50.80% | 50.67% | +0.13 pp | 10/20 |

The compact feature set did not consistently improve results. Accuracy-driven selection and probability averaging also failed to produce a robust gain. The final app keeps both feature sets as validation candidates rather than declaring either universally superior.

## Backtesting on the single holdout

Signals are formed after close(t), entered at the next session's open, and exited at that session's close. ML and Momentum charge 0.10% on entry and exit for every one-session trade. Buy & Hold charges entry and final liquidation costs.

| Stock | ML strategy return | Buy & Hold return | Momentum return |
|---|---:|---:|---:|
| HDFC Bank | -9.11% | -14.15% | -14.26% |
| ICICI Bank | -1.37% | +19.00% | -12.10% |
| Infosys | -27.42% | -27.12% | -34.78% |
| Reliance | -2.98% | -10.51% | -20.82% |
| TCS | -7.77% | -19.19% | -32.25% |

The ML strategy had negative returns on all five single-holdout periods. It did not beat Buy & Hold for ICICI Bank or Infosys. Better drawdown or lower loss in some samples is not evidence of a profitable strategy.

## Final engineering decisions

1. Replace the previous broad GridSearchCV with four fixed, conservative model configurations and compare them over two feature sets. This reduces repeated fitting and avoids suggesting that a larger hyperparameter sweep improves predictive power.
2. Keep validation Macro F1 as the primary model-selection metric; accuracy-driven selection was worse in the multi-window experiment.
3. Add a four-window expanding walk-forward report to the app.
4. Keep accuracy, baseline, Macro F1, balanced accuracy, ROC-AUC, confusion matrix and fold-level results visible.
5. Keep the live direction/probability visible, but require a positive holdout edge, a positive walk-forward edge, at least 3/4 folds above baseline, and mean balanced accuracy and ROC-AUC of at least 0.52 before displaying BUY/SELL. These are conservative UI guardrails, not formal statistical significance thresholds. Given the supplied benchmark, all five tickers remain HOLD / insufficient historical edge.
6. Keep the next-open-to-close executable backtest and explicitly distinguish it from the close-to-close prediction target.

## Verification limitations

- The final Python modelling, walk-forward, evaluation, feature-importance and backtesting modules were executed against the five supplied OHLCV datasets.
- `validate_project.py` passed on deterministic synthetic data. This is an integration test, not a financial-performance test.
- Python source files were syntax-compiled.
- This environment did not have Streamlit or yfinance installed and could not reach package repositories. Therefore, the interactive Streamlit browser UI and a fresh Yahoo Finance download were not launched here. The real-data results above were produced by running the same project modelling modules directly against the user's uploaded CSV files.

## Decision

Freeze the one-day project. Do not add more models, indicators, new horizons, or repeated test-set tuning. Market Intel is a complete educational research platform, but the supplied five-stock evidence does not support claiming a reliable daily prediction advantage or profitability.
