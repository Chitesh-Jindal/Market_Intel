# Market Intel

**AI-Powered Indian Stock Intelligence**

Market Intel is a Streamlit research application for analysing Indian equities and exploring whether historical price, momentum, volatility and volume data can help classify the direction of the next trading session.

It combines technical analysis, supervised machine learning, chronological model evaluation and transaction-cost-aware backtesting in one application.

> **Research note:** The supplied five-stock benchmark did not establish a reliable, stock-independent predictive edge. Market Intel reports evaluation results transparently and defaults to **HOLD — insufficient historical edge** when historical reliability checks are not met. This is an educational research project, not financial advice.

## Features

### Market dashboard
- Daily OHLCV data for selected NSE-listed stocks
- Closing-price chart with SMA 20 and SMA 50
- RSI, MACD, Bollinger Bands, ADX, ATR, Stochastic and CCI

### One-day direction estimate
- Predicts whether the next trading session's close-to-close return will be positive or non-positive
- Displays predicted direction and estimated probability
- Displays BUY / SELL-AVOID only when historical reliability checks pass; otherwise displays HOLD

### Machine-learning comparison
Four classifiers are compared:
- Logistic Regression
- Random Forest
- Gradient Boosting
- HistGradientBoosting

The pipeline compares a compact 12-feature set with a full 30-feature set. Candidate models and classification thresholds are selected using validation data, not the final test period.

### Evaluation and explainability
- Accuracy and majority-class baseline
- Macro F1, balanced accuracy, ROC-AUC and log loss
- Confusion matrix
- Four expanding-window walk-forward periods
- Validation model comparison
- Permutation feature importance

### Backtesting
Compares ML Signal, Buy & Hold and Momentum. Signals are generated after a session closes and entered at the next session's open. A trade exits at that session's close, with transaction costs deducted.

## Repository structure

```text
Market_Intel/
├── app.py                         # Streamlit application
├── data_loader.py                 # Yahoo Finance OHLCV download
├── technical_analysis.py          # Technical indicator calculations
├── ml_pipeline.py                 # Features, models and validation
├── evaluation.py                  # Metrics, baseline and importance
├── backtesting.py                 # Strategy backtesting
├── requirements.txt               # Python dependencies
├── runtime.txt                    # Deployment Python runtime
├── VERSION.txt                    # Release information
├── validate_project.py             # Synthetic integration smoke test
├── README.md
├── docs/
│   ├── AUDIT_RESULTS.md             # Five-stock benchmark and limitations
│   └── LAB_EVALUATION_GUIDE.md      # Methodology and viva preparation
├── results/
│   ├── benchmark_holdout_results.csv
│   ├── benchmark_walk_forward_results.csv
│   └── benchmark_backtest_results.csv
├── scripts/
│   └── benchmark_validation.py     # Reproduce the benchmark from local CSV ZIP
└── archive/
    └── previous-version/            # Earlier implementation, retained for reference
```

## Installation

Python 3.11 is recommended.

### macOS / Linux

```bash
git clone https://github.com/Chitesh-Jindal/Market_Intel.git
cd Market_Intel

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Run the application

```bash
python -m streamlit run app.py
```

Choose an NSE ticker, select a historical period (five years is recommended), and click **Run Market Intel**.

### Run the software integration check

```bash
python validate_project.py
```

The check uses deterministic synthetic OHLCV data to exercise feature engineering, model selection, holdout evaluation, walk-forward validation, feature importance and backtesting. It validates software integration only; synthetic accuracy is not evidence of stock-market predictive performance.

## Methodology

### Prediction target

For a completed session (t), the target is the sign of the next close-to-close return:

```text
Next_Day_Return(t) = Close(t+1) / Close(t) - 1

Target(t) = 1, if Next_Day_Return(t) > 0
            0, otherwise
```

Every model feature is calculated using information available through the close of session (t). Forward prices are excluded from the feature matrix.

### Feature engineering

Features cover:
- Lagged returns and rolling volatility
- Distance from SMA / EMA trend levels
- RSI and MACD
- Bollinger Band position and width
- ADX and ATR
- Stochastic Oscillator and CCI
- OBV and volume features
- Daily range and open-to-close movement

### Model selection and evaluation

- Chronological train / validation / test split
- One-observation purge at split boundaries because labels use the following session
- Validation Macro F1 used for candidate model and threshold selection
- Final test set kept separate from model selection
- Four expanding-window walk-forward test periods
- Training-majority baseline recalculated for every evaluation period

The app enables directional BUY / SELL-AVOID labels only when the single holdout and repeated walk-forward checks meet configured reliability guardrails. These are conservative display rules, not a formal statistical significance test or a guarantee of future performance.

### Backtest execution assumptions

Features are known after close (t), so a simulated trade cannot be entered at that already-observed close. The backtest enters at the next session's open and exits at that session's close. It reports total return, maximum drawdown, Sharpe ratio, win rate and exposure, with transaction costs.

The classification target is close-to-close, while the executable backtest segment is next-open-to-close. These are related but not identical measurements.

## Benchmark summary

The included audit uses five supplied NSE datasets (Reliance, TCS, HDFC Bank, Infosys and ICICI Bank), each spanning 25 November 2021 to 25 November 2026.

Across 20 walk-forward test windows:
- Mean test accuracy: **50.92%**
- Mean majority-baseline accuracy: **50.67%**
- Mean Macro F1: **0.452**
- Mean balanced accuracy: **0.506**
- Mean ROC-AUC: **0.506**
- Windows beating their own baseline: **10 / 20**

The evidence does **not** support a claim of reliable daily predictive skill or strategy profitability. See [docs/AUDIT_RESULTS.md](docs/AUDIT_RESULTS.md) and the CSV files in [results/](results/).

## Limitations

- Daily stock returns are noisy and market relationships change over time.
- A small average difference over a baseline may not be statistically meaningful.
- Model probabilities are not necessarily calibrated.
- Backtests are sensitive to execution, transaction-cost and data assumptions.
- Historical performance does not guarantee future results.
- Yahoo Finance availability and ticker coverage can change.

## Documentation

- [Five-stock audit and experimental results](docs/AUDIT_RESULTS.md)
- [Lab evaluation and viva guide](docs/LAB_EVALUATION_GUIDE.md)

## License

See [LICENSE](LICENSE).
