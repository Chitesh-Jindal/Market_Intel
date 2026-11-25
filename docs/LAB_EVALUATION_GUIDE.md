# Market Intel — Lab Evaluation Guide

## Project title

**Market Intel — AI-Powered Indian Stock Intelligence**

## 1. Project introduction

Market Intel is a machine-learning-based stock analysis research platform for Indian equities. It retrieves historical OHLCV data, calculates technical indicators, engineers features, compares classification algorithms, predicts the direction of the next trading session, evaluates the model against chronological baselines, and tests a simple strategy using historical prices.

## 2. Problem statement

Daily stock returns are noisy and difficult to predict. The project investigates whether historical price, momentum, volatility and volume features contain useful information about the direction of the next trading session's close-to-close return.

The application does not forecast an exact future stock price. It performs binary classification:

- **Positive (1):** next-session close-to-close return is greater than zero.
- **Non-positive (0):** next-session return is zero or negative.

## 3. Data used

The application downloads daily Open, High, Low, Close and Volume data for NSE-listed stocks using Yahoo Finance. The user can select a ticker and historical period.

OHLCV means:
- Open: first traded price in the session
- High: highest price in the session
- Low: lowest price in the session
- Close: final price in the session
- Volume: quantity of shares traded

## 4. Feature engineering

The model uses returns, rolling volatility, price distance from moving averages, RSI, MACD, Bollinger Band position and width, ADX, ATR, Stochastic Oscillator, CCI, OBV and volume features.

Feature engineering converts raw price/volume history into numeric inputs that describe momentum, trend, volatility and trading activity.

The application compares a compact 12-feature set with a full 30-feature set. It does not assume that adding more indicators always improves a model.

## 5. Machine-learning algorithms

Four classification algorithms are compared:

1. **Logistic Regression:** estimates the probability of a positive class using a linear decision boundary. Standardisation is included in a pipeline.
2. **Random Forest:** combines multiple decision trees to reduce reliance on one tree.
3. **Gradient Boosting:** adds shallow trees sequentially to improve the ensemble.
4. **HistGradientBoosting:** a histogram-based implementation of gradient boosting.

The selected algorithm and feature set are chosen from validation data using Macro F1. The probability threshold is also selected using validation data. The test set is kept separate from this selection.

## 6. Why chronological splitting?

Random splitting mixes older and newer observations. For time-series prediction, this can make evaluation unrealistic.

Market Intel uses:
- Chronological training data
- A later validation period
- A final unseen test period
- A one-observation purge at split boundaries because each label uses the next trading session's close

The app also performs four expanding-window walk-forward tests. Each fold uses a later validation block to select a model and threshold, then evaluates on a subsequent test block.

## 7. Evaluation metrics

- **Accuracy:** fraction of predictions that are correct.
- **Majority-class baseline:** accuracy from always predicting the most common class in the training period.
- **Macro F1:** F1 score calculated for each class and averaged equally.
- **Balanced accuracy:** average recall across both classes.
- **ROC-AUC:** measures how well predicted probabilities rank positive cases above negative cases.
- **Confusion matrix:** counts true positives, true negatives, false positives and false negatives.

A model should not be judged by accuracy alone. It is important to compare it with the baseline and inspect class-sensitive metrics.

## 8. Backtesting

The model signal is generated after close(t). The backtest therefore enters at the next trading session's open and exits at that session's close. This avoids assuming that the strategy can trade at a closing price after that close has already been observed.

The strategy is compared with Buy & Hold and Momentum. Transaction costs are included, and the application reports total return, maximum drawdown, Sharpe ratio, win rate and exposure.

The backtest return segment is next-open-to-close, while the classifier's target is close-to-close. They are related but not identical.

## 9. What the five-stock benchmark showed

The supplied benchmark contains five stocks—Reliance, TCS, HDFC Bank, Infosys and ICICI Bank—with daily data from 25 November 2021 to 25 November 2026.

Across 20 walk-forward test windows:
- Mean test accuracy: 50.92%
- Mean majority baseline accuracy: 50.67%
- Mean Macro F1: 0.452
- Mean balanced accuracy: 0.506
- Mean ROC-AUC: 0.506
- Folds beating their own baseline: 10 out of 20

This indicates weak and inconsistent predictive performance. The project should not claim that it reliably predicts daily stock direction or produces profitable trading signals. Its value is the end-to-end research pipeline and the transparent evaluation of model limitations.

## 10. Common viva questions

**Q: Why did you choose Logistic Regression?**  
A: It is an interpretable classification baseline and provides probability estimates. It also helps compare linear behaviour with tree-based ensembles.

**Q: Why use Random Forest and Gradient Boosting?**  
A: They can model nonlinear relationships and interactions between technical features.

**Q: What is data leakage?**  
A: It occurs when information unavailable at prediction time enters model training or evaluation. We calculate features only through close(t), keep forward data out of the feature list, and split chronologically.

**Q: Why not use a random train-test split?**  
A: It can mix future and past observations, making the evaluation less representative of predicting future market sessions.

**Q: Why compare against a majority baseline?**  
A: A classifier can appear accurate simply by predicting the most frequent class. The baseline tells us whether the model adds value beyond that simple strategy.

**Q: Why is accuracy not very high?**  
A: Daily market direction is noisy and non-stationary. Technical indicators do not guarantee predictive information. Our multi-stock walk-forward evaluation shows that the model has only a small average advantage over the baseline, so we report the limitation rather than overstate results.

**Q: What does BUY/HOLD/SELL mean in this project?**  
A: The application shows a cautious action label based on predicted probability and historical reliability checks. If the model fails those checks, it returns HOLD / insufficient historical edge. The labels are educational signals, not financial advice.

**Q: What would you improve next?**  
A: Evaluate additional market-context features such as broad-index returns and sector-relative performance, then test them with a pre-registered walk-forward design on new unseen periods. I would not keep tuning on the same test set.

## 11. Suggested 45-second presentation

“Market Intel is an AI-based stock analysis research application for Indian equities. It downloads historical OHLCV data, engineers technical and statistical features, and compares four classification algorithms to estimate whether the next trading session's return will be positive or non-positive. We use chronological training, validation, holdout testing and repeated walk-forward evaluation to avoid future-data leakage. The application also compares predictions with a majority-class baseline and includes a transaction-cost-aware backtest against Buy & Hold and Momentum. Our experiments show that one-day stock direction is difficult to predict consistently, so the application reports model limitations and defaults to HOLD when historical evidence is insufficient.”
