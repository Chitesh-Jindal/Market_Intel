# Market Intel — AI-Powered Indian Stock Intelligence

A Streamlit research application for next-trading-day direction analysis of Indian equities.

## Final validated version

The final one-day release is maintained in the `final-release/` folder. See its README for setup, methodology, and limitations.

## Quick start

```bash
cd final-release
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python validate_project.py
streamlit run app.py
```

## Scope

- Daily OHLCV data and technical feature engineering
- One-day positive/non-positive return classification
- Logistic Regression, Random Forest, Gradient Boosting, and HistGradientBoosting
- Chronological validation, holdout evaluation, and walk-forward testing
- Conservative BUY/HOLD/SELL-Avoid action gate
- Next-session executable backtesting

## Research limitation

The supplied five-stock benchmark did not establish a reliable predictive edge over baseline. The project reports that limitation and defaults to HOLD when its historical reliability checks are not met. It is an educational research prototype, not investment advice.
