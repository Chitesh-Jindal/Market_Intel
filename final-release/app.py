import pandas as pd
import streamlit as st

from data_loader import load_stock_data
from technical_analysis import calculate_technical_indicators
from ml_pipeline import (
    FEATURES,
    build_ml_dataset,
    train_and_evaluate,
    walk_forward_evaluate,
)
from evaluation import evaluate_test, build_feature_importance
from backtesting import run_backtest
from threadpoolctl import threadpool_limits

st.set_page_config(page_title="Market Intel", page_icon="📈", layout="wide")
st.title("📈 Market Intel")
st.caption("AI-Powered Indian Stock Intelligence | One-day market direction research")

with st.sidebar:
    st.header("Analysis settings")
    ticker = st.selectbox(
        "Select an Indian stock",
        [
            "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
            "ICICIBANK.NS", "SBIN.NS", "ITC.NS", "LT.NS", "BHARTIARTL.NS",
        ],
    )
    period = st.selectbox("Historical data", ["2y", "3y", "5y"], index=2)
    analyse = st.button("Run Market Intel", type="primary", use_container_width=True)
    st.caption("The model predicts the next trading session only.")


@st.cache_data(ttl=1800, show_spinner=False)
def get_market_data(symbol: str, history: str):
    raw = load_stock_data(symbol, history)
    return calculate_technical_indicators(raw)


if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if analyse or st.session_state.analysis_result is None:
    try:
        with st.spinner("Downloading data, comparing models and running walk-forward checks..."):
            market_data = get_market_data(ticker, period)
            X, y, labelled, latest_features = build_ml_dataset(market_data)
            # Keep CPU usage predictable on student laptops and hosted runners.
            with threadpool_limits(limits=1):
                modelling = train_and_evaluate(X, y)
                walk_forward = walk_forward_evaluate(X, y, folds=4)
                latest_probability = float(
                    modelling["latest_model"].predict_proba(
                        latest_features[modelling["selected_features"]]
                    )[:, 1][0]
                )
            st.session_state.analysis_result = {
                "ticker": ticker,
                "period": period,
                "market": market_data,
                "X": X,
                "y": y,
                "labelled": labelled,
                "latest_features": latest_features,
                "modelling": modelling,
                "walk_forward": walk_forward,
                "latest_probability": latest_probability,
            }
    except Exception as error:
        st.error(f"Analysis could not be completed: {error}")
        st.info("Try five years of history, another liquid NSE stock, or check your internet connection.")
        st.stop()

result = st.session_state.analysis_result
if result["ticker"] != ticker or result["period"] != period:
    st.info("Your selection has changed. Click **Run Market Intel** to analyse the new selection.")
    st.stop()

market = result["market"]
labelled = result["labelled"]
modelling = result["modelling"]
walk_forward = result["walk_forward"]
latest_probability = result["latest_probability"]
selected_threshold = modelling["threshold"]

test_metrics, baseline_metrics, confusion, test_predictions = evaluate_test(
    modelling["y_test"],
    modelling["test_probabilities"],
    selected_threshold,
    modelling["y_train"],
)

wf = walk_forward["summary"]
holdout_accuracy_edge = test_metrics["Accuracy"] > baseline_metrics["Baseline Accuracy"]
walk_forward_gate = (
    wf["Mean Test Accuracy"] > wf["Mean Baseline Accuracy"]
    and wf["Mean Balanced Accuracy"] >= 0.52
    and wf["Mean ROC-AUC"] >= 0.52
    and wf["Folds Beating Baseline"] >= 3
)
historical_edge_confirmed = holdout_accuracy_edge and walk_forward_gate

if latest_probability >= selected_threshold:
    direction = "Positive"
else:
    direction = "Non-positive"

if not historical_edge_confirmed:
    action = "HOLD — INSUFFICIENT HISTORICAL EDGE"
    action_message = (
        "The model has not passed the combined held-out and repeated walk-forward checks. "
        "Treat its direction estimate as experimental, not as a trading recommendation."
    )
elif latest_probability >= max(0.60, selected_threshold):
    action = "BUY SIGNAL"
    action_message = "The probability clears both the selected classification threshold and the cautious BUY band."
elif latest_probability <= min(0.40, selected_threshold):
    action = "SELL / AVOID"
    action_message = "The probability clears both the selected classification threshold and the cautious SELL band."
else:
    action = "HOLD"
    action_message = "The probability does not show enough separation for a strong directional action."

tabs = st.tabs([
    "Dashboard", "1-Day Prediction", "Model Evaluation",
    "Backtesting", "Methodology"
])

with tabs[0]:
    st.subheader(f"{result['ticker']} — Market overview")
    latest = market.iloc[-1]
    daily_change = market["Close"].pct_change().iloc[-1] * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest close", f"₹{latest['Close']:,.2f}")
    c2.metric("Latest daily change", f"{daily_change:+.2f}%")
    c3.metric("RSI (14)", f"{latest['RSI']:.1f}" if pd.notna(latest["RSI"]) else "N/A")
    c4.metric("ADX (14)", f"{latest['ADX']:.1f}" if pd.notna(latest["ADX"]) else "N/A")

    st.markdown("**Price and moving averages**")
    st.line_chart(market[["Close", "SMA_20", "SMA_50"]].dropna(), use_container_width=True)

    st.markdown("**Latest technical indicators**")
    indicator_columns = [
        "RSI", "MACD", "MACD_Signal", "BB_Upper", "BB_Middle", "BB_Lower",
        "ADX", "ATR", "Stochastic", "CCI",
    ]
    st.dataframe(market[indicator_columns].tail(8).round(3), use_container_width=True)

with tabs[1]:
    st.subheader("Next trading-day prediction")
    p1, p2, p3 = st.columns(3)
    p1.metric("Predicted direction", direction)
    p2.metric("Probability of positive return", f"{latest_probability * 100:.1f}%")
    p3.metric("Validation-selected threshold", f"{selected_threshold:.3f}")

    if action == "BUY SIGNAL":
        st.success(f"{action} — {action_message}")
    elif action == "SELL / AVOID":
        st.error(f"{action} — {action_message}")
    else:
        st.warning(f"{action} — {action_message}")

    st.progress(min(max(latest_probability, 0.0), 1.0))
    st.caption(
        "BUY requires probability ≥60% and above the selected threshold. "
        "SELL / AVOID requires probability ≤40% and below the selected threshold. "
        "Otherwise the action is HOLD. Probability is not a calibrated guarantee."
    )
    st.info(
        "Prediction target: whether the next trading session's close-to-close return "
        "will be positive (class 1) or non-positive (class 0)."
    )

with tabs[2]:
    st.subheader("Out-of-sample model evaluation")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Holdout accuracy", f"{test_metrics['Accuracy'] * 100:.2f}%")
    m2.metric("Holdout Macro F1", f"{test_metrics['Macro F1']:.3f}")
    m3.metric("Majority baseline", f"{baseline_metrics['Baseline Accuracy'] * 100:.2f}%")
    m4.metric("Holdout ROC-AUC", f"{test_metrics['ROC-AUC']:.3f}")

    st.markdown(f"**Selected model:** {modelling['model_name']} · **Feature set:** {modelling['feature_set_name']} · **Features:** {len(modelling['selected_features'])}")
    edge = (test_metrics["Accuracy"] - baseline_metrics["Baseline Accuracy"]) * 100
    if edge > 0:
        st.success(f"Holdout accuracy is {edge:+.2f} percentage points above the training-majority baseline.")
    else:
        st.warning(f"Holdout accuracy is {edge:+.2f} percentage points versus the baseline. The model did not beat it.")

    st.markdown("### Repeated walk-forward evaluation")
    wf1, wf2, wf3, wf4 = st.columns(4)
    wf1.metric("Mean accuracy", f"{wf['Mean Test Accuracy'] * 100:.2f}%")
    wf2.metric("Mean baseline", f"{wf['Mean Baseline Accuracy'] * 100:.2f}%")
    wf3.metric("Mean balanced accuracy", f"{wf['Mean Balanced Accuracy']:.3f}")
    wf4.metric("Mean ROC-AUC", f"{wf['Mean ROC-AUC']:.3f}")
    st.write(
        f"Accuracy edge: **{wf['Mean Accuracy Edge (pp)']:+.2f} percentage points** · "
        f"Beats fold baseline: **{wf['Folds Beating Baseline']}/{wf['Folds']}**"
    )
    st.caption(
        "The app requires a positive walk-forward accuracy edge, at least 3 of 4 folds "
        "above their baselines, and mean balanced accuracy and ROC-AUC of at least 0.52 "
        "before enabling BUY/SELL labels. These are conservative display guardrails, "
        "not a formal statistical significance test."
    )
    if walk_forward_gate and holdout_accuracy_edge:
        st.success("The model passed the historical reliability checks for this stock and period.")
    else:
        st.warning("The model has not demonstrated a consistent edge. Keep the live recommendation at HOLD.")

    st.dataframe(
        walk_forward["fold_results"].style.format({
            "Threshold": "{:.3f}",
            "Validation Macro F1": "{:.3f}",
            "Test Accuracy": "{:.1%}",
            "Baseline Accuracy": "{:.1%}",
            "Accuracy Edge (pp)": "{:+.2f}",
            "Macro F1": "{:.3f}",
            "Balanced Accuracy": "{:.3f}",
            "ROC-AUC": "{:.3f}",
        }),
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("Validation model comparison"):
        st.caption("Four algorithms and two feature sets are compared on validation data only. The final test periods are not used to select the model.")
        st.dataframe(
            modelling["validation_results"].style.format({
                "Validation Macro F1": "{:.3f}",
                "Validation Accuracy": "{:.1%}",
                "Threshold": "{:.3f}",
            }),
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("Single holdout metrics and baseline"):
        st.dataframe(
            pd.DataFrame({
                "Metric": list(test_metrics.keys()),
                "Selected model": list(test_metrics.values()),
            }),
            hide_index=True,
            use_container_width=True,
        )
        st.dataframe(
            pd.DataFrame({
                "Baseline metric": list(baseline_metrics.keys()),
                "Baseline": list(baseline_metrics.values()),
            }),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("**Confusion matrix**")
    st.dataframe(
        pd.DataFrame(
            confusion,
            index=["Actual non-positive", "Actual positive"],
            columns=["Predicted non-positive", "Predicted positive"],
        ),
        use_container_width=True,
    )

    with st.expander("Feature importance"):
        importance = build_feature_importance(
            modelling["model"],
            modelling["X_test"],
            modelling["y_test"],
            modelling["selected_features"],
        )
        if not importance.empty:
            st.bar_chart(importance.head(12).set_index("Feature")["Importance"])
            st.caption(importance.iloc[0]["Method"])
        else:
            st.write("Feature importance is unavailable for this run.")

with tabs[3]:
    st.subheader("One-day strategy backtesting")
    test_rows = labelled.loc[modelling["X_test"].index]
    equity, backtest_metrics, counts = run_backtest(
        test_rows,
        modelling["test_probabilities"],
        buy_threshold=0.60,
    )
    st.markdown("**Growth of ₹1 over the held-out test period**")
    st.line_chart(equity, use_container_width=True)
    st.dataframe(
        backtest_metrics.style.format({
            "Total Return (%)": "{:.2f}%",
            "Max Drawdown (%)": "{:.2f}%",
            "Sharpe Ratio": "{:.3f}",
            "Win Rate (%)": "{:.1f}%",
            "Exposure (%)": "{:.1f}%",
        }),
        hide_index=True,
        use_container_width=True,
    )
    st.write(
        f"ML signal counts: **BUY {counts['BUY']}**, "
        f"**HOLD {counts['HOLD']}**, **SELL / CASH {counts['SELL / CASH']}**."
    )
    st.caption(
        "The signal is generated after close(t), entered at the next session's open, "
        "and exited at that session's close. ML and Momentum trades pay 0.10% on entry "
        "and exit. Buy & Hold pays entry and final liquidation costs. Win rate counts "
        "invested sessions; exposure shows the percentage of sessions invested."
    )

with tabs[4]:
    st.subheader("Project methodology")
    st.markdown(
        """
        **Objective:** predict whether the next trading session's close-to-close return is positive or non-positive.

        **Data:** daily OHLCV history for selected NSE-listed stocks, downloaded through Yahoo Finance.

        **Feature engineering:** lagged returns, rolling volatility, moving-average distances, RSI, MACD, Bollinger Bands, ADX, ATR, Stochastic, CCI, OBV and volume behaviour.

        **Algorithms:** Logistic Regression, Random Forest, Gradient Boosting and HistGradientBoosting.

        **Model selection:** compare four algorithms across a compact 12-feature set and the full 30-feature set. Select using validation Macro F1 and choose the probability threshold on validation data only. Fixed, conservative model configurations keep runtime manageable.

        **Single holdout:** chronological 60/20/20 training, validation and test split with a one-row purge at boundaries.

        **Walk-forward robustness:** four expanding-window folds, each with a training period, later validation period for model/threshold selection, and subsequent untouched test period. Each fold uses a training-majority baseline.

        **Reliability gate:** directional actions are enabled only if the single holdout beats its baseline, walk-forward mean accuracy beats its mean baseline, at least three of four folds beat their own baselines, and mean balanced accuracy and ROC-AUC are each at least 0.52. These are conservative display guardrails, not a formal statistical significance test. Otherwise the app shows HOLD / insufficient historical edge.

        **Backtest execution:** features use information through close(t); trades enter at next session's open and exit at next session's close. This avoids assuming an impossible fill at an already-observed close.

        **Limitations:** daily stock returns are noisy and market regimes change. Historical performance does not guarantee future performance. The app is an educational research prototype, not financial advice.
        """
    )
    st.write(f"**Labelled observations:** {len(result['X']):,}")
    st.write(f"**Available features:** {len(FEATURES)}")
    st.write(f"**Selected features:** {len(modelling['selected_features'])}")
    st.write(
        f"**Holdout rows:** train {len(modelling['split']['X_train']):,} · "
        f"validation {len(modelling['split']['X_val']):,} · test {len(modelling['split']['X_test']):,}"
    )
