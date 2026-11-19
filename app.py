
import streamlit as st
import pandas as pd

from data_loader import load_stock_data
from technical_analysis import (
    calculate_technical_indicators,
    get_technical_signals
)
from ml_pipeline import (
    prepare_ml_dataset,
    EXISTING_FEATURES,
    ALL_FEATURES,
    get_models
)
from evaluation import walk_forward_validation


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Market Intel",
    page_icon="📈",
    layout="wide"
)

st.title("Market Intel")
st.caption("AI-Powered Indian Stock Intelligence")

st.write(
    "Analyze historical stock data, technical indicators, "
    "and machine-learning model performance."
)


# --------------------------------------------------
# 2. STOCK SELECTION
# --------------------------------------------------

stock = st.text_input(
    "Enter an NSE stock symbol",
    value="RELIANCE.NS"
).strip().upper()

analyze = st.button(
    "Analyze Stock",
    type="primary"
)


# --------------------------------------------------
# 3. MAIN ANALYSIS
# --------------------------------------------------

if analyze:

    if not stock:
        st.error("Please enter a valid stock symbol.")
        st.stop()

    with st.spinner("Downloading historical stock data..."):

        data = load_stock_data(stock, period="2y")

    if data is None or data.empty:
        st.error("No stock data found. Check the symbol and try again.")
        st.stop()

    data = data.copy()

    st.success(f"Historical data loaded for {stock}")

    # --------------------------------------------------
    # 4. HISTORICAL DATA
    # --------------------------------------------------

    st.header("1. Historical Market Data")

    col1, col2, col3 = st.columns(3)

    col1.metric("Trading Days", len(data))
    col2.metric("Latest Close", f"₹{data['Close'].iloc[-1]:,.2f}")
    col3.metric(
        "Latest Daily Return",
        f"{data['Daily_Return (%)'].iloc[-1]:.2f}%"
    )

    with st.expander("View historical data"):
        st.dataframe(data, use_container_width=True)

    # --------------------------------------------------
    # 5. KEY STATISTICS
    # --------------------------------------------------

    st.subheader("Key Statistics")

    average_return = data["Daily_Return (%)"].mean()
    daily_volatility = data["Daily_Return (%)"].std()

    highest_price = data["Close"].max()
    lowest_price = data["Close"].min()

    first_price = data["Close"].iloc[0]
    latest_price = data["Close"].iloc[-1]

    total_change = (
        (latest_price / first_price) - 1
    ) * 100

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Daily Return",
        f"{average_return:.3f}%"
    )

    col2.metric(
        "Daily Volatility",
        f"{daily_volatility:.3f}%"
    )

    col3.metric(
        "Total Price Change",
        f"{total_change:.2f}%"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Highest Close", f"₹{highest_price:,.2f}")
    col2.metric("Lowest Close", f"₹{lowest_price:,.2f}")
    col3.metric("Total Volume", f"{data['Volume'].sum():,.0f}")

    # --------------------------------------------------
    # 6. TECHNICAL ANALYSIS
    # --------------------------------------------------

    st.header("2. Technical Analysis")

    data = calculate_technical_indicators(data)

    st.subheader("Price and Moving Averages")

    st.line_chart(
        data[["Close", "SMA_10", "SMA_20", "EMA_20"]]
    )

    st.subheader("Relative Strength Index")

    st.line_chart(data["RSI"])

    st.subheader("MACD")

    st.line_chart(
        data[["MACD", "MACD_Signal"]]
    )

    st.subheader("Bollinger Bands")

    st.line_chart(
        data[["Close", "BB_Upper", "BB_Middle", "BB_Lower"]]
    )

    # Latest indicator values
    latest = data.iloc[-1]

    st.subheader("Latest Technical Indicators")

    indicator_values = {
        "SMA 10": latest["SMA_10"],
        "SMA 20": latest["SMA_20"],
        "EMA 20": latest["EMA_20"],
        "RSI": latest["RSI"],
        "MACD": latest["MACD"],
        "MACD Signal": latest["MACD_Signal"],
        "Bollinger Upper": latest["BB_Upper"],
        "Bollinger Middle": latest["BB_Middle"],
        "Bollinger Lower": latest["BB_Lower"]
    }

    indicator_df = pd.DataFrame(
        indicator_values.items(),
        columns=["Indicator", "Latest Value"]
    )

    st.dataframe(
        indicator_df.round(3),
        use_container_width=True,
        hide_index=True
    )

    # Technical signal summary
    signals = get_technical_signals(data)

    st.subheader("Technical Signal Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Trend", signals["Trend"])
    col2.metric("Momentum", signals["Momentum"])
    col3.metric("Volatility", signals["Volatility"])

    # --------------------------------------------------
    # 7. MACHINE-LEARNING DATASET
    # --------------------------------------------------

    st.header("3. Machine Learning Dataset")

    X, y, ml_data = prepare_ml_dataset(data)

    if len(X) < 100:
        st.error(
            "Not enough usable observations for reliable "
            "walk-forward evaluation."
        )
        st.stop()

    st.write("The target is whether the next trading day's return is positive.")

    col1, col2, col3 = st.columns(3)

    col1.metric("Usable Observations", len(X))
    col2.metric("Original Features", len(EXISTING_FEATURES))
    col3.metric("Expanded Features", len(ALL_FEATURES))

    with st.expander("View the original 7 features"):
        st.dataframe(
            X[EXISTING_FEATURES],
            use_container_width=True
        )

    with st.expander("View all expanded features"):
        st.dataframe(
            X[ALL_FEATURES],
            use_container_width=True
        )

    with st.expander("View target distribution"):
        target_distribution = (
            y.value_counts()
            .sort_index()
            .rename({
                0: "Non-positive return",
                1: "Positive return"
            })
            .to_frame("Observations")
        )

        st.dataframe(target_distribution)

    st.caption(
        "The next-day return is used to create the target. "
        "It is not included among the model input features."
    )

    # --------------------------------------------------
    # 8. MODEL COMPARISON
    # --------------------------------------------------

    st.header("4. Machine Learning Model Comparison")

    models = get_models()

    st.write(
        "Models included in this experiment:",
        ", ".join(models.keys())
    )

    st.write(
        "Each model is evaluated using both the original "
        "and expanded feature sets."
    )

    st.info(
        "This comparison uses chronological walk-forward "
        "validation. Thresholds are selected using validation "
        "data, and performance is measured on later test data."
    )

    # --------------------------------------------------
    # 9. WALK-FORWARD VALIDATION
    # --------------------------------------------------

    with st.spinner(
        "Training and evaluating models across chronological folds..."
    ):

        walk_forward = walk_forward_validation(X, y)

    summary = walk_forward["summary"]
    fold_results = walk_forward["fold_results"]
    skipped_folds = walk_forward["skipped_folds"]

    if summary.empty:
        st.error(
            "No models could be evaluated. "
            "Check the available observations and class distribution."
        )
        st.stop()

    # --------------------------------------------------
    # 10. OVERALL RESULTS
    # --------------------------------------------------

    st.header("5. Overall Model Performance")

    display_summary = summary.rename(columns={
        "Average_Accuracy": "Average Accuracy",
        "Average_Macro_F1": "Average Macro F1",
        "Average_Balanced_Accuracy": "Average Balanced Accuracy",
        "Average_ROC_AUC": "Average ROC-AUC",
        "Average_Log_Loss": "Average Log Loss",
        "Average_Baseline_Accuracy": "Average Baseline Accuracy",
        "Average_Baseline_Macro_F1": "Average Baseline Macro F1"
    })

    st.dataframe(
        display_summary.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Higher Accuracy, Macro F1, Balanced Accuracy and ROC-AUC "
        "are generally better. Lower Log Loss is better."
    )

    # --------------------------------------------------
    # 11. FEATURE SET COMPARISON
    # --------------------------------------------------

    st.header("6. Original Features vs Expanded Features")

    feature_comparison = summary.pivot(
        index="Model",
        columns="Feature Set",
        values="Average Macro F1"
    )

    st.subheader("Macro F1 Comparison")

    st.dataframe(
        feature_comparison.round(4),
        use_container_width=True
    )

    if (
        "Original 7 features" in feature_comparison.columns
        and "Expanded features" in feature_comparison.columns
    ):

        feature_comparison["Difference"] = (
            feature_comparison["Expanded features"]
            - feature_comparison["Original 7 features"]
        )

        st.subheader("Change in Macro F1")

        st.dataframe(
            feature_comparison[["Difference"]].round(4),
            use_container_width=True
        )

    # --------------------------------------------------
    # 12. INDIVIDUAL FOLD RESULTS
    # --------------------------------------------------

    st.header("7. Individual Walk-Forward Results")

    st.dataframe(
        fold_results.round(4),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # 13. MODEL VS BASELINE
    # --------------------------------------------------

    st.header("8. Model vs Majority-Class Baseline")

    baseline_comparison = summary[
        [
            "Feature Set",
            "Model",
            "Average_Accuracy",
            "Average_Baseline_Accuracy",
            "Average_Macro_F1",
            "Average_Baseline_Macro_F1"
        ]
    ].copy()

    baseline_comparison["Accuracy Difference"] = (
        baseline_comparison["Average_Accuracy"]
        - baseline_comparison["Average_Baseline_Accuracy"]
    )

    baseline_comparison["Macro F1 Difference"] = (
        baseline_comparison["Average_Macro_F1"]
        - baseline_comparison["Average_Baseline_Macro_F1"]
    )

    st.dataframe(
        baseline_comparison.round(4),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # 14. SKIPPED FOLDS
    # --------------------------------------------------

    st.header("9. Evaluation Warnings")

    if not skipped_folds.empty:

        st.warning(
            "Some folds or model runs were skipped. "
            "Review the reasons below."
        )

        st.dataframe(
            skipped_folds,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success("No folds or model runs were skipped.")

    # --------------------------------------------------
    # 15. INTERPRETATION
    # --------------------------------------------------

    st.header("10. Initial Interpretation")

    best_result = summary.iloc[0]

    st.write(
        f"Highest average Macro F1 in this experiment: "
        f"**{best_result['Model']}** using "
        f"**{best_result['Feature Set']}**."
    )

    st.write(
        f"Average Macro F1: "
        f"**{best_result['Average_Macro_F1']:.4f}**"
    )

    st.write(
        f"Average accuracy: "
        f"**{best_result['Average_Accuracy']:.2%}**"
    )

    st.warning(
        "This is a historical classification experiment, not "
        "evidence of a profitable trading strategy. We still need "
        "a separate backtest with transaction costs, slippage, "
        "and risk metrics."
    )

    # --------------------------------------------------
    # 16. DOWNLOAD RESULTS
    # --------------------------------------------------

    st.header("11. Export Results")

    st.download_button(
        label="Download Model Comparison CSV",
        data=summary.to_csv(index=False),
        file_name=f"{stock}_model_comparison.csv",
        mime="text/csv"
    )

    st.download_button(
        label="Download Walk-Forward Fold Results CSV",
        data=fold_results.to_csv(index=False),
        file_name=f"{stock}_walk_forward_results.csv",
        mime="text/csv"
    )

    st.success("Analysis completed.")
