
import streamlit as st
import pandas as pd

from data_loader import load_stock_data
from technical_analysis import (
    calculate_technical_indicators,
    get_technical_signals
)
from ml_pipeline import (
    prepare_ml_dataset,
    train_model
)
from evaluation import (
    evaluate_model,
    walk_forward_validation
)


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Market Intel",
    page_icon="📈",
    layout="wide"
)

st.title("Market Intel")
st.write("AI-Powered Indian Stock Intelligence")

stock = st.text_input(
    "Enter stock symbol",
    "RELIANCE.NS"
)


# --------------------------------------------------
# 2. DOWNLOAD HISTORICAL DATA
# --------------------------------------------------

if st.button("Analyze"):

    if not stock.strip():
        st.error("Please enter a valid stock symbol.")
        st.stop()

    with st.spinner("Downloading stock data..."):

        data = load_stock_data(stock.strip())

    if data is None:
        st.error("No data found for this stock.")
        st.stop()

    # Work with a separate copy
    data = data.copy()

    st.success(f"Successfully loaded data for {stock.upper()}")

    # --------------------------------------------------
    # 3. HISTORICAL DATA
    # --------------------------------------------------

    st.subheader("Historical Data")
    st.dataframe(data, use_container_width=True)

    st.write("Missing Values")
    st.dataframe(data.isnull().sum().to_frame("Missing Values"))

    st.divider()

    # --------------------------------------------------
    # 4. KEY STATISTICS
    # --------------------------------------------------

    st.subheader(f"Analysis for {stock.upper()}")
    st.caption("Key Statistics")

    average_return = data["Daily_Return (%)"].mean()
    volatility = data["Daily_Return (%)"].std()

    highest_price = data["Close"].max()
    lowest_price = data["Close"].min()
    latest_price = data["Close"].iloc[-1]

    price_range = highest_price - lowest_price

    total_volume = data["Volume"].sum()
    trading_days = len(data)

    price_change = latest_price - data["Close"].iloc[0]

    price_change_percent = (
        price_change / data["Close"].iloc[0]
    ) * 100

    col1, col2, col3 = st.columns(3)

    col1.metric("Latest Closing Price", f"{latest_price:.2f}")
    col2.metric("Price Change", f"{price_change:.2f}")
    col3.metric("Price Change (%)", f"{price_change_percent:.2f}%")

    col1, col2, col3 = st.columns(3)

    col1.metric("Highest Closing Price", f"{highest_price:.2f}")
    col2.metric("Lowest Closing Price", f"{lowest_price:.2f}")
    col3.metric("Closing Price Range", f"{price_range:.2f}")

    col1, col2, col3 = st.columns(3)

    col1.metric("Average Daily Return", f"{average_return:.2f}%")
    col2.metric("Daily Volatility", f"{volatility:.2f}%")
    col3.metric("Trading Days", trading_days)

    st.write("Total Trading Volume:", f"{total_volume:,.0f}")

    st.divider()

    # --------------------------------------------------
    # 5. TECHNICAL ANALYSIS
    # --------------------------------------------------

    data = calculate_technical_indicators(data)

    st.subheader("Price and Moving Averages")

    st.line_chart(
        data[["Close", "SMA_10", "SMA_20", "EMA_20"]]
    )

    st.subheader("RSI (Relative Strength Index)")
    st.line_chart(data["RSI"])

    st.subheader("MACD (Moving Average Convergence Divergence)")
    st.line_chart(data[["MACD", "MACD_Signal"]])

    st.subheader("Bollinger Bands")

    st.line_chart(
        data[["Close", "BB_Upper", "BB_Middle", "BB_Lower"]]
    )

    # Latest indicators
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

    for name, value in indicator_values.items():
        st.write(f"{name}: {value:.2f}")

    # Technical signal summary
    signals = get_technical_signals(data)

    st.subheader("Technical Signal Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Trend", signals["Trend"])
    col2.metric("Momentum", signals["Momentum"])
    col3.metric("Volatility", signals["Volatility"])

    st.divider()

    # --------------------------------------------------
    # 6. MACHINE LEARNING DATASET
    # --------------------------------------------------

    st.subheader("ML Dataset")

    X, y, ml_data = prepare_ml_dataset(data)

    st.write("ML Features")
    st.dataframe(X, use_container_width=True)

    st.subheader("ML Features and Target")

    st.write("Features (X):")
    st.dataframe(X, use_container_width=True)

    st.write("Target (y):")
    st.dataframe(y.to_frame(), use_container_width=True)

    st.divider()

    # --------------------------------------------------
    # 7. TRAINING AND TESTING
    # --------------------------------------------------

    st.subheader("Training, Validation and Testing Data")

    if len(X) < 30:
        st.error("Insufficient data for the existing ML evaluation.")
        st.stop()

    results = train_model(X, y)

    st.write("Training rows:", len(results["X_train"]))
    st.write("Validation rows:", len(results["X_val"]))
    st.write("Testing rows:", len(results["X_test"]))

    # --------------------------------------------------
    # 8. FEATURE DRIFT
    # --------------------------------------------------

    st.subheader("Model Diagnostics: Feature Drift")

    st.dataframe(
        results["drift_data"].round(4),
        use_container_width=True
    )

    # --------------------------------------------------
    # 9. MODEL COEFFICIENTS
    # --------------------------------------------------

    st.subheader("Model Diagnostics: Feature Coefficients")

    st.write(
        "Model Intercept:",
        round(results["model_intercept"], 4)
    )

    st.dataframe(
        results["coefficient_data"].round(4),
        use_container_width=True
    )

    # --------------------------------------------------
    # 10. FEATURE CONTRIBUTIONS
    # --------------------------------------------------

    st.subheader("Average Feature Contributions")

    st.dataframe(
        results["contribution_data"].round(4),
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------
    # 11. MODEL EVALUATION
    # --------------------------------------------------

    st.subheader("Model Evaluation")

    evaluation = evaluate_model(results)

    # Threshold analysis
    st.subheader("Validation Threshold Analysis")

    st.dataframe(
        evaluation["threshold_df"].round(4),
        use_container_width=True
    )

    st.write(
        "Selected Threshold:",
        evaluation["selected_threshold"]
    )

    # Validation predictions
    st.subheader("Validation Prediction Analysis")

    st.dataframe(
        evaluation["validation_results"].round(4),
        use_container_width=True
    )

    # Validation evaluation
    st.subheader("Validation Evaluation")

    st.write(
        "Validation Accuracy:",
        f"{evaluation['val_accuracy'] * 100:.2f}%"
    )

    st.write(
        "Baseline Prediction Class:",
        evaluation["val_baseline_class"]
    )

    st.write(
        "Validation Baseline Accuracy:",
        f"{evaluation['val_baseline_accuracy'] * 100:.2f}%"
    )

    st.write("Validation Confusion Matrix:")
    st.write(evaluation["val_confusion_matrix"])

    # Class distributions
    st.subheader("Class Distribution")

    st.write("Training Set:")
    st.write(evaluation["training_class_distribution"])

    st.write("Validation Set:")
    st.write(evaluation["validation_class_distribution"])

    # Probability metrics
    st.subheader("Model Diagnostics: Probability Metrics")

    st.write("Validation ROC-AUC:", round(evaluation["val_auc"], 4))
    st.write("Validation Log Loss:", round(evaluation["val_log_loss"], 4))
    st.write("Prior-Only Baseline Log Loss:", round(evaluation["prior_log_loss"], 4))

    # Positive-return rates
    st.subheader("Model Diagnostics: Positive-Return Rates")

    for name, rates in evaluation["positive_return_rates"].items():

        st.write(name, rates)

    # Probability statistics
    st.subheader("Model Diagnostics: Predicted Probabilities")

    for name, statistics in evaluation["probability_statistics"].items():

        st.write(name, statistics)

    # --------------------------------------------------
    # 12. FINAL TEST EVALUATION
    # --------------------------------------------------

    st.subheader("Final Test Evaluation")

    accuracy = evaluation["test_accuracy"]

    st.metric(
        "Test Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    st.write("Confusion Matrix:")
    st.write(evaluation["test_confusion_matrix"])

    st.subheader("Detailed Classification Metrics")

    st.dataframe(
        evaluation["classification_metrics"].round(3),
        use_container_width=True
    )

    # Model versus baseline
    st.subheader("Model vs Baseline")

    col1, col2 = st.columns(2)

    col1.metric(
        "Logistic Regression Accuracy",
        f"{accuracy:.2%}"
    )

    col2.metric(
        "Majority-Class Baseline",
        f"{evaluation['baseline_accuracy']:.2%}"
    )

    if accuracy > evaluation["baseline_accuracy"]:
        st.success(
            "The model outperforms the majority-class baseline."
        )

    elif accuracy == evaluation["baseline_accuracy"]:
        st.info(
            "The model matches the majority-class baseline."
        )

    else:
        st.warning(
            "The model performs below the majority-class baseline."
        )

    st.divider()

    # --------------------------------------------------
    # 13. WALK-FORWARD VALIDATION
    # --------------------------------------------------

    st.subheader("Walk-Forward Validation")

    walk_forward = walk_forward_validation(X, y)

    if walk_forward is not None:

        st.subheader("Individual Fold Results")

        st.dataframe(
            walk_forward["results"].round(4),
            use_container_width=True
        )

        st.subheader("Overall Walk-Forward Performance")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Average Test Accuracy",
            f"{walk_forward['avg_accuracy']:.2%}"
        )

        col2.metric(
            "Average Macro F1",
            f"{walk_forward['avg_f1']:.3f}"
        )

        col3.metric(
            "Average Baseline Accuracy",
            f"{walk_forward['avg_baseline']:.2%}"
        )

        st.subheader("Walk-Forward Model vs Baseline")

        if walk_forward["avg_accuracy"] > walk_forward["avg_baseline"]:

            st.success(
                "Model outperforms the baseline on average."
            )

        elif walk_forward["avg_accuracy"] == walk_forward["avg_baseline"]:

            st.info(
                "Model matches the baseline on average."
            )

        else:

            st.warning(
                "Model performs below the baseline on average."
            )

    else:

        st.warning(
            "Not enough data to perform walk-forward validation."
        )
