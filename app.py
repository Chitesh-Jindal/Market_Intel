import streamlit as st
import yfinance as yf;
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_auc_score, log_loss
import pandas as pd;
from sklearn.metrics import f1_score

#ta is technical analysis library provides implementations of common financial technical indicators.
st.title("Market Intel")
st.write("AI-Powered Indian Stock Intelligence")

stock = st.text_input("Enter stock symbol", "RELIANCE.NS")

if st.button("Analyze"):
    data = yf.download(stock, period="2y")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    #1mo means 1 month of data
    if data.empty:
        #if the stock entered is invalid
        st.error("No data found for this stock")
    else:
        data["Daily_Return (%)"]=data["Close"].pct_change()*100
        
        st.subheader("Historical Data")
        st.write(data)
        missing = data.isnull().sum()
        #checking values in data set that are missing
        st.write("Missing Values")
        st.write(missing)

        st.divider()
        st.subheader("Analysis for " + stock)

        st.caption("Key Statistics")
        
        average_return = data["Daily_Return (%)"].mean()
        volatility=data["Daily_Return (%)"].std()
        highest_price=data["Close"].max().item()
        lowest_price=data["Close"].min().item()
        latest_price=data["Close"].iloc[-1].item()
        price_range = highest_price - lowest_price
        total_volume=data["Volume"].sum().item()
        trading_days = len(data)
        price_change = latest_price - data["Close"].iloc[0].item()
        price_change_percent = (price_change / data["Close"].iloc[0].item()) * 100

        st.write("Trading Days:", trading_days)

        st.write("Latest Closing Price: ",round(latest_price,2))
        st.write("Price Change:", round(price_change, 2))
        st.write("Highest Closing Price: ",round(highest_price,2))
        st.write("Lowest Closing Price: ",round(lowest_price,2))
        st.write("Closing Price Range:", round(price_range, 2))
        st.write("Price Change (%):", round(price_change_percent, 2), "%")
        
        st.write("Average Daily Return: ",round(average_return,2),"%")
        st.write("Daily Volatility: ",round(volatility,2),"%")
        
        
        st.write("Total Trading Volume:", total_volume)
        st.divider()

        data["SMA_10"]=data["Close"].rolling(window=10).mean()
        #SIMPLE MOVING AVERAGE
        data["SMA_20"]=data["Close"].rolling(window=20).mean()
        data["SMA_Spread"]=data["SMA_10"]-data["SMA_20"]

        #Exponential moving average
        #similar to SMA but gives more weight to recent prices, so repond faster to recent changes
        data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()

        st.line_chart(data[["Close", "SMA_10", "SMA_20", "EMA_20"]])


        #RELATIVE STRENGTH INDEX measures how strongly the price is going upwards or downwards
        #ranges from 0 to 100
        rsi_indicator= RSIIndicator(close=data["Close"], window=14)
        data["RSI"]=rsi_indicator.rsi()
        st.subheader("RSI (Relative Strength Index)")
        st.line_chart(data["RSI"])

        
        #Moving average convergence divergence
        #gives us meaure of trend/momentum
        #MACD=12 day EMA - 26 day EMA
        #+ve means upward momentum and vice versa similarly macd signal tells 9 day EMA a smoother EMA
        #if macd crosses upward macd signal means increasing upward momentuma and downward means increasing downward momentum
        macd_indicator=MACD(close=data["Close"])
        data["MACD"]=macd_indicator.macd()
        data["MACD_Signal"]=macd_indicator.macd_signal();
        st.subheader("MACD (Moving Average Converge Divergence)")
        st.line_chart(data[["MACD","MACD_Signal"]])

        
        #Bollinger Bands 
        #- Middle Band → usually 20-day SMA
        #- Upper Band → middle band + 2 standard deviations
        #- Lower Band → middle band − 2 standard deviations
        bb_indicator = BollingerBands(close=data["Close"], window=20, window_dev=2)

        data["BB_Upper"] = bb_indicator.bollinger_hband()
        data["BB_Middle"] = bb_indicator.bollinger_mavg()
        data["BB_Lower"] = bb_indicator.bollinger_lband()
        st.subheader("Bollinger Bands")
        st.line_chart(data[["Close", "BB_Upper", "BB_Middle", "BB_Lower"]])


        
        # Alternative: Relative and Normalized Features
        
        data["Volume_SMA20"] = data["Volume"].rolling(window=20).mean()
        
        data["Close_vs_SMA20"] = data["Close"] / data["SMA_20"] - 1
        
        data["SMA_Spread_Relative"] = data["SMA_Spread"] / data["Close"]
        
        data["Close_vs_EMA20"] = data["Close"] / data["EMA_20"] - 1
        
        data["MACD_Hist_Relative"] = (data["MACD"] - data["MACD_Signal"]) / data["Close"]
        
        bb_range = (data["BB_Upper"] - data["BB_Lower"]).replace(0, float("nan"))
        
        data["BB_Percent_B"] = (data["Close"] - data["BB_Lower"]) / bb_range
        
        data["BB_Width_Relative"] = (data["BB_Upper"] - data["BB_Lower"]) / data["BB_Middle"]
        
        data["Volume_Relative"] = data["Volume"] / data["Volume_SMA20"].replace(0, float("nan"))

        
        #combined tech analysis section
        st.subheader("Latest Technical Indicators")

        latest = data.iloc[-1]

        st.write("SMA 10:", round(latest["SMA_10"], 2))
        st.write("SMA 20:", round(latest["SMA_20"], 2))
        st.write("EMA 20:", round(latest["EMA_20"], 2))
        st.write("RSI:", round(latest["RSI"], 2))
        st.write("MACD:", round(latest["MACD"], 2))
        st.write("MACD Signal:", round(latest["MACD_Signal"], 2))
        st.write("Bollinger Upper:", round(latest["BB_Upper"], 2))
        st.write("Bollinger Middle:", round(latest["BB_Middle"], 2))
        st.write("Bollinger Lower:", round(latest["BB_Lower"], 2))

        #what these values indicate
        #trend= sma/ema
        st.subheader("Technical Signal Summary")
        if latest["SMA_10"]> latest["SMA_20"] and latest["EMA_20"]> latest["SMA_20"]:
            trend_signal = "Positive"
        elif latest["SMA_10"] < latest["SMA_20"] and latest["EMA_20"] < latest["SMA_20"]:
            trend_signal = "Negative"
        else:
            trend_signal = "Mixed"
            
        #momentum= rsi/macd
        if latest["RSI"] > 70:
            momentum_signal = "Strong / Overbought"
        elif latest["RSI"] < 30:
            momentum_signal = "Weak / Oversold"
        elif latest["MACD"] > latest["MACD_Signal"]:
            momentum_signal = "Positive"  
        else:
            momentum_signal = "Negative" 

        #volatility= bollinger bands
        if latest["Close"] > latest["BB_Upper"]:
            volatility_signal = "High / Above Upper Band"
        elif latest["Close"] < latest["BB_Lower"]:
            volatility_signal = "High / Below Lower Band"
        else:
            volatility_signal = "Normal"

        st.write("Trend:", trend_signal)
        st.write("Momentum:", momentum_signal)
        st.write("Volatility:", volatility_signal)

        
        #ml dataset
        st.subheader("ML Dataset")
        data["Next_Day_Return (%)"] = (data["Close"].shift(-1) / data["Close"] - 1) * 100
    
        data = data.dropna(subset=["Next_Day_Return (%)"])
    
        data["Target"] = (data["Next_Day_Return (%)"] > 0).astype(int)
        
        ml_data = data[
            [
                "Close_vs_SMA20",
                "SMA_Spread_Relative",
                "Close_vs_EMA20",
                "MACD_Hist_Relative",
                "BB_Percent_B",
                "BB_Width_Relative",
                "Volume_Relative"
            ]
        ].dropna()

        
        st.write(ml_data)

        #input and output for classification
        X=ml_data
        y=data.loc[ml_data.index, "Target"]
        st.subheader("ML Features and Target")
        st.write("Features (X):", X)
        st.write("Target (y):", y)


        # Split data chronologically into training, validation and testing
        train_end = int(len(X) * 0.64)
        validation_end = int(len(X) * 0.80)
        
        X_train = X.iloc[:train_end]
        X_val = X.iloc[train_end:validation_end]
        X_test = X.iloc[validation_end:]
        
        y_train = y.iloc[:train_end]
        y_val = y.iloc[train_end:validation_end]
        y_test = y.iloc[validation_end:]
        
        st.subheader("Training, Validation and Testing Data")
        
        st.write("Training rows:", len(X_train))
        st.write("Validation rows:", len(X_val))
        st.write("Testing rows:", len(X_test))


        #adding logistic regression
        
        #feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Feature drift diagnostics
        st.subheader("Model Diagnostics: Feature Drift")
        
        feature_names = X_train.columns
        
        train_scaled_df = pd.DataFrame(
            X_train_scaled,
            columns=feature_names,
            index=X_train.index
        )
        
        val_scaled_df = pd.DataFrame(
            X_val_scaled,
            columns=feature_names,
            index=X_val.index
        )
        
        drift_data = pd.DataFrame({
            "Training Mean": train_scaled_df.mean(),
            "Validation Mean": val_scaled_df.mean(),
            "Validation Min": val_scaled_df.min(),
            "Validation Max": val_scaled_df.max()
        })
        
        # Calculate mean shift first
        drift_data["Validation Mean Shift"] = (
            drift_data["Validation Mean"] - drift_data["Training Mean"]
        )
        
        # Calculate absolute mean shift
        drift_data["Absolute Mean Shift"] = (
            drift_data["Validation Mean Shift"].abs()
        )
        
        # Identify observations outside 3 standard deviations
        drift_data["Outside +/-3 SD"] = (
            (drift_data["Validation Min"] < -3) |
            (drift_data["Validation Max"] > 3)
        )
        
        # Sort features by largest distribution shift
        drift_data = drift_data.sort_values(
            "Absolute Mean Shift",
            ascending=False
        )
        
        st.dataframe(drift_data.round(3))

        
        # Train Logistic Regression using training data only
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)


        # Model coefficient diagnostics
        st.subheader("Model Diagnostics: Feature Coefficients")
        
        coefficient_data = pd.DataFrame({
            "Feature": X_train.columns,
            "Coefficient": model.coef_[0]
        })
        
        coefficient_data["Absolute Coefficient"] = (
            coefficient_data["Coefficient"].abs()
        )
        
        coefficient_data = coefficient_data.sort_values(
            "Absolute Coefficient",
            ascending=False
        )
        
        st.write("Model Intercept:", round(model.intercept_[0], 4))
        st.dataframe(coefficient_data.round(4))
        
        # Average contribution of each feature to the model score
        st.subheader("Average Feature Contributions")
        
        contribution_data = pd.DataFrame({
            "Feature": X_train.columns,
            "Training Contribution": (
                X_train_scaled.mean(axis=0) * model.coef_[0]
            ),
            "Validation Contribution": (
                X_val_scaled.mean(axis=0) * model.coef_[0]
            )
        })
        
        contribution_data["Contribution Shift"] = (
            contribution_data["Validation Contribution"]
            - contribution_data["Training Contribution"]
        )
        
        contribution_data["Absolute Shift"] = (
            contribution_data["Contribution Shift"].abs()
        )
        
        contribution_data = contribution_data.sort_values(
            "Absolute Shift",
            ascending=False
        )
        
        st.dataframe(contribution_data.round(4))


        # Predicted probability of a positive return
        y_train_prob = model.predict_proba(X_train_scaled)[:, 1]
        y_val_prob = model.predict_proba(X_val_scaled)[:, 1]
        y_test_prob = model.predict_proba(X_test_scaled)[:, 1]

        # Generate predictions for diagnostic analysis
        y_train_pred = model.predict(X_train_scaled)

        # Generate validation predictions
        y_val_pred = model.predict(X_val_scaled)


        # Threshold analysis using validation probabilities
        st.subheader("Validation Threshold Analysis")
        
        thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
        
        threshold_results = []
        
        for threshold in thresholds:
            threshold_predictions = (y_val_prob >= threshold).astype(int)
        
            threshold_results.append({
                "Threshold": threshold,
                "Accuracy": accuracy_score(
                    y_val, threshold_predictions
                ),
                "Macro F1": classification_report(
                    y_val,
                    threshold_predictions,
                    output_dict=True,
                    zero_division=0
                )["macro avg"]["f1-score"],
                "Positive Predictions (%)": (
                    threshold_predictions.mean() * 100
                )
            })
        
        threshold_df = pd.DataFrame(threshold_results)
        
        st.dataframe(threshold_df.round(4))

        selected_threshold = 0.55
        y_train_pred = (
            y_train_prob >= selected_threshold
        ).astype(int)
        y_val_pred = (
            y_val_prob >= selected_threshold
        ).astype(int)
        st.write("Selected Threshold:", selected_threshold)
        
        # Validation prediction diagnostics
        st.subheader("Validation Prediction Analysis")

        validation_results = pd.DataFrame({
            "Actual Target": y_val,
            "Predicted Probability": y_val_prob,
            "Predicted Target": y_val_pred
        })

        st.dataframe(validation_results.round(4))

        st.subheader("Validation Evaluation")
        val_accuracy = accuracy_score(y_val, y_val_pred)
        st.write("Validation Accuracy:", round(val_accuracy * 100, 2), "%")
        
        class_counts = y_train.value_counts()

        if class_counts.get(0, 0) >= class_counts.get(1, 0):
            val_baseline_class = 0
        else:
            val_baseline_class = 1
        
        val_baseline_predictions = [val_baseline_class] * len(y_val)
        val_baseline_accuracy = accuracy_score(
            y_val, val_baseline_predictions
        )
        
        st.write("Baseline prediction class:", val_baseline_class)
        st.write(
            "Validation Baseline Accuracy:",
            round(val_baseline_accuracy * 100, 2),
            "%"
        )
        
        st.write("Validation Confusion Matrix:")
        st.write(confusion_matrix(y_val, y_val_pred, labels=[0, 1]))

        st.subheader("Class Distribution")

        st.write("Training set:")
        st.write(y_train.value_counts().sort_index().rename({
            0: "Non-positive return (0)",
            1: "Positive return (1)"
        }))
        
        st.write("Validation set:")
        st.write(y_val.value_counts().sort_index().rename({
            0: "Non-positive return (0)",
            1: "Positive return (1)"
        }))

        # Validation probability evaluation
        st.subheader("Model Diagnostics: Probability Metrics")
        
        val_auc = roc_auc_score(y_val, y_val_prob)
        val_log_loss = log_loss(y_val, y_val_prob)
        
        training_positive_rate = y_train.mean()
        
        prior_probabilities = [
            training_positive_rate
        ] * len(y_val)
        
        prior_log_loss = log_loss(y_val, prior_probabilities)
        
        st.write("Validation ROC-AUC:", round(val_auc, 4))
        st.write("Validation Log Loss:", round(val_log_loss, 4))
        st.write("Prior-Only Baseline Log Loss:", round(prior_log_loss, 4))
        
        # Final test predictions — retained for final evaluation
        y_pred = (y_test_prob >= selected_threshold).astype(int)

        # Positive-return prediction rates
        st.subheader("Model Diagnostics: Positive-Return Rates")
        
        diagnostic_sets = [
            ("Training", y_train, y_train_pred),
            ("Validation", y_val, y_val_pred),
            ("Test", y_test, y_pred)
        ]
        
        for name, actual, predicted in diagnostic_sets:
            st.write(f"{name} actual positive-return rate:",
                     round(actual.mean() * 100, 2), "%")
            st.write(f"{name} predicted positive-return rate:",
                     round(predicted.mean() * 100, 2), "%")

        # Predicted probability diagnostics
        st.subheader("Model Diagnostics: Predicted Probabilities")
        
        probability_sets = [
            ("Training", y_train_prob),
            ("Validation", y_val_prob),
            ("Test", y_test_prob)
        ]
        
        for name, probabilities in probability_sets:
            st.write(f"{name} probability statistics")
            st.write({
                "Minimum": round(probabilities.min(), 4),
                "Maximum": round(probabilities.max(), 4),
                "Mean": round(probabilities.mean(), 4),
                "Standard deviation": round(probabilities.std(), 4)
            })


        #evaluate the predictions
        st.subheader("Model Evaluation")
        
        accuracy=accuracy_score(y_test, y_pred)
        st.write("Test Accuracy: ", round(accuracy * 100, 2), "%")

        st.write("Confusion Matrix: ")
        st.write(confusion_matrix(y_test, y_pred, labels=[0, 1]))

        st.subheader("Detailed Classification Metrics")

        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Non-positive", "Positive"],
            output_dict=True,
            zero_division=0
        )
        
        st.dataframe(pd.DataFrame(report).transpose().round(3))

        
        #compare accuracy based on baseline
        majority_class = y_train.value_counts().idxmax()
        baseline_predictions = [majority_class] * len(y_test)
        baseline_accuracy = accuracy_score(y_test, baseline_predictions)
        
        st.markdown("#### Model vs Baseline")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Logistic Regression Accuracy",
                f"{accuracy:.2%}"
            )
        
        with col2:
            st.metric(
                "Majority-Class Baseline",
                f"{baseline_accuracy:.2%}"
            )
        
        if accuracy > baseline_accuracy:
            st.success("The model outperforms the majority-class baseline.")
        elif accuracy == baseline_accuracy:
            st.info("The model matches the majority-class baseline.")
        else:
            st.warning("The model performs below the majority-class baseline.")


        
        # WALK-FORWARD VALIDATION

        st.divider()
        st.subheader("Walk-Forward Validation")

        n = len(X)

        initial_train_size = int(n * 0.50)
        validation_size = int(n * 0.10)
        test_size = int(n * 0.10)

        thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

        walk_forward_results = []

        train_end = initial_train_size
        fold = 1

        while train_end + validation_size + test_size <= n:

            # Chronological indices
            val_start = train_end
            val_end = val_start + validation_size

            test_start = val_end
            test_end = test_start + test_size

            # Split the data
            X_train_fold = X.iloc[:train_end]
            X_val_fold = X.iloc[val_start:val_end]
            X_test_fold = X.iloc[test_start:test_end]

            y_train_fold = y.iloc[:train_end]
            y_val_fold = y.iloc[val_start:val_end]
            y_test_fold = y.iloc[test_start:test_end]


            # Check whether validation and test contain both classes
            if y_val_fold.nunique() < 2 or y_test_fold.nunique() < 2:
                st.warning(
                    f"Fold {fold} skipped: validation or test "
                    "contains only one class."
                )
                train_end += test_size
                fold += 1
                continue

    
            # Ensure training data contains both classes
            if y_train_fold.nunique() < 2:
                st.warning(f"Fold {fold} skipped: training has only one class.")
                train_end += test_size
                fold += 1
                continue

            # Scale features using training data only
            fold_scaler = StandardScaler()

            X_train_fold_scaled = fold_scaler.fit_transform(X_train_fold)
            X_val_fold_scaled = fold_scaler.transform(X_val_fold)
            X_test_fold_scaled = fold_scaler.transform(X_test_fold)

            # Train a fresh model
            fold_model = LogisticRegression(max_iter=1000)

            fold_model.fit(
                X_train_fold_scaled,
                y_train_fold
            )

            # Validation probabilities
            val_prob = fold_model.predict_proba(
                X_val_fold_scaled
            )[:, 1]

            # Select threshold using validation Macro F1
            best_threshold = 0.50
            best_val_f1 = -1

            for threshold in thresholds:

                val_predictions = (
                    val_prob >= threshold
                ).astype(int)

                val_f1 = f1_score(
                    y_val_fold,
                    val_predictions,
                    labels=[0, 1],
                    average="macro",
                    zero_division=0
                )

                if (
                    val_f1 > best_val_f1
                    or (
                        val_f1 == best_val_f1
                        and abs(threshold - 0.50)
                        < abs(best_threshold - 0.50)
                    )
                ):
                    best_val_f1 = val_f1
                    best_threshold = threshold

            # Evaluate on unseen test data
            test_prob = fold_model.predict_proba(
                X_test_fold_scaled
            )[:, 1]

            test_predictions = (
                test_prob >= best_threshold
            ).astype(int)

            # Model metrics
            test_accuracy = accuracy_score(
                y_test_fold,
                test_predictions
            )

            test_f1 = f1_score(
                y_test_fold,
                test_predictions,
                labels=[0, 1],
                average="macro",
                zero_division=0
            )

            # Majority-class baseline
            majority_class_fold = (
                y_train_fold.value_counts().idxmax()
            )

            baseline_predictions = [
                majority_class_fold
            ] * len(y_test_fold)

            baseline_accuracy = accuracy_score(
                y_test_fold,
                baseline_predictions
            )

            # Save fold results
            walk_forward_results.append({
                "Fold": fold,
                "Training Rows": len(X_train_fold),
                "Validation Rows": len(X_val_fold),
                "Testing Rows": len(X_test_fold),
                "Test Start": str(X_test_fold.index[0])[:10],
                "Test End": str(X_test_fold.index[-1])[:10],
                "Selected Threshold": best_threshold,
                "Test Accuracy": test_accuracy,
                "Test Macro F1": test_f1,
                "Baseline Accuracy": baseline_accuracy
            })

            # Move forward in time
            train_end += test_size
            fold += 1

        # Display results
        if walk_forward_results:

            walk_forward_df = pd.DataFrame(
                walk_forward_results
            )

            st.subheader("Individual Fold Results")

            st.dataframe(
                walk_forward_df.round(4),
                use_container_width=True
            )

            # Average performance
            st.subheader("Overall Walk-Forward Performance")

            avg_accuracy = walk_forward_df[
                "Test Accuracy"
            ].mean()

            avg_f1 = walk_forward_df[
                "Test Macro F1"
            ].mean()

            avg_baseline = walk_forward_df[
                "Baseline Accuracy"
            ].mean()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Average Test Accuracy",
                    f"{avg_accuracy:.2%}"
                )

            with col2:
                st.metric(
                    "Average Macro F1",
                    f"{avg_f1:.3f}"
                )

            with col3:
                st.metric(
                    "Average Baseline Accuracy",
                    f"{avg_baseline:.2%}"
                )

            st.subheader("Walk-Forward Model vs Baseline")

            if avg_accuracy > avg_baseline:
                st.success(
                    "Model outperforms the baseline on average."
                )
            elif avg_accuracy == avg_baseline:
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


        
