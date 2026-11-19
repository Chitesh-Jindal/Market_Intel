
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    log_loss,
    f1_score
)


# Main model evaluation
def evaluate_model(results):

    # Retrieve data from ML pipeline
    y_train = results["y_train"]
    y_val = results["y_val"]
    y_test = results["y_test"]

    y_train_prob = results["y_train_prob"]
    y_val_prob = results["y_val_prob"]
    y_test_prob = results["y_test_prob"]

    # Threshold analysis using validation probabilities
    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

    threshold_results = []

    for threshold in thresholds:

        predictions = (
            y_val_prob >= threshold
        ).astype(int)

        threshold_results.append({
            "Threshold": threshold,

            "Accuracy": accuracy_score(
                y_val, predictions
            ),

            "Macro F1": f1_score(
                y_val,
                predictions,
                labels=[0, 1],
                average="macro",
                zero_division=0
            ),

            "Positive Predictions (%)": (
                predictions.mean() * 100
            )
        })

    threshold_df = pd.DataFrame(threshold_results)

    # Preserve the existing selected threshold
    selected_threshold = 0.55

    # Generate predictions
    y_train_pred = (
        y_train_prob >= selected_threshold
    ).astype(int)

    y_val_pred = (
        y_val_prob >= selected_threshold
    ).astype(int)

    y_test_pred = (
        y_test_prob >= selected_threshold
    ).astype(int)

    # Validation prediction diagnostics
    validation_results = pd.DataFrame({
        "Actual Target": y_val,
        "Predicted Probability": y_val_prob,
        "Predicted Target": y_val_pred
    })

    # Validation accuracy
    val_accuracy = accuracy_score(
        y_val, y_val_pred
    )

    # Majority-class baseline from training data
    val_baseline_class = y_train.value_counts().idxmax()

    val_baseline_predictions = [
        val_baseline_class
    ] * len(y_val)

    val_baseline_accuracy = accuracy_score(
        y_val, val_baseline_predictions
    )

    # Validation confusion matrix
    val_confusion_matrix = confusion_matrix(
        y_val, y_val_pred, labels=[0, 1]
    )

    # Class distributions
    training_class_distribution = (
        y_train.value_counts().sort_index().rename({
            0: "Non-positive return (0)",
            1: "Positive return (1)"
        })
    )

    validation_class_distribution = (
        y_val.value_counts().sort_index().rename({
            0: "Non-positive return (0)",
            1: "Positive return (1)"
        })
    )

    # Validation probability metrics
    val_auc = roc_auc_score(
        y_val, y_val_prob
    )

    val_log_loss = log_loss(
        y_val, y_val_prob
    )

    training_positive_rate = y_train.mean()

    prior_probabilities = [
        training_positive_rate
    ] * len(y_val)

    prior_log_loss = log_loss(
        y_val, prior_probabilities
    )

    # Positive-return prediction rates
    diagnostic_sets = [
        ("Training", y_train, y_train_pred),
        ("Validation", y_val, y_val_pred),
        ("Test", y_test, y_test_pred)
    ]

    positive_return_rates = {}

    for name, actual, predicted in diagnostic_sets:

        positive_return_rates[name] = {
            "Actual Positive-Return Rate (%)": (
                actual.mean() * 100
            ),

            "Predicted Positive-Return Rate (%)": (
                predicted.mean() * 100
            )
        }

    # Predicted probability diagnostics
    probability_sets = [
        ("Training", y_train_prob),
        ("Validation", y_val_prob),
        ("Test", y_test_prob)
    ]

    probability_statistics = {}

    for name, probabilities in probability_sets:

        probability_statistics[name] = {
            "Minimum": probabilities.min(),
            "Maximum": probabilities.max(),
            "Mean": probabilities.mean(),
            "Standard deviation": probabilities.std()
        }

    # Test evaluation
    test_accuracy = accuracy_score(
        y_test, y_test_pred
    )

    test_confusion_matrix = confusion_matrix(
        y_test, y_test_pred, labels=[0, 1]
    )

    # Detailed classification metrics
    report = classification_report(
        y_test,
        y_test_pred,
        labels=[0, 1],
        target_names=["Non-positive", "Positive"],
        output_dict=True,
        zero_division=0
    )

    classification_metrics = (
        pd.DataFrame(report).transpose()
    )

    # Majority-class baseline
    majority_class = y_train.value_counts().idxmax()

    baseline_predictions = [
        majority_class
    ] * len(y_test)

    baseline_accuracy = accuracy_score(
        y_test, baseline_predictions
    )

    # Return evaluation results
    return {
        "selected_threshold": selected_threshold,

        "threshold_df": threshold_df,

        "y_train_pred": y_train_pred,
        "y_val_pred": y_val_pred,
        "y_test_pred": y_test_pred,

        "validation_results": validation_results,

        "val_accuracy": val_accuracy,
        "val_baseline_class": val_baseline_class,
        "val_baseline_accuracy": val_baseline_accuracy,
        "val_confusion_matrix": val_confusion_matrix,

        "training_class_distribution": training_class_distribution,
        "validation_class_distribution": validation_class_distribution,

        "val_auc": val_auc,
        "val_log_loss": val_log_loss,
        "prior_log_loss": prior_log_loss,

        "positive_return_rates": positive_return_rates,
        "probability_statistics": probability_statistics,

        "test_accuracy": test_accuracy,
        "test_confusion_matrix": test_confusion_matrix,
        "classification_metrics": classification_metrics,

        "majority_class": majority_class,
        "baseline_accuracy": baseline_accuracy
    }


# Walk-forward validation
def walk_forward_validation(X, y):

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

            train_end += test_size
            fold += 1
            continue

        # Ensure training data contains both classes
        if y_train_fold.nunique() < 2:

            train_end += test_size
            fold += 1
            continue

        # Scale using training data only
        fold_scaler = StandardScaler()

        X_train_fold_scaled = fold_scaler.fit_transform(
            X_train_fold
        )

        X_val_fold_scaled = fold_scaler.transform(
            X_val_fold
        )

        X_test_fold_scaled = fold_scaler.transform(
            X_test_fold
        )

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

        # Test accuracy
        test_accuracy = accuracy_score(
            y_test_fold,
            test_predictions
        )

        # Test Macro F1
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

    # Prepare results
    if walk_forward_results:

        walk_forward_df = pd.DataFrame(
            walk_forward_results
        )

        avg_accuracy = walk_forward_df[
            "Test Accuracy"
        ].mean()

        avg_f1 = walk_forward_df[
            "Test Macro F1"
        ].mean()

        avg_baseline = walk_forward_df[
            "Baseline Accuracy"
        ].mean()

        return {
            "results": walk_forward_df,
            "avg_accuracy": avg_accuracy,
            "avg_f1": avg_f1,
            "avg_baseline": avg_baseline
        }

    return None

