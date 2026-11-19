
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    log_loss
)

from ml_pipeline import (
    get_models,
    EXISTING_FEATURES,
    ALL_FEATURES
)


# --------------------------------------------------
# 1. CLASSIFICATION THRESHOLD SELECTION
# --------------------------------------------------

THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]


def select_threshold(y_true, probabilities):
    """
    Select the threshold that gives the best Macro F1
    on validation data.

    If multiple thresholds have the same Macro F1,
    prefer the one closest to 0.50.
    """

    best_threshold = 0.50
    best_f1 = -1

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)

        score = f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        )

        if (
            score > best_f1
            or (
                score == best_f1
                and abs(threshold - 0.50)
                < abs(best_threshold - 0.50)
            )
        ):
            best_f1 = score
            best_threshold = threshold

    return best_threshold, best_f1


# --------------------------------------------------
# 2. MODEL EVALUATION
# --------------------------------------------------

def calculate_metrics(y_true, probabilities, threshold):
    """
    Calculate classification metrics for a set of
    predicted probabilities and a selected threshold.
    """

    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_true, predictions),
        "Macro F1": f1_score(
            y_true, predictions,
            average="macro",
            zero_division=0
        ),
        "Balanced Accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "Precision": precision_score(
            y_true, predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions,
            zero_division=0
        )
    }

    # ROC-AUC requires both classes to be present.
    if len(np.unique(y_true)) == 2:
        metrics["ROC-AUC"] = roc_auc_score(
            y_true, probabilities
        )
    else:
        metrics["ROC-AUC"] = np.nan

    # Clip probabilities to avoid log(0).
    clipped_probabilities = np.clip(
        probabilities, 1e-7, 1 - 1e-7
    )

    metrics["Log Loss"] = log_loss(
        y_true,
        clipped_probabilities,
        labels=[0, 1]
    )

    return metrics


# --------------------------------------------------
# 3. MAJORITY-CLASS BASELINE
# --------------------------------------------------

def calculate_baseline(y_train, y_test):
    """
    Predict the most common class in the training data
    for every observation in the test period.
    """

    majority_class = int(y_train.mode().iloc[0])

    predictions = np.full(
        len(y_test),
        majority_class
    )

    return {
        "Baseline Accuracy": accuracy_score(
            y_test, predictions
        ),
        "Baseline Macro F1": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )
    }


# --------------------------------------------------
# 4. WALK-FORWARD VALIDATION
# --------------------------------------------------

def walk_forward_validation(X, y, feature_sets=None):
    """
    Compare multiple machine-learning models using
    expanding-window walk-forward validation.

    Each fold has:
        - Expanding training period
        - Validation period
        - Unseen test period

    The threshold is selected on validation data.
    The selected threshold is then evaluated on test data.
    """

    if feature_sets is None:
        feature_sets = {
            "Original 7 features": EXISTING_FEATURES,
            "Expanded features": ALL_FEATURES
        }

    X = X.copy()
    y = y.copy()

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")

    if not X.index.equals(y.index):
        raise ValueError("X and y must have matching indexes.")

    n = len(X)

    initial_train_size = int(n * 0.50)
    validation_size = int(n * 0.10)
    test_size = int(n * 0.10)

    if (
        initial_train_size < 1
        or validation_size < 1
        or test_size < 1
    ):
        raise ValueError(
            "Not enough observations for walk-forward validation."
        )

    models = get_models()

    fold_results = []
    skipped_folds = []

    fold_number = 1
    train_end = initial_train_size

    while train_end + validation_size + test_size <= n:

        validation_start = train_end
        validation_end = validation_start + validation_size

        test_start = validation_end
        test_end = test_start + test_size

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]

        X_val = X.iloc[validation_start:validation_end]
        y_val = y.iloc[validation_start:validation_end]

        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]

        # Every model and feature set uses these same dates.
        if (
            y_train.nunique() < 2
            or y_val.nunique() < 2
            or y_test.nunique() < 2
        ):
            skipped_folds.append({
                "Fold": fold_number,
                "Reason": "Training, validation, or test period has only one class."
            })

            fold_number += 1
            train_end += test_size
            continue

        for feature_set_name, feature_columns in feature_sets.items():

            missing_columns = [
                column for column in feature_columns
                if column not in X.columns
            ]

            if missing_columns:
                raise ValueError(
                    f"Missing features for {feature_set_name}: "
                    f"{missing_columns}"
                )

            X_train_features = X_train[feature_columns]
            X_val_features = X_val[feature_columns]
            X_test_features = X_test[feature_columns]

            for model_name, base_model in models.items():

                try:
                    # Create a fresh model for every fold.
                    model = clone(base_model)

                    model.fit(
                        X_train_features,
                        y_train
                    )

                    val_probabilities = model.predict_proba(
                        X_val_features
                    )[:, 1]

                    test_probabilities = model.predict_proba(
                        X_test_features
                    )[:, 1]

                    # Select threshold using validation data only.
                    selected_threshold, validation_macro_f1 = (
                        select_threshold(
                            y_val,
                            val_probabilities
                        )
                    )

                    # Evaluate on the unseen test period.
                    test_metrics = calculate_metrics(
                        y_test,
                        test_probabilities,
                        selected_threshold
                    )

                    baseline_metrics = calculate_baseline(
                        y_train,
                        y_test
                    )

                    result = {
                        "Fold": fold_number,
                        "Feature Set": feature_set_name,
                        "Model": model_name,
                        "Training Rows": len(X_train),
                        "Validation Rows": len(X_val),
                        "Test Rows": len(X_test),
                        "Selected Threshold": selected_threshold,
                        "Validation Macro F1": validation_macro_f1
                    }

                    result.update(test_metrics)
                    result.update(baseline_metrics)

                    fold_results.append(result)

                except Exception as error:
                    skipped_folds.append({
                        "Fold": fold_number,
                        "Feature Set": feature_set_name,
                        "Model": model_name,
                        "Reason": str(error)
                    })

        fold_number += 1

        # Expand training data by one test window.
        train_end += test_size

    if not fold_results:
        return {
            "fold_results": pd.DataFrame(),
            "summary": pd.DataFrame(),
            "skipped_folds": pd.DataFrame(skipped_folds)
        }

    results_df = pd.DataFrame(fold_results)

    # Aggregate performance across all successful folds.
    summary_df = (
        results_df
        .groupby(["Feature Set", "Model"])
        .agg(
            Folds=("Fold", "count"),
            Average_Accuracy=("Accuracy", "mean"),
            Average_Macro_F1=("Macro F1", "mean"),
            Average_Balanced_Accuracy=("Balanced Accuracy", "mean"),
            Average_ROC_AUC=("ROC-AUC", "mean"),
            Average_Log_Loss=("Log Loss", "mean"),
            Average_Baseline_Accuracy=("Baseline Accuracy", "mean"),
            Average_Baseline_Macro_F1=("Baseline Macro F1", "mean")
        )
        .reset_index()
    )

    summary_df = summary_df.sort_values(
        by="Average_Macro_F1",
        ascending=False
    ).reset_index(drop=True)

    return {
        "fold_results": results_df,
        "summary": summary_df,
        "skipped_folds": pd.DataFrame(skipped_folds)
    }
