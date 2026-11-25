import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_test(y_true, probabilities, threshold, y_train):
    """Calculate held-out test metrics and a training-majority baseline."""
    y_true_array = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "Accuracy": float(accuracy_score(y_true_array, predictions)),
        "Macro F1": float(f1_score(y_true_array, predictions, average="macro", zero_division=0)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true_array, predictions)),
        "Precision (positive class)": float(precision_score(y_true_array, predictions, zero_division=0)),
        "Recall (positive class)": float(recall_score(y_true_array, predictions, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true_array, probabilities))
            if len(np.unique(y_true_array)) == 2 else np.nan,
        "Log Loss": float(log_loss(
            y_true_array, np.clip(probabilities, 1e-7, 1 - 1e-7), labels=[0, 1]
        )),
    }

    majority_class = int(pd.Series(y_train).mode().iloc[0])
    baseline_predictions = np.full(len(y_true_array), majority_class, dtype=int)
    baseline = {
        "Baseline Accuracy": float(accuracy_score(y_true_array, baseline_predictions)),
        "Baseline Macro F1": float(f1_score(
            y_true_array, baseline_predictions, average="macro", zero_division=0
        )),
        "Baseline Class": "Positive" if majority_class == 1 else "Non-positive",
    }

    matrix = confusion_matrix(y_true_array, predictions, labels=[0, 1])
    return metrics, baseline, matrix, predictions


def build_feature_importance(model, X_test, y_test, feature_names):
    """Permutation importance for the selected model's actual feature subset."""
    try:
        result = permutation_importance(
            model, X_test[feature_names], y_test,
            scoring="f1_macro", n_repeats=3,
            random_state=42, n_jobs=1,
        )
    except Exception:
        return pd.DataFrame(columns=["Feature", "Importance", "Method"])

    table = pd.DataFrame({
        "Feature": feature_names,
        "Importance": result.importances_mean,
        "Method": "Permutation importance (mean decrease in Macro F1)",
    })
    return table.sort_values("Importance", ascending=False).reset_index(drop=True)
