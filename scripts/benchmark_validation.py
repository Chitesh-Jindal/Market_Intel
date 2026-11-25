"""Reproduce the five-stock offline benchmark from benchmark_data.zip.

Usage:
    python -m scripts.benchmark_validation --input /path/to/benchmark_data.zip

The script does not download data. It uses the OHLCV CSVs exported from Yahoo
Finance and included in the supplied benchmark archive.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd

from market_intel.technical_analysis import calculate_technical_indicators
from market_intel.ml_pipeline import build_ml_dataset, train_and_evaluate, walk_forward_evaluate
from market_intel.evaluation import evaluate_test
from market_intel.backtesting import run_backtest
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="benchmark_data.zip", help="Path to benchmark_data.zip")
    parser.add_argument("--output", default="benchmark_output", help="Output folder")
    parser.add_argument(
        "--tickers", nargs="*", default=None,
        help="Optional list such as RELIANCE.NS TCS.NS. By default all CSVs are tested."
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Put benchmark_data.zip beside this script or pass --input."
        )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    holdout_rows, fold_rows, backtest_rows = [], [], []
    with zipfile.ZipFile(input_path) as archive, threadpool_limits(limits=1):
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("No CSV files were found in the benchmark ZIP.")
        if args.tickers:
            requested = {ticker.replace(".", "_") for ticker in args.tickers}
            csv_names = [
                name for name in csv_names
                if Path(name).stem in requested
            ]
            if not csv_names:
                raise ValueError("None of the requested tickers were found in the ZIP.")

        for csv_name in sorted(csv_names):
            ticker = Path(csv_name).stem.replace("_", ".")
            raw = pd.read_csv(archive.open(csv_name), parse_dates=["Date"]).set_index("Date").sort_index()
            required = ["Open", "High", "Low", "Close", "Volume"]
            missing = [col for col in required if col not in raw.columns]
            if missing:
                raise ValueError(f"{ticker}: missing columns {missing}")

            indicators = calculate_technical_indicators(raw[required])
            X, y, labelled, latest = build_ml_dataset(indicators)

            model_result = train_and_evaluate(X, y)
            metrics, baseline, _, _ = evaluate_test(
                model_result["y_test"],
                model_result["test_probabilities"],
                model_result["threshold"],
                model_result["y_train"],
            )
            holdout_rows.append({
                "Ticker": ticker,
                "Rows": len(raw),
                "First date": raw.index.min().date().isoformat(),
                "Last date": raw.index.max().date().isoformat(),
                "Selected model": model_result["model_name"],
                "Feature set": model_result["feature_set_name"],
                "Threshold": model_result["threshold"],
                "Test accuracy": metrics["Accuracy"],
                "Baseline accuracy": baseline["Baseline Accuracy"],
                "Accuracy edge (pp)": (
                    metrics["Accuracy"] - baseline["Baseline Accuracy"]
                ) * 100,
                "Macro F1": metrics["Macro F1"],
                "Balanced accuracy": metrics["Balanced Accuracy"],
                "ROC-AUC": metrics["ROC-AUC"],
            })

            walk = walk_forward_evaluate(X, y, folds=4)
            for row in walk["fold_results"].to_dict("records"):
                fold_rows.append({"Ticker": ticker, **row})

            test_rows = labelled.loc[model_result["X_test"].index]
            _, backtest_metrics, _ = run_backtest(
                test_rows, model_result["test_probabilities"], buy_threshold=0.60
            )
            for row in backtest_metrics.to_dict("records"):
                backtest_rows.append({"Ticker": ticker, **row})

            print(
                f"{ticker}: holdout {metrics['Accuracy']:.1%} vs baseline "
                f"{baseline['Baseline Accuracy']:.1%}; walk-forward "
                f"{walk['summary']['Mean Test Accuracy']:.1%} vs "
                f"{walk['summary']['Mean Baseline Accuracy']:.1%} "
                f"({walk['summary']['Folds Beating Baseline']}/4 folds above baseline)",
                flush=True,
            )

    holdout = pd.DataFrame(holdout_rows)
    folds = pd.DataFrame(fold_rows)
    backtests = pd.DataFrame(backtest_rows)
    holdout.to_csv(out / "holdout_results.csv", index=False)
    folds.to_csv(out / "walk_forward_folds.csv", index=False)
    backtests.to_csv(out / "backtest_results.csv", index=False)

    summary = folds[[
        "Test Accuracy", "Baseline Accuracy", "Accuracy Edge (pp)",
        "Macro F1", "Balanced Accuracy", "ROC-AUC"
    ]].mean()
    wins = int((folds["Accuracy Edge (pp)"] > 0).sum())
    print("\nAggregate across ticker-fold windows")
    print(summary.to_string())
    print(f"Folds beating their majority baseline: {wins}/{len(folds)}")
    print(f"\nCSV results written to: {out.resolve()}")


if __name__ == "__main__":
    main()
