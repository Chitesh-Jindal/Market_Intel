import numpy as np
import pandas as pd

TRADING_DAYS = 252
TRANSACTION_COST = 0.001  # 0.10% per side


def _strategy_metrics(equity: pd.Series, daily_returns: pd.Series, name: str,
                     invested: pd.Series):
    daily_returns = daily_returns.dropna()
    if equity.empty:
        return {
            "Strategy": name, "Total Return (%)": 0.0,
            "Max Drawdown (%)": 0.0, "Sharpe Ratio": 0.0,
            "Win Rate (%)": 0.0, "Exposure (%)": 0.0,
        }

    total_return = (equity.iloc[-1] - 1.0) * 100
    high_water = equity.cummax().clip(lower=1.0)
    drawdown = (equity / high_water - 1.0).min() * 100
    volatility = daily_returns.std()
    sharpe = (
        daily_returns.mean() / volatility * np.sqrt(TRADING_DAYS)
        if pd.notna(volatility) and volatility > 0 else 0.0
    )
    active_returns = daily_returns.loc[
        invested.reindex(daily_returns.index).fillna(False).astype(bool)
    ]
    win_rate = (active_returns > 0).mean() * 100 if len(active_returns) else 0.0
    exposure = invested.mean() * 100 if len(invested) else 0.0

    return {
        "Strategy": name,
        "Total Return (%)": float(total_return),
        "Max Drawdown (%)": float(drawdown),
        "Sharpe Ratio": float(sharpe),
        "Win Rate (%)": float(win_rate),
        "Exposure (%)": float(exposure),
    }


def run_backtest(test_rows: pd.DataFrame, probabilities, buy_threshold=0.60,
                 transaction_cost=TRANSACTION_COST):
    """Backtest signals generated at close(t), executed at open(t+1), exited at close(t+1).

    The model target remains close(t) -> close(t+1). The executable backtest
    measures the intraday return available after the signal is known, from the
    next session's open to that session's close. Thus it does not pretend the
    strategy can trade at close(t) after observing that close.
    """
    rows = test_rows.copy().sort_index()
    probabilities = np.asarray(probabilities, dtype=float)
    if len(rows) != len(probabilities):
        raise ValueError("Test rows and prediction probabilities must have equal lengths.")
    if not 0.0 < buy_threshold < 1.0:
        raise ValueError("buy_threshold must be between 0 and 1.")
    if not 0.0 <= transaction_cost < 1.0:
        raise ValueError("transaction_cost must be between 0 and 1.")
    required = ["Next_Session_Open", "Next_Session_Close", "Return_20D"]
    missing = [c for c in required if c not in rows.columns]
    if missing:
        raise ValueError(f"Backtest data is missing required columns: {missing}")

    next_open = pd.to_numeric(rows["Next_Session_Open"], errors="coerce")
    next_close = pd.to_numeric(rows["Next_Session_Close"], errors="coerce")
    intraday_return = (next_close / next_open) - 1.0
    if intraday_return.isna().any() or not np.isfinite(intraday_return.to_numpy()).all():
        raise ValueError("The backtest contains missing or invalid next-session prices.")

    ml_position = pd.Series(probabilities >= buy_threshold, index=rows.index)
    momentum_position = pd.Series(rows["Return_20D"].to_numpy() > 0, index=rows.index)

    def intraday_equity(position: pd.Series):
        # Each signal is a one-session trade: enter at next open, exit at close.
        cost_factor = (1.0 - transaction_cost) ** 2  # entry and exit
        gross = 1.0 + intraday_return
        growth = pd.Series(
            np.where(position.to_numpy(), gross.to_numpy() * cost_factor, 1.0),
            index=rows.index,
        )
        daily = growth - 1.0
        equity = growth.cumprod()
        return daily, equity, position.astype(bool)

    ml_daily, ml_equity, ml_invested = intraday_equity(ml_position)
    mom_daily, mom_equity, mom_invested = intraday_equity(momentum_position)

    # Buy & Hold enters at the first next-session open and remains invested
    # through the final next-session close, including overnight price gaps.
    bh_gross = pd.Series(index=rows.index, dtype=float)
    bh_gross.iloc[0] = next_close.iloc[0] / next_open.iloc[0]
    if len(rows) > 1:
        bh_gross.iloc[1:] = next_close.iloc[1:].to_numpy() / next_close.iloc[:-1].to_numpy()
    bh_growth = bh_gross.copy()
    bh_growth.iloc[0] *= (1.0 - transaction_cost)  # initial entry
    bh_growth.iloc[-1] *= (1.0 - transaction_cost)  # final liquidation
    bh_daily = bh_growth - 1.0
    bh_equity = bh_growth.cumprod()
    bh_invested = pd.Series(True, index=rows.index)

    strategies = {
        "ML Signal": (ml_daily, ml_equity, ml_invested),
        "Buy & Hold": (bh_daily, bh_equity, bh_invested),
        "Momentum": (mom_daily, mom_equity, mom_invested),
    }
    equity_curves = pd.DataFrame({name: value[1] for name, value in strategies.items()})
    metrics = pd.DataFrame([
        _strategy_metrics(value[1], value[0], name, value[2])
        for name, value in strategies.items()
    ])

    signal_counts = {
        "BUY": int(np.sum(probabilities >= buy_threshold)),
        "HOLD": int(np.sum((probabilities > 1 - buy_threshold) & (probabilities < buy_threshold))),
        "SELL / CASH": int(np.sum(probabilities <= 1 - buy_threshold)),
    }
    return equity_curves, metrics, signal_counts
