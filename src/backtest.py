from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore(series: pd.Series, lookback: int = 60) -> pd.Series:
    rolling_mean = series.rolling(lookback).mean()
    rolling_std = series.rolling(lookback).std().replace(0, np.nan)
    return (series - rolling_mean) / rolling_std


def backtest_pair(
    close_df: pd.DataFrame,
    pair: tuple[str, str],
    train_end_date: str,
    z_entry: float = 1.5,
    z_exit: float = 0.2,
) -> pd.Series:
    t1, t2 = pair
    px = close_df[[t1, t2]].dropna()
    px = px.loc[px.index > pd.Timestamp(train_end_date)]
    if len(px) < 100:
        return pd.Series(dtype=float)

    log_s1 = np.log(px[t1])
    log_s2 = np.log(px[t2])
    spread = log_s1 - log_s2
    z = _zscore(spread).fillna(0.0)

    position = pd.Series(0.0, index=px.index)
    for i in range(1, len(position)):
        prev = position.iloc[i - 1]
        zi = z.iloc[i]
        if prev == 0.0:
            if zi > z_entry:
                position.iloc[i] = -1.0
            elif zi < -z_entry:
                position.iloc[i] = 1.0
            else:
                position.iloc[i] = 0.0
        else:
            if abs(zi) < z_exit:
                position.iloc[i] = 0.0
            else:
                position.iloc[i] = prev

    ret1 = px[t1].pct_change().fillna(0.0)
    ret2 = px[t2].pct_change().fillna(0.0)
    pnl = position.shift(1).fillna(0.0) * (ret1 - ret2)
    return pnl


def aggregate_backtest(
    close_df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    train_end_date: str,
    z_entry: float = 1.5,
    z_exit: float = 0.2,
) -> pd.Series:
    pnl_frames: list[pd.Series] = []
    for pair in pairs:
        pnl = backtest_pair(close_df, pair, train_end_date, z_entry=z_entry, z_exit=z_exit)
        if not pnl.empty:
            pnl_frames.append(pnl)

    if not pnl_frames:
        return pd.Series(dtype=float)

    aligned = pd.concat(pnl_frames, axis=1).fillna(0.0)
    return aligned.mean(axis=1)


def sharpe_ratio(daily_returns: pd.Series, annualization: int = 252) -> float:
    if daily_returns.empty:
        return float("nan")
    mu = daily_returns.mean()
    sigma = daily_returns.std()
    if sigma == 0 or np.isnan(sigma):
        return float("nan")
    return np.sqrt(annualization) * mu / sigma


def max_drawdown(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return float("nan")
    equity = (1 + daily_returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def summarize_strategy(daily_returns: pd.Series, annualization: int = 252) -> dict[str, float]:
    if daily_returns.empty:
        return {
            "sharpe": float("nan"),
            "total_return": float("nan"),
            "max_drawdown": float("nan"),
            "volatility": float("nan"),
            "mean_daily_return": float("nan"),
        }

    equity = (1 + daily_returns).cumprod()
    return {
        "sharpe": sharpe_ratio(daily_returns, annualization=annualization),
        "total_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": max_drawdown(daily_returns),
        "volatility": float(daily_returns.std() * np.sqrt(annualization)),
        "mean_daily_return": float(daily_returns.mean()),
    }
