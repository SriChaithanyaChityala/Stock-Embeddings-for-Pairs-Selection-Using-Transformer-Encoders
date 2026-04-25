from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import yfinance as yf


def download_ohlc(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
    )
    if raw.empty:
        raise ValueError("No market data returned from yfinance.")
    return raw


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean()
    sigma = df.std().replace(0.0, np.nan)
    z = (df - mu) / (sigma + 1e-8)
    return z.fillna(0.0)


def build_yfinance_feature_panel(
    raw: pd.DataFrame, tickers: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    close_data: dict[str, pd.Series] = {}
    ret_data: dict[str, pd.Series] = {}
    panel: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        if ticker not in raw.columns.get_level_values(0):
            continue
        df_t = raw[ticker].copy()
        df_t = df_t[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        close = df_t["Close"].astype(float)
        returns = close.pct_change().fillna(0.0)
        close_data[ticker] = close
        ret_data[ticker] = returns
        feats = pd.DataFrame(index=df_t.index)
        feats["ret_1d"] = returns
        feats["oc_move"] = (df_t["Close"] - df_t["Open"]) / (df_t["Open"] + 1e-8)
        feats["hl_range"] = (df_t["High"] - df_t["Low"]) / (df_t["Close"] + 1e-8)
        feats["vol_chg"] = df_t["Volume"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        panel[ticker] = feats

    close_df = pd.DataFrame(close_data).dropna(how="all")
    ret_df = pd.DataFrame(ret_data).dropna(how="all")

    common_index = close_df.index.intersection(ret_df.index)
    close_df = close_df.loc[common_index].sort_index()
    ret_df = ret_df.loc[common_index].sort_index()

    close_df = close_df.ffill().dropna(axis=1)
    ret_df = ret_df[close_df.columns].fillna(0.0)

    clean_panel: dict[str, pd.DataFrame] = {}
    for ticker in close_df.columns:
        feats = panel[ticker].loc[common_index].copy()
        feats = feats.fillna(0.0)
        clean_panel[ticker] = feats
    return close_df, ret_df, clean_panel


def load_returns_baseline(
    returns_csv_path: str,
    tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_stocks: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Load a returns matrix CSV (dates x tickers) and derive close-like series.

    The baseline file is expected to have trading dates in the first column and
    tickers in the remaining columns.
    """
    df = pd.read_csv(returns_csv_path)
    if df.empty:
        raise ValueError("Returns CSV is empty.")

    # First column is expected to be date-like; keep robust fallback behavior.
    idx_col = df.columns[0]
    df[idx_col] = pd.to_datetime(df[idx_col], errors="coerce")
    df = df.dropna(subset=[idx_col]).set_index(idx_col).sort_index()

    ret_df = df.apply(pd.to_numeric, errors="coerce")
    ret_df = ret_df.replace([np.inf, -np.inf], np.nan)
    ret_df = ret_df.dropna(axis=1, how="all")

    if tickers:
        present = [t for t in tickers if t in ret_df.columns]
        if present:
            ret_df = ret_df[present]

    if start_date:
        ret_df = ret_df.loc[ret_df.index >= pd.Timestamp(start_date)]
    if end_date:
        ret_df = ret_df.loc[ret_df.index <= pd.Timestamp(end_date)]

    ret_df = ret_df.fillna(0.0)
    if max_stocks is not None and max_stocks > 0:
        ret_df = ret_df.iloc[:, :max_stocks]

    if ret_df.empty or ret_df.shape[1] < 2:
        raise ValueError("Not enough usable ticker columns in returns CSV.")

    close_df = (1.0 + ret_df).cumprod() * 100.0
    panel: dict[str, pd.DataFrame] = {}
    for ticker in ret_df.columns:
        panel[ticker] = pd.DataFrame({"ret_1d": ret_df[ticker]}, index=ret_df.index)
    return close_df, ret_df, panel


class FeatureWindowDataset(Dataset):
    def __init__(
        self,
        feature_panel: dict[str, pd.DataFrame],
        returns: pd.DataFrame,
        train_end_date: str,
        window_size: int,
        train: bool = True,
    ) -> None:
        self.samples: list[tuple[np.ndarray, float]] = []
        train_end_ts = pd.Timestamp(train_end_date)

        for ticker, feats in feature_panel.items():
            if ticker not in returns.columns:
                continue
            series = returns[ticker].astype(float)
            feat_df = feats.copy().astype(float)

            if train:
                series = series.loc[series.index <= train_end_ts]
                feat_df = feat_df.loc[feat_df.index <= train_end_ts]
            else:
                series = series.loc[series.index > train_end_ts]
                feat_df = feat_df.loc[feat_df.index > train_end_ts]

            common = feat_df.index.intersection(series.index)
            if len(common) <= window_size + 1:
                continue

            series = series.loc[common]
            feat_df = _normalize_df(feat_df.loc[common]).fillna(0.0)
            feat_values = feat_df.to_numpy(dtype=np.float32)
            target_values = series.to_numpy(dtype=np.float32)

            for i in range(window_size, len(common) - 1):
                window = feat_values[i - window_size : i]
                target = target_values[i + 1]
                self.samples.append((window, target))

        if not self.samples:
            split = "train" if train else "test"
            raise ValueError(f"No {split} samples were built. Check dates/window size.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        # Ensure writable contiguous storage for safe tensor conversion.
        x_safe = np.array(x, dtype=np.float32, copy=True)
        return torch.from_numpy(x_safe), torch.tensor([y], dtype=torch.float32)


def latest_windows_from_panel(
    feature_panel: dict[str, pd.DataFrame],
    window_size: int,
    end_date: str,
) -> tuple[list[str], np.ndarray]:
    tickers: list[str] = []
    windows: list[np.ndarray] = []
    end_ts = pd.Timestamp(end_date)

    for ticker, feats in feature_panel.items():
        feat_df = feats.loc[feats.index <= end_ts].copy()
        if len(feat_df) < window_size:
            continue
        feat_df = _normalize_df(feat_df).fillna(0.0)
        windows.append(feat_df.to_numpy(dtype=np.float32)[-window_size:])
        tickers.append(ticker)

    if not windows:
        raise ValueError("No windows available for embedding extraction.")
    return tickers, np.stack(windows, axis=0)
