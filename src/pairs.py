from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def _cosine_similarity_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    x_norm = x / norms
    return x_norm @ x_norm.T


def select_pairs_by_embedding(
    tickers: list[str],
    embeddings: np.ndarray,
    top_k: int,
) -> list[tuple[str, str, float]]:
    sim = _cosine_similarity_matrix(embeddings)
    pairs: list[tuple[str, str, float]] = []

    for i, j in combinations(range(len(tickers)), 2):
        pairs.append((tickers[i], tickers[j], float(sim[i, j])))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def select_pairs_by_cointegration(
    close_df: pd.DataFrame,
    candidate_tickers: Iterable[str],
    top_k: int,
) -> list[tuple[str, str, float]]:
    tickers = [t for t in candidate_tickers if t in close_df.columns]
    results: list[tuple[str, str, float]] = []

    for t1, t2 in combinations(tickers, 2):
        s1 = close_df[t1].dropna()
        s2 = close_df[t2].dropna()
        common = s1.index.intersection(s2.index)
        if len(common) < 120:
            continue
        try:
            stat, pvalue, _ = coint(s1.loc[common], s2.loc[common])
            results.append((t1, t2, float(pvalue)))
        except Exception:
            continue

    results.sort(key=lambda x: x[2])  # lower p-value is stronger evidence
    return results[:top_k]


def select_pairs_by_johansen(
    close_df: pd.DataFrame,
    candidate_tickers: Iterable[str],
    top_k: int,
) -> list[tuple[str, str, float]]:
    tickers = [t for t in candidate_tickers if t in close_df.columns]
    results: list[tuple[str, str, float]] = []

    for t1, t2 in combinations(tickers, 2):
        pair_df = close_df[[t1, t2]].dropna()
        if len(pair_df) < 120:
            continue
        try:
            joh = coint_johansen(pair_df, det_order=0, k_ar_diff=1)
            trace_stat = float(joh.lr1[0])  # larger is stronger evidence
            results.append((t1, t2, trace_stat))
        except Exception:
            continue

    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_k]


def pair_relationship_table(
    pair_scores: list[tuple[str, str, float]],
    close_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for t1, t2, emb_sim in pair_scores:
        df = close_df[[t1, t2]].dropna()
        if len(df) < 30:
            continue
        corr = float(df[t1].pct_change().corr(df[t2].pct_change()))
        try:
            _, pvalue, _ = coint(df[t1], df[t2])
            coint_score = float(pvalue)
        except Exception:
            coint_score = float("nan")
        try:
            joh = coint_johansen(df[[t1, t2]], det_order=0, k_ar_diff=1)
            johansen_trace = float(joh.lr1[0])
        except Exception:
            johansen_trace = float("nan")

        rows.append(
            {
                "ticker_1": t1,
                "ticker_2": t2,
                "embedding_cosine": emb_sim,
                "return_corr": corr,
                "engle_granger_pvalue": coint_score,
                "johansen_trace_stat": johansen_trace,
            }
        )
    return pd.DataFrame(rows)
