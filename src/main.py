from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from backtest import aggregate_backtest, summarize_strategy
from config import DEFAULT_TICKERS, ExperimentConfig
from data import (
    FeatureWindowDataset,
    build_yfinance_feature_panel,
    download_ohlc,
    latest_windows_from_panel,
    load_returns_baseline,
)
from model import StockTransformerEncoder, extract_embeddings, train_model
from pairs import (
    pair_relationship_table,
    select_pairs_by_cointegration,
    select_pairs_by_embedding,
    select_pairs_by_johansen,
)


def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    tickers = tuple([t.strip().upper() for t in args.tickers.split(",") if t.strip()])
    if not tickers and not args.returns_csv:
        tickers = DEFAULT_TICKERS
    return ExperimentConfig(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        train_end_date=args.train_end_date,
        window_size=args.window_size,
        embedding_dim=args.embedding_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        top_k_pairs=args.top_k_pairs,
        z_entry=args.z_entry,
        z_exit=args.z_exit,
    )


def run_experiment(cfg: ExperimentConfig, args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.returns_csv:
        print(f"Loading baseline returns dataset from: {args.returns_csv}")
        close_df, ret_df, feature_panel = load_returns_baseline(
            returns_csv_path=args.returns_csv,
            tickers=list(cfg.tickers) if cfg.tickers else None,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            max_stocks=args.max_stocks,
        )
    else:
        print("Downloading market data from yfinance...")
        raw = download_ohlc(list(cfg.tickers), cfg.start_date, cfg.end_date)
        close_df, ret_df, feature_panel = build_yfinance_feature_panel(raw, list(cfg.tickers))

    print(f"Data shape | close={close_df.shape} returns={ret_df.shape}")
    train_ds = FeatureWindowDataset(
        feature_panel=feature_panel,
        returns=ret_df,
        train_end_date=cfg.train_end_date,
        window_size=cfg.window_size,
        train=True,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch, _ = next(iter(train_loader))
    model = StockTransformerEncoder(
        input_dim=sample_batch.shape[-1],
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        embedding_dim=cfg.embedding_dim,
    )
    print(f"Training transformer on {device}...")
    train_model(
        model,
        train_loader,
        epochs=cfg.epochs,
        lr=cfg.lr,
        device=device,
        objective=args.objective,
        contrastive_weight=args.contrastive_weight,
    )

    tickers, windows_np = latest_windows_from_panel(
        feature_panel=feature_panel,
        window_size=cfg.window_size,
        end_date=cfg.train_end_date,
    )
    windows = torch.from_numpy(windows_np)
    embeddings = extract_embeddings(model, windows, device=device).numpy()

    emb_pairs_scored = select_pairs_by_embedding(tickers, embeddings, cfg.top_k_pairs)
    emb_pairs = [(a, b) for a, b, _ in emb_pairs_scored]

    close_train = close_df.loc[: cfg.train_end_date]
    engle_scored = select_pairs_by_cointegration(close_train, tickers, cfg.top_k_pairs)
    johansen_scored = select_pairs_by_johansen(close_train, tickers, cfg.top_k_pairs)
    engle_pairs = [(a, b) for a, b, _ in engle_scored]
    johansen_pairs = [(a, b) for a, b, _ in johansen_scored]

    if args.baseline_method == "engle":
        baseline_scored = engle_scored
    elif args.baseline_method == "johansen":
        baseline_scored = johansen_scored
    else:
        baseline_scored = engle_scored if len(engle_scored) >= len(johansen_scored) else johansen_scored
    baseline_pairs = [(a, b) for a, b, _ in baseline_scored]

    emb_returns = aggregate_backtest(
        close_df=close_df,
        pairs=emb_pairs,
        train_end_date=cfg.train_end_date,
        z_entry=cfg.z_entry,
        z_exit=cfg.z_exit,
    )
    engle_returns = aggregate_backtest(
        close_df=close_df,
        pairs=engle_pairs,
        train_end_date=cfg.train_end_date,
        z_entry=cfg.z_entry,
        z_exit=cfg.z_exit,
    )
    johansen_returns = aggregate_backtest(
        close_df=close_df,
        pairs=johansen_pairs,
        train_end_date=cfg.train_end_date,
        z_entry=cfg.z_entry,
        z_exit=cfg.z_exit,
    )
    baseline_returns = aggregate_backtest(
        close_df=close_df,
        pairs=baseline_pairs,
        train_end_date=cfg.train_end_date,
        z_entry=cfg.z_entry,
        z_exit=cfg.z_exit,
    )

    emb_metrics = summarize_strategy(emb_returns)
    engle_metrics = summarize_strategy(engle_returns)
    johansen_metrics = summarize_strategy(johansen_returns)
    baseline_metrics = summarize_strategy(baseline_returns)

    print("\nTop embedding pairs (ticker1, ticker2, cosine_similarity):")
    for row in emb_pairs_scored:
        print(row)

    print("\nTop Engle-Granger pairs (ticker1, ticker2, pvalue):")
    for row in engle_scored:
        print(row)

    print("\nTop Johansen pairs (ticker1, ticker2, trace_stat):")
    for row in johansen_scored:
        print(row)

    print(f"\nTop baseline pairs [{args.baseline_method}] (ticker1, ticker2, score):")
    for row in baseline_scored:
        print(row)

    print("\nOut-of-sample results")
    print(f"Embedding strategy Sharpe:    {emb_metrics['sharpe']:.4f}")
    print(f"Engle-Granger Sharpe:         {engle_metrics['sharpe']:.4f}")
    print(f"Johansen Sharpe:              {johansen_metrics['sharpe']:.4f}")
    print(f"Baseline strategy Sharpe:     {baseline_metrics['sharpe']:.4f}")
    print(f"Embedding total return:       {emb_metrics['total_return']:.2%}")
    print(f"Engle-Granger total return:   {engle_metrics['total_return']:.2%}")
    print(f"Johansen total return:        {johansen_metrics['total_return']:.2%}")

    emb_pairs_df = pd.DataFrame(emb_pairs_scored, columns=["ticker_1", "ticker_2", "embedding_cosine"])
    engle_pairs_df = pd.DataFrame(engle_scored, columns=["ticker_1", "ticker_2", "engle_granger_pvalue"])
    johansen_pairs_df = pd.DataFrame(johansen_scored, columns=["ticker_1", "ticker_2", "johansen_trace_stat"])
    base_pairs_df = pd.DataFrame(
        baseline_scored,
        columns=["ticker_1", "ticker_2", f"{args.baseline_method}_score"],
    )
    relation_df = pair_relationship_table(emb_pairs_scored, close_df)
    perf_df = pd.DataFrame(
        [
            {"strategy": "embedding", **emb_metrics},
            {"strategy": "engle_granger", **engle_metrics},
            {"strategy": "johansen", **johansen_metrics},
            {"strategy": f"baseline_{args.baseline_method}", **baseline_metrics},
        ]
    )
    returns_df = pd.concat(
        [
            emb_returns.rename("embedding"),
            engle_returns.rename("engle_granger"),
            johansen_returns.rename("johansen"),
        ],
        axis=1,
    ).fillna(0.0)
    equity_df = (1 + returns_df).cumprod()

    emb_pairs_df.to_csv(out_dir / "embedding_pairs.csv", index=False)
    engle_pairs_df.to_csv(out_dir / "engle_granger_pairs.csv", index=False)
    johansen_pairs_df.to_csv(out_dir / "johansen_pairs.csv", index=False)
    base_pairs_df.to_csv(out_dir / "baseline_pairs.csv", index=False)
    relation_df.to_csv(out_dir / "embedding_vs_corr_vs_coint.csv", index=False)
    perf_df.to_csv(out_dir / "performance_summary.csv", index=False)
    returns_df.to_csv(out_dir / "daily_returns.csv")
    equity_df.to_csv(out_dir / "equity_curves.csv")
    _save_plots(equity_df, perf_df, out_dir)
    print(f"\nSaved report artifacts to: {out_dir.resolve()}")


def _save_plots(equity_df: pd.DataFrame, perf_df: pd.DataFrame, out_dir: Path) -> None:
    plt.style.use("default")

    fig, ax = plt.subplots(figsize=(10, 5))
    equity_df.plot(ax=ax)
    ax.set_title("Out-of-Sample Equity Curves")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "equity_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    perf_plot = perf_df.set_index("strategy")[["sharpe", "total_return"]]
    perf_plot.plot(kind="bar", ax=ax)
    ax.set_title("Strategy Performance Summary")
    ax.set_xlabel("Strategy")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "performance_summary.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stock embeddings for pairs trading")
    parser.add_argument("--tickers", type=str, default="")
    parser.add_argument("--start-date", type=str, default="2000-01-01")
    parser.add_argument("--end-date", type=str, default="2018-12-31")
    parser.add_argument("--train-end-date", type=str, default="2014-12-31")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--top-k-pairs", type=int, default=10)
    parser.add_argument("--z-entry", type=float, default=1.5)
    parser.add_argument("--z-exit", type=float, default=0.2)
    parser.add_argument("--returns-csv", type=str, default="")
    parser.add_argument("--max-stocks", type=int, default=0)
    parser.add_argument("--baseline-method", choices=["engle", "johansen", "best"], default="engle")
    parser.add_argument("--objective", choices=["mse", "mse_contrastive"], default="mse")
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    pd.options.display.float_format = "{:.6f}".format
    args = parse_args()
    cfg = build_config_from_args(args)
    if args.max_stocks <= 0:
        args.max_stocks = None
    run_experiment(cfg, args)
