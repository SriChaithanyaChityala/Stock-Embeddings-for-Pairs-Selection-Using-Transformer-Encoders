# Stock Embeddings for Pairs Trading

This project implements a complete research pipeline for **pairs selection** and **out-of-sample (OOS) backtesting**:

1. Build rolling windows of stock features
2. Train a transformer encoder to produce stock embeddings
3. Select candidate pairs by embedding cosine similarity
4. Compare against statistical baselines (Engle-Granger, Johansen)
5. Backtest all strategies on the same OOS period
6. Export report-ready CSV and PNG artifacts

The bundled 611-stock returns matrix is at:

- `data/returns_df_611.csv`

---

## What This Project Answers

Main question:

- Can learned stock embeddings improve pair selection for mean-reversion trading compared with standard cointegration methods?

What is delivered:

- End-to-end reproducible code in `src/`
- Two data paths (returns matrix and Yahoo Finance)
- Side-by-side strategy comparison on shared OOS backtests
- Exported metrics, pair lists, and plots for report/presentation use

---

## Quick Start

From project root (folder containing `src/` and `requirements.txt`):

```bash
cd ~/Desktop/ML
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/main.py [options]
```

Windows activation:

```bash
.\.venv\Scripts\activate
```

Important shell note:

- Do **not** wrap commands in backticks.

---

## Method Overview

### Inputs

- **Returns path** (`--returns-csv`): matrix with first column as date, remaining columns as ticker returns
- **Yahoo path** (no `--returns-csv`): OHLCV downloaded with `yfinance`

### Features

- Returns path: return-based windows
- Yahoo path: engineered features including:
  - daily return
  - open-close move
  - high-low range
  - volume change

### Model

- Transformer encoder creates a latent embedding per stock window
- Embeddings are compared by cosine similarity
- Training objective:
  - `mse` (default)
  - `mse_contrastive` (optional exploratory setting)

### Pair Selection

- Proposed: top-k embedding pairs by cosine similarity
- Baselines:
  - Engle-Granger (p-value ranking)
  - Johansen (trace-stat ranking)

### Evaluation

- Train on dates `<= --train-end-date`
- Evaluate OOS on dates `> --train-end-date`
- Backtest: spread z-score entry/exit rule
- Metrics: Sharpe, total return, max drawdown, volatility

---

## Run Examples

### A) Full 611-stock returns baseline

```bash
python3 src/main.py \
  --returns-csv "data/returns_df_611.csv" \
  --start-date 2000-01-01 --end-date 2018-12-31 --train-end-date 2014-12-31 \
  --window-size 30 --epochs 4 --top-k-pairs 5 \
  --baseline-method engle --objective mse \
  --output-dir outputs/returns_baseline
```

### B) Yahoo Finance (10 tickers)

```bash
python3 src/main.py \
  --tickers "AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,JPM,BAC,XOM" \
  --start-date 2015-01-01 --end-date 2024-12-31 --train-end-date 2022-12-31 \
  --window-size 30 --epochs 4 --top-k-pairs 5 \
  --baseline-method johansen --objective mse_contrastive \
  --output-dir outputs/yfinance
```

---

## Fair Comparison: Returns vs Yahoo

Your earlier runs used different date windows and sometimes different ticker universes. That is valid for separate experiments, but not a strict apples-to-apples comparison.

For a controlled comparison, align:

- `--start-date`
- `--end-date`
- `--train-end-date`
- `--tickers`
- `--baseline-method`
- `--objective`

Example aligned setup:

```bash
START=2015-01-01
END=2018-12-31
TRAIN_END=2016-12-31
TICKERS="AAPL,MSFT,GOOGL,AMZN,FB,NVDA,TSLA,JPM,BAC,XOM"

python3 src/main.py \
  --tickers "$TICKERS" \
  --start-date "$START" --end-date "$END" --train-end-date "$TRAIN_END" \
  --window-size 30 --epochs 4 --top-k-pairs 5 \
  --baseline-method engle --objective mse \
  --output-dir outputs/yfinance_2015_2018

python3 src/main.py \
  --returns-csv "data/returns_df_611.csv" \
  --tickers "$TICKERS" \
  --start-date "$START" --end-date "$END" --train-end-date "$TRAIN_END" \
  --window-size 30 --epochs 4 --top-k-pairs 5 \
  --baseline-method engle --objective mse \
  --output-dir outputs/returns_2015_2018
```

Notes:

- In older data, Facebook may appear as `FB` instead of `META`.
- If tickers are missing in returns CSV, the loader keeps available columns.

---

## Output Artifacts

Each run writes these files under `--output-dir`:

- `embedding_pairs.csv`
- `engle_granger_pairs.csv`
- `johansen_pairs.csv`
- `baseline_pairs.csv`
- `embedding_vs_corr_vs_coint.csv`
- `performance_summary.csv`
- `daily_returns.csv`
- `equity_curves.csv`
- `equity_curves.png`
- `performance_summary.png`

---

## Repository Layout

- `src/main.py` - CLI entry point and end-to-end runner
- `src/data.py` - data loading and feature window creation
- `src/model.py` - transformer encoder, training, embedding extraction
- `src/pairs.py` - embedding and cointegration pair selection
- `src/backtest.py` - spread backtest and metric aggregation
- `src/config.py` - default config and ticker universe
- `data/` - dataset files and notes
- `outputs/` - generated experiment artifacts

---

## Limitations (Important)

This is a research/course pipeline, not production trading software.

- No full transaction-cost or slippage model
- Results depend on date regime, ticker universe, and objective
- Better prediction loss does not always imply better tradability

---

## Troubleshooting

### 1) Module errors (`ModuleNotFoundError`)

Activate venv and reinstall dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Matplotlib cache warning

Set writable config directory:

```bash
mkdir -p .mplconfig
export MPLCONFIGDIR="$(pwd)/.mplconfig"
```

### 3) Slow full-dataset run

Use smaller runs for smoke tests:

- lower `--epochs`
- reduce `--top-k-pairs`
- set `--max-stocks N`

