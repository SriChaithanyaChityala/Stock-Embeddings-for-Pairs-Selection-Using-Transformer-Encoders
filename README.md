# Stock Embeddings for Pairs Trading

Transformer embeddings → cosine pair selection → Engle–Granger / Johansen baselines → OOS backtest → CSV/PNG under `--output-dir`. The 611-stock return matrix is in **`data/returns_df_611.csv`** (bundled in this repo).

## Setup

From the **project root** (the folder that contains `src/` and `requirements.txt`), for example:

```bash
cd ~/Desktop/ML
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, use `.\.venv\Scripts\activate` instead of `source .venv/bin/activate`.

Run from project root (no backticks around the command):

```bash
python3 src/main.py [options]
```

---

## Comparing `returns_df_611` vs Yahoo on the **same** period

Your two commands used **different** windows:

| Run | `--start-date` | `--end-date` | `--train-end-date` |
|-----|----------------|--------------|---------------------|
| Yahoo | 2015-01-01 | **2024-12-31** | **2022-12-31** |
| Returns CSV | **2000-01-01** | **2018-12-31** | **2014-12-31** |

That is **valid** for two separate experiments, but **not** a controlled comparison: OOS years, train length, and market regime differ. The Yahoo run also used **10 tickers** while the returns file can load **many** columns unless you pass `--tickers`.

**Aligned example (2015–2018, same train cutoff, same universe):** use identical dates and pass the **same** `--tickers` on the returns path so the CSV is restricted to those columns. For Facebook in older panels the column is often `FB` (not `META`); if a ticker is missing from the CSV, the loader keeps only columns that exist.

```bash
# Shared settings (edit paths as needed)
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

Use the **same** `--baseline-method` and `--objective` on both if you want the same experimental protocol. Small differences can still remain (Yahoo OHLC-derived features vs return-only features on the CSV path).

---

## Other examples

**Full returns matrix (no `--tickers`):** uses all ticker columns in range (not comparable to a 10-name Yahoo run unless you add `--max-stocks` or `--tickers`).

```bash
python3 src/main.py \
  --returns-csv "data/returns_df_611.csv" \
  --start-date 2000-01-01 --end-date 2018-12-31 --train-end-date 2014-12-31 \
  --window-size 30 --epochs 4 --top-k-pairs 5 \
  --baseline-method engle --objective mse \
  --output-dir outputs/returns_baseline
```

**Yahoo, long window:**

```bash
python3 src/main.py \
  --tickers "AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,JPM,BAC,XOM" \
  --start-date 2015-01-01 --end-date 2024-12-31 --train-end-date 2022-12-31 \
  --window-size 30 --epochs 4 --top-k-pairs 5 \
  --baseline-method johansen --objective mse_contrastive \
  --output-dir outputs/yfinance
```

## Outputs

Under `--output-dir`: `embedding_pairs.csv`, `engle_granger_pairs.csv`, `johansen_pairs.csv`, `baseline_pairs.csv`, `embedding_vs_corr_vs_coint.csv`, `performance_summary.csv`, `daily_returns.csv`, `equity_curves.csv`, `equity_curves.png`, `performance_summary.png`.

## Layout

`src/main.py`, `src/data.py`, `src/model.py`, `src/pairs.py`, `src/backtest.py`, `src/config.py`.

If matplotlib warns about config dirs: `export MPLCONFIGDIR=/path/to/ML/.mplconfig` (create the directory first).
