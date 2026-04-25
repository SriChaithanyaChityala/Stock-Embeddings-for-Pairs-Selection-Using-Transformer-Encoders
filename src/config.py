from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    tickers: tuple[str, ...]
    start_date: str = "2000-01-01"
    end_date: str = "2018-12-31"
    train_end_date: str = "2014-12-31"
    window_size: int = 30
    embedding_dim: int = 32
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 8
    top_k_pairs: int = 10
    z_entry: float = 1.5
    z_exit: float = 0.2
    initial_capital: float = 100000.0


DEFAULT_TICKERS = (
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "JPM",
    "BAC",
    "XOM",
    "CVX",
    "KO",
    "PEP",
    "WMT",
    "UNH",
    "PFE",
    "V",
    "MA",
    "INTC",
    "CSCO",
)
