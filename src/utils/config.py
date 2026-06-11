"""
Central configuration for the CSRF pipeline.

All hyperparameters live here. Scripts read from config.yaml (via load_config)
and instantiate this dataclass. Nothing is hardcoded in individual modules.

fast_dev mode: when True, every stage subsamples to a tiny slice of the data
so the full pipeline can be exercised end-to-end in minutes on a CPU.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Lag rules: number of months to lag each characteristic type before use.
# These enforce the look-ahead bias prevention described in Gu et al. (2020).
# Monthly: 1 month, Quarterly: 4 months (filing delay), Annual: 6 months.
# ---------------------------------------------------------------------------
DEFAULT_LAG_RULES = {
    "monthly": 1,
    "quarterly": 4,
    "annual": 6,
}

# Characteristic names from the GKX dataset grouped by update frequency.
# These lists are used by preprocess.py to apply the correct lag per column.
# Source: Appendix Table A1 of Gu, Kelly & Xiu (2020).
MONTHLY_CHARS = [
    "beta", "betasq", "chmom", "dolvol", "idiovol", "ill", "indmom",
    "maxret", "mom12m", "mom1m", "mom36m", "mom6m", "mve", "retvol",
    "std_dolvol", "std_turn", "turn", "zerotrade",
]

QUARTERLY_CHARS = [
    "aeavol", "baspread", "cash", "cfp", "chcsho", "chinv", "ep",
    "gma", "lgr", "mve_ia", "nincr", "pchcapx_ia", "pchgm_pchsale",
    "pchquick", "pchsale_pchinvt", "pchsale_pchrect", "pchsale_pchxsga",
    "pchsaleinv", "quick", "rd_sale", "roaq", "roeq", "rsup", "sgr",
    "sp", "sue",
]

# Everything not in monthly or quarterly is treated as annual (6-month lag).
# preprocess.py applies this residual classification automatically.


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    splits_dir: str = "data/splits"

    # Date boundaries for the three non-overlapping sets
    train_start: str = "1957-01"
    train_end: str = "1999-12"
    cal_start: str = "2000-01"
    cal_end: str = "2007-12"
    test_start: str = "2008-01"
    test_end: str = "2021-12"

    # Validation set boundary within training (last 20% by time)
    val_start: str = "1992-01"   # ~1992 onward is the final 20% of 1957-1999

    # Characteristic lag rules
    lag_rules: dict = field(default_factory=lambda: DEFAULT_LAG_RULES)

    # fast_dev subsample sizes
    fast_dev_n_stocks: int = 200
    fast_dev_n_months: int = 24


@dataclass
class ModelConfig:
    # Architecture
    n_chars: int = 168          # 94 firm characteristics + 74 industry dummies
    hidden_dims: List[int] = field(default_factory=lambda: [32, 16, 8, 4])
    dropout: float = 0.10

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-5
    huber_delta: float = 0.5    # Huber loss parameter (in return units, ~0.5%/month)
    batch_size: int = 2048
    max_epochs: int = 200
    early_stopping_patience: int = 20
    lr_scheduler_patience: int = 10
    lr_scheduler_factor: float = 0.5
    lr_min: float = 1e-5

    # CPU-specific
    num_workers: int = 0        # 0 = main process only (required on Windows)
    pin_memory: bool = False

    # Checkpointing (critical for long CPU runs)
    checkpoint_every_n_epochs: int = 5
    checkpoint_dir: str = "data/splits"

    # Random seed for reproducibility
    seed: int = 42


@dataclass
class SPCIConfig:
    alpha: float = 0.10         # miscoverage level → 90% prediction intervals
    L: int = 24                 # lagged residuals used as QRF features
    min_history: int = 12       # minimum months of residuals to use SPCI vs fallback
    n_estimators: int = 200     # GBM trees for quantile regression
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8      # stochastic GBM — reduces overfitting


@dataclass
class PortfolioConfig:
    n_deciles: int = 10
    lambda_grid: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    half_spread_bps: float = 20.0   # conservative half-spread for small caps

    # Fama-MacBeth kernel bandwidth (Newey-West lags)
    fm_bandwidth: int = 6


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    spci: SPCIConfig = field(default_factory=SPCIConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)

    # Master fast_dev switch — all scripts check this before loading data
    fast_dev: bool = False

    results_dir: str = "results"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> Config:
    """Load config.yaml and merge with dataclass defaults."""
    p = Path(path)
    if not p.exists():
        return Config()

    with open(p) as f:
        raw = yaml.safe_load(f) or {}

    cfg = Config()

    if "fast_dev" in raw:
        cfg.fast_dev = bool(raw["fast_dev"])
    if "results_dir" in raw:
        cfg.results_dir = raw["results_dir"]

    for section, dc in [("data", cfg.data), ("model", cfg.model),
                        ("spci", cfg.spci), ("portfolio", cfg.portfolio)]:
        if section in raw:
            for k, v in raw[section].items():
                if hasattr(dc, k):
                    setattr(dc, k, v)

    return cfg
