"""
Parquet save/load helpers and results directory management.

All pipeline stages read and write parquet. Parquet is chosen over CSV because:
- ~10x smaller files for the GKX panel (~30k stocks × 780 months)
- Preserves dtypes (dates, floats, categoricals) without round-trip loss
- Columnar format — loading only the columns you need is fast
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_parquet(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False, **kwargs)


def load_parquet(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_parquet(path, **kwargs)


def results_path(cfg_results_dir: str, *parts: str) -> Path:
    """Construct a path under the results directory and ensure it exists."""
    p = Path(cfg_results_dir).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_table(df: pd.DataFrame, cfg_results_dir: str, *parts: str) -> Path:
    """Save a DataFrame as CSV inside results/tables/."""
    p = results_path(cfg_results_dir, "tables", *parts)
    df.to_csv(p)
    return p
