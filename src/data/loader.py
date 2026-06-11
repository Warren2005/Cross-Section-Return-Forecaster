"""
Data loading for the CSRF pipeline.

Primary source: Gu, Kelly & Xiu (2020) characteristics panel, freely available
at https://dachxiu.chicagobooth.edu/#rp

The file on Xiu's site is a zip containing a SAS (.sas7bdat) or CSV file.
This loader handles both formats and normalises the result to a standard
DataFrame with lowercase column names and a pandas Period date index.

CRSP abstraction
----------------
Some downstream steps (value-weighted portfolios, size-decile coverage) need
market capitalisation (`me`) and possibly other CRSP fields. We define a
`ReturnLoader` Protocol so that the CRSP source can be swapped in later
without touching any analysis code.

  GKXLoader  — default, no WRDS required; uses the `me` column already in
               the GKX file as the market-cap proxy.
  CRSPLoader — # CRSP-DEPENDENT: requires WRDS access. Stub only.

To switch to CRSP: set loader = CRSPLoader(...) in 01_build_data.py.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd


# ---------------------------------------------------------------------------
# Column metadata
# ---------------------------------------------------------------------------

# Columns that are identifiers or the outcome — never preprocessed as features
ID_COLS = {"permno", "date", "ret"}

# Columns in the GKX file that are industry dummies (74 SIC-based indicators).
# The GKX dataset names them with their SIC integer value as a string prefix.
# This set is used to separate characteristics from dummies during preprocessing.
# The actual column names will be detected dynamically by `detect_industry_cols`.
INDUSTRY_COL_PREFIXES = ("sic",)


def detect_industry_cols(df: pd.DataFrame) -> list[str]:
    """
    Return columns that look like industry dummies.
    The GKX file names them starting with a SIC prefix (e.g. 'sic10', 'sic12').
    Falls back to: any binary {0, 1} column not in ID_COLS.
    """
    sic_cols = [c for c in df.columns
                if any(c.startswith(p) for p in INDUSTRY_COL_PREFIXES)]
    if sic_cols:
        return sic_cols

    # Fallback: detect by dtype/cardinality — binary columns with exactly 2 unique values
    binary_cols = []
    non_id = [c for c in df.columns if c not in ID_COLS]
    for c in non_id:
        uq = df[c].dropna().unique()
        if set(uq).issubset({0, 1, 0.0, 1.0}):
            binary_cols.append(c)
    return binary_cols


def detect_char_cols(df: pd.DataFrame, industry_cols: list[str]) -> list[str]:
    """Return the 94 continuous characteristic columns (not IDs, not industry dummies)."""
    exclude = ID_COLS | set(industry_cols)
    return [c for c in df.columns if c not in exclude]


# ---------------------------------------------------------------------------
# Protocol — all loaders must satisfy this interface
# ---------------------------------------------------------------------------

@runtime_checkable
class ReturnLoader(Protocol):
    def load(self) -> pd.DataFrame:
        """
        Return a DataFrame with at minimum these columns:
          permno  — int, CRSP permanent stock identifier
          date    — pd.Period (monthly frequency, e.g. Period('2001-03', 'M'))
          ret     — float, monthly excess return
          me      — float, market capitalisation (used for VW portfolios)
        Plus all characteristic and industry dummy columns.
        Missing values are allowed; downstream code handles them.
        """
        ...


# ---------------------------------------------------------------------------
# GKX loader (no WRDS required)
# ---------------------------------------------------------------------------

class GKXLoader:
    """
    Load the Gu, Kelly & Xiu (2020) characteristics panel.

    Supported input formats (auto-detected by file extension):
      .zip  — zip archive containing a single .sas7bdat or .csv file
      .sas7bdat — SAS data file (requires the `pyreadstat` package)
      .csv, .csv.gz — comma-separated, optionally gzip-compressed
      .parquet — pre-converted parquet (fastest)

    Parameters
    ----------
    path : path to the downloaded GKX file (any format above)
    date_col : name of the date column in the raw file (default 'DATE')
    ret_col  : name of the return column in the raw file (default 'RET')
    """

    def __init__(
        self,
        path: str | Path,
        date_col: str = "DATE",
        ret_col: str = "RET",
    ):
        self.path = Path(path)
        self.date_col = date_col
        self.ret_col = ret_col

    def load(self) -> pd.DataFrame:
        df = self._read_raw()
        df = self._normalise_columns(df)
        df = self._parse_date(df)
        df = self._cast_types(df)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_raw(self) -> pd.DataFrame:
        suffix = self.path.suffix.lower()

        if suffix == ".zip":
            return self._read_zip()
        elif suffix == ".parquet":
            return pd.read_parquet(self.path)
        elif suffix in (".csv", ".gz"):
            return pd.read_csv(self.path, low_memory=False)
        elif suffix == ".sas7bdat":
            return self._read_sas(self.path)
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                "Supported: .zip, .parquet, .csv, .csv.gz, .sas7bdat"
            )

    def _read_zip(self) -> pd.DataFrame:
        with zipfile.ZipFile(self.path) as zf:
            names = zf.namelist()
            # Pick the first non-directory member
            inner = next(n for n in names if not n.endswith("/"))
            inner_suffix = Path(inner).suffix.lower()
            with zf.open(inner) as f:
                if inner_suffix == ".sas7bdat":
                    import tempfile, shutil
                    # pyreadstat needs a real file path, not a file object
                    with tempfile.NamedTemporaryFile(suffix=".sas7bdat", delete=False) as tmp:
                        shutil.copyfileobj(f, tmp)
                        tmp_path = Path(tmp.name)
                    try:
                        return self._read_sas(tmp_path)
                    finally:
                        tmp_path.unlink(missing_ok=True)
                else:
                    return pd.read_csv(f, low_memory=False)

    @staticmethod
    def _read_sas(path: Path) -> pd.DataFrame:
        try:
            import pyreadstat
            df, _ = pyreadstat.read_sas7bdat(str(path))
            return df
        except ImportError as e:
            raise ImportError(
                "Reading SAS files requires pyreadstat: pip install pyreadstat"
            ) from e

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lower-case all column names; rename date and return columns."""
        df = df.rename(columns=str.lower)
        if self.date_col.lower() != "date":
            df = df.rename(columns={self.date_col.lower(): "date"})
        if self.ret_col.lower() != "ret":
            df = df.rename(columns={self.ret_col.lower(): "ret"})
        return df

    def _parse_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the raw date column to pd.Period('M').

        The GKX file encodes dates as integers like 195701 (YYYYMM) or
        as float SAS date values. We handle both.
        """
        col = df["date"]

        if pd.api.types.is_float_dtype(col) and col.max() < 100_000:
            # SAS date (days since 1960-01-01)
            df["date"] = (
                pd.to_datetime(col, unit="D", origin="1960-01-01")
                .dt.to_period("M")
            )
        else:
            # YYYYMM integer or string
            col_str = col.astype(int).astype(str)
            df["date"] = pd.PeriodIndex(
                col_str.apply(lambda s: f"{s[:4]}-{s[4:6]}"), freq="M"
            )
        return df

    def _cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df["permno"] = df["permno"].astype(int)
        df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
        return df


# ---------------------------------------------------------------------------
# CRSP loader stub  # CRSP-DEPENDENT
# ---------------------------------------------------------------------------

class CRSPLoader:
    """
    Load stock data from CRSP via WRDS.

    # CRSP-DEPENDENT: requires WRDS access and the `wrds` Python package.
    # To activate: pip install wrds, then authenticate with your WRDS credentials.
    # Replace GKXLoader with CRSPLoader in scripts/01_build_data.py.

    This loader returns a DataFrame in the same format as GKXLoader so that
    all downstream code works without modification.
    """

    def __init__(self, wrds_username: str):
        self.wrds_username = wrds_username

    def load(self) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError(
            "CRSPLoader requires WRDS access. "
            "See LEARNING.md §3.1 for setup instructions. "
            "Use GKXLoader until WRDS access is available."
        )


# ---------------------------------------------------------------------------
# Factor data loaders (Ken French Data Library — always free)
# ---------------------------------------------------------------------------

def load_ff5_factors(path: str | Path) -> pd.DataFrame:
    """
    Load Fama-French 5-factor monthly returns from the Ken French Data Library.

    Download: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
    File: "Fama/French 5 Factors (2x3)" → monthly CSV

    Returns DataFrame with columns: date (Period M), mkt_rf, smb, hml, rmw, cma, rf
    Values are in decimal (already divided by 100).
    """
    df = pd.read_csv(path, skiprows=3, header=0)
    # French's CSV has a blank line before the annual section — stop there
    end = df[df.iloc[:, 0].astype(str).str.strip() == ""].index
    if len(end):
        df = df.iloc[: end[0]]

    df.columns = df.columns.str.strip().str.lower().str.replace("-", "_")
    df = df.rename(columns={"unnamed: 0": "date", "mkt-rf": "mkt_rf"})
    df["date"] = pd.PeriodIndex(df["date"].astype(str).str.strip(), freq="M")

    for c in ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0

    return df.dropna(subset=["date"]).reset_index(drop=True)


def load_vix(path: str | Path) -> pd.DataFrame:
    """
    Load monthly VIX from FRED (series VIXCLS, downloaded as CSV).
    Download: https://fred.stlouisfed.org/series/VIXCLS

    Returns DataFrame with columns: date (Period M), vix
    """
    df = pd.read_csv(path, parse_dates=["DATE"])
    df = df.rename(columns={"DATE": "date", "VIXCLS": "vix"})
    df["date"] = df["date"].dt.to_period("M")
    df["vix"] = pd.to_numeric(df["vix"], errors="coerce")
    # Monthly average — FRED gives daily values, so resample
    df = df.groupby("date")["vix"].mean().reset_index()
    return df
