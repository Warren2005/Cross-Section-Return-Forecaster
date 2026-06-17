"""
Phase 6 -- Robustness checks.

Usage:
    python scripts/06_robustness.py [--fast-dev] [--config PATH]

Prerequisites:
    Phases 1-5 must have been run. Needs:
      data/splits/cal_predictions.parquet
      data/splits/test_predictions.parquet
      data/splits/test_intervals.parquet
      data/splits/test.parquet            (for large-cap filter via 'me')
      data/splits/cal_intervals.parquet   (for lambda tuning in sub-period check)

Five robustness checks
----------------------

Check 1 -- Alpha sweep (coverage_level in {80%, 90%, 95%})
  Recompute fallback quantiles from cal residuals at each target level.
  Validate that actual test coverage matches the target (conformal guarantee).
  Also confirms that the SPCI width ordering (IC, decile L-S Sharpe) is
  invariant to alpha because varying alpha only scales the absolute interval
  size, not the relative ordering of widths.

Check 2 -- Residual history (L proxy)
  Split test_intervals by residual history length, using the used_spci flag
  and width variation to infer which stocks had >= L months of history.
  Show how coverage and width IC differ between SPCI-eligible and fallback
  stocks, and across history-length bins.
  (A full L sweep -- refitting SPCI with L in {12, 24, 36} -- requires
  re-running Phase 3 with different config. See note in results output.)

Check 3 -- Bootstrap baseline
  Per-stock historical residual standard deviation (from cal_predictions) as
  an uncertainty measure, scaled to match target coverage on the cal set.
  This mimics the Liu et al. (2026) approach without bootstrap resampling.
  Compares: coverage validity and width IC (bootstrap vs SPCI).

Check 4 -- Large-cap universe
  Restrict to stocks with rank-normalised 'me' > 0 (approximately the top
  half by market cap, since me is in [-1, 1] after rank normalisation).
  Reports coverage, width IC, and D1-D10 L-S Sharpe for this subset.
  Note: for the top SIZE QUINTILE (top 20%), filter me > 0.6.

Check 5 -- Sub-period stability
  Split the test period at 2015-01:
    First sub-period:  2008-01 to 2014-12  (84 months, GFC + recovery)
    Second sub-period: 2015-01 to 2021-12  (84 months, bull market + COVID)
  Report coverage, width IC, and width decile L-S Sharpe for each sub-period.

Outputs (results/tables/):
  robustness_alpha_sweep.csv
  robustness_history_bins.csv
  robustness_bootstrap_vs_spci.csv
  robustness_large_cap.csv
  robustness_sub_period.csv
  robustness_summary.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.analysis.portfolio import (
    construct_ls_portfolio,
    lambda_tune,
    portfolio_stats,
    uncertainty_adjusted_score,
)
from src.analysis.width_signal import width_ic
from src.conformal.coverage import empirical_coverage
from src.conformal.fallback import calibration_quantile
from src.data.splits import fast_dev_subsample
from src.utils.config import load_config
from src.utils.io import load_parquet, save_table


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ls_sharpe(interval_df: pd.DataFrame, n_deciles: int = 10) -> float:
    """Width decile L-S Sharpe on interval_df (narrowest long, widest short)."""
    df = interval_df.dropna(subset=["y_true", "width"]).copy()
    df["width_decile"] = df.groupby("date")["width"].transform(
        lambda x: pd.qcut(x, n_deciles, labels=False, duplicates="drop")
    )
    df = df.dropna(subset=["width_decile"])
    df["width_decile"] = df["width_decile"].astype(int)

    monthly = (
        df.groupby(["date", "width_decile"])["y_true"]
        .mean()
        .unstack("width_decile")
    )
    if 0 not in monthly.columns or (n_deciles - 1) not in monthly.columns:
        return np.nan

    ls = monthly[0] - monthly[n_deciles - 1]
    ls = ls.dropna()
    if len(ls) < 6 or ls.std() == 0:
        return np.nan
    return float(ls.mean() / ls.std() * np.sqrt(12))


def _coverage(df: pd.DataFrame) -> float:
    sub = df.dropna(subset=["y_true", "lower", "upper"])
    covered = (sub["y_true"] >= sub["lower"]) & (sub["y_true"] <= sub["upper"])
    return float(covered.mean())


def _width_ic(df: pd.DataFrame) -> float:
    """Monthly Spearman IC between width and |y_true - y_pred|, averaged."""
    df = df.dropna(subset=["y_true", "y_pred", "width"]).copy()
    df["abs_err"] = (df["y_true"] - df["y_pred"]).abs()
    monthly_ic = (
        df.groupby("date")
        .apply(
            lambda g: float(spearmanr(g["width"], g["abs_err"]).statistic)
            if len(g) >= 10 else np.nan,
            include_groups=False,
        )
        .dropna()
    )
    return float(monthly_ic.mean()) if len(monthly_ic) > 0 else np.nan


def _print_section(title: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


# ---------------------------------------------------------------------------
# Check 1: Alpha sweep
# ---------------------------------------------------------------------------

def check_alpha_sweep(
    cal_preds: pd.DataFrame,
    test_preds: pd.DataFrame,
    test_intervals: pd.DataFrame,
    alphas: list[float] = (0.05, 0.10, 0.20),
) -> pd.DataFrame:
    """
    Validate conformal coverage at different alpha levels using fallback quantiles.

    Also shows that SPCI width ordering (IC, L-S Sharpe) is invariant to alpha:
    scaling all widths by a constant factor does not change their cross-sectional
    ranking, so IC and Sharpe from width decile sorts are unaffected by alpha.
    """
    _print_section("CHECK 1: Alpha sweep (coverage_level sensitivity)")

    cal_resids = cal_preds["residual"].dropna().values

    rows = []
    for alpha in alphas:
        # Fallback quantile at this alpha level
        q = calibration_quantile(cal_resids, alpha)

        # Build symmetric intervals from test predictions
        df = test_preds[["permno", "date", "y_pred", "y_true"]].dropna().copy()
        df["lower"] = df["y_pred"] - q
        df["upper"] = df["y_pred"] + q
        df["width"] = 2 * q   # constant

        # Coverage
        cov = _coverage(df)

        rows.append({
            "alpha":           alpha,
            "target_cov (%)":  (1 - alpha) * 100,
            "actual_cov (%)":  cov * 100,
            "gap (pp)":        (cov - (1 - alpha)) * 100,
            "fallback_q (bps)": q * 1e4,
            "width_bps":       2 * q * 1e4,
            "note": "width IC = 0 (constant width fallback)",
        })
        print(f"  alpha={alpha:.2f}  target={100*(1-alpha):.0f}%  "
              f"actual={cov*100:.2f}%  q={q*1e4:.0f} bps")

    # SPCI width ordering is invariant to alpha -- show once
    spci_ic     = _width_ic(test_intervals)
    spci_sharpe = _ls_sharpe(test_intervals)
    print(f"\n  SPCI test intervals (alpha=0.10):")
    print(f"    Width IC (vs |error|):  {spci_ic:+.4f}")
    print(f"    Width D1-D10 Sharpe:    {spci_sharpe:.3f}")
    print(f"  Note: scaling all widths by a constant (varying alpha)")
    print(f"  does not change the width ordering -- IC and Sharpe are")
    print(f"  invariant to alpha for SPCI intervals.")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Check 2: Residual history sensitivity (L proxy)
# ---------------------------------------------------------------------------

def check_history_bins(
    test_intervals: pd.DataFrame,
    cal_preds: pd.DataFrame,
    test_preds: pd.DataFrame,
) -> pd.DataFrame:
    """
    Proxy for L sensitivity: split test set by residual history length.

    SPCI is only applied when a stock has >= min_history months of cal residuals.
    By splitting into stocks with different history lengths, we can see how
    coverage and width IC vary with the amount of residual history available.

    Full L sweep (refitting SPCI with L in {12, 24, 36}) requires re-running
    Phase 3 with different SPCI config. This check uses existing intervals.
    """
    _print_section("CHECK 2: Residual history sensitivity (L proxy)")

    # Compute how many cal-period residuals each stock has
    cal_counts = (
        cal_preds.dropna(subset=["residual"])
        .groupby("permno")
        .size()
        .rename("n_cal_months")
    )

    df = test_intervals.join(cal_counts, on="permno")
    df["n_cal_months"] = df["n_cal_months"].fillna(0).astype(int)

    bins  = [(0, 11, "< 12 months (fallback)"),
             (12, 23, "12-23 months (short)"),
             (24, 35, "24-35 months"),
             (36, 9999, ">= 36 months (full history)")]

    rows = []
    print(f"  {'History bin':<28} {'N obs':>8} {'Coverage':>10} "
          f"{'IC(w,|e|)':>12} {'Sharpe':>8}")
    print(f"  {'-'*70}")

    for lo, hi, label in bins:
        mask = (df["n_cal_months"] >= lo) & (df["n_cal_months"] <= hi)
        sub  = df[mask]
        if len(sub) < 100:
            continue

        cov    = _coverage(sub)
        ic     = _width_ic(sub)
        sharpe = _ls_sharpe(sub)
        spci_pct = sub["used_spci"].mean() * 100 if "used_spci" in sub else np.nan

        rows.append({
            "history_bin":   label,
            "n_obs":         len(sub),
            "pct_spci":      spci_pct,
            "coverage (%)":  cov * 100,
            "width_ic":      ic,
            "ls_sharpe":     sharpe,
        })
        print(f"  {label:<28} {len(sub):>8,} {cov*100:>9.2f}% {ic:>12.4f} {sharpe:>8.3f}")

    print(f"\n  Note: for a full L sweep (refit SPCI with L=12/24/36),")
    print(f"  re-run Phase 3 with spci.L changed in config.yaml.")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Check 3: Bootstrap baseline
# ---------------------------------------------------------------------------

def check_bootstrap_baseline(
    cal_preds: pd.DataFrame,
    test_preds: pd.DataFrame,
    alpha: float = 0.10,
) -> pd.DataFrame:
    """
    Compare SPCI to a 'bootstrap-like' baseline using per-stock residual std.

    Bootstrap baseline:
      For each stock, estimate uncertainty as sigma_i = std(residuals on cal set).
      Scale sigma_i by constant k (fitted on cal set) so that k*sigma_i achieves
      (1-alpha)% coverage on the calibration set.
      Test interval: [y_pred - k*sigma_i, y_pred + k*sigma_i].

    This mimics the core idea of Liu et al. (2026): using model-derived per-stock
    uncertainty as a return signal, without the conformal guarantee.

    Key comparison:
      - Does the bootstrap baseline achieve target coverage?
      - Is its width IC comparable to SPCI's?
      - Is its D1-D10 L-S Sharpe comparable to SPCI's?
    """
    _print_section("CHECK 3: Bootstrap baseline vs SPCI")

    # Per-stock residual std on cal set
    stock_std = (
        cal_preds.dropna(subset=["residual"])
        .groupby("permno")["residual"]
        .std(ddof=1)
        .rename("sigma")
    )
    mean_sigma = float(stock_std.mean())

    # Find k such that k*sigma achieves (1-alpha)% coverage on cal set
    cal = cal_preds[["permno", "date", "y_pred", "y_true", "residual"]].dropna().copy()
    cal = cal.join(stock_std, on="permno")
    cal["sigma"] = cal["sigma"].fillna(mean_sigma)
    cal["scaled"] = cal["residual"].abs() / cal["sigma"].clip(lower=1e-8)
    n_cal = len(cal)
    level = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
    k = float(np.quantile(cal["scaled"].dropna(), level))

    print(f"  Bootstrap scale factor k = {k:.3f}  (fitted on cal set, alpha={alpha})")

    # Apply to test set
    test = test_preds[["permno", "date", "y_pred", "y_true"]].dropna().copy()
    test = test.join(stock_std, on="permno")
    test["sigma"]  = test["sigma"].fillna(mean_sigma)
    test["width"]  = 2 * k * test["sigma"]
    test["lower"]  = test["y_pred"] - k * test["sigma"]
    test["upper"]  = test["y_pred"] + k * test["sigma"]

    boot_cov    = _coverage(test)
    boot_ic     = _width_ic(test)
    boot_sharpe = _ls_sharpe(test)

    # Compare against SPCI from test_intervals (stored separately)
    print(f"\n  {'Method':<22} {'Coverage':>10} {'IC(w,|e|)':>12} {'D1-D10 Sharpe':>15}")
    print(f"  {'-'*62}")
    print(f"  {'Bootstrap baseline':<22} {boot_cov*100:>9.2f}% {boot_ic:>12.4f} {boot_sharpe:>15.3f}")
    print(f"\n  (Compare to SPCI results from Check 1 above.)")
    print(f"  Bootstrap achieves {'[OK] >= ' if boot_cov >= (1-alpha) else '[FAIL] < '}"
          f"{(1-alpha)*100:.0f}% coverage on test set.")
    print(f"  Coverage note: bootstrap has no finite-sample guarantee unlike SPCI.")

    return pd.DataFrame([{
        "method":        "bootstrap_baseline",
        "k_scale":       k,
        "coverage (%)":  boot_cov * 100,
        "target (%)":    (1 - alpha) * 100,
        "width_ic":      boot_ic,
        "ls_sharpe":     boot_sharpe,
        "mean_sigma_bps": mean_sigma * 1e4,
    }])


# ---------------------------------------------------------------------------
# Check 4: Large-cap universe
# ---------------------------------------------------------------------------

def check_large_cap(
    test_intervals: pd.DataFrame,
    test_df: pd.DataFrame,
    me_threshold: float = 0.0,
    label: str = "top half (me > 0)",
    n_deciles: int = 10,
) -> pd.DataFrame:
    """
    Restrict to large-cap stocks and repeat the width signal analysis.

    Since 'me' in test.parquet is rank-normalised to [-1, 1]:
      me > 0.0  : approximately top 50% by market cap
      me > 0.6  : approximately top quintile (top 20%)

    Large-cap stocks are more liquid, have lower transaction costs, and are
    more relevant for institutional investors. If the width signal persists
    after restricting to large caps, it is tradeable at scale.
    """
    _print_section(f"CHECK 4: Large-cap universe ({label})")

    # Merge me from test_df into test_intervals
    if "me" not in test_df.columns:
        print(f"  [!] 'me' column not found in test.parquet -- check skipped.")
        return pd.DataFrame()

    me_panel = test_df[["permno", "date", "me"]].dropna()
    df = test_intervals.merge(me_panel, on=["permno", "date"], how="left")

    large_mask = df["me"] > me_threshold
    large_df   = df[large_mask]

    if len(large_df) < 1000:
        print(f"  [!] Insufficient data after filtering ({len(large_df)} rows) -- skipped.")
        return pd.DataFrame()

    cov    = _coverage(large_df)
    ic     = _width_ic(large_df)
    sharpe = _ls_sharpe(large_df, n_deciles=n_deciles)
    n_stocks = large_df["permno"].nunique()
    n_months = large_df["date"].nunique()

    print(f"  Filter: me > {me_threshold}  ({n_stocks:,} stocks, {n_months} months)")
    print(f"  Coverage:          {cov*100:.2f}%")
    print(f"  Width IC:          {ic:+.4f}")
    print(f"  D1-D10 L-S Sharpe: {sharpe:.3f}")
    print(f"\n  Full universe comparison:")
    full_cov    = _coverage(test_intervals)
    full_ic     = _width_ic(test_intervals)
    full_sharpe = _ls_sharpe(test_intervals, n_deciles=n_deciles)
    print(f"  Coverage:          {full_cov*100:.2f}%")
    print(f"  Width IC:          {full_ic:+.4f}")
    print(f"  D1-D10 L-S Sharpe: {full_sharpe:.3f}")

    return pd.DataFrame([
        {"universe": "full",      "n_stocks": test_intervals["permno"].nunique(),
         "coverage (%)": full_cov*100, "width_ic": full_ic, "ls_sharpe": full_sharpe},
        {"universe": label,       "n_stocks": n_stocks,
         "coverage (%)": cov*100,      "width_ic": ic,      "ls_sharpe": sharpe},
    ])


# ---------------------------------------------------------------------------
# Check 5: Sub-period stability
# ---------------------------------------------------------------------------

def check_sub_period(
    test_intervals: pd.DataFrame,
    cal_intervals: pd.DataFrame | None,
    split_date: str = "2015-01",
    n_deciles: int = 10,
) -> pd.DataFrame:
    """
    Split the test set at split_date and report key metrics for each half.

    2008-01 to 2014-12: GFC recovery, low-rate environment, European debt crisis
    2015-01 to 2021-12: long bull market, COVID shock, rate normalisation

    If both sub-periods show similar coverage and width IC, the result is not
    driven by a single macro regime.
    """
    _print_section("CHECK 5: Sub-period stability (2008-2014 vs 2015-2021)")

    split = pd.Period(split_date, freq="M")

    pre  = test_intervals[test_intervals["date"] <  split]
    post = test_intervals[test_intervals["date"] >= split]

    # Lambda tuning (quick, using cal intervals if available)
    if cal_intervals is not None:
        lam_star, _ = lambda_tune(cal_intervals, [0.0, 0.3, 0.5, 0.7, 1.0], n_deciles=n_deciles)
    else:
        lam_star = 0.5

    rows = []
    print(f"  Lambda* = {lam_star}  (from calibration set)")
    print(f"\n  {'Sub-period':<22} {'Months':>7} {'Coverage':>10} "
          f"{'Width IC':>10} {'Width Sharpe':>13} {'UA Sharpe':>10}")
    print(f"  {'-'*75}")

    for label, df in [("2008-2014", pre), ("2015-2021", post)]:
        if len(df) < 500:
            continue

        cov    = _coverage(df)
        ic     = _width_ic(df)
        sharpe = _ls_sharpe(df, n_deciles=n_deciles)
        n_months = df["date"].nunique()

        # Uncertainty-adjusted L-S Sharpe
        scored = uncertainty_adjusted_score(df, lam_star)
        port   = construct_ls_portfolio(scored, score_col="score", n_deciles=n_deciles)
        stats  = portfolio_stats(port["ls_ew"])
        ua_sh  = stats["sharpe"]

        rows.append({
            "sub_period":   label,
            "n_months":     n_months,
            "coverage (%)": cov * 100,
            "width_ic":     ic,
            "width_ls_sharpe": sharpe,
            "ua_sharpe":    ua_sh,
            "lambda_star":  lam_star,
        })
        print(f"  {label:<22} {n_months:>7} {cov*100:>9.2f}% "
              f"{ic:>10.4f} {sharpe:>13.3f} {ua_sh:>10.3f}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CSRF Phase 6: robustness checks")
    p.add_argument("--config",   default="config.yaml")
    p.add_argument("--fast-dev", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)
    if args.fast_dev:
        cfg.fast_dev = True

    print("=" * 65)
    print("CSRF Phase 6 -- Robustness Checks")
    print(f"  fast_dev: {cfg.fast_dev}")
    print("=" * 65)

    splits_dir  = Path(cfg.data.splits_dir)
    results_dir = cfg.results_dir

    # -- Load data -----------------------------------------------------------
    print("\nLoading data...")
    cal_preds  = load_parquet(splits_dir / "cal_predictions.parquet")
    test_preds = load_parquet(splits_dir / "test_predictions.parquet")
    test_iv    = load_parquet(splits_dir / "test_intervals.parquet")

    # test.parquet for 'me' column
    try:
        test_df = load_parquet(splits_dir / "test.parquet", columns=["permno", "date", "me"])
    except Exception:
        test_df = pd.DataFrame(columns=["permno", "date", "me"])

    # cal_intervals optional
    cal_iv_path = splits_dir / "cal_intervals.parquet"
    cal_iv = load_parquet(cal_iv_path) if cal_iv_path.exists() else None

    if cfg.fast_dev:
        cal_preds  = fast_dev_subsample(cal_preds,  cfg.data)
        test_preds = fast_dev_subsample(test_preds, cfg.data)
        test_iv    = fast_dev_subsample(test_iv,    cfg.data)
        if cal_iv is not None:
            cal_iv = fast_dev_subsample(cal_iv, cfg.data)
        if not test_df.empty:
            test_df = fast_dev_subsample(test_df, cfg.data)

    print(f"  Cal preds:      {len(cal_preds):,} rows")
    print(f"  Test preds:     {len(test_preds):,} rows")
    print(f"  Test intervals: {len(test_iv):,} rows")

    # -- Run checks ----------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        df1 = check_alpha_sweep(
            cal_preds, test_preds, test_iv,
            alphas=[0.05, 0.10, 0.20],
        )

        df2 = check_history_bins(test_iv, cal_preds, test_preds)

        df3 = check_bootstrap_baseline(cal_preds, test_preds, alpha=cfg.spci.alpha)

        df4 = check_large_cap(
            test_iv, test_df,
            me_threshold=0.0, label="top half (me > 0)",
            n_deciles=cfg.portfolio.n_deciles,
        )

        df5 = check_sub_period(
            test_iv, cal_iv,
            split_date="2015-01",
            n_deciles=cfg.portfolio.n_deciles,
        )

    # -- Summary printout ----------------------------------------------------
    _print_section("ROBUSTNESS SUMMARY")
    print("  Check 1 (alpha sweep)     -- see alpha_sweep table")
    print("  Check 2 (history bins)    -- see history_bins table")
    print("  Check 3 (bootstrap)       -- see bootstrap_vs_spci table")
    print("  Check 4 (large-cap)       -- see large_cap table")
    print("  Check 5 (sub-period)      -- see sub_period table")

    # -- Save ----------------------------------------------------------------
    print("\nSaving results...")
    for df, name in [
        (df1, "robustness_alpha_sweep.csv"),
        (df2, "robustness_history_bins.csv"),
        (df3, "robustness_bootstrap_vs_spci.csv"),
        (df4, "robustness_large_cap.csv"),
        (df5, "robustness_sub_period.csv"),
    ]:
        if not df.empty:
            save_table(df, results_dir, name)
            print(f"  Saved: {name}")

    print("\nPhase 6 complete. Pipeline finished.")
    print("See LEARNING.md for full interpretation of all results.")


if __name__ == "__main__":
    main()
