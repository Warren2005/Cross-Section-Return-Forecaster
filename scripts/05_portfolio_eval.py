"""
Phase 5 -- Portfolio construction and evaluation.

Usage:
    python scripts/05_portfolio_eval.py [--fast-dev] [--config PATH]

Prerequisites:
    Phases 1-4 must have been run. Specifically:
      data/splits/cal_intervals.parquet   (Phase 3 output -- for lambda tuning)
      data/splits/test_intervals.parquet  (Phase 3 output -- test evaluation)
      data/raw/F-F_Research_Data_5_Factors_2x3.csv  (Ken French, optional)

What this script does:
  1. Tune lambda on the calibration set (grid search, frozen before test)
  2. Construct three L-S portfolios on the test set:
       - Gu baseline       (lambda=0)
       - Width-only        (lambda=1)
       - Uncertainty-adj   (lambda=lambda*)
  3. Compute gross performance statistics for each portfolio
  4. Deduct transaction costs (default 20 bps half-spread)
  5. Compute Fama-French 5-factor alpha for each (if FF5 data is available)
  6. Print and save the full results table

Outputs (results/tables/):
  lambda_tuning.csv         -- calibration Sharpe for each lambda candidate
  portfolio_results.csv     -- full comparison table (all three strategies)
  portfolio_ew_returns.csv  -- monthly EW L-S returns for each strategy
  portfolio_turnover.csv    -- monthly turnover per strategy

Verification gate:
  Uncertainty-adjusted Sharpe (gross EW) should exceed Gu baseline Sharpe.
  If not, the width signal did not improve portfolio construction.

FF5 alpha note:
  Download "Fama/French 5 Factors (2x3) [Monthly]" from Ken French's data
  library and save to data/raw/F-F_Research_Data_5_Factors_2x3.csv.
  If the file is absent, FF5 alpha is skipped and a warning is printed.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.analysis.portfolio import (
    apply_transaction_costs,
    build_results_table,
    construct_ls_portfolio,
    ff5_alpha,
    lambda_tune,
    portfolio_stats,
    print_results_table,
    uncertainty_adjusted_score,
)
from src.data.loader import load_ff5_factors
from src.data.splits import fast_dev_subsample
from src.utils.config import load_config
from src.utils.io import load_parquet, save_table


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CSRF Phase 5: portfolio evaluation")
    p.add_argument("--config",   default="config.yaml")
    p.add_argument("--fast-dev", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# FF5 loader with graceful fallback
# ---------------------------------------------------------------------------

def _try_load_ff5(raw_dir: Path) -> pd.DataFrame | None:
    candidates = [
        raw_dir / "F-F_Research_Data_5_Factors_2x3.CSV",
        raw_dir / "F-F_Research_Data_5_Factors_2x3.csv",
        raw_dir / "ff5_factors.csv",
    ]
    for path in candidates:
        if path.exists():
            try:
                df = load_ff5_factors(path)
                print(f"  Loaded FF5 factors: {path.name}  ({len(df)} months)")
                return df
            except Exception as e:
                print(f"  [!] Failed to load {path.name}: {e}")
    print(
        "  [!] FF5 factors file not found in data/raw/ -- FF5 alpha will be skipped.\n"
        "      Download from: mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        "data_library.html\n"
        "      File: 'Fama/French 5 Factors (2x3) [Monthly]'\n"
        "      Save to: data/raw/F-F_Research_Data_5_Factors_2x3.csv"
    )
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    if args.fast_dev:
        cfg.fast_dev = True

    print("=" * 65)
    print("CSRF Phase 5 -- Portfolio Construction and Evaluation")
    print(f"  fast_dev       : {cfg.fast_dev}")
    print(f"  lambda grid    : {cfg.portfolio.lambda_grid}")
    print(f"  half-spread    : {cfg.portfolio.half_spread_bps} bps")
    print(f"  n_deciles      : {cfg.portfolio.n_deciles}")
    print("=" * 65)

    splits_dir  = Path(cfg.data.splits_dir)
    raw_dir     = Path(cfg.data.raw_dir)
    results_dir = cfg.results_dir

    # -- 1. Load data ----------------------------------------------------------
    print("\n[1/6] Loading interval files...")

    # Calibration intervals for lambda tuning
    cal_path = splits_dir / "cal_intervals.parquet"
    if not cal_path.exists():
        print(
            "  [!] cal_intervals.parquet not found.\n"
            "      Re-run scripts/03_conformal_calibrate.py to generate it.\n"
            "      Proceeding with lambda=0 (Gu baseline only)."
        )
        cal_intervals = None
    else:
        cal_intervals = load_parquet(cal_path)
        if cfg.fast_dev:
            cal_intervals = fast_dev_subsample(cal_intervals, cfg.data)
        print(f"  Cal intervals:  {len(cal_intervals):,} rows  "
              f"({cal_intervals['date'].nunique()} months, "
              f"{cal_intervals['permno'].nunique():,} stocks)")

    # Test intervals for evaluation
    test_intervals = load_parquet(splits_dir / "test_intervals.parquet")
    if cfg.fast_dev:
        test_intervals = fast_dev_subsample(test_intervals, cfg.data)
    print(f"  Test intervals: {len(test_intervals):,} rows  "
          f"({test_intervals['date'].nunique()} months, "
          f"{test_intervals['permno'].nunique():,} stocks)")
    print(f"  Test period:    {test_intervals['date'].min()} -- "
          f"{test_intervals['date'].max()}")

    # Width statistics
    w_mean = test_intervals["width"].mean() * 1e4
    w_med  = test_intervals["width"].median() * 1e4
    print(f"  Width:          mean={w_mean:.0f} bps, median={w_med:.0f} bps")

    # -- 2. Optional FF5 factors -----------------------------------------------
    print("\n[2/6] Loading Fama-French 5 factors...")
    ff5_df = _try_load_ff5(raw_dir)

    # -- 3. Lambda tuning on calibration set -----------------------------------
    print("\n[3/6] Tuning lambda on calibration set...")

    if cal_intervals is not None:
        lambda_star, tune_df = lambda_tune(
            cal_df       = cal_intervals,
            lambda_grid  = cfg.portfolio.lambda_grid,
            n_deciles    = cfg.portfolio.n_deciles,
        )
        save_table(tune_df, results_dir, "lambda_tuning.csv")

        print(f"\n  Lambda grid search (cal set):")
        print(f"  {'lambda':>8}  {'Sharpe':>8}  {'Mean (%/mo)':>12}  {'t-stat':>8}")
        print(f"  {'-'*44}")
        for _, row in tune_df.iterrows():
            marker = " <-- SELECTED" if abs(row["lambda"] - lambda_star) < 1e-6 else ""
            print(f"  {row['lambda']:>8.1f}  {row['sharpe']:>8.3f}  "
                  f"{row['mean_monthly']*100:>12.3f}  {row['tstat']:>8.2f}{marker}")

        print(f"\n  lambda* = {lambda_star}  (maximises calibration Sharpe)")
    else:
        lambda_star = 0.5  # neutral fallback
        tune_df     = pd.DataFrame()
        print(f"  Calibration data unavailable -- using lambda* = {lambda_star} (default fallback)")

    # -- 4. Build three test-set portfolios ------------------------------------
    print("\n[4/6] Constructing L-S portfolios on test set...")

    strategies = {
        f"Gu (lam=0)":                 0.0,
        f"Width-only (lam=1)":         1.0,
        f"Uncert-adj (lam={lambda_star:.1f})": lambda_star,
    }

    port_results  = {}    # label -> construct_ls_portfolio output
    gross_stats   = {}    # label -> portfolio_stats dict
    net_stats     = {}    # label -> portfolio_stats dict (after costs)
    ff5_results   = {}    # label -> ff5_alpha dict or None
    turnover_means= {}    # label -> float

    ew_returns_dict = {}  # for saving monthly returns CSV

    for label, lam in strategies.items():
        print(f"\n  Strategy: {label}")

        scored = uncertainty_adjusted_score(test_intervals, lam)
        port   = construct_ls_portfolio(
            scored,
            score_col      = "score",
            n_deciles      = cfg.portfolio.n_deciles,
            market_cap_col = None,   # CRSP-DEPENDENT: set to 'me' if raw caps available
        )

        port_results[label] = port
        gross_stats[label]  = portfolio_stats(port["ls_ew"])

        to_mean = float(port["turnover"].mean()) if len(port["turnover"]) > 0 else 0.5
        turnover_means[label] = to_mean

        # Transaction cost adjustment
        net_ls = apply_transaction_costs(
            port["ls_ew"],
            port["turnover"],
            half_spread_bps = cfg.portfolio.half_spread_bps,
        )
        net_stats[label] = portfolio_stats(net_ls)

        # FF5 alpha
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ff5_results[label] = ff5_alpha(port["ls_ew"], ff5_df, n_lags=6)

        ew_returns_dict[label] = port["ls_ew"]

        s = gross_stats[label]
        print(f"    Gross EW Sharpe: {s['sharpe']:.3f}  "
              f"(mean={s['mean_monthly']*100:.3f}%/mo, "
              f"t={s['tstat']:.2f}, n={s['n_months']}mo)")
        print(f"    Mean turnover:   {to_mean*100:.1f}%")
        print(f"    Net EW Sharpe:   {net_stats[label]['sharpe']:.3f}")
        if ff5_results[label]:
            fa = ff5_results[label]
            print(f"    FF5 alpha:       {fa['alpha_monthly']*100:.3f}%/mo  "
                  f"(t={fa['t_stat']:.2f})")

    # -- 5. Print full results table -------------------------------------------
    print("\n[5/6] Results summary:")

    results_tbl = build_results_table(
        strategies     = gross_stats,
        ff5_results    = ff5_results,
        cost_stats     = net_stats,
        turnover_means = turnover_means,
    )
    print_results_table(results_tbl, half_spread_bps=cfg.portfolio.half_spread_bps)

    # Gate check
    gu_sharpe = gross_stats[f"Gu (lam=0)"]["sharpe"]
    ua_label  = f"Uncert-adj (lam={lambda_star:.1f})"
    ua_sharpe = gross_stats[ua_label]["sharpe"]
    if not np.isnan(ua_sharpe) and not np.isnan(gu_sharpe):
        if ua_sharpe > gu_sharpe:
            print(f"\n  [OK]  Uncertainty-adjusted Sharpe ({ua_sharpe:.3f}) > "
                  f"Gu baseline ({gu_sharpe:.3f}) -- width signal adds value")
        else:
            print(f"\n  [!]  Uncertainty-adjusted Sharpe ({ua_sharpe:.3f}) does NOT exceed "
                  f"Gu baseline ({gu_sharpe:.3f}) -- investigate width signal")

    # -- 6. Save outputs -------------------------------------------------------
    print("\n[6/6] Saving results...")

    save_table(results_tbl, results_dir, "portfolio_results.csv")
    print(f"  Saved: portfolio_results.csv")

    ew_df = pd.DataFrame(ew_returns_dict)
    ew_df.index.name = "date"
    save_table(ew_df, results_dir, "portfolio_ew_returns.csv")
    print(f"  Saved: portfolio_ew_returns.csv")

    to_df = pd.DataFrame({k: port_results[k]["turnover"] for k in port_results})
    to_df.index.name = "date"
    save_table(to_df, results_dir, "portfolio_turnover.csv")
    print(f"  Saved: portfolio_turnover.csv")

    if not tune_df.empty:
        print(f"  Saved: lambda_tuning.csv")

    # FF5 factor betas for reference
    ff5_rows = []
    for label, fa in ff5_results.items():
        if fa:
            row = {"strategy": label, "alpha_monthly": fa["alpha_monthly"],
                   "alpha_annual": fa["alpha_annual"], "t_stat": fa["t_stat"],
                   "r2": fa["r2"], "n_months": fa["n_months"]}
            row.update(fa["betas"])
            ff5_rows.append(row)
    if ff5_rows:
        ff5_out = pd.DataFrame(ff5_rows)
        save_table(ff5_out, results_dir, "portfolio_ff5_alpha.csv")
        print(f"  Saved: portfolio_ff5_alpha.csv")

    print(f"\n  All results in: {results_dir}/tables/")
    print("\nPhase 5 complete.")
    print("Next step: python scripts/06_robustness.py")


if __name__ == "__main__":
    main()
