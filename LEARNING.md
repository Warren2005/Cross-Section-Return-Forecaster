# CSRF Learning Document
### Calibrated Cross-Sectional Return Forecaster — Theory, Design Decisions, and Codebase Guide

> **Purpose:** This document is the single source of truth for anyone trying to understand *why* this project is built the way it is. It is written for a reader who is technically literate but has no prior familiarity with this specific project. A new conversation with Claude should be able to reconstruct the full context of the project from this document alone.
>
> **Maintenance rule:** Every non-obvious design decision made during implementation is recorded here in the relevant section *before* moving on. This document grows alongside the code.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Background Theory](#2-background-theory)
   - 2.1 [ML in Asset Pricing](#21-ml-in-asset-pricing)
   - 2.2 [Conformal Prediction](#22-conformal-prediction)
   - 2.3 [SPCI for Time Series Panels](#23-spci-for-time-series-panels)
   - 2.4 [The Width-as-Signal Hypothesis](#24-the-width-as-signal-hypothesis)
   - 2.5 [Portfolio Evaluation Metrics](#25-portfolio-evaluation-metrics)
3. [Data](#3-data)
   - 3.1 [The GKX Dataset](#31-the-gkx-dataset)
   - 3.2 [Rank Normalization](#32-rank-normalization)
   - 3.3 [Lag Enforcement and Look-Ahead Bias](#33-lag-enforcement-and-look-ahead-bias)
   - 3.4 [Train / Calibration / Test Split](#34-train--calibration--test-split)
4. [Base Model](#4-base-model)
   - 4.1 [Architecture Choices](#41-architecture-choices)
   - 4.2 [Training Protocol](#42-training-protocol)
   - 4.3 [Validating Against Gu et al.](#43-validating-against-gu-et-al)
5. [Conformal Layer](#5-conformal-layer)
   - 5.1 [Why SPCI over Standard Split Conformal](#51-why-spci-over-standard-split-conformal)
   - 5.2 [Feature Design for the Quantile Model](#52-feature-design-for-the-quantile-model)
   - 5.3 [Fallback for Short-History Stocks](#53-fallback-for-short-history-stocks)
   - 5.4 [Interpreting the Coverage Table](#54-interpreting-the-coverage-table)
6. [Width Signal](#6-width-signal)
   - 6.1 [Fama-MacBeth Regression Setup](#61-fama-macbeth-regression-setup)
   - 6.2 [Double-Sort Interpretation](#62-double-sort-interpretation)
   - 6.3 [FF5 Alpha and What It Proves](#63-ff5-alpha-and-what-it-proves)
7. [Portfolio Construction](#7-portfolio-construction)
   - 7.1 [Why λ Is Tuned on the Calibration Set and Frozen](#71-why-λ-is-tuned-on-the-calibration-set-and-frozen)
   - 7.2 [Transaction Cost Assumptions](#72-transaction-cost-assumptions)
8. [Codebase Navigation](#8-codebase-navigation)
9. [Glossary](#9-glossary)

---

## 1. Project Overview

### What problem are we solving?

Every major machine learning paper in asset pricing — including the landmark Gu, Kelly & Xiu (2020) paper published in the *Review of Financial Studies* — predicts stock returns as a single number: a *point estimate*. Given 94 firm characteristics (things like momentum, book-to-market ratio, and earnings surprises), the model outputs one number: the expected return for that stock next month. Stocks are then sorted into deciles by this number, and a long-short portfolio is formed by buying the top decile (predicted winners) and shorting the bottom decile (predicted losers).

This approach has a blind spot: **it ignores the model's uncertainty**. Suppose stock A and stock B both have a predicted return of +1%. But the model is very confident about A (it has a clear signal from its characteristics) and quite uncertain about B (its characteristics are unusual, or contradictory). Under pure point-estimate sorting, A and B are treated identically. The uncertainty — which contains real information — is thrown away.

### What does this project do differently?

We add a *conformal prediction layer* on top of the base ML model. Conformal prediction doesn't just produce a point estimate; it produces a **prediction interval**: a range of returns that is guaranteed to contain the true realized return at least 90% of the time (or whatever confidence level we choose). Crucially, this guarantee is *distribution-free* — it holds regardless of the statistical assumptions the model makes, and it holds with finite samples, not just asymptotically.

The interval for stock A might be `[+0.2%, +1.8%]` (narrow, confident) and for stock B it might be `[-3%, +5%]` (wide, uncertain). Both have the same midpoint (+1%) but very different widths.

We then test a novel empirical hypothesis: **does interval width, by itself, predict subsequent returns?** Specifically, does sorting stocks by interval width — holding the narrow-interval stocks and shorting the wide-interval stocks — produce positive risk-adjusted returns? This is the central research claim.

### Why does this matter?

Three reasons:

1. **Distributional robustness.** Prior work on uncertainty-adjusted portfolios (Liu et al. 2026) uses bootstrap or Bayesian uncertainty estimates. These are mathematically valid only under specific assumptions about the data-generating process — assumptions that are almost certainly violated during market crises. Conformal prediction gives a coverage guarantee with *no* assumptions. This is especially valuable during the 2008 financial crisis and the 2020 COVID shock.

2. **A new cross-sectional signal.** If the width signal survives Fama-French 5-factor adjustment, it represents genuinely new information — a model-derived measure of prediction difficulty that is not captured by size, value, momentum, or profitability factors.

3. **A cleaner portfolio.** By down-weighting uncertain stocks in portfolio construction, we get higher risk-adjusted returns than pure point-estimate sorting.

---

## 2. Background Theory

### 2.1 ML in Asset Pricing

#### What Gu, Kelly & Xiu (2020) did

Gu, Kelly & Xiu (GKX) published "Empirical Asset Pricing via Machine Learning" in the *Review of Financial Studies* in 2020. It is the most influential ML asset pricing paper to date (3,000+ citations).

Their setup:
- Dataset: ~30,000 US stocks, monthly observations from 1957 to 2016
- Features: 94 firm-level characteristics (accounting ratios, price-based signals, etc.) + 74 industry dummies
- Outcome: next-month excess return (return above the risk-free rate)
- Models tested: OLS, PCR, PLS, Ridge, Lasso, Elastic Net, Random Forest, Gradient Boosting, and 5 neural network architectures (NN1 through NN5)

Their key finding: neural networks, especially the deeper ones (NN4, NN5), produce the highest out-of-sample R² (~0.40%) and the best long-short Sharpe ratio (~1.4 annualized). A 0.40% R² sounds tiny but is highly significant in a setting where returns are extremely noisy — the predictable component of monthly stock returns is genuinely small.

#### Why 0.40% R² is impressive

Monthly stock returns have a signal-to-noise ratio close to 0. A typical stock has monthly return volatility of ~8% and an expected return of ~0.7%. Most of the variation in returns is random — it cannot be predicted by any model. An R² of 0.40% means the model explains 0.40% of total variance, which corresponds to capturing essentially all of the *predictable* component. For context, simpler models like OLS achieve R² ≈ -0.10% (they destroy predictive value on the test set).

#### What this project builds on top of GKX

We replicate the GKX NN5 architecture and use it as the base model. We then add a conformal calibration layer and test whether uncertainty (interval width) is itself a signal. Our baseline comparison is GKX's NN5 — if our uncertainty-conditioned portfolio doesn't beat it, the conformal layer adds no economic value.

---

### 2.2 Conformal Prediction

#### The core guarantee

Conformal prediction is a framework for constructing *statistically guaranteed* prediction intervals. The key result (Vovk et al. 2005, Angelopoulos & Bates 2023 tutorial) is:

Given:
- A training set used to fit a model
- A calibration set not used during training (hold-out)
- A test point drawn from the same distribution

The conformal prediction interval `C_α(X)` satisfies:

```
P(Y_new ∈ C_α(X_new)) ≥ 1 - α
```

This means: if we set α = 0.10, then at least 90% of new observations will fall inside the interval. This holds *exactly* (not approximately), *for finite samples* (not just asymptotically), and *for any distribution of (X, Y)* (not just Gaussian or any parametric family).

Compare this to:
- **Bootstrap confidence intervals**: approximately valid for large samples; wrong if the model is misspecified
- **Bayesian credible intervals**: valid under the model's prior and likelihood; wrong if the model is misspecified

Conformal prediction's guarantee is unconditional and non-parametric.

#### How split conformal prediction works

The simplest version is **split conformal prediction**:

1. Train a model `f` on the training set.
2. On the calibration set (not seen during training), compute nonconformity scores:
   ```
   s_i = |y_i - f(x_i)|    (absolute residual)
   ```
3. Compute the empirical quantile:
   ```
   q̂ = the ⌈(n+1)(1-α)⌉/n quantile of {s_1, ..., s_n}
   ```
   (The adjustment from `n` to `n+1` corrects for finite-sample coverage.)
4. For any new test point x:
   ```
   Interval = [f(x) - q̂,  f(x) + q̂]
   ```

The interval is symmetric around the point estimate and has the same width for every test point (it uses the *global* calibration quantile).

#### Why symmetric, fixed-width intervals are limiting

This "standard" conformal approach has a significant limitation for our use case: every stock gets the same interval width. If we want interval width to be a *signal* — carrying information about individual stocks — we need the width to vary across stocks. This requires **conditional** conformal prediction: the interval width adapts to the characteristics of each specific stock.

SPCI (described in Section 2.3) achieves this by conditioning the width on the stock's history of past prediction errors.

---

### 2.3 SPCI for Time Series Panels

#### Why standard conformal fails for financial panels

Standard split conformal prediction requires **exchangeability** — roughly, the calibration and test points must be interchangeable (drawn i.i.d. from the same distribution). This is violated in two ways in our setting:

1. **Temporal dependence:** The return of a stock in month t is correlated with its return in month t-1. The error of our model in month t is correlated with the error in month t-1 (because the model's systematic misspecification is persistent). This violates exchangeability across time.

2. **Cross-sectional dependence:** In any given month, all stock returns are correlated through the market return and sector factors. If the market crashes, all stocks fall together — the residuals of all stocks in month t are correlated with each other.

If we naively apply standard split conformal to this panel, the coverage guarantee breaks down. In practice, coverage will be below 90% during systematic stress events (exactly when we need it most).

#### SPCI: Sequential Predictive Conformal Inference

Xu & Xie (ICML 2023) developed SPCI specifically for time series settings where exchangeability fails. The key insight: instead of using a fixed global quantile of calibration residuals, SPCI **learns a model for the conditional quantile of the next residual** using the history of past residuals.

In words: if stock i has had very large prediction errors in recent months (|ε_{t-1}|, |ε_{t-2}|, ...), SPCI will predict a wider interval for it in month t. If it has had small, stable errors, SPCI predicts a narrower interval. This is adaptive conformal prediction conditioned on residual history.

The procedure:

1. On the calibration set, compute residuals `ε_{i,t} = y_{i,t} - ŷ_{i,t}` for all stocks i and months t.

2. Build a training set for a **quantile regression model**:
   - Features for stock i at time t: `[|ε_{i,t-1}|, ..., |ε_{i,t-L}|, sign(ε_{i,t-1}), ..., rolling_std, rolling_MAE]`
   - Target: `|ε_{i,t}|` (the absolute residual we want to predict)

3. Fit **two** quantile regression models:
   - One for the lower α/2 quantile (e.g., 5th percentile)
   - One for the upper 1-α/2 quantile (e.g., 95th percentile)
   
   We use Gradient Boosting with `loss='quantile'` (also called pinball/quantile loss), which directly minimizes the quantile regression objective.

4. At test time, for stock i in month t:
   - Look up the stock's last L residuals (from calibration period)
   - Run both quantile models to get `lo_i` and `hi_i`
   - Prediction interval: `[ŷ_{i,t} + lo_i,  ŷ_{i,t} + hi_i]`
   - Interval width: `hi_i - lo_i`  ← **this is our signal**

#### Adapting SPCI to the panel setting

The original SPCI paper considers a single time series. We have a panel of ~3,000 stocks, each with its own residual history. We treat each stock's history as an independent time series and pool all stocks' training examples into a single quantile model. This is the correct approach when stocks share a common volatility regime (which they do — during the 2008 crisis, all stocks' residuals blow up together) but have idiosyncratic residual autocorrelation.

The pooled model learns two things simultaneously:
- Cross-sectional: high-volatility stocks (e.g., small caps) have systematically larger residuals than low-volatility stocks
- Temporal: a stock whose recent residuals were large will tend to have large residuals again (volatility clustering)

---

### 2.4 The Width-as-Signal Hypothesis

#### The core claim

The central empirical claim of this project is: **conformal interval width, measured in the cross-section each month, predicts subsequent stock returns.** Specifically:

- Stocks with *narrow* conformal intervals (model is confident) tend to have *higher* subsequent risk-adjusted returns
- Stocks with *wide* conformal intervals (model is uncertain) tend to have *lower* subsequent risk-adjusted returns
- This relationship is *orthogonal* to the point estimate itself and to known risk factors

If true, this means the ML model's own uncertainty signal contains economically valuable information — over and above the point estimate that practitioners already use.

#### Economic intuition

Why would model uncertainty predict returns? Three mechanisms:

**1. Idiosyncratic noise loading.** Stocks with unusual characteristic combinations — firms that don't fit neatly into the patterns learned by the model — have higher model uncertainty. These firms tend to have higher idiosyncratic volatility, which means their returns are noisier and harder to predict. Investors demand a premium for holding idiosyncratic noise, but empirically (the "idiosyncratic volatility puzzle," Ang et al. 2006), high-IVOL stocks *underperform* after controlling for systematic factors. Model uncertainty may capture this effect through a more model-aware lens.

**2. Information asymmetry.** Stocks where the model is highly uncertain are often those with less available public information — small caps, firms with opaque accounting, firms in unusual industries. These stocks are underanalyzed and the model's uncertainty reflects genuine information gaps. Lower analyst coverage → lower price efficiency → potentially lower risk-adjusted returns (market microstructure costs dominate).

**3. Regime instability.** A wide interval may signal that the stock is in a regime where the historical relationship between characteristics and returns has broken down. Firms in financial distress, firms undergoing restructuring, or firms in nascent industries have characteristics that behave differently from historical norms. The model is uncertain because the past is not a reliable guide to the future for these firms.

#### Connection to but distinction from the IVOL puzzle

Ang, Hodrick, Xing & Zhang (2006) documented that stocks with high idiosyncratic volatility (IVOL — variance of residuals from a Fama-French 3-factor regression) have *lower* subsequent returns. This is the "IVOL puzzle" — a violation of the intuition that higher risk should earn higher return.

Conformal interval width is related to IVOL: both measure how hard a stock's return is to predict. But they differ in three important ways:

1. IVOL uses a linear Fama-French model; interval width uses a deep neural network with 94 characteristics. The NN-based width captures non-linear interactions between characteristics that IVOL ignores.
2. IVOL measures historical realized variance; interval width is a *model-derived* forward-looking measure of prediction difficulty.
3. IVOL is computed on past returns; width is computed fresh each month using current characteristics.

The Fama-MacBeth regression (Section 6.1) directly tests whether width has incremental explanatory power after controlling for IVOL and other known signals.

---

### 2.5 Portfolio Evaluation Metrics

#### Out-of-sample R²

The standard metric for return predictability:

```
OOS R² = 1 - MSE(model) / Var(realized returns)
       = 1 - Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²
```

A positive R² means the model beats the historical mean as a forecast. A negative R² means the model is *worse* than just predicting the historical average return for every stock. Gu et al. report R² ≈ 0.40% for NN5. We target the same range.

#### Annualized Sharpe Ratio

The primary economic metric for portfolio performance:

```
Sharpe = (mean monthly return / std of monthly returns) × √12
```

The √12 converts from monthly to annualized. A Sharpe of 1.4 (Gu et al. NN5 long-short) means the portfolio earns 1.4 standard deviations of return per year. For context:
- A Sharpe > 1.0 is considered excellent for a long-short equity strategy
- The S&P 500 has a long-run Sharpe of about 0.4-0.5
- Hedge funds report Sharpes of 0.5-1.5 in their best strategies

#### Coverage

The fraction of realized returns that fall inside the prediction interval:

```
Empirical coverage = (# observations where lower ≤ y ≤ upper) / total observations
```

For a 90% conformal interval (α = 0.10), we require coverage ≥ 90%. Coverage below 90% means the intervals are too narrow — they're not honoring the guarantee. Coverage significantly above 90% means intervals are too wide (conservative).

#### Fama-French 5-Factor Alpha

The FF5 model (Fama & French 2015) explains stock returns through five factors:
- MKT: Market excess return (beta)
- SMB: Small Minus Big (size)
- HML: High Minus Low (value / book-to-market)
- RMW: Robust Minus Weak (profitability)
- CMA: Conservative Minus Aggressive (investment)

We regress our portfolio's monthly returns on these five factors:
```
R_{portfolio,t} = α + β₁·MKT_t + β₂·SMB_t + β₃·HML_t + β₄·RMW_t + β₅·CMA_t + ε_t
```

The **alpha (α)** is the return unexplained by the five factors — the "abnormal return." A positive alpha with t-stat > 2.0 means the portfolio earns returns that cannot be explained by exposure to known risk factors. This is the strongest evidence that the width signal contains genuinely new information.

#### Information Coefficient (IC)

Spearman rank correlation between predicted values (or signal values) and realized returns:

```
IC = Spearman_corr(signal_{i,t}, return_{i,t+1})
```

Positive IC means the signal is directionally correct more often than not. Monthly ICs above 0.03 are considered meaningful in equity research. The Information Ratio (IR = mean(IC) / std(IC)) measures the consistency of the signal.

---

## 3. Data

### 3.1 The GKX Dataset

**Download:** https://dachxiu.chicagobooth.edu/#rp (free, no login required). The file is a zip containing a SAS `.sas7bdat` file or CSV. Place it in `data/raw/`; the loader auto-detects the format.

**Structure:** ~30,000 US stocks, monthly observations from 1957-01 to 2021-12. Each row is a (permno, month) pair with:
- `permno` — CRSP permanent stock identifier (integer, stable across corporate events)
- `date` — encoded as YYYYMM integer in the raw file; parsed to `pd.Period('M')` by the loader
- `ret` — monthly excess return (already net of the risk-free rate)
- 94 firm-level characteristics (accounting ratios, price-based signals, etc.)
- 74 industry dummy columns (binary 0/1, based on 2-digit SIC codes)

**Missing values:** Characteristics are frequently missing for small or newly-listed firms. Roughly 10–30% of characteristic-stock-month cells are NaN in the raw data. Rank normalization preserves these NaNs (using `na_option='keep'`). The neural network imputes NaN to 0 in its DataLoader, which equals the cross-sectional median after normalization to [-1, 1].

**Column detection:** `loader.py` detects industry dummy columns by SIC prefix naming (`sic10`, `sic12`, etc.) or falls back to identifying binary {0,1} columns. Characteristics are everything that is neither an identifier nor a dummy. Column metadata is saved to `data/splits/column_meta.json` by `01_build_data.py` for use by all downstream scripts.

**Why parquet for storage?** The GKX panel with all columns is ~2 GB as CSV. Parquet compresses this to ~200 MB, preserves dtypes (critical for `pd.Period` date columns), and supports column-subset loading — loading only the feature columns without reading `ret` avoids unnecessary RAM use during model training.

### 3.2 Rank Normalization

**What it does:** Within each month, for each characteristic, rank all stocks (average rank for ties) and map the ranks to `[-1, 1]` using:

```
normalized = 2 × (rank - 1) / (n_valid - 1) - 1
```

where `n_valid` is the number of non-missing values in that month. A stock at the cross-sectional median gets 0; the lowest-ranked stock gets -1; the highest-ranked gets +1.

**Why rank normalization instead of winsorization or z-scoring?**

Three reasons:

1. **Outlier robustness without arbitrary thresholds.** Winsorizing (capping at e.g. the 1st/99th percentile) requires choosing thresholds; the choice materially affects results and different papers use different levels. Z-scoring doesn't remove outliers at all — a stock with book-to-market of 100× the median remains 100× after z-scoring. Rank normalization maps every stock to [-1, 1] regardless of the magnitude of its raw value.

2. **Temporal comparability.** Raw characteristic values drift over time (e.g., average P/E ratios are higher in 2020 than 1960). After rank normalization, the cross-sectional distribution is always uniform on [-1, 1] in every month, making the feature space stationary over the 65-year sample.

3. **Preserves ordinal information.** We care about relative ordering (is stock A richer than stock B on this metric?) not absolute levels. Rank normalization is a monotone transformation — it preserves ordering exactly.

**Industry dummies:** The 74 binary industry indicators are also rank-normalized. For a binary column, this separates the in-industry stocks (which receive a value near +1) from out-of-industry stocks (near -1), with the exact values depending on the proportion of stocks in that industry that month. This is consistent with Gu et al. and allows the NN to learn industry-adjusted signals uniformly.

**Implementation:** `preprocess.py:rank_normalize()`. Applied date-by-date with a tqdm progress bar (the full panel takes several minutes on CPU).

### 3.3 Lag Enforcement and Look-Ahead Bias

**What look-ahead bias means:** If we use a characteristic at its current value (e.g., the book-to-market ratio as of the balance sheet date) to predict the stock's return in the same month, we are using information that was not publicly available at the time of trading. This artificially inflates predictive performance and makes the strategy impossible to implement live.

**The specific problem in financial data:** Accounting data is not published instantaneously. A firm's annual earnings for fiscal year ending December 31 are typically filed with the SEC 60–90 days later (March/April). If we use December 31 accounting data to form a January portfolio, we are using data that didn't exist yet in January.

**Our lag rules** (following Gu et al. 2020 exactly):

| Characteristic frequency | Filing delay | Lag applied |
|--------------------------|-------------|-------------|
| Monthly (price, volume)  | 1 month     | 1-month row shift per stock |
| Quarterly (earnings)     | 4 months    | 4-month row shift per stock |
| Annual (balance sheet)   | 6 months    | 6-month row shift per stock |

**Implementation detail:** Lags are applied as row shifts within each stock's time series (`groupby('permno').shift(N)`). This is valid when each stock appears at most once per month, which holds for the GKX panel. The shift introduces NaN values for the first N months of each stock's history — these are handled by the NaN-preserving rank normalization and the NN's zero imputation.

**Order matters:** Lags are applied BEFORE rank normalization. If we normalized first, then lagged, we would be rank-normalizing the unlagged raw values and then discarding them — we would lose the cross-sectional ordering information that normalization preserves.

**Characteristic classification:** The 94 characteristics are assigned to frequency categories using the lists in `src/utils/config.py` (`MONTHLY_CHARS`, `QUARTERLY_CHARS`). Anything not in either list is treated as annual (6-month lag) — this is conservative, erring on the side of longer delays when unsure.

### 3.4 Train / Calibration / Test Split

**The three sets:**

| Set | Dates | Months | Purpose |
|-----|-------|--------|---------|
| Train | 1957-01 to 1999-12 | 504 | NN weight learning |
| Calibration | 2000-01 to 2007-12 | 96 | SPCI quantile model; λ tuning |
| Test | 2008-01 to 2021-12 | 168 | All evaluation |

**Why 1999/2000 as the train/cal boundary?**

The calibration set needs to be large enough to estimate stable conformal quantiles. With ~3,000 stocks per month over 96 months, the calibration set contains approximately 288,000 stock-month residuals — more than enough for the SPCI quantile model. Starting calibration in 2000 also gives the NN 43 years of training data (1957–1999), which is the same range used by Gu et al. for their main results.

**Why 2008 as the cal/test boundary?**

2008 is where the interesting evaluation happens. The test set contains three major market stress regimes: the 2008–2009 financial crisis, the 2020 COVID shock, and the 2022 interest rate shock. These are the hardest cases for any prediction model and the most important cases for testing the conformal coverage guarantee. Starting the test set in 2008 maximises the stress-test content of the evaluation period.

**The calibration set quarantine rule:** The calibration set is used for exactly two things:
1. Computing the SPCI residual quantile model (Phase 3)
2. Tuning the portfolio weight λ (Phase 5)

It is never used to train the NN and never used to report final results. Violating this rule — e.g., by looking at test-set results and then adjusting parameters — would constitute data snooping and invalidate the research findings.

**Validation:** `splits.py:_validate_splits()` asserts that no date appears in more than one split before returning. This is a hard guard against date-boundary bugs.

---

## 4. Base Model

### 4.1 Architecture Choices

The network is a 5-layer feed-forward network matching the Gu et al. (2020) NN5 specification: Input → 32 → 16 → 8 → 4 → 1. We add three modifications; each is justified below.

**BatchNorm1d before each activation**

Cross-sectional features have very different scales even after rank normalisation — a stock's characteristics vary across industries, size groups, and time periods in ways that aren't fully removed by the per-column normalisation. BatchNorm rescales each hidden layer's activations to zero mean and unit variance before the ReLU, which prevents the gradient from vanishing in deeper layers and speeds up convergence substantially. Without BatchNorm, training on the full GKX panel with 168 features typically diverges or converges very slowly on CPU. Gu et al. do not use BatchNorm (they use a simpler architecture and likely had GPU resources), but it is standard practice in modern MLPs for tabular data.

**Skip connection from layer 1 to layer 3**

```
h3 = layer3(layer2(h1)) + skip_proj(h1)
```

The skip connection allows layer 4 and the output layer to receive a direct gradient signal from layer 1, bypassing the two intermediate transformations. Without it, the gradient signal from the output (which is a very noisy single-month return) must pass through four non-linear layers before reaching the first layer's weights — causing slow learning of the early feature representations. The skip projection (`Linear(32, 8)`) is a learned linear map, not a residual identity, because the dimensions differ.

**Huber loss (δ = 0.5) instead of MSE**

Monthly stock returns have heavy tails: kurtosis is typically 10–20× that of a normal distribution. Crisis months (October 2008, March 2020) produce returns of -30% or worse for some stocks. Under MSE loss, these extreme observations dominate the gradient — a single month can move the parameters more than dozens of normal months. Huber loss is quadratic (like MSE) for errors smaller than δ and linear (like MAE) for errors larger than δ. With δ = 0.5 (in return units, so 0.5% monthly), the loss limits the influence of outliers beyond ±0.5% returns while still producing unbiased gradients for the typical range of returns.

**Kaiming initialisation**

All Linear layers use Kaiming uniform initialisation (He et al. 2015), which is designed for ReLU activations and ensures the variance of activations is preserved at initialisation. Random initialisation with standard normal would cause gradients to vanish immediately for a network this deep.

**Implementation:** `src/models/network.py:ReturnPredictor`

### 4.2 Training Protocol

**Why chronological batching is mandatory**

Standard PyTorch DataLoader shuffles the entire dataset randomly before each epoch. In our setting, this would allow the model to see stock A's return in month t+1 before it processes stock A's characteristics in month t — information leakage. More subtly, random shuffling means the model can observe the *distribution* of future returns during training, which inflates out-of-sample performance.

`ChronologicalBatchSampler` enforces two constraints:
1. All observations in a batch belong to the same calendar month.
2. Months are emitted in ascending chronological order.

Within each month, stocks are shuffled randomly (controlled by seed + epoch for reproducibility). This within-month shuffle is safe because all stocks in a given month share the same calendar date — there is no temporal ordering to violate.

**Train / validation split within training**

The 1957–1999 training period is divided at `val_start = 1992-01`:
- **Pure training (1957–1991, ~35 years):** gradient updates
- **Validation (1992–1999, ~8 years):** early stopping only — no gradients

This split gives the validation set a different market regime from the pure training set (1990s bull market vs. the mixed 1957–1991 period), which makes early stopping more informative. The validation loss is computed on a forward-pass with `torch.no_grad()` — the model never fits to validation data.

**Early stopping:** If validation loss does not improve for 20 consecutive epochs, training stops and the best weights are restored. This prevents overfitting on the noisy return signal without requiring a fixed epoch count.

**Checkpointing (critical for CPU)**

The script saves the best model to `data/splits/best_model.pt` whenever validation loss improves, and saves a periodic snapshot every 5 epochs to `data/splits/checkpoint_epoch_NNN.pt`. If training is interrupted (power outage, closed terminal), re-running `02_train_model.py` automatically resumes from `best_model.pt`. This is essential given that a full CPU training run can take 6–15 hours.

**CPU-specific settings**

- `num_workers=0`: PyTorch's multiprocess data loading is unreliable on Windows. Using the main process avoids deadlocks.
- `pin_memory=False`: `pin_memory` accelerates GPU transfers; on CPU it wastes time.
- `batch_size=2048`: large enough that each month (~3,000 stocks) fits in 1–2 batches, small enough to keep per-batch gradient updates meaningful.

**Implementation:** `src/models/train.py:ChronologicalBatchSampler`, `train()`

### 4.3 Validating Against Gu et al.

**The target**

Gu et al. (2020) report OOS R² ≈ 0.40% for NN5 in their Table 4. Our implementation should land in [0.20%, 0.65%]:
- Below 0.20%: likely a preprocessing bug (look-ahead bias not properly prevented, or rank normalisation applied in the wrong order)
- Above 0.65%: almost certainly data leakage — future information contaminating training

We evaluate R² on the *calibration set* (2000–2007), not the test set, to preserve test-set integrity during development.

**OOS R² formula**

```
R² = 1 - Σ(y - ŷ)² / Σ(y - ȳ_train)²
```

The denominator uses `ȳ_train` — the mean return from the training set — not the mean from the calibration or test set. This is Gu et al.'s exact formula. It is more conservative than using the in-sample mean because the historical mean changes over time.

**Common failure modes**

| Symptom | Most likely cause |
|---------|------------------|
| R² < 0 (worse than mean forecast) | Lag enforcement not applied; features contain future data |
| R² ≈ 0 (tiny positive) | Characteristics overfit to training period; try smaller model |
| R² > 1% | Data leakage — test data is contaminating the training set |
| L-S Sharpe < 0.5 | Batch sampler bug (shuffling across months); check `ChronologicalBatchSampler` |
| Training loss diverges | BatchNorm or learning rate issue; try lr=1e-4 |

**Inference outputs**

The script generates three prediction files, each with columns `[permno, date, y_true, y_pred, residual]`:
- `train_predictions.parquet` — needed by Phase 3 to build SPCI residual histories
- `cal_predictions.parquet` — used for SPCI fitting and λ tuning
- `test_predictions.parquet` — all final evaluation in Phase 4 and 5

---

## 5. Conformal Layer

*This section will be filled in during Phase 3 implementation.*

### 5.1 Why SPCI over Standard Split Conformal

*[To be written]*

### 5.2 Feature Design for the Quantile Model

*[To be written]*

### 5.3 Fallback for Short-History Stocks

*[To be written]*

### 5.4 Interpreting the Coverage Table

*[To be written]*

---

## 6. Width Signal

*This section will be filled in during Phase 4 implementation.*

### 6.1 Fama-MacBeth Regression Setup

*[To be written]*

### 6.2 Double-Sort Interpretation

*[To be written]*

### 6.3 FF5 Alpha and What It Proves

*[To be written]*

---

## 7. Portfolio Construction

*This section will be filled in during Phase 5 implementation.*

### 7.1 Why λ Is Tuned on the Calibration Set and Frozen

*[To be written]*

### 7.2 Transaction Cost Assumptions

*[To be written]*

---

## 8. Codebase Navigation

*This section will be filled in at the end of Phase 6.*

---

## 9. Glossary

| Term | Definition |
|------|-----------|
| **alpha (α)** | In conformal prediction: the miscoverage level (e.g., α=0.10 → 90% intervals). In finance: abnormal return unexplained by risk factors. Context determines which meaning applies. |
| **calibration set** | The held-out dataset used *only* to compute conformal quantiles and tune λ. Never used for training the NN or evaluating final results. |
| **conformal prediction** | A framework for constructing statistically guaranteed prediction intervals without distributional assumptions. |
| **coverage** | The fraction of realized outcomes that fall inside the prediction interval. For a 90% interval, coverage should be ≥ 90%. |
| **decile** | When stocks are sorted by a signal into 10 equally sized groups. D1 = bottom 10%, D10 = top 10%. |
| **exchangeability** | A statistical property (weaker than i.i.d.) required for conformal prediction guarantees. Roughly: the joint distribution of data points is symmetric with respect to permutations. |
| **Fama-MacBeth regression** | A two-step procedure: (1) run a cross-sectional regression each month; (2) report time-series averages of the monthly coefficients with Newey-West standard errors. |
| **FF5** | Fama-French 5-factor model (2015): MKT, SMB, HML, RMW, CMA. The standard risk model for equity alphas. |
| **GKX** | Abbreviation for Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine Learning." The benchmark paper this project builds on. |
| **Huber loss** | A loss function that is quadratic for small errors (like MSE) and linear for large errors (like MAE). More robust to outliers than MSE. Controlled by the δ parameter. |
| **IC** | Information Coefficient — Spearman rank correlation between signal and subsequent return. |
| **IVOL** | Idiosyncratic Volatility — variance of stock returns not explained by systematic factor exposure. |
| **L-S** | Long-Short — a portfolio strategy that buys (long) the top-ranked stocks and shorts the bottom-ranked stocks. |
| **look-ahead bias** | A backtesting error where future data leaks into the model inputs, making results appear better than they would be in live trading. |
| **nonconformity score** | A measure of how unusual a new observation is relative to the calibration set. For regression, typically the absolute residual |y - ŷ|. |
| **OOS R²** | Out-of-sample R-squared. Measures how well the model explains return variance on data it was never trained on. |
| **permno** | CRSP's permanent identifier for a US stock. Stable across corporate events (unlike ticker symbols). |
| **SPCI** | Sequential Predictive Conformal Inference (Xu & Xie, ICML 2023). A conformal prediction method for time series that models temporal dependence in the nonconformity scores. |
| **split conformal** | The basic version of conformal prediction: split the data into train and calibration, use the global calibration quantile as the interval half-width. |
| **width** | The length of the conformal prediction interval: `upper - lower`. The main signal in this project. |
| **λ (lambda)** | The weight on interval width in the uncertainty-adjusted portfolio score. Tuned on the calibration set, frozen for all test evaluation. |
