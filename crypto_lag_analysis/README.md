# Cross-Correlation and Time-Lag Analysis of Cryptocurrency Time Series

This project estimates lead/lag structure between crypto assets using:

- normalized series (returns or z-score)
- discrete cross-correlation function (CCF)
- time-resolved rolling CCF (lag spectrogram)
- directional information-flow proxy via Granger causality
- Monte Carlo significance testing with phase-randomized surrogates
- robustness tests under both random and deterministic periodic missing data

The codebase now has a shared analysis core (`core_analysis.py`) used by both entry scripts:

- `analyze_crypto_lags.py` for trade/tick-style CSVs
- `analyze_candlestick_lags.py` for OHLCV candlestick CSVs

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

### Trade data pipeline

```bash
python analyze_crypto_lags.py --data-root .. --interval 1min --normalization returns --mc-sims 1000
```

### Candlestick pipeline

```bash
python analyze_candlestick_lags.py --data-root .. --candles-timeframe 1 --interval 1min --price-field close --normalization returns --mc-sims 1000
```

## Key Options

Common options:

- `--assets`: asset list (default `BTC ETH ADA`)
- `--pairs`: explicit pairs, e.g. `BTC-ETH BTC-ADA` (default is all combinations)
- `--interval`: resample interval (`1s`, `5s`, `1min`, `5min`, `15min`, `1h`, ...)
- `--normalization`: `returns` or `zscore`
- `--max-lag`: max lag in bins for CCF
- `--ccf-window`: displayed half-window around 0 lag (default `15` bins)
- `--mc-sims`: Monte Carlo simulations
- `--vol-window`: rolling volatility window for subplot overlays
- `--gap-densities`: missing-data fractions for robustness runs
- `--gap-repeats`: replications per density
- `--periodic-gap-period`: periodic outage cycle length
- `--periodic-gap-drop`: legacy fixed periodic drop duration (its implied density is auto-included)
- `--periodic-gap-phase`: periodic outage phase offset
- `--rolling-ccf-window`: rolling window duration for lag spectrogram (default `24h`)
- `--rolling-ccf-step`: rolling window step duration (default `30min`)
- `--granger-maxlag`: max VAR lag order for Granger tests (default `20`)
- `--granger-alpha`: significance level for directionality inference (default `0.05`)

Trade-script deep-dive option:

- `--resolution-sweep`: extra intervals for lag-resolution probing, e.g. `30s 10s 5s 1s`

Candlestick-specific options:

- `--candles-timeframe`: filename suffix (`1`, `5`, `15`, `60`, ...)
- `--price-field`: `open`, `high`, `low`, `close`, `hlc3`, `ohlc4`

## Output Artifacts

- `normalized_series.png`
- `ccf_with_significance.png`
- `correlation_vs_gap_density.png`
- `lag_vs_gap_density.png` (backward-compatible alias of correlation plot)
- `time_resolved_ccf_heatmap.png`
- `granger_causality_matrix.png`
- `summary.txt`
- `results.json`

`results.json` now also includes:

- `time_resolved_ccf`: rolling-window lag matrix, ridge trajectory, and timestamps per pair
- `granger_causality`: directed p-value matrix, best lag matrix, significance matrix, and edge list

### Plot behavior updates

1. `normalized_series.png`
- Uses per-asset subplots.
- Plots cumulative returns (macroscopic trajectory) rather than raw return hairballs.
- Overlays rolling volatility on a secondary axis.

2. `ccf_with_significance.png`
- Uses stem plots (discrete lag bins).
- Zooms to a central lag window (default ±15 bins).
- Keeps per-lag and global 95% significance references.

3. `correlation_vs_gap_density.png`
- Tracks degradation in peak correlation amplitude (`|r|`) versus missing-data density.
- Overlays random-gap and periodic-gap behavior for direct comparison.

## Periodic Gaps (Deterministic)

Periodic outages are applied by a deterministic square-wave mask:

- period: `T` bins
- gap length: `D` bins (`0 < D < T`)
- optional phase offset: `phi`

For each configured density `d`, periodic dropout length is set to `round(d * T)` and clipped to `[1, T-1]`.
This allows direct random-vs-periodic comparisons at matched data-loss levels.

## Zero-Lag Deep Dive

If dominant lag is 0 at 1-minute cadence, true delay may be below resolution. Use:

### 1) Higher resolution on trade data

```bash
python analyze_crypto_lags.py --data-root .. --interval 1s --max-lag 120 --assets BTC ETH ADA --pairs BTC-ETH BTC-ADA ETH-ADA
```

or sweep multiple intervals in one run:

```bash
python analyze_crypto_lags.py --data-root .. --interval 1min --resolution-sweep 30s 10s 5s 1s
```

### 2) Asymmetric liquidity pairs

```bash
python analyze_crypto_lags.py --data-root .. --interval 1s --assets BTC ACH --pairs BTC-ACH
```

Lower-liquidity assets can widen reaction horizons and expose non-zero lags that are hidden at coarse cadence.

## Advanced Directionality and Regime Analysis

### 1) Time-Resolved CCF (Lag Spectrogram)

Instead of a single global CCF, the analysis computes a rolling CCF over windows (default `24h`) and stores:

- $r_{k,t}$ matrix per pair (lag $k$ vs center time $t$)
- dominant lag ridge over time
- corresponding per-window peak correlation

The heatmap reveals transient lag shifts that can be hidden by a single global average.

### 2) Granger Causality (Linear Transfer Entropy Proxy)

For each directed pair $X \to Y$, the pipeline runs `statsmodels` Granger tests over lags up to `--granger-maxlag` and records:

- minimum p-value across tested lags
- lag at which that minimum occurs
- significance at `--granger-alpha`
- intensity proxy $-\log_{10}(p)$ for matrix visualization

Interpretation: if $p_{X\to Y}$ is significant and smaller than $p_{Y\to X}$, evidence supports stronger information flow from $X$ into $Y$ in the linear VAR sense.

## Data Notes

- Trade CSV format expected: `timestamp_unix,price,volume` (no header).
- Candlestick CSV format expected: `timestamp_unix,open,high,low,close,volume,trades` (no header).
- Files are discovered recursively under `--data-root`.

## Significance Null (Phase-Randomized Surrogates)

Given a zero-mean series $x_t$, compute FFT coefficients
$X_k = A_k e^{i\theta_k}$.

Surrogates keep amplitudes $A_k$ and randomize interior phases,
$\phi_k \sim U(0, 2\pi)$, then invert FFT.

This preserves power spectrum (red-noise character) while breaking deterministic cross-series phase coupling, enabling robust empirical 95% thresholds for $|r_k|$.
