# Cross-Correlation and Time-Lag Analysis of Cryptocurrency Time Series

This project provides a single Python script to estimate lead/lag relationships among:

- XBT/USDT (Bitcoin)
- ETH/USDT (Ethereum)
- ADA/USDT (Cardano)

using normalized returns, cross-correlation, Monte Carlo significance testing, and gap robustness checks.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python analyze_crypto_lags.py --data-root .. --interval 1min --normalization returns --mc-sims 1000
```

### Common options

- `--interval`: `1min`, `5min`, `15min`, `1h`, etc.
- `--normalization`: `returns` or `zscore`
- `--max-lag`: max lag in bins for CCF (default: 240)
- `--mc-sims`: Monte Carlo simulations (default: 1000)
- `--seed`: random seed for reproducibility
- `--output-dir`: where figures and summary are written

## Outputs

- `normalized_series.png`
- `ccf_with_significance.png`
- `lag_vs_gap_density.png`
- `summary.txt`
- `results.json`

## Notes

- Input CSV format is inferred as: `timestamp_unix,price,volume` with no header.
- The script discovers `XBTUSDT.csv`, `ETHUSDT.csv`, and `ADAUSDT.csv` recursively under `--data-root` and concatenates all quarters.
