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

### Candlestick dataset

```bash
python analyze_candlestick_lags.py --data-root .. --candles-timeframe 1 --interval 1min --price-field close --normalization returns --mc-sims 1000
```

### Common options

- `--interval`: `1min`, `5min`, `15min`, `1h`, etc.
- `--normalization`: `returns` or `zscore`
- `--max-lag`: max lag in bins for CCF (default: 240)
- `--mc-sims`: Monte Carlo simulations (default: 1000)
- `--seed`: random seed for reproducibility
- `--output-dir`: where figures and summary are written

Candlestick-specific options:

- `--candles-timeframe`: filename suffix for candles, e.g. `1`, `5`, `15`, `60`, `240`, `1440`
- `--price-field`: one of `open`, `high`, `low`, `close`, `hlc3`, `ohlc4`

## Outputs

- `normalized_series.png`
- `ccf_with_significance.png`
- `lag_vs_gap_density.png`
- `summary.txt`
- `results.json`

## Notes

- Input CSV format is inferred as: `timestamp_unix,price,volume` with no header.
- The script discovers `XBTUSDT.csv`, `ETHUSDT.csv`, and `ADAUSDT.csv` recursively under `--data-root` and concatenates all quarters.
- Candlestick CSV format is inferred as: `timestamp_unix,open,high,low,close,volume,trades` with no header.
- The candlestick script discovers files like `XBTUSD_1.csv`, `ETHUSD_1.csv`, and `ADAUSD_1.csv` recursively under `--data-root`.
