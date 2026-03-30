#!/usr/bin/env python3
"""Cross-correlation and time-lag analysis for candlestick crypto time series.

Pipeline:
1) Discover and load candlestick CSV feeds for BTC/XBT, ETH, ADA.
2) Select a price field and regularize to a fixed interval.
3) Normalize (fractional returns or z-score).
4) Compute cross-correlation function (CCF) and dominant lag.
5) Estimate significance with Monte Carlo phase-randomized red-noise surrogates.
6) Stress-test lag robustness under random gap densities.
7) Save plots + summary artifacts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore


@dataclass
class PairAnalysis:
    pair: str
    lags: np.ndarray
    ccf: np.ndarray
    dominant_lag_bins: int
    dominant_lag_timedelta: str
    dominant_corr: float
    sig_global_95: float
    sig_per_lag_95: np.ndarray
    is_significant: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-correlation and time-lag analysis for BTC/ETH/ADA candlesticks"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(".."),
        help="Root folder containing quarter candlestick folders with CSV files.",
    )
    parser.add_argument(
        "--candles-timeframe",
        type=str,
        default="1",
        help="Candlestick timeframe suffix in filenames, e.g. 1, 5, 15, 60, 240, 1440.",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1min",
        help="Resampling interval, e.g. 1min, 5min, 15min, 1h.",
    )
    parser.add_argument(
        "--price-field",
        type=str,
        choices=["open", "high", "low", "close", "hlc3", "ohlc4"],
        default="close",
        help="Candlestick price field used to build the series.",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        choices=["returns", "zscore"],
        default="returns",
        help="Normalization method.",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=240,
        help="Maximum lag in bins for CCF.",
    )
    parser.add_argument(
        "--mc-sims",
        type=int,
        default=1000,
        help="Number of Monte Carlo simulations.",
    )
    parser.add_argument(
        "--max-interp-steps",
        type=int,
        default=3,
        help="Max consecutive missing bins to fill with linear interpolation.",
    )
    parser.add_argument(
        "--gap-densities",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.5],
        help="Random gap fractions for robustness test.",
    )
    parser.add_argument(
        "--gap-repeats",
        type=int,
        default=80,
        help="Replications per gap density.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_candlestick"),
        help="Output directory for figures and summary files.",
    )
    return parser.parse_args()


def discover_symbol_files(
    data_root: Path,
    symbol_candidates: Sequence[str],
    candles_timeframe: str,
) -> List[Path]:
    files: List[Path] = []
    for candidate in symbol_candidates:
        files.extend(data_root.rglob(f"{candidate}_{candles_timeframe}.csv"))
    deduped = sorted(set(files))
    if not deduped:
        raise FileNotFoundError(
            f"No candlestick files found for candidates: {symbol_candidates} "
            f"with timeframe={candles_timeframe} under {data_root}"
        )
    return deduped


def load_candlestick_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        header=None,
        names=["timestamp", "open", "high", "low", "close", "volume", "trades"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    numeric_cols = ["open", "high", "low", "close", "volume", "trades"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df[["timestamp", "open", "high", "low", "close", "volume", "trades"]].sort_values(
        "timestamp"
    )


def compute_price_field(df: pd.DataFrame, price_field: str) -> pd.Series:
    if price_field in {"open", "high", "low", "close"}:
        return df[price_field]
    if price_field == "hlc3":
        return (df["high"] + df["low"] + df["close"]) / 3.0
    if price_field == "ohlc4":
        return (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    raise ValueError(f"Unsupported price field: {price_field}")


def build_price_series(
    files: Sequence[Path],
    interval: str,
    max_interp_steps: int,
    price_field: str,
) -> pd.Series:
    frames = [load_candlestick_csv(p) for p in files]
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp")

    # If duplicate candle open-times exist, keep the last observation.
    selected_price = compute_price_field(df, price_field)
    instant_price = pd.Series(selected_price.to_numpy(dtype=float), index=df["timestamp"]).groupby(
        level=0
    ).last()

    regular = instant_price.resample(interval).last()
    regular = regular.interpolate(method="time", limit=max_interp_steps, limit_area="inside")
    return regular


def normalize_series(series: pd.Series, method: str) -> pd.Series:
    if method == "returns":
        out = series.pct_change()
    elif method == "zscore":
        z = zscore(series.to_numpy(dtype=float), nan_policy="omit")
        out = pd.Series(z, index=series.index)
    else:
        raise ValueError(f"Unsupported normalization: {method}")
    return out.dropna()


def align_series(series_map: Dict[str, pd.Series]) -> pd.DataFrame:
    df = pd.concat(series_map, axis=1, sort=False).sort_index()
    df = df.dropna(how="any")
    if df.empty:
        raise ValueError("Aligned DataFrame is empty after dropping missing rows.")
    return df


def ccf_discrete(x: np.ndarray, y: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute discrete normalized CCF for lags in [-max_lag, max_lag].

    Sign convention follows r_k = corr(x_t, y_{t-k}).
    Positive k implies x leads y by k bins.
    """
    if x.shape != y.shape:
        raise ValueError("x and y must have identical shape")

    lags = np.arange(-max_lag, max_lag + 1)
    ccf_vals = np.full_like(lags, fill_value=np.nan, dtype=float)

    for i, k in enumerate(lags):
        if k > 0:
            x_seg = x[k:]
            y_seg = y[:-k]
        elif k < 0:
            k_abs = -k
            x_seg = x[:-k_abs]
            y_seg = y[k_abs:]
        else:
            x_seg = x
            y_seg = y

        if x_seg.size < 3:
            continue

        x_centered = x_seg - np.mean(x_seg)
        y_centered = y_seg - np.mean(y_seg)
        denom = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
        ccf_vals[i] = np.nan if denom == 0 else np.sum(x_centered * y_centered) / denom

    return lags, ccf_vals


def dominant_lag(lags: np.ndarray, ccf_vals: np.ndarray) -> Tuple[int, float]:
    idx = int(np.nanargmax(np.abs(ccf_vals)))
    return int(lags[idx]), float(ccf_vals[idx])


def phase_randomized_surrogate(series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate a surrogate preserving amplitude spectrum with randomized phase."""
    x = np.asarray(series, dtype=float)
    n = x.size

    spectrum = np.fft.rfft(x)
    amplitudes = np.abs(spectrum)

    phase = np.zeros_like(amplitudes)
    if amplitudes.size > 2:
        phase[1:-1] = rng.uniform(0, 2 * np.pi, size=amplitudes.size - 2)

    surrogate_spec = amplitudes * np.exp(1j * phase)
    sur = np.fft.irfft(surrogate_spec, n=n)

    sur = sur - np.mean(sur)
    std = np.std(sur)
    if std > 0:
        sur = sur / std
    return sur


def monte_carlo_significance(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int,
    n_sims: int,
    rng: np.random.Generator,
) -> Tuple[float, np.ndarray]:
    """Return global and per-lag 95% significance thresholds for |r_k|."""
    abs_max_values: List[float] = []
    abs_per_lag: List[np.ndarray] = []

    for _ in range(n_sims):
        sx = phase_randomized_surrogate(x, rng)
        sy = phase_randomized_surrogate(y, rng)
        _, sim_ccf = ccf_discrete(sx, sy, max_lag=max_lag)
        sim_abs = np.abs(sim_ccf)
        abs_max_values.append(float(np.nanmax(sim_abs)))
        abs_per_lag.append(sim_abs)

    per_lag_arr = np.vstack(abs_per_lag)
    global_95 = float(np.percentile(abs_max_values, 95))
    per_lag_95 = np.percentile(per_lag_arr, 95, axis=0)
    return global_95, per_lag_95


def timedelta_from_lag_bins(lag_bins: int, interval: str) -> str:
    step = pd.to_timedelta(interval)
    delta = lag_bins * step
    return str(delta)


def simulate_random_gaps(series: pd.Series, density: float, rng: np.random.Generator) -> pd.Series:
    mask = rng.random(series.size) < density
    gapped = series.copy()
    gapped.iloc[mask] = np.nan
    return gapped


def run_gap_robustness(
    aligned_df: pd.DataFrame,
    pairs: Sequence[Tuple[str, str]],
    max_lag: int,
    gap_densities: Sequence[float],
    repeats: int,
    rng: np.random.Generator,
    baseline_thresholds: Dict[str, float],
) -> Dict[str, Dict[str, List[float]]]:
    results: Dict[str, Dict[str, List[float]]] = {}

    for a, b in pairs:
        key = f"{a}-{b}"
        results[key] = {
            "density": [],
            "mean_lag_bins": [],
            "std_lag_bins": [],
            "significant_fraction": [],
        }

        for density in gap_densities:
            lag_samples: List[int] = []
            sig_count = 0
            valid = 0

            for _ in range(repeats):
                ga = simulate_random_gaps(aligned_df[a], density=density, rng=rng)
                gb = simulate_random_gaps(aligned_df[b], density=density, rng=rng)

                # Interpolate to mimic practical recovery from sparse missing chunks.
                tmp = pd.concat([ga, gb], axis=1).interpolate(
                    method="time", limit_area="inside"
                )
                tmp = tmp.dropna(how="any")
                if len(tmp) < (2 * max_lag + 10):
                    continue

                lags, ccf_vals = ccf_discrete(
                    tmp.iloc[:, 0].to_numpy(dtype=float),
                    tmp.iloc[:, 1].to_numpy(dtype=float),
                    max_lag=max_lag,
                )
                lag_bin, corr = dominant_lag(lags, ccf_vals)
                lag_samples.append(lag_bin)
                valid += 1
                if abs(corr) >= baseline_thresholds[key]:
                    sig_count += 1

            if valid == 0:
                results[key]["density"].append(float(density))
                results[key]["mean_lag_bins"].append(float("nan"))
                results[key]["std_lag_bins"].append(float("nan"))
                results[key]["significant_fraction"].append(float("nan"))
            else:
                results[key]["density"].append(float(density))
                results[key]["mean_lag_bins"].append(float(np.mean(lag_samples)))
                results[key]["std_lag_bins"].append(float(np.std(lag_samples, ddof=1)))
                results[key]["significant_fraction"].append(float(sig_count / valid))

    return results


def plot_normalized_series(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    for col in df.columns:
        ax.plot(df.index, df[col], label=col, linewidth=1.0)
    ax.set_title("Normalized Crypto Candlestick Time Series")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Normalized Value")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_ccf(
    analyses: Sequence[PairAnalysis],
    interval: str,
    output_path: Path,
) -> None:
    n = len(analyses)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, analyses):
        x_lags = res.lags
        ax.plot(x_lags, res.ccf, color="tab:blue", label="CCF")
        ax.plot(x_lags, res.sig_per_lag_95, color="tab:red", linestyle="--", label="+95% per-lag")
        ax.plot(x_lags, -res.sig_per_lag_95, color="tab:red", linestyle="--", label="-95% per-lag")
        ax.axhline(res.sig_global_95, color="tab:orange", linestyle=":", label="Global 95%")
        ax.axhline(-res.sig_global_95, color="tab:orange", linestyle=":")
        ax.axvline(res.dominant_lag_bins, color="tab:green", linestyle="-.", label="Dominant lag")
        ax.set_ylabel("r_k")
        ax.set_title(
            f"{res.pair}: dominant lag={res.dominant_lag_bins} bins ({res.dominant_lag_timedelta}), "
            f"r={res.dominant_corr:.4f}, significant={res.is_significant}"
        )
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel(f"Lag (bins of {interval})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_lag_vs_gap_density(
    gap_results: Dict[str, Dict[str, List[float]]], output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for pair, vals in gap_results.items():
        x = np.array(vals["density"], dtype=float)
        y = np.array(vals["mean_lag_bins"], dtype=float)
        yerr = np.array(vals["std_lag_bins"], dtype=float)
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=4, label=pair)

    ax.set_title("Lag vs Gap Density")
    ax.set_xlabel("Gap Density")
    ax.set_ylabel("Estimated Dominant Lag (bins)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_summary_text(
    analyses: Sequence[PairAnalysis],
    interval: str,
    gap_results: Dict[str, Dict[str, List[float]]],
) -> str:
    lines: List[str] = []
    lines.append("Candlestick Cross-Correlation and Time-Lag Analysis Summary")
    lines.append("=" * 64)
    lines.append(f"Sampling interval: {interval}")
    lines.append("")
    lines.append("Dominant lag findings:")

    any_significant = False
    for res in analyses:
        status = "SIGNIFICANT" if res.is_significant else "not significant"
        if res.is_significant:
            any_significant = True
        lines.append(
            f"- {res.pair}: lag={res.dominant_lag_bins} bins ({res.dominant_lag_timedelta}), "
            f"r={res.dominant_corr:.4f}, |r| threshold95={res.sig_global_95:.4f} -> {status}."
        )

    lines.append("")
    if any_significant:
        lines.append(
            "Interpretation: At least one pair shows a statistically significant lead/lag under the "
            "Monte Carlo red-noise null hypothesis."
        )
    else:
        lines.append(
            "Interpretation: No pair shows a statistically significant dominant lag at the 95% level "
            "under the Monte Carlo red-noise null hypothesis."
        )

    lines.append("")
    lines.append("Gap robustness (mean lag bins, std, significant fraction):")
    for pair, vals in gap_results.items():
        for density, mean_lag, std_lag, sig_frac in zip(
            vals["density"],
            vals["mean_lag_bins"],
            vals["std_lag_bins"],
            vals["significant_fraction"],
        ):
            lines.append(
                f"- {pair}, gap={density:.0%}: mean_lag={mean_lag:.2f}, std={std_lag:.2f}, "
                f"significant_fraction={sig_frac:.2f}"
            )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    symbol_map = {
        "BTC": ["XBTUSD", "BTCUSD", "XBTUSDT", "BTCUSDT"],
        "ETH": ["ETHUSD", "ETHUSDT"],
        "ADA": ["ADAUSD", "ADAUSDT"],
    }

    price_series: Dict[str, pd.Series] = {}
    discovered_files: Dict[str, List[str]] = {}

    for sym, candidates in symbol_map.items():
        files = discover_symbol_files(
            args.data_root,
            candidates,
            candles_timeframe=args.candles_timeframe,
        )
        discovered_files[sym] = [str(p) for p in files]
        price_series[sym] = build_price_series(
            files=files,
            interval=args.interval,
            max_interp_steps=args.max_interp_steps,
            price_field=args.price_field,
        )

    normalized = {
        sym: normalize_series(s, method=args.normalization) for sym, s in price_series.items()
    }
    aligned = align_series(normalized)

    pairs = [("BTC", "ETH"), ("BTC", "ADA"), ("ETH", "ADA")]

    analyses: List[PairAnalysis] = []
    baseline_thresholds: Dict[str, float] = {}

    for a, b in pairs:
        x = aligned[a].to_numpy(dtype=float)
        y = aligned[b].to_numpy(dtype=float)

        lags, ccf_vals = ccf_discrete(x, y, max_lag=args.max_lag)
        lag_bin, corr = dominant_lag(lags, ccf_vals)

        global_95, per_lag_95 = monte_carlo_significance(
            x, y, max_lag=args.max_lag, n_sims=args.mc_sims, rng=rng
        )
        is_sig = abs(corr) >= global_95

        pair_name = f"{a}-{b}"
        baseline_thresholds[pair_name] = global_95

        analyses.append(
            PairAnalysis(
                pair=pair_name,
                lags=lags,
                ccf=ccf_vals,
                dominant_lag_bins=lag_bin,
                dominant_lag_timedelta=timedelta_from_lag_bins(lag_bin, args.interval),
                dominant_corr=corr,
                sig_global_95=global_95,
                sig_per_lag_95=per_lag_95,
                is_significant=is_sig,
            )
        )

    gap_results = run_gap_robustness(
        aligned_df=aligned,
        pairs=pairs,
        max_lag=args.max_lag,
        gap_densities=args.gap_densities,
        repeats=args.gap_repeats,
        rng=rng,
        baseline_thresholds=baseline_thresholds,
    )

    plot_normalized_series(aligned, args.output_dir / "normalized_series.png")
    plot_ccf(analyses, interval=args.interval, output_path=args.output_dir / "ccf_with_significance.png")
    plot_lag_vs_gap_density(gap_results, args.output_dir / "lag_vs_gap_density.png")

    summary_text = build_summary_text(analyses, args.interval, gap_results)
    (args.output_dir / "summary.txt").write_text(summary_text, encoding="utf-8")

    serializable = {
        "config": {
            "data_root": str(args.data_root),
            "candles_timeframe": args.candles_timeframe,
            "interval": args.interval,
            "price_field": args.price_field,
            "normalization": args.normalization,
            "max_lag": args.max_lag,
            "mc_sims": args.mc_sims,
            "max_interp_steps": args.max_interp_steps,
            "gap_densities": args.gap_densities,
            "gap_repeats": args.gap_repeats,
            "seed": args.seed,
        },
        "files": discovered_files,
        "pair_results": [
            {
                "pair": r.pair,
                "dominant_lag_bins": r.dominant_lag_bins,
                "dominant_lag_timedelta": r.dominant_lag_timedelta,
                "dominant_corr": r.dominant_corr,
                "sig_global_95": r.sig_global_95,
                "is_significant": r.is_significant,
            }
            for r in analyses
        ],
        "gap_results": gap_results,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )

    print(summary_text)
    print("\nSaved artifacts to:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
