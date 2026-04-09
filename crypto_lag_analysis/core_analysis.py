#!/usr/bin/env python3
"""Shared analytics and plotting utilities for crypto lag analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def normalize_series(series: pd.Series, method: str) -> pd.Series:
    if method == "returns":
        out = series.pct_change()
    elif method == "zscore":
        z = zscore(series.to_numpy(dtype=float), nan_policy="omit")
        out = pd.Series(np.asarray(z, dtype=float), index=series.index, dtype=float)
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


def timedelta_to_bins(duration: str, interval: str, allow_zero: bool = False) -> int:
    duration_td = pd.to_timedelta(duration)
    step_td = pd.to_timedelta(interval)
    bins_float = duration_td / step_td
    bins = int(round(float(bins_float)))
    if bins < 0 or (bins == 0 and not allow_zero):
        min_msg = "0" if allow_zero else "1"
        raise ValueError(
            f"Duration {duration} must map to at least {min_msg} bin(s) for interval {interval}."
        )
    if not np.isclose(float(bins_float), float(bins), rtol=0, atol=1e-9):
        raise ValueError(
            f"Duration {duration} is not an integer number of bins for interval {interval}."
        )
    return bins


def simulate_random_gaps(series: pd.Series, density: float, rng: np.random.Generator) -> pd.Series:
    mask = rng.random(series.size) < density
    return series.mask(pd.Series(mask, index=series.index), np.nan)


def simulate_periodic_gaps(
    series: pd.Series,
    period_bins: int,
    gap_length_bins: int,
    phase_bins: int = 0,
) -> pd.Series:
    """Simulate deterministic periodic outages in the time series."""
    if period_bins <= 0:
        raise ValueError("period_bins must be positive.")
    if gap_length_bins <= 0:
        raise ValueError("gap_length_bins must be positive.")
    if gap_length_bins >= period_bins:
        raise ValueError("gap_length_bins must be smaller than period_bins.")

    idx = np.arange(series.size)
    keep_mask = ((idx - phase_bins) % period_bins) >= gap_length_bins
    drop_positions = np.flatnonzero(~keep_mask).tolist()
    gapped = series.copy()
    gapped.iloc[drop_positions] = np.nan
    return gapped


def analyze_pairs(
    aligned_df: pd.DataFrame,
    pairs: Sequence[Tuple[str, str]],
    max_lag: int,
    mc_sims: int,
    rng: np.random.Generator,
    interval: str,
) -> Tuple[List[PairAnalysis], Dict[str, float]]:
    analyses: List[PairAnalysis] = []
    baseline_thresholds: Dict[str, float] = {}

    for a, b in pairs:
        x = aligned_df[a].to_numpy(dtype=float)
        y = aligned_df[b].to_numpy(dtype=float)

        lags, ccf_vals = ccf_discrete(x, y, max_lag=max_lag)
        lag_bin, corr = dominant_lag(lags, ccf_vals)

        global_95, per_lag_95 = monte_carlo_significance(
            x, y, max_lag=max_lag, n_sims=mc_sims, rng=rng
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
                dominant_lag_timedelta=timedelta_from_lag_bins(lag_bin, interval),
                dominant_corr=corr,
                sig_global_95=global_95,
                sig_per_lag_95=per_lag_95,
                is_significant=is_sig,
            )
        )

    return analyses, baseline_thresholds


def _evaluate_gapped_pair(
    ga: pd.Series,
    gb: pd.Series,
    max_lag: int,
) -> Optional[Tuple[int, float]]:
    tmp = pd.concat([ga, gb], axis=1).interpolate(method="time", limit_area="inside")
    tmp = tmp.dropna(how="any")
    if len(tmp) < (2 * max_lag + 10):
        return None

    lags, ccf_vals = ccf_discrete(
        tmp.iloc[:, 0].to_numpy(dtype=float),
        tmp.iloc[:, 1].to_numpy(dtype=float),
        max_lag=max_lag,
    )
    return dominant_lag(lags, ccf_vals)


def _sample_stats(
    lag_samples: Sequence[int],
    corr_samples: Sequence[float],
    threshold: float,
) -> Dict[str, float]:
    if not lag_samples or not corr_samples:
        return {
            "mean_lag_bins": float("nan"),
            "std_lag_bins": float("nan"),
            "mean_peak_abs_corr": float("nan"),
            "std_peak_abs_corr": float("nan"),
            "significant_fraction": float("nan"),
            "valid": 0.0,
        }

    lags = np.asarray(lag_samples, dtype=float)
    abs_corr = np.abs(np.asarray(corr_samples, dtype=float))

    lag_std = float(np.std(lags, ddof=1)) if lags.size > 1 else 0.0
    corr_std = float(np.std(abs_corr, ddof=1)) if abs_corr.size > 1 else 0.0

    return {
        "mean_lag_bins": float(np.mean(lags)),
        "std_lag_bins": lag_std,
        "mean_peak_abs_corr": float(np.mean(abs_corr)),
        "std_peak_abs_corr": corr_std,
        "significant_fraction": float(np.mean(abs_corr >= threshold)),
        "valid": float(lags.size),
    }


def run_gap_robustness(
    aligned_df: pd.DataFrame,
    pairs: Sequence[Tuple[str, str]],
    max_lag: int,
    gap_densities: Sequence[float],
    repeats: int,
    rng: np.random.Generator,
    baseline_thresholds: Dict[str, float],
    periodic_period_bins: int,
    periodic_phase_bins: int = 0,
) -> Dict[str, Dict[str, List[float]]]:
    """Compare CCF robustness under random and periodic gaps at matched densities."""
    results: Dict[str, Dict[str, List[float]]] = {}

    for a, b in pairs:
        key = f"{a}-{b}"
        results[key] = {
            "density": [],
            "periodic_effective_density": [],
            "periodic_gap_length_bins": [],
            "random_mean_lag_bins": [],
            "random_std_lag_bins": [],
            "random_mean_peak_abs_corr": [],
            "random_std_peak_abs_corr": [],
            "random_significant_fraction": [],
            "periodic_mean_lag_bins": [],
            "periodic_std_lag_bins": [],
            "periodic_mean_peak_abs_corr": [],
            "periodic_std_peak_abs_corr": [],
            "periodic_significant_fraction": [],
            "random_valid_samples": [],
            "periodic_valid_samples": [],
        }

        for density in gap_densities:
            random_lags: List[int] = []
            random_corrs: List[float] = []
            periodic_lags: List[int] = []
            periodic_corrs: List[float] = []

            gap_length_bins = int(round(float(density) * periodic_period_bins))
            gap_length_bins = max(1, min(gap_length_bins, periodic_period_bins - 1))
            periodic_effective_density = float(gap_length_bins / periodic_period_bins)

            for rep in range(repeats):
                ga = simulate_random_gaps(aligned_df[a], density=float(density), rng=rng)
                gb = simulate_random_gaps(aligned_df[b], density=float(density), rng=rng)
                evaluated = _evaluate_gapped_pair(ga, gb, max_lag=max_lag)
                if evaluated is not None:
                    lag_bin, corr = evaluated
                    random_lags.append(int(lag_bin))
                    random_corrs.append(float(corr))

                periodic_phase = (periodic_phase_bins + rep) % periodic_period_bins
                pga = simulate_periodic_gaps(
                    aligned_df[a],
                    period_bins=periodic_period_bins,
                    gap_length_bins=gap_length_bins,
                    phase_bins=periodic_phase,
                )
                pgb = simulate_periodic_gaps(
                    aligned_df[b],
                    period_bins=periodic_period_bins,
                    gap_length_bins=gap_length_bins,
                    phase_bins=periodic_phase,
                )
                pevaluated = _evaluate_gapped_pair(pga, pgb, max_lag=max_lag)
                if pevaluated is not None:
                    lag_bin, corr = pevaluated
                    periodic_lags.append(int(lag_bin))
                    periodic_corrs.append(float(corr))

            random_stats = _sample_stats(
                random_lags,
                random_corrs,
                threshold=baseline_thresholds[key],
            )
            periodic_stats = _sample_stats(
                periodic_lags,
                periodic_corrs,
                threshold=baseline_thresholds[key],
            )

            results[key]["density"].append(float(density))
            results[key]["periodic_effective_density"].append(periodic_effective_density)
            results[key]["periodic_gap_length_bins"].append(float(gap_length_bins))
            results[key]["random_mean_lag_bins"].append(random_stats["mean_lag_bins"])
            results[key]["random_std_lag_bins"].append(random_stats["std_lag_bins"])
            results[key]["random_mean_peak_abs_corr"].append(random_stats["mean_peak_abs_corr"])
            results[key]["random_std_peak_abs_corr"].append(random_stats["std_peak_abs_corr"])
            results[key]["random_significant_fraction"].append(random_stats["significant_fraction"])
            results[key]["periodic_mean_lag_bins"].append(periodic_stats["mean_lag_bins"])
            results[key]["periodic_std_lag_bins"].append(periodic_stats["std_lag_bins"])
            results[key]["periodic_mean_peak_abs_corr"].append(
                periodic_stats["mean_peak_abs_corr"]
            )
            results[key]["periodic_std_peak_abs_corr"].append(periodic_stats["std_peak_abs_corr"])
            results[key]["periodic_significant_fraction"].append(
                periodic_stats["significant_fraction"]
            )
            results[key]["random_valid_samples"].append(random_stats["valid"])
            results[key]["periodic_valid_samples"].append(periodic_stats["valid"])

    return results


def plot_normalized_series(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
    vol_window: int = 60,
) -> None:
    n = len(df.columns)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.6 * n), sharex=True)
    if n == 1:
        axes = [axes]

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:brown", "tab:gray"]
    vol_window = max(2, int(vol_window))

    for i, (ax, col) in enumerate(zip(axes, df.columns)):
        series = df[col].astype(float)
        cumulative_returns = (1.0 + series.fillna(0.0)).cumprod() - 1.0
        ax.plot(
            df.index,
            cumulative_returns,
            label=f"Cumulative {col}",
            linewidth=1.0,
            color=colors[i % len(colors)],
        )
        ax.set_ylabel("Cumulative")

        ax2 = ax.twinx()
        rolling_vol = series.rolling(window=vol_window, min_periods=max(4, vol_window // 4)).std()
        ax2.plot(
            df.index,
            rolling_vol,
            color="tab:red",
            alpha=0.35,
            linewidth=0.9,
            label=f"{vol_window}-bin volatility",
        )
        ax2.set_ylabel("Volatility")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
        ax.grid(alpha=0.25)

    axes[0].set_title(title)
    axes[-1].set_xlabel("Timestamp")
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.98))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_ccf(
    analyses: Sequence[PairAnalysis],
    interval: str,
    output_path: Path,
    window_bins: int = 15,
) -> None:
    n = len(analyses)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    window_bins = max(1, int(window_bins))

    for ax, res in zip(axes, analyses):
        x_lags = res.lags
        mask = (x_lags >= -window_bins) & (x_lags <= window_bins)

        markerline, stemlines, baseline = ax.stem(
            x_lags[mask],
            res.ccf[mask],
            linefmt="tab:blue",
            markerfmt="bo",
            basefmt="k-",
            label="CCF",
        )
        plt.setp(stemlines, linewidth=1.2)
        plt.setp(markerline, markersize=4)
        plt.setp(baseline, linewidth=0.8)

        ax.plot(
            x_lags[mask],
            res.sig_per_lag_95[mask],
            color="tab:red",
            linestyle="--",
            label="Per-lag 95%",
        )
        ax.plot(x_lags[mask], -res.sig_per_lag_95[mask], color="tab:red", linestyle="--")
        ax.axhline(res.sig_global_95, color="tab:orange", linestyle=":", label="Global 95%")
        ax.axhline(-res.sig_global_95, color="tab:orange", linestyle=":")
        ax.axvline(res.dominant_lag_bins, color="tab:green", linestyle="-.", label="Dominant lag")

        ax.set_xlim(-window_bins, window_bins)
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


def plot_correlation_vs_gap_density(
    gap_results: Dict[str, Dict[str, List[float]]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for pair, vals in gap_results.items():
        x_random = np.asarray(vals["density"], dtype=float)
        y_random = np.asarray(vals["random_mean_peak_abs_corr"], dtype=float)
        yerr_random = np.asarray(vals["random_std_peak_abs_corr"], dtype=float)

        x_periodic = np.asarray(vals["periodic_effective_density"], dtype=float)
        y_periodic = np.asarray(vals["periodic_mean_peak_abs_corr"], dtype=float)
        yerr_periodic = np.asarray(vals["periodic_std_peak_abs_corr"], dtype=float)

        ax.errorbar(
            x_random,
            y_random,
            yerr=yerr_random,
            marker="o",
            linestyle="-",
            capsize=4,
            label=f"{pair} random",
        )
        ax.errorbar(
            x_periodic,
            y_periodic,
            yerr=yerr_periodic,
            marker="s",
            linestyle="--",
            capsize=4,
            label=f"{pair} periodic",
        )

    ax.set_title("Signal Degradation: Peak |CCF| vs Data Loss")
    ax.set_xlabel("Fraction of Missing Data (Gap Density)")
    ax.set_ylabel("Mean Peak Cross-Correlation Amplitude |r|")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_summary_text(
    headline: str,
    analyses: Sequence[PairAnalysis],
    interval: str,
    gap_results: Dict[str, Dict[str, List[float]]],
    periodic_gap_period: str,
    periodic_gap_phase: str,
    extra_sections: Optional[Sequence[str]] = None,
) -> str:
    lines: List[str] = []
    lines.append(headline)
    lines.append("=" * len(headline))
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
    zero_lag_pairs = [res.pair for res in analyses if res.dominant_lag_bins == 0]
    if zero_lag_pairs:
        if len(zero_lag_pairs) == len(analyses):
            lines.append(
                "Zero-lag note: all pairs peak at 0 bins. At interval="
                f"{interval}, this means any true lead/lag is likely below one sample step. "
                "Resolving sub-bin lags requires higher-frequency data."
            )
        else:
            lines.append(
                "Zero-lag note: some pairs peak at 0 bins, suggesting effectively simultaneous "
                "behavior at this sampling interval."
            )

    lines.append("")
    lines.append(
        "Gap robustness (random vs periodic): peak |r| and lag stability versus missing-data density"
    )
    lines.append(
        f"Periodic mask settings: period={periodic_gap_period}, phase={periodic_gap_phase}."
    )
    lines.append(
        "For each density d, periodic gap length is round(d * period_bins), then clipped to [1, period-1]."
    )

    for pair, vals in gap_results.items():
        for i, density in enumerate(vals["density"]):
            lines.append(
                f"- {pair}, density={density:.0%}: "
                f"random |r|={vals['random_mean_peak_abs_corr'][i]:.4f}"
                f"±{vals['random_std_peak_abs_corr'][i]:.4f}, "
                f"periodic |r|={vals['periodic_mean_peak_abs_corr'][i]:.4f}"
                f"±{vals['periodic_std_peak_abs_corr'][i]:.4f}; "
                f"random lag={vals['random_mean_lag_bins'][i]:.2f}"
                f"±{vals['random_std_lag_bins'][i]:.2f}, "
                f"periodic lag={vals['periodic_mean_lag_bins'][i]:.2f}"
                f"±{vals['periodic_std_lag_bins'][i]:.2f}"
            )

    if extra_sections:
        for section in extra_sections:
            lines.append("")
            lines.append(section)

    return "\n".join(lines)
