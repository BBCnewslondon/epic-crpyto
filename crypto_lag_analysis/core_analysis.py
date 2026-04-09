#!/usr/bin/env python3
"""Shared analytics and plotting utilities for crypto lag analysis."""

from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def _rolling_window_positions(
    n_points: int,
    window_bins: int,
    step_bins: int,
) -> List[Tuple[int, int]]:
    if window_bins <= 0:
        raise ValueError("window_bins must be positive.")
    if step_bins <= 0:
        raise ValueError("step_bins must be positive.")
    if n_points < window_bins:
        return []

    out: List[Tuple[int, int]] = []
    start = 0
    while start + window_bins <= n_points:
        end = start + window_bins
        out.append((start, end))
        start += step_bins
    return out


def compute_time_resolved_ccf(
    aligned_df: pd.DataFrame,
    pairs: Sequence[Tuple[str, str]],
    max_lag: int,
    window_bins: int,
    step_bins: int,
    interval: str,
) -> Dict[str, Dict[str, Any]]:
    windows = _rolling_window_positions(len(aligned_df), window_bins=window_bins, step_bins=step_bins)

    lag_axis = np.arange(-max_lag, max_lag + 1, dtype=int)
    outputs: Dict[str, Dict[str, Any]] = {}

    for a, b in pairs:
        pair_name = f"{a}-{b}"
        matrix = np.full((len(windows), lag_axis.size), np.nan, dtype=float)
        center_times: List[pd.Timestamp] = []
        dominant_lags: List[int] = []
        dominant_corrs: List[float] = []

        x_full = aligned_df[a].to_numpy(dtype=float)
        y_full = aligned_df[b].to_numpy(dtype=float)

        for i, (start, end) in enumerate(windows):
            x = x_full[start:end]
            y = y_full[start:end]
            lags, ccf_vals = ccf_discrete(x, y, max_lag=max_lag)
            matrix[i, :] = ccf_vals

            lag_bin, corr = dominant_lag(lags, ccf_vals)
            dominant_lags.append(int(lag_bin))
            dominant_corrs.append(float(corr))

            center_idx = start + (window_bins // 2)
            center_times.append(aligned_df.index[center_idx])

        outputs[pair_name] = {
            "lags": lag_axis,
            "center_times": pd.DatetimeIndex(center_times),
            "ccf_matrix": matrix,
            "dominant_lag_bins": np.asarray(dominant_lags, dtype=int),
            "dominant_lag_timedelta": [timedelta_from_lag_bins(v, interval) for v in dominant_lags],
            "dominant_corr": np.asarray(dominant_corrs, dtype=float),
            "window_bins": int(window_bins),
            "step_bins": int(step_bins),
            "window_duration": str(pd.to_timedelta(interval) * window_bins),
            "step_duration": str(pd.to_timedelta(interval) * step_bins),
        }

    return outputs


def _granger_pvalue_for_direction(
    src: np.ndarray,
    dst: np.ndarray,
    maxlag: int,
) -> Tuple[float, int]:
    if src.size != dst.size:
        raise ValueError("src and dst must have identical length")

    # statsmodels expects a two-column array [target, exog] and tests whether
    # lagged exog terms improve target prediction.
    data = np.column_stack([dst, src])
    granger_tests = getattr(
        importlib.import_module("statsmodels.tsa.stattools"),
        "grangercausalitytests",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="verbose is deprecated since functions should not print results",
            category=FutureWarning,
        )
        tests = granger_tests(data, maxlag=maxlag, verbose=False)

    best_p = float("inf")
    best_lag = 1
    for lag, payload in tests.items():
        pval = float(payload[0]["ssr_ftest"][1])
        if np.isfinite(pval) and pval < best_p:
            best_p = pval
            best_lag = int(lag)

    return best_p, best_lag


def compute_granger_causality_matrix(
    aligned_df: pd.DataFrame,
    assets: Sequence[str],
    maxlag: int,
    alpha: float,
) -> Dict[str, Any]:
    if maxlag < 1:
        raise ValueError("maxlag must be >= 1")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    labels = [a.upper() for a in assets]
    n = len(labels)
    pvals = np.full((n, n), np.nan, dtype=float)
    best_lags = np.full((n, n), np.nan, dtype=float)
    significant = np.zeros((n, n), dtype=float)
    te_proxy = np.full((n, n), np.nan, dtype=float)

    for i, src in enumerate(labels):
        for j, dst in enumerate(labels):
            if i == j:
                continue

            src_vals = aligned_df[src].to_numpy(dtype=float)
            dst_vals = aligned_df[dst].to_numpy(dtype=float)
            min_required = max(20, maxlag * 5)
            if src_vals.size <= min_required:
                continue

            try:
                pval, lag = _granger_pvalue_for_direction(src_vals, dst_vals, maxlag=maxlag)
            except (ValueError, np.linalg.LinAlgError):
                continue

            pvals[i, j] = pval
            best_lags[i, j] = float(lag)
            significant[i, j] = 1.0 if pval < alpha else 0.0
            te_proxy[i, j] = max(0.0, -np.log10(max(pval, 1e-16)))

    net_flow = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.isfinite(te_proxy[i, j]) and np.isfinite(te_proxy[j, i]):
                net_flow[i, j] = float(te_proxy[i, j] - te_proxy[j, i])

    directed_edges: List[Dict[str, Any]] = []
    for i, src in enumerate(labels):
        for j, dst in enumerate(labels):
            if i == j:
                continue
            if not np.isfinite(pvals[i, j]):
                continue
            directed_edges.append(
                {
                    "source": src,
                    "target": dst,
                    "p_value": float(pvals[i, j]),
                    "best_lag": int(best_lags[i, j]),
                    "is_significant": bool(significant[i, j] > 0.5),
                    "te_proxy": float(te_proxy[i, j]),
                }
            )

    return {
        "assets": labels,
        "alpha": float(alpha),
        "maxlag": int(maxlag),
        "p_value_matrix": pvals,
        "best_lag_matrix": best_lags,
        "significant_matrix": significant,
        "te_proxy_matrix": te_proxy,
        "net_te_proxy_matrix": net_flow,
        "directed_edges": directed_edges,
    }


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


def plot_time_resolved_ccf_heatmap(
    time_ccf_results: Dict[str, Dict[str, Any]],
    interval: str,
    output_path: Path,
    show_ridge: bool = True,
) -> None:
    if not time_ccf_results:
        return

    pairs = list(time_ccf_results.keys())
    n = len(pairs)
    fig, axes = plt.subplots(n, 1, figsize=(14, max(4, 3.8 * n)), sharex=True)
    if n == 1:
        axes = [axes]

    mesh = None
    for ax, pair in zip(axes, pairs):
        payload = time_ccf_results[pair]
        matrix = np.asarray(payload["ccf_matrix"], dtype=float)
        lags = np.asarray(payload["lags"], dtype=float)
        centers = pd.DatetimeIndex(payload["center_times"])

        if matrix.size == 0 or len(centers) == 0:
            ax.set_title(f"{pair}: no windows available")
            ax.set_ylabel("Lag bins")
            ax.grid(alpha=0.25)
            continue

        time_num = centers.view("int64") / 1e9
        if len(time_num) == 1:
            # Expand a single center time into a narrow span to keep pcolormesh happy.
            step = float(pd.to_timedelta(interval).total_seconds())
            t_edges = np.array([time_num[0] - step / 2.0, time_num[0] + step / 2.0])
        else:
            t_mid = (time_num[:-1] + time_num[1:]) / 2.0
            first_edge = time_num[0] - (t_mid[0] - time_num[0])
            last_edge = time_num[-1] + (time_num[-1] - t_mid[-1])
            t_edges = np.concatenate([[first_edge], t_mid, [last_edge]])

        if len(lags) == 1:
            lag_edges = np.array([lags[0] - 0.5, lags[0] + 0.5])
        else:
            lag_mid = (lags[:-1] + lags[1:]) / 2.0
            first_lag = lags[0] - (lag_mid[0] - lags[0])
            last_lag = lags[-1] + (lags[-1] - lag_mid[-1])
            lag_edges = np.concatenate([[first_lag], lag_mid, [last_lag]])

        mesh = ax.pcolormesh(
            t_edges,
            lag_edges,
            matrix.T,
            cmap="RdBu_r",
            shading="auto",
            vmin=-1.0,
            vmax=1.0,
        )

        if show_ridge:
            ridge = np.asarray(payload["dominant_lag_bins"], dtype=float)
            ax.plot(time_num, ridge, color="black", linewidth=1.2, alpha=0.85, label="Dominant ridge")
            ax.legend(loc="upper right", fontsize=8)

        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_ylabel("Lag bins")
        ax.set_title(
            f"{pair}: rolling CCF lag spectrogram (window={payload['window_duration']}, "
            f"step={payload['step_duration']})"
        )
        ax.grid(alpha=0.2)

    # Convert matplotlib date-like float axis labels back to readable timestamps.
    xticks = axes[-1].get_xticks()
    if xticks.size:
        xt_labels = [
            pd.to_datetime(int(v * 1e9), unit="ns", utc=True).strftime("%Y-%m-%d\n%H:%M") for v in xticks
        ]
        axes[-1].set_xticks(xticks)
        axes[-1].set_xticklabels(xt_labels, rotation=0)
    axes[-1].set_xlabel("Window center time (UTC)")

    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes, shrink=0.96)
        cbar.set_label("CCF r_k,t")
    fig.subplots_adjust(left=0.07, right=0.93, top=0.96, bottom=0.09, hspace=0.3)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_granger_matrix(
    granger_result: Dict[str, Any],
    output_path: Path,
) -> None:
    assets = list(granger_result.get("assets", []))
    matrix = np.asarray(granger_result.get("te_proxy_matrix", np.empty((0, 0))), dtype=float)
    if not assets or matrix.size == 0:
        return

    fig, ax = plt.subplots(figsize=(1.8 + 1.2 * len(assets), 1.4 + 1.2 * len(assets)))
    masked = np.ma.array(matrix, mask=~np.isfinite(matrix))
    im = ax.imshow(masked, cmap="YlOrRd", interpolation="nearest")

    ax.set_xticks(np.arange(len(assets)))
    ax.set_yticks(np.arange(len(assets)))
    ax.set_xticklabels(assets)
    ax.set_yticklabels(assets)
    ax.set_xlabel("Target (predicted asset)")
    ax.set_ylabel("Source (informational driver)")
    ax.set_title("Granger Information-Flow Intensity (-log10 p-value)")

    alpha = float(granger_result.get("alpha", 0.05))
    threshold = -np.log10(alpha)
    for i in range(len(assets)):
        for j in range(len(assets)):
            val = matrix[i, j]
            if i == j or not np.isfinite(val):
                continue
            marker = "*" if val >= threshold else ""
            ax.text(j, i, f"{val:.2f}{marker}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("-log10(p)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def summarize_time_resolved_ccf(
    time_ccf_results: Dict[str, Dict[str, Any]],
) -> str:
    lines: List[str] = ["Time-resolved CCF (lag spectrogram) summary:"]
    if not time_ccf_results:
        lines.append("No rolling windows were available for the configured settings.")
        return "\n".join(lines)

    for pair, payload in time_ccf_results.items():
        ridge = np.asarray(payload["dominant_lag_bins"], dtype=float)
        corr = np.asarray(payload["dominant_corr"], dtype=float)
        if ridge.size == 0:
            lines.append(f"- {pair}: no valid windows.")
            continue

        lines.append(
            f"- {pair}: windows={ridge.size}, ridge lag mean={np.nanmean(ridge):.2f} bins, "
            f"ridge lag std={np.nanstd(ridge):.2f}, mean |peak r|={np.nanmean(np.abs(corr)):.4f}."
        )

    return "\n".join(lines)


def summarize_granger(granger_result: Dict[str, Any]) -> str:
    lines: List[str] = ["Granger causality (directionality proxy) summary:"]
    assets = list(granger_result.get("assets", []))
    if not assets:
        lines.append("No Granger result available.")
        return "\n".join(lines)

    alpha = float(granger_result.get("alpha", 0.05))
    pvals = np.asarray(granger_result.get("p_value_matrix", np.empty((0, 0))), dtype=float)
    lags = np.asarray(granger_result.get("best_lag_matrix", np.empty((0, 0))), dtype=float)

    any_sig = False
    for i, src in enumerate(assets):
        for j, dst in enumerate(assets):
            if i == j or not np.isfinite(pvals[i, j]):
                continue
            sig = bool(pvals[i, j] < alpha)
            if sig:
                any_sig = True
            lines.append(
                f"- {src} -> {dst}: p={pvals[i, j]:.4g}, best_lag={int(lags[i, j])} "
                f"({'SIGNIFICANT' if sig else 'not significant'} at alpha={alpha:.3f})."
            )

    if not any_sig:
        lines.append(
            "No directed edge reached significance; observed co-movement may be dominated by "
            "contemporaneous common factors."
        )

    return "\n".join(lines)


def serialize_time_resolved_ccf(
    time_ccf_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for pair, payload in time_ccf_results.items():
        centers = pd.DatetimeIndex(payload["center_times"])
        out[pair] = {
            "window_bins": int(payload["window_bins"]),
            "step_bins": int(payload["step_bins"]),
            "window_duration": str(payload["window_duration"]),
            "step_duration": str(payload["step_duration"]),
            "lags": np.asarray(payload["lags"], dtype=int).tolist(),
            "center_times_utc": [t.isoformat() for t in centers],
            "dominant_lag_bins": np.asarray(payload["dominant_lag_bins"], dtype=int).tolist(),
            "dominant_lag_timedelta": list(payload["dominant_lag_timedelta"]),
            "dominant_corr": np.asarray(payload["dominant_corr"], dtype=float).tolist(),
            "ccf_matrix": np.asarray(payload["ccf_matrix"], dtype=float).tolist(),
        }
    return out


def serialize_granger_result(
    granger_result: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "assets": list(granger_result.get("assets", [])),
        "alpha": float(granger_result.get("alpha", 0.05)),
        "maxlag": int(granger_result.get("maxlag", 1)),
        "p_value_matrix": np.asarray(granger_result.get("p_value_matrix", np.empty((0, 0))), dtype=float).tolist(),
        "best_lag_matrix": np.asarray(granger_result.get("best_lag_matrix", np.empty((0, 0))), dtype=float).tolist(),
        "significant_matrix": np.asarray(granger_result.get("significant_matrix", np.empty((0, 0))), dtype=float).tolist(),
        "te_proxy_matrix": np.asarray(granger_result.get("te_proxy_matrix", np.empty((0, 0))), dtype=float).tolist(),
        "net_te_proxy_matrix": np.asarray(granger_result.get("net_te_proxy_matrix", np.empty((0, 0))), dtype=float).tolist(),
        "directed_edges": list(granger_result.get("directed_edges", [])),
    }


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
