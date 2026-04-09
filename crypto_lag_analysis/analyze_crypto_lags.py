#!/usr/bin/env python3
"""Cross-correlation and time-lag analysis for crypto trade time series."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from core_analysis import (
    align_series,
    analyze_pairs,
    build_summary_text,
    ccf_discrete,
    dominant_lag,
    normalize_series,
    plot_ccf,
    plot_correlation_vs_gap_density,
    plot_normalized_series,
    run_gap_robustness,
    timedelta_from_lag_bins,
    timedelta_to_bins,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-correlation and time-lag analysis for crypto trade feeds"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(".."),
        help="Root folder containing quarter folders with CSV files.",
    )
    parser.add_argument(
        "--assets",
        type=str,
        nargs="+",
        default=["BTC", "ETH", "ADA"],
        help="Assets to load. Example: BTC ETH ADA or BTC ACH.",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        nargs="+",
        default=None,
        help="Optional pair list like BTC-ETH BTC-ADA. Defaults to all combinations.",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1min",
        help="Resampling interval, e.g. 1s, 5s, 1min, 5min, 15min, 1h.",
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
        "--ccf-window",
        type=int,
        default=15,
        help="Half-window in bins shown around lag=0 in the CCF plot.",
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
        "--vol-window",
        type=int,
        default=60,
        help="Rolling window (in bins) used for volatility overlay.",
    )
    parser.add_argument(
        "--gap-densities",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.5],
        help="Gap fractions for robustness test.",
    )
    parser.add_argument(
        "--gap-repeats",
        type=int,
        default=80,
        help="Replications per gap density.",
    )
    parser.add_argument(
        "--periodic-gap-period",
        type=str,
        default="90min",
        help="Periodic gap cycle length, e.g. 90min.",
    )
    parser.add_argument(
        "--periodic-gap-drop",
        type=str,
        default="30min",
        help="Legacy fixed periodic drop duration. Its implied density is included for comparison.",
    )
    parser.add_argument(
        "--periodic-gap-phase",
        type=str,
        default="0min",
        help="Phase offset for periodic gaps, e.g. 0min or 10min.",
    )
    parser.add_argument(
        "--resolution-sweep",
        type=str,
        nargs="*",
        default=[],
        help=(
            "Optional additional intervals to probe unresolved zero-lag behavior, "
            "e.g. 30s 10s 5s 1s."
        ),
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
        default=Path("outputs"),
        help="Output directory for figures and summary files.",
    )
    return parser.parse_args()


def symbol_candidates_for_asset(asset: str) -> List[str]:
    asset = asset.upper()
    bases = [asset]
    if asset in {"BTC", "XBT"}:
        bases = ["XBT", "BTC"]

    quotes = ["USDT", "USD", "USDC"]
    candidates: List[str] = []
    for base in bases:
        for quote in quotes:
            candidates.append(f"{base}{quote}")
    return candidates


def discover_symbol_files(data_root: Path, symbol_candidates: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for candidate in symbol_candidates:
        files.extend(data_root.rglob(f"{candidate}.csv"))

    deduped = sorted(set(files))
    if not deduped:
        raise FileNotFoundError(
            f"No files found for candidates: {symbol_candidates} under {data_root}"
        )
    return deduped


def load_trade_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=["timestamp", "price", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df[["timestamp", "price", "volume"]].sort_values("timestamp")


def build_price_series(files: Sequence[Path], interval: str, max_interp_steps: int) -> pd.Series:
    frames = [load_trade_csv(p) for p in files]
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp")

    # Multiple trades can share a timestamp. Keep the last trade price for that instant.
    instant_price = df.groupby("timestamp", as_index=True)["price"].last()
    regular = instant_price.resample(interval).last()
    regular = regular.interpolate(method="time", limit=max_interp_steps, limit_area="inside")
    return regular


def build_pairs(assets: Sequence[str], pair_specs: Sequence[str] | None) -> List[Tuple[str, str]]:
    canonical_assets = [a.upper() for a in assets]

    if not pair_specs:
        pairs = list(combinations(canonical_assets, 2))
        if not pairs:
            raise ValueError("At least two assets are required to form a pair.")
        return [(a, b) for a, b in pairs]

    out: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    available = set(canonical_assets)
    for spec in pair_specs:
        token = spec.strip().upper().replace(":", "-").replace("/", "-")
        parts = token.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid pair specification: {spec}")
        a, b = parts[0], parts[1]
        if a == b:
            raise ValueError(f"Pair must contain two different assets: {spec}")
        if a not in available or b not in available:
            raise ValueError(
                f"Pair {spec} references unknown assets. Available assets: {sorted(available)}"
            )
        pair = (a, b)
        if pair not in seen:
            out.append(pair)
            seen.add(pair)

    if not out:
        raise ValueError("No valid pairs were parsed from --pairs.")
    return out


def run_resolution_sweep(
    files_by_asset: Mapping[str, Sequence[Path]],
    pairs: Sequence[Tuple[str, str]],
    intervals: Sequence[str],
    normalization: str,
    max_interp_steps: int,
    max_lag: int,
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}

    for interval in intervals:
        try:
            price_series = {
                asset: build_price_series(
                    files=files,
                    interval=interval,
                    max_interp_steps=max_interp_steps,
                )
                for asset, files in files_by_asset.items()
            }
            normalized = {
                asset: normalize_series(series, method=normalization)
                for asset, series in price_series.items()
            }
            aligned = align_series(normalized)
        except (FileNotFoundError, pd.errors.ParserError) as exc:
            results[interval] = {"error": str(exc)}
            continue

        pair_results: Dict[str, Dict[str, Any]] = {}
        for a, b in pairs:
            lags, ccf_vals = ccf_discrete(
                aligned[a].to_numpy(dtype=float),
                aligned[b].to_numpy(dtype=float),
                max_lag=max_lag,
            )
            lag_bin, corr = dominant_lag(lags, ccf_vals)
            pair_results[f"{a}-{b}"] = {
                "dominant_lag_bins": int(lag_bin),
                "dominant_lag_timedelta": timedelta_from_lag_bins(lag_bin, interval),
                "dominant_corr": float(corr),
                "peak_abs_corr": float(abs(corr)),
            }

        results[interval] = {
            "n_points": int(len(aligned)),
            "pair_results": pair_results,
        }

    return results


def build_resolution_section(resolution_results: Dict[str, Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("Resolution sweep (zero-lag deep dive):")

    for interval, payload in resolution_results.items():
        if "error" in payload:
            lines.append(f"- {interval}: failed ({payload['error']})")
            continue

        lines.append(f"- {interval}: n_points={payload['n_points']}")
        for pair, pair_res in payload["pair_results"].items():
            lines.append(
                f"  {pair}: lag={pair_res['dominant_lag_bins']} bins "
                f"({pair_res['dominant_lag_timedelta']}), "
                f"r={pair_res['dominant_corr']:.4f}, |r|={pair_res['peak_abs_corr']:.4f}"
            )

    lines.append(
        "If lag remains 0 at finer bins, the reaction time is still below resolution; "
        "test lower-liquidity pairs via --assets/--pairs (e.g., BTC ACH with BTC-ACH)."
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    assets = [a.upper() for a in args.assets]
    pairs = build_pairs(assets, args.pairs)

    files_by_asset: Dict[str, List[Path]] = {}
    serializable_files: Dict[str, List[str]] = {}
    price_series: Dict[str, pd.Series] = {}

    for asset in assets:
        files = discover_symbol_files(args.data_root, symbol_candidates_for_asset(asset))
        files_by_asset[asset] = files
        serializable_files[asset] = [str(p) for p in files]
        price_series[asset] = build_price_series(
            files=files,
            interval=args.interval,
            max_interp_steps=args.max_interp_steps,
        )

    normalized = {
        asset: normalize_series(series, method=args.normalization)
        for asset, series in price_series.items()
    }
    aligned = align_series(normalized)

    period_bins = timedelta_to_bins(args.periodic_gap_period, args.interval)
    drop_bins = timedelta_to_bins(args.periodic_gap_drop, args.interval)
    if drop_bins >= period_bins:
        raise ValueError("periodic-gap-drop must be shorter than periodic-gap-period.")
    phase_bins = timedelta_to_bins(args.periodic_gap_phase, args.interval, allow_zero=True)
    phase_bins = phase_bins % period_bins

    gap_densities = sorted(set(float(d) for d in args.gap_densities))
    fixed_periodic_density = float(drop_bins / period_bins)
    if fixed_periodic_density not in gap_densities:
        gap_densities.append(fixed_periodic_density)
        gap_densities.sort()

    analyses, baseline_thresholds = analyze_pairs(
        aligned_df=aligned,
        pairs=pairs,
        max_lag=args.max_lag,
        mc_sims=args.mc_sims,
        rng=rng,
        interval=args.interval,
    )

    gap_results = run_gap_robustness(
        aligned_df=aligned,
        pairs=pairs,
        max_lag=args.max_lag,
        gap_densities=gap_densities,
        repeats=args.gap_repeats,
        rng=rng,
        baseline_thresholds=baseline_thresholds,
        periodic_period_bins=period_bins,
        periodic_phase_bins=phase_bins,
    )

    resolution_results: Dict[str, Dict[str, Any]] = {}
    if args.resolution_sweep:
        resolution_results = run_resolution_sweep(
            files_by_asset=files_by_asset,
            pairs=pairs,
            intervals=args.resolution_sweep,
            normalization=args.normalization,
            max_interp_steps=args.max_interp_steps,
            max_lag=args.max_lag,
        )

    plot_normalized_series(
        aligned,
        args.output_dir / "normalized_series.png",
        title="Macroscopic Trajectory and Volatility Regimes (Trades)",
        vol_window=args.vol_window,
    )
    plot_ccf(
        analyses,
        interval=args.interval,
        output_path=args.output_dir / "ccf_with_significance.png",
        window_bins=args.ccf_window,
    )
    plot_correlation_vs_gap_density(gap_results, args.output_dir / "correlation_vs_gap_density.png")
    # Backward-compatible output name.
    plot_correlation_vs_gap_density(gap_results, args.output_dir / "lag_vs_gap_density.png")

    extra_sections: List[str] = []
    if resolution_results:
        extra_sections.append(build_resolution_section(resolution_results))

    summary_text = build_summary_text(
        headline="Cross-Correlation and Time-Lag Analysis Summary",
        analyses=analyses,
        interval=args.interval,
        gap_results=gap_results,
        periodic_gap_period=args.periodic_gap_period,
        periodic_gap_phase=args.periodic_gap_phase,
        extra_sections=extra_sections,
    )
    (args.output_dir / "summary.txt").write_text(summary_text, encoding="utf-8")

    serializable = {
        "config": {
            "data_root": str(args.data_root),
            "assets": assets,
            "pairs": [f"{a}-{b}" for a, b in pairs],
            "interval": args.interval,
            "normalization": args.normalization,
            "max_lag": args.max_lag,
            "ccf_window": args.ccf_window,
            "mc_sims": args.mc_sims,
            "max_interp_steps": args.max_interp_steps,
            "vol_window": args.vol_window,
            "gap_densities_input": args.gap_densities,
            "gap_densities_used": gap_densities,
            "gap_repeats": args.gap_repeats,
            "periodic_gap_period": args.periodic_gap_period,
            "periodic_gap_drop": args.periodic_gap_drop,
            "periodic_gap_phase": args.periodic_gap_phase,
            "resolution_sweep": args.resolution_sweep,
            "seed": args.seed,
        },
        "files": serializable_files,
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
        "resolution_sweep": resolution_results,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )

    print(summary_text)
    print("\nSaved artifacts to:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
