"""Features de mercado y estado agregado para modelo."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_returns(prices: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    features = {}
    for p in periods:
        rets = prices.pct_change(periods=p)
        for col in prices.columns:
            features[f"{col}_ret_{p}d"] = rets[col]
    return pd.DataFrame(features, index=prices.index)


def compute_volatility(prices: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    simple_ret = prices.pct_change()
    features = {}
    for w in windows:
        vol = simple_ret.rolling(window=w, min_periods=max(5, w // 2)).std() * np.sqrt(252)
        for col in prices.columns:
            features[f"{col}_vol_{w}d"] = vol[col]
    return pd.DataFrame(features, index=prices.index)


def compute_momentum(prices: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    features = {}
    shifted = prices.shift(1)
    for w in windows:
        mom = shifted / shifted.shift(w) - 1.0
        for col in prices.columns:
            features[f"{col}_mom_{w}d"] = mom[col]
    return pd.DataFrame(features, index=prices.index)


def compute_rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    ret = prices.pct_change()
    features = {}
    for col in prices.columns:
        delta = ret[col]
        gain = delta.clip(lower=0).rolling(window=window, min_periods=window).mean()
        loss = (-delta).clip(lower=0).rolling(window=window, min_periods=window).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        features[f"{col}_rsi_{window}"] = rsi / 100.0
    return pd.DataFrame(features, index=prices.index)


def compute_volume_ratio(volumes: Optional[pd.DataFrame], window: int = 20) -> pd.DataFrame:
    if volumes is None or volumes.empty:
        return pd.DataFrame()
    base = volumes.replace(0, np.nan)
    avg = base.rolling(window=window, min_periods=max(5, window // 2)).mean()
    ratio = base / avg
    features = {f"{col}_volume_ratio_{window}d": ratio[col] for col in volumes.columns}
    return pd.DataFrame(features, index=volumes.index)


def _pairwise_average_correlation(window_values: np.ndarray) -> float:
    if window_values.ndim != 2 or window_values.shape[0] < 2 or window_values.shape[1] < 2:
        return 0.0
    mask = ~np.isnan(window_values).all(axis=0)
    window_values = window_values[:, mask]
    if window_values.shape[1] < 2:
        return 0.0
    corr = np.corrcoef(window_values, rowvar=False)
    if corr.ndim != 2:
        return 0.0
    off_diag = corr[np.triu_indices(corr.shape[0], k=1)]
    off_diag = off_diag[~np.isnan(off_diag)]
    if off_diag.size == 0:
        return 0.0
    return float(np.mean(off_diag))


def compute_market_state_features(
    prices: pd.DataFrame,
    benchmark_prices: Optional[pd.Series] = None,
    correlation_window: int = 20,
    dispersion_window: int = 20,
    benchmark_lookback: int = 20,
    realized_vol_window: int = 20,
    correlation_min_periods: int = 10,
) -> pd.DataFrame:
    returns = prices.pct_change()
    realized_vol = returns.rolling(realized_vol_window, min_periods=max(5, realized_vol_window // 2)).std().mean(axis=1) * np.sqrt(252)
    dispersion = returns.std(axis=1).rolling(dispersion_window, min_periods=max(5, dispersion_window // 2)).mean()

    avg_corr_values = np.zeros(len(returns), dtype=float)
    ret_values = returns.values
    for idx in range(len(returns)):
        start = max(0, idx - correlation_window + 1)
        window_values = ret_values[start : idx + 1]
        if len(window_values) < correlation_min_periods:
            avg_corr_values[idx] = 0.0
            continue
        avg_corr_values[idx] = _pairwise_average_correlation(window_values)
    avg_corr = pd.Series(avg_corr_values, index=returns.index, name="__market_avg_corr_20d")

    if benchmark_prices is None:
        benchmark_returns = returns.mean(axis=1).fillna(0.0)
    else:
        benchmark_series = benchmark_prices.squeeze().reindex(prices.index).ffill().bfill()
        benchmark_returns = benchmark_series.pct_change().fillna(0.0)

    benchmark_values = (1.0 + benchmark_returns).cumprod()
    benchmark_drawdown = benchmark_values / benchmark_values.cummax() - 1.0

    return pd.DataFrame(
        {
            "__market_realized_vol_20d": realized_vol,
            "__market_avg_corr_20d": avg_corr,
            "__market_dispersion_20d": dispersion,
            "__benchmark_return_20d": benchmark_returns.rolling(benchmark_lookback, min_periods=max(5, benchmark_lookback // 2)).sum(),
            "__benchmark_drawdown": benchmark_drawdown,
            "__benchmark_return_1d": benchmark_returns,
        },
        index=prices.index,
    )


def build_market_features(
    prices: pd.DataFrame,
    config: dict,
    volumes: Optional[pd.DataFrame] = None,
    benchmark_prices: Optional[pd.Series] = None,
) -> pd.DataFrame:
    feature_names = config["market_features"]["features"]
    state_cfg = config["market_features"].get("market_state", {})

    parts = []
    if "return_1d" in feature_names:
        parts.append(compute_returns(prices, periods=[1]))
    if "return_5d" in feature_names:
        parts.append(compute_returns(prices, periods=[5]))
    if "return_20d" in feature_names:
        parts.append(compute_returns(prices, periods=[20]))
    if "volatility_20d" in feature_names:
        parts.append(compute_volatility(prices, windows=[20]))
    if "momentum_20d" in feature_names:
        parts.append(compute_momentum(prices, windows=[20]))
    if "rsi_14" in feature_names:
        parts.append(compute_rsi(prices, window=14))
    if "volume_ratio" in feature_names:
        volume_ratio = compute_volume_ratio(volumes, window=20)
        if not volume_ratio.empty:
            parts.append(volume_ratio)
    parts.append(
        compute_market_state_features(
            prices=prices,
            benchmark_prices=benchmark_prices,
            correlation_window=state_cfg.get("correlation_window", 20),
            dispersion_window=state_cfg.get("dispersion_window", 20),
            benchmark_lookback=state_cfg.get("benchmark_lookback", 20),
            realized_vol_window=state_cfg.get("realized_vol_window", 20),
            correlation_min_periods=state_cfg.get("correlation_min_periods", 10),
        )
    )
    market = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    logger.info("Market features: %s", market.shape)
    return market
