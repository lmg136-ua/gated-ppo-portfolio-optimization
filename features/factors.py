"""Factores y exposiciones para modelo"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, 1.0).fillna(1.0)
    return df.sub(mean, axis=0).div(std, axis=0).fillna(0.0)


def compute_factor_loadings(
    prices: pd.DataFrame,
    benchmark_prices: Optional[pd.Series] = None,
    beta_window: int = 63,
    momentum_window: int = 63,
    low_vol_window: int = 63,
) -> dict[str, pd.DataFrame]:
    returns = prices.pct_change().fillna(0.0)
    if benchmark_prices is None:
        benchmark_returns = returns.mean(axis=1)
    else:
        benchmark_series = benchmark_prices.squeeze().reindex(prices.index).ffill().bfill()
        benchmark_returns = benchmark_series.pct_change().fillna(0.0)

    bench_var = benchmark_returns.rolling(beta_window, min_periods=max(10, beta_window // 3)).var().replace(0, np.nan)
    beta = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for col in prices.columns:
        cov = returns[col].rolling(beta_window, min_periods=max(10, beta_window // 3)).cov(benchmark_returns)
        beta[col] = cov / bench_var

    momentum = prices.shift(1).pct_change(momentum_window)
    low_vol = -returns.rolling(low_vol_window, min_periods=max(10, low_vol_window // 3)).std() * np.sqrt(252)

    return {
        "beta": _cross_sectional_zscore(beta),
        "momentum": _cross_sectional_zscore(momentum),
        "low_vol": _cross_sectional_zscore(low_vol),
    }
