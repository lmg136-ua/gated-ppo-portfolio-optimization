"""Preprocesado V2 anti look-ahead para el TFG."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import yaml

from data.context_builder import build_context_features
from features import (
    RollingNormalizer,
    build_market_features,
    build_regime_features,
    compute_factor_loadings,
)

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class PreparedDataBundle:
    market_features: pd.DataFrame
    context_features: pd.DataFrame
    returns: pd.DataFrame
    factor_loadings: dict[str, pd.DataFrame]
    benchmark_returns: pd.Series
    metadata: dict


class DataPreprocessor:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.market_normalizer = RollingNormalizer()
        self.context_normalizer = RollingNormalizer()

    def prepare_bundle(
        self,
        prices: pd.DataFrame,
        context: pd.DataFrame,
        train_end_date: str,
        volumes: Optional[pd.DataFrame] = None,
        benchmark_prices: Optional[pd.DataFrame] = None,
    ) -> PreparedDataBundle:
        prices = prices.ffill().bfill()
        if volumes is not None:
            volumes = volumes.reindex(index=prices.index, columns=prices.columns).ffill().bfill()

        exp_cfg = self.config.get("experiment", {})
        feature_names = self.config["market_features"]["features"]
        if "volume_ratio" in feature_names and (volumes is None or volumes.empty):
            if exp_cfg.get("require_real_volume", False) and exp_cfg.get("mode", "experiment") == "experiment":
                raise RuntimeError("El modelo requiere volumen real para construir volume_ratio en modo experimento.")
            logger.warning("volume_ratio solicitado pero no hay volumen real disponible. Se omitira esa familia de features.")

        benchmark_series = None
        if benchmark_prices is not None and not benchmark_prices.empty:
            benchmark_series = benchmark_prices.squeeze().reindex(prices.index).ffill().bfill()
        elif exp_cfg.get("require_real_benchmark", False) and exp_cfg.get("mode", "experiment") == "experiment":
            raise RuntimeError("El modelo requiere benchmark real (SPY) en modo experimento.")

        context_base = build_context_features(
            fred_data=context,
            add_derived_features=self.config.get("context", {}).get("derived_features", True),
        )
        market_features_raw = build_market_features(
            prices=prices,
            volumes=volumes,
            benchmark_prices=benchmark_series,
            config=self.config,
        )

        common_idx = market_features_raw.index.intersection(context_base.index).intersection(prices.index)
        market_features_raw = market_features_raw.loc[common_idx]
        context_base = context_base.loc[common_idx]
        prices = prices.loc[common_idx]
        returns = prices.pct_change().fillna(0.0)

        benchmark_returns = (
            benchmark_series.reindex(common_idx).pct_change().fillna(0.0)
            if benchmark_series is not None
            else returns.mean(axis=1).fillna(0.0)
        )

        train_mask = market_features_raw.index <= pd.Timestamp(train_end_date)
        if train_mask.sum() < 30:
            raise ValueError("Muy pocas observaciones de train para ajustar el preprocesador.")

        market_norm = self.market_normalizer.fit(market_features_raw.loc[train_mask]).transform(market_features_raw).clip(-5, 5)
        context_base_norm = self.context_normalizer.fit(context_base.loc[train_mask]).transform(context_base).clip(-5, 5)
        regime_features = build_regime_features(
            context_raw=context_base,
            market_features_raw=market_features_raw,
            train_mask=train_mask,
            config=self.config,
        )
        context_final = pd.concat([context_base_norm, regime_features.loc[context_base_norm.index]], axis=1).fillna(0.0)

        aggregate_cols = [c for c in market_norm.columns if c.startswith("__market_") or c.startswith("__benchmark_")]
        factor_cfg = self.config.get("factor_features", {})
        factor_loadings = compute_factor_loadings(
            prices=prices,
            benchmark_prices=benchmark_series.reindex(common_idx) if benchmark_series is not None else None,
            beta_window=int(factor_cfg.get("beta_window", 63)),
            momentum_window=int(factor_cfg.get("momentum_window", 63)),
            low_vol_window=int(factor_cfg.get("low_vol_window", 63)),
        )
        factor_loadings = {name: frame.reindex(common_idx).fillna(0.0) for name, frame in factor_loadings.items()}

        metadata = {
            "train_end_date": str(train_end_date),
            "market_feature_count": int(market_norm.shape[1]),
            "context_feature_count": int(context_final.shape[1]),
            "factor_names": list(factor_loadings.keys()),
            "market_state_features": aggregate_cols,
            "regime_features": list(regime_features.columns),
        }

        logger.info(
            "Preprocesado | market=%s | context=%s | returns=%s | factors=%s",
            market_norm.shape,
            context_final.shape,
            returns.shape,
            list(factor_loadings.keys()),
        )
        return PreparedDataBundle(
            market_features=market_norm,
            context_features=context_final,
            returns=returns,
            factor_loadings=factor_loadings,
            benchmark_returns=benchmark_returns,
            metadata=metadata,
        )

    def prepare(
        self,
        prices: pd.DataFrame,
        context: pd.DataFrame,
        train_end_date: str,
        volumes: Optional[pd.DataFrame] = None,
        benchmark_prices: Optional[pd.DataFrame] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        bundle = self.prepare_bundle(
            prices=prices,
            context=context,
            train_end_date=train_end_date,
            volumes=volumes,
            benchmark_prices=benchmark_prices,
        )
        return bundle.market_features, bundle.context_features, bundle.returns
