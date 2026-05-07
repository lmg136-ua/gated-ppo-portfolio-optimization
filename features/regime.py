"""Detector explicito de regimen."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _select_regime_columns(context: pd.DataFrame, market_state: pd.DataFrame) -> pd.DataFrame:
    cols = [name for name in ["VIXCLS", "T10Y2Y", "DGS10", "DGS2", "FEDFUNDS"] if name in context.columns]
    regime_df = context[cols].copy() if cols else pd.DataFrame(index=context.index)
    for name in ["__market_realized_vol_20d", "__market_avg_corr_20d", "__market_dispersion_20d"]:
        if name in market_state.columns:
            regime_df[name] = market_state[name]
    if "T10Y2Y" not in regime_df.columns and {"DGS10", "DGS2"}.issubset(context.columns):
        regime_df["T10Y2Y"] = context["DGS10"] - context["DGS2"]
    return regime_df.ffill().bfill().fillna(0.0)


@dataclass
class ExplicitRegimeDetector:
    n_states: int = 3
    covariance_type: str = "diag"
    random_state: int = 42

    def __post_init__(self) -> None:
        self.scaler = StandardScaler()
        self.model = GaussianMixture(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
        )
        self._fitted = False
        self.stress_state_: Optional[int] = None

    def fit(self, X_train: pd.DataFrame) -> "ExplicitRegimeDetector":
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled)
        means = pd.DataFrame(self.model.means_, columns=X_train.columns)
        stress_score = pd.Series(0.0, index=means.index)
        for col in means.columns:
            if any(token in col for token in ["VIX", "vol", "corr"]):
                stress_score += means[col]
            elif "T10Y2Y" in col:
                stress_score -= means[col]
        self.stress_state_ = int(stress_score.idxmax())
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("ExplicitRegimeDetector no ajustado")
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)
        states = self.model.predict(X_scaled)
        out = pd.DataFrame(index=X.index)
        for i in range(self.n_states):
            out[f"regime_prob_{i}"] = probs[:, i]
        out["regime_state"] = states.astype(float)
        if self.stress_state_ is not None:
            out["regime_stress_prob"] = probs[:, self.stress_state_]
            out["regime_stress_state"] = float(self.stress_state_)
        return out


def build_regime_features(
    context_raw: pd.DataFrame,
    market_features_raw: pd.DataFrame,
    train_mask: pd.Series,
    config: dict,
) -> pd.DataFrame:
    reg_cfg = config.get("context", {}).get("regime_detector", {})
    if not reg_cfg.get("enabled", True):
        return pd.DataFrame(index=context_raw.index)

    market_state = market_features_raw[[c for c in market_features_raw.columns if c.startswith("__market_") or c.startswith("__benchmark_")]]
    regime_input = _select_regime_columns(context_raw, market_state)
    detector = ExplicitRegimeDetector(
        n_states=int(reg_cfg.get("n_states", 3)),
        covariance_type=reg_cfg.get("covariance_type", "diag"),
        random_state=int(reg_cfg.get("random_state", 42)),
    )
    detector.fit(regime_input.loc[train_mask])
    regime_features = detector.transform(regime_input)
    logger.info("Regime detector ajustado: %s estados", detector.n_states)
    return regime_features
