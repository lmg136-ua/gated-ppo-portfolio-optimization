"""Normalizacion de features para el pipeline """

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class RollingNormalizer:
    mean_: Optional[pd.Series] = None
    std_: Optional[pd.Series] = None
    _fitted: bool = False

    def fit(self, X: pd.DataFrame) -> "RollingNormalizer":
        self.mean_ = X.mean()
        self.std_ = X.std().replace(0, 1.0).fillna(1.0)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("RollingNormalizer no ajustado")
        return ((X - self.mean_) / self.std_).fillna(0.0)
