"""Construccion de features de contexto macrofinanciero."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def build_context_features(
    fred_data: pd.DataFrame,
    add_derived_features: bool = True,
) -> pd.DataFrame:
    """
    Construye el vector de contexto a partir de series FRED.

    Las features derivadas recogen proxies compactos de inflacion,
    pendiente de la curva, tension de mercado y ciclo monetario.
    """
    context = fred_data.copy()

    if add_derived_features and len(fred_data.columns) > 1:
        if "CPIAUCSL" in context.columns:
            context["CPI_mom"] = context["CPIAUCSL"].pct_change(21)

        if "DGS10" in context.columns and "DGS2" in context.columns and "T10Y2Y" not in context.columns:
            context["T10Y2Y_calc"] = context["DGS10"] - context["DGS2"]

        if "VIXCLS" in context.columns:
            context["VIX_high_stress"] = (context["VIXCLS"] > 25).astype(float)
            context["VIX_change_5d"] = context["VIXCLS"].pct_change(5)

        if "FEDFUNDS" in context.columns:
            context["FEDFUNDS_change_21d"] = context["FEDFUNDS"].diff(21)

    context = context.ffill().bfill().dropna()
    logger.info("Contexto construido: %s", context.shape)
    return context
