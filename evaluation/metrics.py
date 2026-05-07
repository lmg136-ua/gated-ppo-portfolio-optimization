"""
metrics.py
----------
Métricas de evaluación para estrategias de cartera.

Métricas implementadas:
  - Sharpe Ratio (anualizado)
  - Sortino Ratio
  - CAGR (Compound Annual Growth Rate)
  - Max Drawdown
  - Calmar Ratio (CAGR / MaxDD)
  - CVaR 95% (Conditional Value at Risk)
  - Turnover (promedio diario)
  - Coste total pagado
  - Información por ventana walk-forward 
"""

import numpy as np
import pandas as pd
from typing import Optional, Union


def compute_returns_from_values(portfolio_values: np.ndarray) -> np.ndarray:
    """Calcula retornos diarios a partir de valores de cartera."""
    values = np.array(portfolio_values)
    returns = np.diff(values) / values[:-1]
    return returns


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """
    Sharpe Ratio anualizado.
    SR = (E[r] - rf) / std(r) * sqrt(T)
    """
    if len(returns) < 2:
        return 0.0
    rf_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - rf_daily
    std = np.std(excess, ddof=1)
    if std < 1e-10:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """
    Sortino Ratio: solo penaliza volatilidad negativa.
    """
    if len(returns) < 2:
        return 0.0
    rf_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - rf_daily
    downside = returns[returns < 0]
    if len(downside) < 2:
        return float(np.mean(excess) * periods_per_year / 1e-10)
    downside_std = np.std(downside, ddof=1) * np.sqrt(periods_per_year)
    if downside_std < 1e-10:
        return 0.0
    return float(np.mean(excess) * periods_per_year / downside_std)


def cagr(
    portfolio_values: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """
    Compound Annual Growth Rate.
    CAGR = (V_T / V_0)^(1/n_years) - 1
    """
    values = np.array(portfolio_values)
    if len(values) < 2 or values[0] <= 0:
        return 0.0
    n_years = (len(values) - 1) / periods_per_year
    if n_years <= 0:
        return 0.0
    return float((values[-1] / values[0]) ** (1 / n_years) - 1)


def max_drawdown(portfolio_values: np.ndarray) -> float:
    """
    Maximum Drawdown: mayor caída desde un máximo.
    MDD = max((peak - trough) / peak)
    """
    values = np.array(portfolio_values)
    cummax = np.maximum.accumulate(values)
    drawdowns = (values - cummax) / cummax
    return float(drawdowns.min())


def calmar_ratio(
    portfolio_values: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """Calmar = CAGR / |Max Drawdown|."""
    mdd = abs(max_drawdown(portfolio_values))
    if mdd < 1e-10:
        return 0.0
    return float(cagr(portfolio_values, periods_per_year) / mdd)


def cvar(
    returns: np.ndarray,
    confidence: float = 0.95,
    periods_per_year: int = 252,
) -> float:
    """
    CVaR (Conditional Value at Risk) al nivel de confianza dado.
    CVaR_α = E[r | r < VaR_α]  (anualizado)
    """
    if len(returns) < 10:
        return 0.0
    var_threshold = np.percentile(returns, (1 - confidence) * 100)
    tail = returns[returns <= var_threshold]
    if len(tail) == 0:
        return float(var_threshold * np.sqrt(periods_per_year))
    return float(np.mean(tail) * np.sqrt(periods_per_year))


def average_turnover(weights_history: np.ndarray) -> float:
    """
    Turnover promedio diario: sum(|Δw_i|) promediado.
    """
    if len(weights_history) < 2:
        return 0.0
    diffs = np.abs(np.diff(weights_history, axis=0))
    return float(0.5 * diffs.sum(axis=1).mean())


def total_cost(
    weights_history: Optional[np.ndarray],
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
    avg_turnover: float = 0.0,
    n_days: int = 252,
) -> float:
    """Coste total pagado durante el período."""
    if weights_history is not None:
        to = average_turnover(weights_history)
        n = len(weights_history) - 1
    else:
        to = avg_turnover
        n = max(1, n_days - 1)
    return float(to * (transaction_cost + slippage) * n)


def compute_all_metrics(
    portfolio_values: np.ndarray,
    returns: Optional[np.ndarray] = None,
    weights_history: Optional[np.ndarray] = None,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
    label: str = "strategy",
    summary_metrics: Optional[dict] = None,
) -> dict:
    """
    Calcula todas las métricas de una vez.

    Parameters
    ----------
    portfolio_values : np.ndarray
        Historial de valores de cartera.
    returns : np.ndarray, optional
        Retornos diarios (si no se provee, se calculan desde portfolio_values).
    weights_history : np.ndarray, optional
        Historial de pesos (T, n_assets) para calcular turnover.
    summary_metrics : dict, optional
        Métricas de diagnóstico (to_raw, alpha, etc.) promediadas durante el periodo.
    ...

    Returns
    -------
    dict
        Diccionario con todas las métricas.
    """
    if returns is None:
        returns = compute_returns_from_values(portfolio_values)

    metrics = {
        "label": label,
        "sharpe": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "cagr": cagr(portfolio_values, periods_per_year),
        "max_drawdown": max_drawdown(portfolio_values),
        "calmar": calmar_ratio(portfolio_values, periods_per_year),
        "cvar_95": cvar(returns, confidence=0.95, periods_per_year=periods_per_year),
        "final_value": float(portfolio_values[-1]),
        "total_return": float(portfolio_values[-1] / portfolio_values[0] - 1),
        "n_days": len(returns),
    }

    if summary_metrics and "to_exec" in summary_metrics:
        to_use = summary_metrics["to_exec"]
        metrics["avg_turnover"] = to_use
        metrics["total_cost"] = total_cost(
            weights_history=None,
            transaction_cost=transaction_cost,
            slippage=slippage,
            avg_turnover=to_use,
            n_days=metrics["n_days"]
        )
    elif weights_history is not None:
        metrics["avg_turnover"] = average_turnover(weights_history)
        metrics["total_cost"] = total_cost(weights_history, transaction_cost, slippage)

    if summary_metrics:
        # Añade todas las diagnosticos (to_raw, alpha, flag, dist) al array final
        metrics.update(summary_metrics)

    return metrics


def summarize_walk_forward(
    fold_metrics: list[dict],
) -> dict:
    """Resume métricas de múltiples folds usando solo columnas numéricas."""
    if not fold_metrics:
        return {}

    df = pd.DataFrame(fold_metrics)
    summary = {"n_folds": len(df)}
    for key in df.columns:
        if key in ("label", "n_days"):
            continue
        numeric_values = pd.to_numeric(df[key], errors="coerce")
        numeric_values = numeric_values.dropna()
        if numeric_values.empty:
            continue
        summary[f"{key}_mean"] = float(numeric_values.mean())
        summary[f"{key}_std"] = float(numeric_values.std(ddof=0))
    return summary


def metrics_to_dataframe(metrics_list: list[dict]) -> pd.DataFrame:
    """Convierte lista de métricas a DataFrame para comparar estrategias."""
    df = pd.DataFrame(metrics_list)
    if "label" in df.columns:
        df = df.set_index("label")

    # Formatear para legibilidad
    pct_cols = ["cagr", "max_drawdown", "cvar_95", "total_return", "avg_turnover"]
    for col in pct_cols:
        if col in df.columns:
            df[col] = (df[col] * 100).round(2)

    ratio_cols = ["sharpe", "sortino", "calmar"]
    for col in ratio_cols:
        if col in df.columns:
            df[col] = df[col].round(3)

    return df


if __name__ == "__main__":
    # Test con datos sintéticos
    np.random.seed(42)
    T = 252 * 3  # 3 años

    # Portfolio con retornos ~8% anual, vol ~15%
    daily_ret = np.random.normal(0.08 / 252, 0.15 / np.sqrt(252), T)
    values = np.cumprod(1 + daily_ret) * 1_000_000

    # Pesos random para turnover
    weights = np.random.dirichlet(np.ones(10), size=T)

    metrics = compute_all_metrics(
        portfolio_values=values,
        returns=daily_ret,
        weights_history=weights,
        label="test_strategy",
    )

    print("=== MÉTRICAS ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
