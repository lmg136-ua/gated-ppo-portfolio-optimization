"""Baselines clasicos con protocolo comparable al RL."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from evaluation.metrics import compute_all_metrics, summarize_walk_forward

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    values: np.ndarray
    weights: np.ndarray
    metrics: dict


def get_baseline_protocol(config: dict) -> dict:
    bc = config.get("baselines", {})
    return {
        "protocol_name": str(bc.get("protocol_name", "version_2")),
        "rebalance_every": int(bc.get("rebalance_every", 21)),
        "estimation_window": int(bc.get("estimation_window", 252)),
    }


class BaseStrategy:
    def __init__(self, n_assets: int, w_max: float = 0.10):
        self.n_assets = n_assets
        self.w_max = w_max
        self.name = self.__class__.__name__

    def get_weights(self, returns_window: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class EqualWeightStrategy(BaseStrategy):
    def get_weights(self, returns_window: np.ndarray) -> np.ndarray:
        return np.ones(self.n_assets) / self.n_assets


class InverseVolatilityStrategy(BaseStrategy):
    def __init__(self, n_assets: int, vol_window: int = 63, w_max: float = 0.10):
        super().__init__(n_assets, w_max)
        self.vol_window = vol_window

    def get_weights(self, returns_window: np.ndarray) -> np.ndarray:
        if len(returns_window) < 5:
            return np.ones(self.n_assets) / self.n_assets
        vol = returns_window[-self.vol_window :].std(axis=0, ddof=1)
        inv_vol = 1.0 / np.clip(vol, 1e-6, None)
        w = inv_vol / inv_vol.sum()
        return _project_weights_simple(w, self.w_max)


class MinimumVarianceStrategy(BaseStrategy):
    def get_weights(self, returns_window: np.ndarray) -> np.ndarray:
        if len(returns_window) < 30:
            return np.ones(self.n_assets) / self.n_assets
        cov = LedoitWolf().fit(returns_window).covariance_
        w = cp.Variable(self.n_assets)
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, cov)), [cp.sum(w) == 1, w >= 0, w <= self.w_max])
        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                return _project_weights_simple(w.value, self.w_max)
        except Exception as exc:
            logger.warning("MinVariance fallback: %s", exc)
        return np.ones(self.n_assets) / self.n_assets


class MeanVarianceStrategy(BaseStrategy):
    def get_weights(self, returns_window: np.ndarray) -> np.ndarray:
        if len(returns_window) < 30:
            return np.ones(self.n_assets) / self.n_assets
        mu = returns_window.mean(axis=0)
        cov = LedoitWolf().fit(returns_window).covariance_
        w = cp.Variable(self.n_assets)
        risk_aversion = 10.0
        objective = cp.Maximize(mu @ w - risk_aversion * cp.quad_form(w, cov))
        prob = cp.Problem(objective, [cp.sum(w) == 1, w >= 0, w <= self.w_max])
        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                return _project_weights_simple(w.value, self.w_max)
        except Exception as exc:
            logger.warning("MeanVariance fallback: %s", exc)
        return np.ones(self.n_assets) / self.n_assets


def _project_weights_simple(weights: np.ndarray, w_max: float) -> np.ndarray:
    w = np.clip(np.asarray(weights, dtype=float), 0.0, w_max)
    s = w.sum()
    if s <= 1e-12:
        return np.ones_like(w) / len(w)
    w = w / s
    w = np.clip(w, 0.0, w_max)
    return w / w.sum()


def simulate_strategy(
    strategy: BaseStrategy,
    returns: pd.DataFrame,
    transaction_cost: float,
    slippage: float,
    rebalance_every: int = 21,
    estimation_window: int = 252,
) -> BaselineResult:
    ret = returns.values.astype(float)
    T, N = ret.shape
    w_prev = np.ones(N) / N
    values = [1.0]
    weights_history = [w_prev.copy()]
    net_returns = []

    for t in range(0, T - 1):
        if t >= estimation_window and (t % rebalance_every == 0):
            target = strategy.get_weights(ret[max(0, t - estimation_window) : t])
        else:
            target = w_prev.copy()

        gross = float(np.dot(target, ret[t + 1]))
        turnover = 0.5 * float(np.abs(target - w_prev).sum())
        net = gross - transaction_cost * turnover - slippage * turnover
        values.append(values[-1] * (1.0 + net))
        net_returns.append(net)
        weights_history.append(target.copy())

        drift = target * (1.0 + ret[t + 1])
        denom = drift.sum()
        w_prev = drift / denom if denom > 1e-12 else target.copy()

    values_arr = np.asarray(values)
    weights_arr = np.asarray(weights_history)
    metrics = compute_all_metrics(
        portfolio_values=values_arr,
        returns=np.asarray(net_returns),
        weights_history=weights_arr,
        transaction_cost=transaction_cost,
        slippage=slippage,
        label=strategy.name,
    )
    return BaselineResult(values=values_arr, weights=weights_arr, metrics=metrics)


def simulate_buy_and_hold_index(benchmark_returns: pd.Series, label: str = "BuyAndHoldIndex") -> BaselineResult:
    benchmark_returns = benchmark_returns.fillna(0.0).astype(float)
    values = np.concatenate([[1.0], np.cumprod(1.0 + benchmark_returns.values)])
    metrics = compute_all_metrics(
        portfolio_values=values,
        returns=benchmark_returns.values,
        weights_history=None,
        transaction_cost=0.0,
        slippage=0.0,
        label=label,
        summary_metrics={"avg_turnover": 0.0, "total_cost": 0.0},
    )
    metrics["avg_turnover"] = 0.0
    metrics["total_cost"] = 0.0
    return BaselineResult(values=values, weights=np.empty((len(values), 0)), metrics=metrics)


def run_walk_forward_baselines(
    returns: pd.DataFrame,
    folds: list,
    config: dict,
    benchmark_prices: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark_returns = None
    if benchmark_prices is not None and not benchmark_prices.empty:
        benchmark_returns = benchmark_prices.squeeze().reindex(returns.index).ffill().bfill().pct_change().fillna(0.0)
    protocol = get_baseline_protocol(config)

    strategy_specs = []
    bc = config.get("baselines", {})
    if bc.get("equal_weight", True):
        strategy_specs.append(("EqualWeight", lambda n: EqualWeightStrategy(n_assets=n, w_max=config["environment"]["w_max"])))
    if bc.get("inverse_volatility", True):
        strategy_specs.append(("InverseVolatility", lambda n: InverseVolatilityStrategy(n_assets=n, w_max=config["environment"]["w_max"])))
    if bc.get("min_variance", True):
        strategy_specs.append(("MinimumVariance", lambda n: MinimumVarianceStrategy(n_assets=n, w_max=config["environment"]["w_max"])))
    if bc.get("mean_variance", True):
        strategy_specs.append(("MeanVariance", lambda n: MeanVarianceStrategy(n_assets=n, w_max=config["environment"]["w_max"])))

    summaries = []
    fold_rows = []
    for name, factory in strategy_specs:
        fold_metrics = []
        for fold in folds:
            test_ret = returns[(returns.index >= fold.test_start) & (returns.index <= fold.test_end)]
            train_ret = returns[(returns.index >= fold.train_start) & (returns.index <= fold.train_end)]
            val_ret = returns[(returns.index >= fold.val_start) & (returns.index <= fold.val_end)]
            wf_ret = pd.concat([train_ret, val_ret, test_ret], axis=0)
            test_start_pos = len(train_ret) + len(val_ret)
            strat = factory(returns.shape[1])
            result = simulate_strategy(
                strat,
                wf_ret,
                transaction_cost=config["environment"]["transaction_cost"],
                slippage=config["environment"]["slippage"],
                rebalance_every=protocol["rebalance_every"],
                estimation_window=protocol["estimation_window"],
            )
            vals = result.values[test_start_pos:]
            weights = result.weights[test_start_pos:]
            rets = np.diff(vals) / vals[:-1] if len(vals) > 1 else np.array([])
            m = compute_all_metrics(
                portfolio_values=vals,
                returns=rets,
                weights_history=weights,
                transaction_cost=config["environment"]["transaction_cost"],
                slippage=config["environment"]["slippage"],
                label=name,
            )
            m["fold_id"] = int(fold.fold_id)
            m["test_start"] = str(fold.test_start.date())
            m["test_end"] = str(fold.test_end.date())
            m["baseline_protocol"] = protocol["protocol_name"]
            m["baseline_rebalance_every"] = int(protocol["rebalance_every"])
            m["baseline_estimation_window"] = int(protocol["estimation_window"])
            fold_metrics.append(m)
            fold_rows.append({"model": name, **m})
        summary = summarize_walk_forward(fold_metrics)
        summary["model"] = name
        summary["baseline_protocol"] = protocol["protocol_name"]
        summary["baseline_rebalance_every"] = int(protocol["rebalance_every"])
        summary["baseline_estimation_window"] = int(protocol["estimation_window"])
        summaries.append(summary)

    if bc.get("buy_and_hold_index", True) and benchmark_returns is not None:
        fold_metrics = []
        for fold in folds:
            test_bench = benchmark_returns[(benchmark_returns.index >= fold.test_start) & (benchmark_returns.index <= fold.test_end)]
            result = simulate_buy_and_hold_index(test_bench, label="BuyAndHoldIndex")
            metrics = dict(result.metrics)
            metrics["fold_id"] = int(fold.fold_id)
            metrics["test_start"] = str(fold.test_start.date())
            metrics["test_end"] = str(fold.test_end.date())
            metrics["baseline_protocol"] = protocol["protocol_name"]
            metrics["baseline_rebalance_every"] = 0
            metrics["baseline_estimation_window"] = 0
            fold_metrics.append(metrics)
            fold_rows.append({"model": "BuyAndHoldIndex", **metrics})
        summary = summarize_walk_forward(fold_metrics)
        summary["model"] = "BuyAndHoldIndex"
        summary["baseline_protocol"] = protocol["protocol_name"]
        summary["baseline_rebalance_every"] = 0
        summary["baseline_estimation_window"] = 0
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries).set_index("model")
    fold_df = pd.DataFrame(fold_rows)
    return summary_df, fold_df
