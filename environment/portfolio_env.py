"""Entorno Gymnasium para optimizacion de carteras del modelo final."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from environment.safety import build_safety_projector

logger = logging.getLogger(__name__)


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        market_features: pd.DataFrame,
        context_features: pd.DataFrame,
        returns: pd.DataFrame,
        config: dict,
        mode: str = "train",
        use_context: bool = True,
        use_safety: bool = True,
        use_turnover_penalty: bool = False,
        use_turnover_shield: bool = False,
        factor_loadings: Optional[dict[str, pd.DataFrame]] = None,
        benchmark_returns: Optional[pd.Series] = None,
    ):
        super().__init__()
        common_idx = market_features.index.intersection(context_features.index).intersection(returns.index)
        factor_loadings = factor_loadings or {}
        for frame in factor_loadings.values():
            common_idx = common_idx.intersection(frame.index)
        if benchmark_returns is not None:
            common_idx = common_idx.intersection(benchmark_returns.index)
        if len(common_idx) < 5:
            raise ValueError("El entorno necesita al menos 5 observaciones alineadas.")

        self.market_feature_names = list(market_features.columns)
        self.context_feature_names = list(context_features.columns)
        self.factor_names = list(factor_loadings.keys()) if factor_loadings else []
        self.include_factor_state = bool(self.factor_names)

        self.market_features = market_features.loc[common_idx].astype(np.float32).values
        self.context_features = context_features.loc[common_idx].astype(np.float32).values
        self.returns = returns.loc[common_idx].astype(np.float32).values
        self.factor_loadings = {
            name: frame.loc[common_idx].astype(np.float32).values for name, frame in factor_loadings.items()
        }
        self.benchmark_returns = (
            benchmark_returns.loc[common_idx].astype(np.float32).values if benchmark_returns is not None else None
        )
        self.dates = common_idx

        self.n_steps = len(common_idx)
        self.n_assets = returns.shape[1]
        self.market_dim = market_features.shape[1]
        self._context_all_dim = context_features.shape[1]
        self.context_dim = context_features.shape[1] if use_context else 0
        self.factor_dim = len(self.factor_names) if self.include_factor_state else 0
        self.config = config
        self.mode = mode
        self.use_context = use_context
        self.use_safety = use_safety
        self.use_turnover_penalty = use_turnover_penalty
        self.use_turnover_shield = use_turnover_shield

        env_cfg = config["environment"]
        safety_cfg = config["safety"]
        reward_cfg = env_cfg.get("reward", {})
        adaptive_cfg = safety_cfg.get("adaptive_shield", {})

        self.transaction_cost = float(env_cfg["transaction_cost"])
        self.slippage = float(env_cfg["slippage"])
        self.initial_capital = float(env_cfg["initial_capital"])
        self.turnover_penalty_coeff = float(env_cfg.get("turnover_penalty_coeff", 0.0))
        self.reward_type = str(env_cfg.get("reward_type", "log_return_net"))
        self.action_type = str(env_cfg.get("action_type", "absolute"))
        self.delta_scale = float(env_cfg.get("delta_scale", 1.0))
        self.lambda_turnover = float(reward_cfg.get("lambda_turnover", self.turnover_penalty_coeff))
        self.lambda_downside = float(reward_cfg.get("lambda_downside", 0.0))
        self.downside_factor = float(reward_cfg.get("downside_factor", 2.0))
        self.lambda_drawdown = float(reward_cfg.get("lambda_drawdown", 0.0))

        self.base_tau_min = float(safety_cfg.get("tau_min", 0.0))
        self.base_tau_max = float(safety_cfg.get("tau_max", 1.0))
        self.adaptive_shield_enabled = bool(adaptive_cfg.get("enabled", False))
        self.stress_sensitivity = float(adaptive_cfg.get("stress_sensitivity", 0.6))
        self.inaction_sensitivity = float(adaptive_cfg.get("inaction_sensitivity", 0.5))
        self.regime_weight = float(adaptive_cfg.get("regime_weight", 0.4))
        self.vol_weight = float(adaptive_cfg.get("vol_weight", 0.35))
        self.corr_weight = float(adaptive_cfg.get("corr_weight", 0.25))
        self.min_trade_budget_ratio = float(adaptive_cfg.get("min_trade_budget_ratio", 0.35))

        self.safety = build_safety_projector(self.n_assets, config)
        if not use_safety:
            self.safety.disable()

        obs_dim = self.market_dim + self.n_assets + self.factor_dim + (self.context_dim if use_context else 0)
        self.observation_space = spaces.Box(low=-20.0, high=20.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)

        self._market_idx = {name: idx for idx, name in enumerate(self.market_feature_names)}
        self._context_idx = {name: idx for idx, name in enumerate(self.context_feature_names)}

        self._current_step = 0
        self._weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self._portfolio_value = self.initial_capital
        self._portfolio_history: list[float] = []
        self._return_history: list[float] = []
        self._turnover_history: list[float] = []
        self._reward_history: list[float] = []
        self._weight_history: list[np.ndarray] = []
        self._info_history: list[dict] = []

    def _compute_factor_exposures(self, t: int) -> np.ndarray:
        if not self.include_factor_state:
            return np.zeros(0, dtype=np.float32)
        exposures = []
        for name in self.factor_names:
            factor_row = self.factor_loadings[name][t]
            exposures.append(float(np.dot(self._weights, factor_row)))
        return np.asarray(exposures, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        t = min(self._current_step, self.n_steps - 1)
        parts = [self.market_features[t], self._weights]
        if self.include_factor_state:
            parts.append(self._compute_factor_exposures(t))
        if self.use_context:
            parts.append(self.context_features[t])
        return np.concatenate(parts, axis=0).astype(np.float32)

    def _sigmoid01(self, x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-np.clip(x, -8.0, 8.0))))

    def _current_stress_signal(self, t: int) -> dict:
        vol_z = (
            float(self.market_features[t, self._market_idx["__market_realized_vol_20d"]])
            if "__market_realized_vol_20d" in self._market_idx
            else 0.0
        )
        corr_z = (
            float(self.market_features[t, self._market_idx["__market_avg_corr_20d"]])
            if "__market_avg_corr_20d" in self._market_idx
            else 0.0
        )
        regime_prob = (
            float(self.context_features[t, self._context_idx["regime_stress_prob"]])
            if "regime_stress_prob" in self._context_idx
            else 0.0
        )

        vol_signal = self._sigmoid01(vol_z)
        corr_signal = self._sigmoid01(corr_z)
        stress_signal = np.clip(
            self.regime_weight * regime_prob + self.vol_weight * vol_signal + self.corr_weight * corr_signal,
            0.0,
            1.0,
        )
        tau_max_t = self.base_tau_max
        tau_min_t = self.base_tau_min
        if self.adaptive_shield_enabled:
            tau_max_t = self.base_tau_max * max(self.min_trade_budget_ratio, 1.0 - self.stress_sensitivity * stress_signal)
            tau_min_t = min(self.base_tau_max, self.base_tau_min * (1.0 + self.inaction_sensitivity * stress_signal))
        return {
            "stress_signal": float(stress_signal),
            "vol_signal": float(vol_signal),
            "corr_signal": float(corr_signal),
            "regime_stress_prob": float(regime_prob),
            "tau_min_t": float(tau_min_t),
            "tau_max_t": float(tau_max_t),
        }

    def _compute_reward(self, w_new, w_old, t, turnover_to_penalize=None) -> Tuple[float, dict]:
        asset_returns_next = self.returns[t + 1]
        gross_return = float(np.sum(w_new * asset_returns_next))
        turnover = 0.5 * np.sum(np.abs(w_new - w_old))
        tc = self.transaction_cost * turnover
        slippage_cost = self.slippage * turnover
        net_return = gross_return - tc - slippage_cost
        to_pen = turnover if turnover_to_penalize is None else turnover_to_penalize

        current_peak = max(self._portfolio_history) if self._portfolio_history else self.initial_capital
        prev_drawdown = max(0.0, 1.0 - self._portfolio_value / max(current_peak, 1e-8))
        next_value = self._portfolio_value * (1.0 + net_return)
        next_peak = max(current_peak, next_value)
        next_drawdown = max(0.0, 1.0 - next_value / max(next_peak, 1e-8))
        drawdown_incremental = max(0.0, next_drawdown - prev_drawdown)

        if self.reward_type == "downside_asymmetric_v2":
            reward = (
                net_return
                - self.lambda_turnover * to_pen
                - self.lambda_downside * max(0.0, -net_return) * self.downside_factor
                - self.lambda_drawdown * drawdown_incremental
            )
        else:
            reward = float(np.log1p(np.clip(net_return, -0.95, 10.0)))
            if self.use_turnover_penalty:
                reward -= self.turnover_penalty_coeff * to_pen

        return reward, {
            "gross_return": gross_return,
            "turnover": turnover,
            "transaction_cost": tc,
            "slippage_cost": slippage_cost,
            "net_return": net_return,
            "asset_returns_next": asset_returns_next,
            "drawdown_incremental": drawdown_incremental,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._current_step = 0
        self._weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self._portfolio_value = self.initial_capital
        self._portfolio_history = [self._portfolio_value]
        self._return_history = []
        self._turnover_history = []
        self._reward_history = []
        self._weight_history = [self._weights.copy()]
        self._info_history = []
        return self._get_obs(), {"step": 0, "portfolio_value": self._portfolio_value}

    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        x = np.asarray(action, dtype=np.float32)
        x = np.clip(x, -10.0, 10.0)
        e = np.exp(x - np.max(x))
        return (e / np.sum(e)).astype(np.float32)

    def _action_to_delta(self, action: np.ndarray, trade_budget: float) -> np.ndarray:
        x = np.asarray(action, dtype=np.float32)
        x = np.clip(x, -1.0, 1.0)
        x = x - x.mean()
        norm = 0.5 * np.sum(np.abs(x))
        if norm < 1e-8:
            return np.zeros_like(x)
        delta = x / norm * trade_budget * self.delta_scale
        return delta.astype(np.float32)

    def step(self, action: np.ndarray):
        t = self._current_step
        w_pre = self._weights.copy()
        stress_info = self._current_stress_signal(t)
        tau_min_t = stress_info["tau_min_t"]
        tau_max_t = stress_info["tau_max_t"]

        if self.action_type == "delta":
            delta_raw = self._action_to_delta(action, trade_budget=tau_max_t)
            w_target = w_pre + delta_raw
        else:
            w_target = self._action_to_weights(action)
        to_raw = 0.5 * np.sum(np.abs(w_target - w_pre))

        if np.any(np.isnan(w_target)) or np.any(np.isinf(w_target)):
            logger.error("NaN/Inf en w_target (paso %s)", t)
            w_target = w_pre.copy()

        if self.use_turnover_shield:
            delta = w_target - w_pre
            if to_raw <= tau_min_t:
                alpha = 0.0
            else:
                alpha = min(1.0, tau_max_t / max(to_raw, 1e-8))
            w_proposed = w_pre + alpha * delta
        else:
            alpha = 1.0
            w_proposed = w_target

        to_shield = 0.5 * np.sum(np.abs(w_proposed - w_pre))
        w_exec = self.safety.project(w_proposed) if self.use_safety else w_proposed
        to_exec = 0.5 * np.sum(np.abs(w_exec - w_pre))

        reward, step_info = self._compute_reward(w_exec, w_pre, t, turnover_to_penalize=to_exec)
        if np.isnan(reward) or np.isinf(reward):
            logger.error("Reward invalida (paso %s). Forzado a 0.0", t)
            reward = 0.0

        net_return = step_info["net_return"]
        self._portfolio_value *= 1.0 + net_return
        self._portfolio_value = max(self._portfolio_value, 1.0)

        next_returns = step_info["asset_returns_next"]
        w_drift = w_exec * (1.0 + next_returns)
        denom = float(np.sum(w_drift))
        self._weights = (w_drift / denom).astype(np.float32) if denom > 1e-8 else w_exec.astype(np.float32)

        date_str = str(self.dates[min(t + 1, len(self.dates) - 1)])
        factor_exp = self._compute_factor_exposures(t)
        step_info.update(
            {
                "step": t,
                "date": date_str,
                "portfolio_value": self._portfolio_value,
                "weights": w_exec.copy(),
                "w_target": w_target.copy(),
                "w_proposed": w_proposed.copy(),
                "w_exec": w_exec.copy(),
                "to_raw": float(to_raw),
                "to_shield": float(to_shield),
                "to_exec": float(to_exec),
                "alpha_t": float(alpha),
                "non_trade_flag": int(alpha == 0),
                "dist_raw_exec": float(0.5 * np.sum(np.abs(w_target - w_exec))),
                "constraint_check": self.safety.check_constraints(w_exec) if self.use_safety else True,
                "trade_budget_used": float(to_exec / max(tau_max_t, 1e-8)) if tau_max_t > 0 else 0.0,
                "factor_exposure_beta": float(factor_exp[0]) if len(factor_exp) > 0 else 0.0,
                "factor_exposure_momentum": float(factor_exp[1]) if len(factor_exp) > 1 else 0.0,
                "factor_exposure_low_vol": float(factor_exp[2]) if len(factor_exp) > 2 else 0.0,
            }
        )
        step_info.update(stress_info)
        if "__market_avg_corr_20d" in self._market_idx:
            step_info["market_avg_corr"] = float(self.market_features[t, self._market_idx["__market_avg_corr_20d"]])
        if "__market_realized_vol_20d" in self._market_idx:
            step_info["market_realized_vol"] = float(
                self.market_features[t, self._market_idx["__market_realized_vol_20d"]]
            )
        if "__benchmark_return_1d" in self._market_idx:
            step_info["benchmark_return_1d"] = float(self.market_features[t, self._market_idx["__benchmark_return_1d"]])

        self._portfolio_history.append(self._portfolio_value)
        self._return_history.append(net_return)
        self._turnover_history.append(to_exec)
        self._reward_history.append(reward)
        self._weight_history.append(w_exec.copy())
        self._info_history.append(step_info)

        self._current_step += 1
        terminated = self._current_step >= self.n_steps - 1
        truncated = False
        return self._get_obs(), reward, terminated, truncated, step_info

    def get_portfolio_history(self) -> dict:
        return {
            "portfolio_values": np.asarray(self._portfolio_history, dtype=np.float64),
            "simple_returns": np.asarray(self._return_history, dtype=np.float64),
            "turnover": np.asarray(self._turnover_history, dtype=np.float64),
            "rewards": np.asarray(self._reward_history, dtype=np.float64),
            "weights": np.asarray(self._weight_history, dtype=np.float64),
            "dates": self.dates[: len(self._portfolio_history)],
        }

    def get_diagnostics_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._info_history)

    def render(self):
        print(
            f"Step {self._current_step}/{self.n_steps} | Value={self._portfolio_value:,.0f} | "
            f"Top weights={sorted(self._weights, reverse=True)[:3]}"
        )

    @property
    def obs_dim(self) -> int:
        return int(self.observation_space.shape[0])

    @property
    def act_dim(self) -> int:
        return int(self.action_space.shape[0])


def make_env(
    market_features: pd.DataFrame,
    context_features: pd.DataFrame,
    returns: pd.DataFrame,
    config: dict,
    mode: str = "train",
    use_context: bool = True,
    use_safety: bool = True,
    use_turnover_penalty: bool = False,
    use_turnover_shield: bool = False,
    factor_loadings: Optional[dict[str, pd.DataFrame]] = None,
    benchmark_returns: Optional[pd.Series] = None,
) -> PortfolioEnv:
    return PortfolioEnv(
        market_features=market_features,
        context_features=context_features,
        returns=returns,
        config=config,
        mode=mode,
        use_context=use_context,
        use_safety=use_safety,
        use_turnover_penalty=use_turnover_penalty,
        use_turnover_shield=use_turnover_shield,
        factor_loadings=factor_loadings,
        benchmark_returns=benchmark_returns,
    )
