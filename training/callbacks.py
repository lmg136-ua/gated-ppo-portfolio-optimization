"""Callbacks y utilidades de seleccion financiera para V2."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


def financial_selection_score(metrics: dict, config: dict) -> float:
    weights = config.get("training", {}).get("model_selection", {})
    sharpe_w = float(weights.get("sharpe_weight", 0.5))
    calmar_w = float(weights.get("calmar_weight", 0.5))
    return sharpe_w * float(metrics.get("sharpe", 0.0)) + calmar_w * float(metrics.get("calmar", 0.0))


def evaluate_financial_policy(model, env, config: dict) -> tuple[dict, float]:
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    history = env.get_portfolio_history()
    diagnostics = env.get_diagnostics_frame()
    summary_metrics = diagnostics.mean(numeric_only=True).to_dict() if not diagnostics.empty else {}
    metrics = compute_all_metrics(
        portfolio_values=history["portfolio_values"],
        returns=history["simple_returns"],
        weights_history=history["weights"],
        transaction_cost=config["environment"]["transaction_cost"],
        slippage=config["environment"]["slippage"],
        label="validation",
        summary_metrics=summary_metrics,
    )
    score = financial_selection_score(metrics, config)
    return metrics, score


class FinancialEvalCallback(BaseCallback):
    def __init__(
        self,
        val_env,
        config: dict,
        save_dir: str,
        eval_freq: int,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.val_env = val_env
        self.config = config
        self.save_dir = Path(save_dir)
        self.eval_freq = int(eval_freq)
        self.best_score = float("-inf")
        self.eval_history: list[dict] = []
        self.max_no_improvement_evals = int(config.get("training", {}).get("model_selection", {}).get("max_no_improvement_evals", 6))
        self.min_evals = int(config.get("training", {}).get("model_selection", {}).get("min_evals", 3))
        self.no_improvement_count = 0
        self.n_evals = 0

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        metrics, score = evaluate_financial_policy(self.model, self.val_env, self.config)
        self.n_evals += 1
        record = {"timesteps": int(self.num_timesteps), "selection_score": score, **metrics}
        self.eval_history.append(record)

        if score > self.best_score:
            self.best_score = score
            self.no_improvement_count = 0
            self.model.save(str(self.save_dir / "best_model"))
            pd.DataFrame([record]).to_csv(self.save_dir / "best_val_metrics.csv", index=False)
            if self.verbose:
                logger.info("Nuevo mejor checkpoint: score=%.4f", score)
        else:
            self.no_improvement_count += 1

        if self.n_evals >= self.min_evals and self.no_improvement_count >= self.max_no_improvement_evals:
            if self.verbose:
                logger.info("Early stop financiero activado tras %d evaluaciones", self.n_evals)
            return False
        return True

    def _on_training_end(self) -> None:
        if self.eval_history:
            pd.DataFrame(self.eval_history).to_csv(self.save_dir / "eval_history.csv", index=False)
