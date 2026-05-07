"""Entrenamiento y evaluacion PPO para el entorno de cartera."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch

from training.callbacks import FinancialEvalCallback, evaluate_financial_policy

logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_ppo_agent(
    env,
    val_env,
    config: dict,
    save_path: str,
    verbose: int = 0,
    seed_override: int | None = None,
):
    from stable_baselines3 import PPO
    from models.policy import build_policy_kwargs

    train_cfg = config["training"]
    seed = int(seed_override if seed_override is not None else train_cfg.get("seed", 42))
    _set_seed(seed)

    os.makedirs(save_path, exist_ok=True)
    policy_kwargs = build_policy_kwargs(env=env, config=config)
    eval_freq = max(1000, int(train_cfg["total_timesteps"] // int(train_cfg.get("eval_freq_fraction", 10))))
    eval_callback = FinancialEvalCallback(
        val_env=val_env,
        config=config,
        save_dir=save_path,
        eval_freq=eval_freq,
        verbose=verbose,
    )

    tb_root = config.get("paths", {}).get("logs", "runtime/logs")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=train_cfg["learning_rate"],
        n_steps=train_cfg["n_steps"],
        batch_size=train_cfg["batch_size"],
        n_epochs=train_cfg["n_epochs"],
        gamma=train_cfg["gamma"],
        gae_lambda=train_cfg["gae_lambda"],
        clip_range=train_cfg["clip_range"],
        ent_coef=train_cfg["ent_coef"],
        vf_coef=train_cfg["vf_coef"],
        max_grad_norm=train_cfg["max_grad_norm"],
        policy_kwargs=policy_kwargs,
        device=train_cfg.get("device", "auto"),
        seed=seed,
        verbose=verbose,
        tensorboard_log=tb_root,
    )

    logger.info(
        "Entrenando PPO | timesteps=%s | seed=%s",
        f"{train_cfg['total_timesteps']:,}",
        seed,
    )

    save_path_p = Path(save_path)
    run_name = f"{save_path_p.parent.name}/{save_path_p.name}" if save_path else f"PPO_seed_{seed}"
    model.learn(
        total_timesteps=train_cfg["total_timesteps"],
        callback=eval_callback,
        progress_bar=False,
        reset_num_timesteps=True,
        tb_log_name=run_name,
    )

    best_model_path = Path(save_path) / "best_model.zip"
    if best_model_path.exists():
        model = PPO.load(str(best_model_path), env=env)
        logger.info("Cargado mejor modelo desde %s", best_model_path)
    return model


def evaluate_agent(
    model,
    env,
    return_gate: bool = False,
    deterministic: bool = True,
):
    obs, _ = env.reset()
    done = False
    gate_values = []

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if return_gate:
            feat_extractor = getattr(model.policy, "features_extractor", None)
            latest_gate = getattr(feat_extractor, "latest_gate", None)
            if latest_gate is not None:
                gate_values.append(np.asarray(latest_gate.squeeze().cpu().numpy(), dtype=float))

    history = env.get_portfolio_history()
    diagnostics = env.get_diagnostics_frame()
    portfolio_values = history["portfolio_values"]
    returns = history["simple_returns"]
    weights_arr = history["weights"]
    gate_arr = np.asarray(gate_values) if gate_values else None
    summary_metrics = diagnostics.mean(numeric_only=True).to_dict() if not diagnostics.empty else {}
    return portfolio_values, returns, weights_arr, gate_arr, summary_metrics, diagnostics


def score_model_on_validation(model, val_env, config: dict) -> tuple[float, dict]:
    metrics, score = evaluate_financial_policy(model, val_env, config)
    return score, metrics


def select_best_seed(candidates: list[dict]) -> dict:
    if not candidates:
        raise ValueError("No hay candidatos para seleccionar.")
    return sorted(candidates, key=lambda item: item["score"], reverse=True)[0]
