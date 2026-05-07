"""Walk-forward validation limpia y sin leakage."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __str__(self):
        return (
            f"Fold {self.fold_id}: Train [{self.train_start.date()}->{self.train_end.date()}] | "
            f"Val [{self.val_start.date()}->{self.val_end.date()}] | Test [{self.test_start.date()}->{self.test_end.date()}]"
        )


def generate_walk_forward_folds(
    dates: pd.DatetimeIndex,
    train_years: int = 4,
    val_years: int = 1,
    test_years: int = 1,
    slide_months: int = 6,
) -> List[WalkForwardFold]:
    folds: List[WalkForwardFold] = []
    start = dates[0]
    end = dates[-1]
    current_train_start = start
    fold_id = 0

    def snap(dt: pd.Timestamp) -> pd.Timestamp:
        idx = dates.searchsorted(dt)
        idx = min(idx, len(dates) - 1)
        return dates[idx]

    while True:
        train_end = current_train_start + pd.DateOffset(years=train_years)
        val_start = train_end + pd.offsets.BDay(1)
        val_end = val_start + pd.DateOffset(years=val_years)
        test_start = val_end + pd.offsets.BDay(1)
        test_end = test_start + pd.DateOffset(years=test_years)
        if test_end > end:
            break
        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train_start=snap(current_train_start),
                train_end=snap(train_end),
                val_start=snap(val_start),
                val_end=snap(val_end),
                test_start=snap(test_start),
                test_end=snap(test_end),
            )
        )
        current_train_start = current_train_start + pd.DateOffset(months=slide_months)
        fold_id += 1
    logger.info("Walk-forward: %d folds generados.", len(folds))
    return folds


class WalkForwardRunner:
    def __init__(self, config: dict, output_dir: str = "runtime/walk_forward"):
        self.config = config
        self.output_dir = str(Path(output_dir))
        os.makedirs(self.output_dir, exist_ok=True)
        wf = config["walk_forward"]
        self.train_years = wf["train_years"]
        self.val_years = wf["val_years"]
        self.test_years = wf["test_years"]
        self.slide_months = wf["slide_months"]

    def _resolve_seeds(self) -> list[int]:
        train_cfg = self.config.get("training", {})
        if train_cfg.get("use_benchmark_seeds", False):
            return [int(seed) for seed in train_cfg.get("benchmark_seeds", [train_cfg.get("seed", 42)])]
        return [int(seed) for seed in train_cfg.get("seeds", [train_cfg.get("seed", 42)])]

    def _slice_factor_loadings(
        self,
        factor_loadings: dict[str, pd.DataFrame],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        return {name: frame[(frame.index >= start) & (frame.index <= end)] for name, frame in factor_loadings.items()}

    def _save_yaml(self, path: Path, payload: dict) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)

    def _build_sanity_table(self, results_df: pd.DataFrame) -> pd.DataFrame:
        preferred_columns = [
            "fold_id",
            "test_start",
            "test_end",
            "selected_seed",
            "validation_selection_score",
            "seed_test_sharpe_mean",
            "seed_test_sharpe_std",
            "seed_test_calmar_mean",
            "seed_test_calmar_std",
            "sharpe",
            "cagr",
            "max_drawdown",
            "avg_turnover",
            "total_cost",
            "gate_mean",
            "gate_regime_mean",
            "gate_asset_mean",
            "to_raw",
            "to_exec",
            "alpha_t",
            "non_trade_flag",
            "trade_budget_used",
            "stress_fraction",
            "shield_reduction_mean",
            "dominant_regime",
            "context_source",
        ]
        available = [column for column in preferred_columns if column in results_df.columns]
        return results_df.loc[:, available].copy()

    def run(
        self,
        prices: pd.DataFrame,
        context_raw: pd.DataFrame,
        model_name: str = "PPO_gated_full",
        volumes: Optional[pd.DataFrame] = None,
        benchmark_prices: Optional[pd.DataFrame] = None,
        run_metadata: Optional[dict] = None,
        max_folds: Optional[int] = None,
        include_fold_ids: Optional[list[int]] = None,
    ) -> list[dict]:
        from data.preprocessor import DataPreprocessor
        from environment.portfolio_env import make_env
        from evaluation.metrics import compute_all_metrics
        from stable_baselines3 import PPO
        from training.trainer import evaluate_agent, score_model_on_validation, select_best_seed, train_ppo_agent

        use_context = True
        use_safety = True
        use_penalty = True
        use_shield = True

        common_idx = prices.index.intersection(context_raw.index)
        prices = prices.loc[common_idx].dropna(axis=1, how="all").ffill().dropna()
        context_raw = context_raw.loc[common_idx].ffill().bfill()
        if volumes is not None and not volumes.empty:
            volumes = volumes.reindex(index=common_idx, columns=prices.columns).ffill().bfill()
        if benchmark_prices is not None and not benchmark_prices.empty:
            benchmark_prices = benchmark_prices.reindex(common_idx).ffill().bfill()

        common_idx = prices.index.intersection(context_raw.index)
        context_raw = context_raw.loc[common_idx].ffill().bfill()
        folds = generate_walk_forward_folds(
            dates=common_idx,
            train_years=self.train_years,
            val_years=self.val_years,
            test_years=self.test_years,
            slide_months=self.slide_months,
        )
        if include_fold_ids is not None:
            include_fold_ids = [int(fold_id) for fold_id in include_fold_ids]
            folds = [fold for fold in folds if int(fold.fold_id) in include_fold_ids]
        if max_folds is not None:
            folds = folds[: int(max_folds)]
        if not folds:
            raise ValueError("No hay suficientes datos para walk-forward.")

        all_fold_metrics = []
        seeds = self._resolve_seeds()

        for fold in folds:
            logger.info("%s", fold)
            fold_prices = prices[(prices.index >= fold.train_start) & (prices.index <= fold.test_end)]
            fold_context = context_raw[(context_raw.index >= fold.train_start) & (context_raw.index <= fold.test_end)]
            fold_volumes = (
                volumes[(volumes.index >= fold.train_start) & (volumes.index <= fold.test_end)]
                if volumes is not None and not volumes.empty
                else None
            )
            fold_benchmark = (
                benchmark_prices[(benchmark_prices.index >= fold.train_start) & (benchmark_prices.index <= fold.test_end)]
                if benchmark_prices is not None and not benchmark_prices.empty
                else None
            )

            preprocessor = DataPreprocessor(self.config)
            bundle = preprocessor.prepare_bundle(
                prices=fold_prices,
                context=fold_context,
                train_end_date=str(fold.train_end.date()),
                volumes=fold_volumes,
                benchmark_prices=fold_benchmark,
            )

            market_features = bundle.market_features
            context_features = bundle.context_features
            returns = bundle.returns
            benchmark_returns = bundle.benchmark_returns
            factor_loadings = bundle.factor_loadings

            train_mf = market_features[(market_features.index >= fold.train_start) & (market_features.index <= fold.train_end)]
            train_cf = context_features[(context_features.index >= fold.train_start) & (context_features.index <= fold.train_end)]
            train_ret = returns[(returns.index >= fold.train_start) & (returns.index <= fold.train_end)]
            train_bench = benchmark_returns[(benchmark_returns.index >= fold.train_start) & (benchmark_returns.index <= fold.train_end)]
            train_factors = self._slice_factor_loadings(factor_loadings, fold.train_start, fold.train_end)

            val_mf = market_features[(market_features.index >= fold.val_start) & (market_features.index <= fold.val_end)]
            val_cf = context_features[(context_features.index >= fold.val_start) & (context_features.index <= fold.val_end)]
            val_ret = returns[(returns.index >= fold.val_start) & (returns.index <= fold.val_end)]
            val_bench = benchmark_returns[(benchmark_returns.index >= fold.val_start) & (benchmark_returns.index <= fold.val_end)]
            val_factors = self._slice_factor_loadings(factor_loadings, fold.val_start, fold.val_end)

            test_mf = market_features[(market_features.index >= fold.test_start) & (market_features.index <= fold.test_end)]
            test_cf = context_features[(context_features.index >= fold.test_start) & (context_features.index <= fold.test_end)]
            test_ret = returns[(returns.index >= fold.test_start) & (returns.index <= fold.test_end)]
            test_bench = benchmark_returns[(benchmark_returns.index >= fold.test_start) & (benchmark_returns.index <= fold.test_end)]
            test_factors = self._slice_factor_loadings(factor_loadings, fold.test_start, fold.test_end)

            train_env = make_env(
                train_mf,
                train_cf,
                train_ret,
                self.config,
                mode="train",
                use_context=use_context,
                use_safety=use_safety,
                use_turnover_penalty=use_penalty,
                use_turnover_shield=use_shield,
                factor_loadings=train_factors,
                benchmark_returns=train_bench,
            )
            val_env = make_env(
                val_mf,
                val_cf,
                val_ret,
                self.config,
                mode="val",
                use_context=use_context,
                use_safety=use_safety,
                use_turnover_penalty=use_penalty,
                use_turnover_shield=use_shield,
                factor_loadings=val_factors,
                benchmark_returns=val_bench,
            )
            test_env = make_env(
                test_mf,
                test_cf,
                test_ret,
                self.config,
                mode="test",
                use_context=use_context,
                use_safety=use_safety,
                use_turnover_penalty=use_penalty,
                use_turnover_shield=use_shield,
                factor_loadings=test_factors,
                benchmark_returns=test_bench,
            )

            ckpt_dir = Path(self.output_dir) / model_name / f"fold_{fold.fold_id}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            self._save_yaml(
                ckpt_dir / "fold_metadata.yaml",
                {
                    "model_name": model_name,
                    "fold_id": int(fold.fold_id),
                    "train_start": str(fold.train_start.date()),
                    "train_end": str(fold.train_end.date()),
                    "val_start": str(fold.val_start.date()),
                    "val_end": str(fold.val_end.date()),
                    "test_start": str(fold.test_start.date()),
                    "test_end": str(fold.test_end.date()),
                    "seeds": seeds,
                    "preprocessor": bundle.metadata,
                    "run_metadata": run_metadata or {},
                },
            )

            candidates = []
            for seed in seeds:
                seed_dir = ckpt_dir / f"seed_{seed}"
                seed_dir.mkdir(parents=True, exist_ok=True)
                model_path = seed_dir / "best_model.zip"
                if model_path.exists():
                    logger.info("Saltando entrenamiento fold %s seed %s, cargando modelo existente.", fold.fold_id, seed)
                    model = PPO.load(model_path, env=train_env)
                else:
                    model = train_ppo_agent(
                        env=train_env,
                        val_env=val_env,
                        config=self.config,
                        save_path=str(seed_dir),
                        verbose=0,
                        seed_override=seed,
                    )
                val_score, val_metrics = score_model_on_validation(model, val_env, self.config)
                candidates.append(
                    {
                        "seed": int(seed),
                        "score": float(val_score),
                        "val_metrics": val_metrics,
                        "model": model,
                    }
                )

            best_candidate = select_best_seed(candidates)
            pd.DataFrame(
                [
                    {"seed": cand["seed"], "selection_score": cand["score"], **cand["val_metrics"]}
                    for cand in candidates
                ]
            ).to_csv(ckpt_dir / "seed_selection.csv", index=False)
            self._save_yaml(
                ckpt_dir / "selected_seed.yaml",
                {
                    "selected_seed": int(best_candidate["seed"]),
                    "selection_score": float(best_candidate["score"]),
                },
            )

            seed_test_rows = []
            for cand in candidates:
                seed_portfolio_values, seed_test_returns, seed_weights, _, seed_summary_m, _ = evaluate_agent(
                    cand["model"], test_env, return_gate=False
                )
                seed_test_metrics = compute_all_metrics(
                    portfolio_values=seed_portfolio_values,
                    returns=seed_test_returns,
                    weights_history=seed_weights,
                    transaction_cost=self.config["environment"]["transaction_cost"],
                    slippage=self.config["environment"]["slippage"],
                    summary_metrics=seed_summary_m,
                    label=model_name,
                )
                seed_test_rows.append({"seed": cand["seed"], **seed_test_metrics})
            pd.DataFrame(seed_test_rows).to_csv(ckpt_dir / "seed_test_metrics.csv", index=False)

            portfolio_values, test_returns, weights_history, gate_values, summary_m, diagnostics = evaluate_agent(
                best_candidate["model"], test_env, return_gate=use_gate
            )
            diagnostics.to_csv(ckpt_dir / "test_diagnostics.csv", index=False)

            metrics = compute_all_metrics(
                portfolio_values=portfolio_values,
                returns=test_returns,
                weights_history=weights_history,
                transaction_cost=self.config["environment"]["transaction_cost"],
                slippage=self.config["environment"]["slippage"],
                summary_metrics=summary_m,
                label=model_name,
            )
            metrics["fold_id"] = fold.fold_id
            metrics["test_start"] = str(fold.test_start.date())
            metrics["test_end"] = str(fold.test_end.date())
            metrics["selected_seed"] = int(best_candidate["seed"])
            metrics["validation_selection_score"] = float(best_candidate["score"])
            metrics["experiment_version"] = self.config.get("experiment", {}).get("version", "full_gated_v2")
            metrics["seed_test_sharpe_mean"] = float(pd.DataFrame(seed_test_rows)["sharpe"].mean())
            metrics["seed_test_sharpe_std"] = float(pd.DataFrame(seed_test_rows)["sharpe"].std(ddof=0))
            metrics["seed_test_calmar_mean"] = float(pd.DataFrame(seed_test_rows)["calmar"].mean())
            metrics["seed_test_calmar_std"] = float(pd.DataFrame(seed_test_rows)["calmar"].std(ddof=0))
            if run_metadata and "context_source" in run_metadata:
                metrics["context_source"] = run_metadata["context_source"]
            if not diagnostics.empty and "stress_signal" in diagnostics.columns:
                metrics["stress_fraction"] = float((diagnostics["stress_signal"] > 0.5).mean())
            if not diagnostics.empty and {"to_raw", "to_exec"}.issubset(diagnostics.columns):
                raw = diagnostics["to_raw"].replace(0.0, np.nan)
                reduction = 1.0 - diagnostics["to_exec"] / raw
                metrics["shield_reduction_mean"] = float(reduction.replace([np.inf, -np.inf], np.nan).fillna(0.0).mean())
            if "regime_state" in test_cf.columns:
                regime_mode = test_cf["regime_state"].dropna().mode()
                if not regime_mode.empty:
                    metrics["dominant_regime"] = int(regime_mode.iloc[0])
            if gate_values is not None:
                np.save(ckpt_dir / "gate_values.npy", gate_values)
                if gate_values.ndim == 2 and gate_values.shape[1] >= 2:
                    metrics["gate_regime_mean"] = float(np.mean(gate_values[:, 0]))
                    metrics["gate_asset_mean"] = float(np.mean(gate_values[:, 1]))
                metrics["gate_mean"] = float(np.mean(gate_values))
                metrics["gate_std"] = float(np.std(gate_values))
            all_fold_metrics.append(metrics)

        self._save_results(all_fold_metrics, model_name)
        return all_fold_metrics

    def _save_results(self, fold_metrics: list[dict], model_name: str) -> None:
        from evaluation.metrics import summarize_walk_forward

        results_df = pd.DataFrame(fold_metrics)
        results_path = Path(self.output_dir) / f"{model_name}_fold_results.csv"
        if results_path.exists():
            existing_df = pd.read_csv(results_path)
            if "fold_id" in existing_df.columns and "fold_id" in results_df.columns:
                existing_df = existing_df[~existing_df["fold_id"].isin(results_df["fold_id"])]
                results_df = pd.concat([existing_df, results_df], ignore_index=True)
                results_df = results_df.sort_values("fold_id").reset_index(drop=True)
        results_df.to_csv(results_path, index=False)
        summary = summarize_walk_forward(results_df.to_dict("records"))
        summary["model"] = model_name
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(Path(self.output_dir) / f"{model_name}_summary.csv", index=False)
        sanity_df = self._build_sanity_table(results_df)
        sanity_df.to_csv(Path(self.output_dir) / f"{model_name}_sanity_by_fold.csv", index=False)
        logger.info("Guardados resultados %s", model_name)
