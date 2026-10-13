"""Build the static data payload used by the architecture demo.

The demo is intentionally read-only: it visualizes the curated final results
and repository configuration without retraining or reconstructing unavailable
daily actions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo" / "demo-data.js"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.preprocessor import DataPreprocessor  # noqa: E402


def _num(value: Any, digits: int = 6) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (int, str)):
        return value
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def _record(row: pd.Series, columns: list[str]) -> dict[str, Any]:
    return {column: _num(row[column]) for column in columns if column in row.index}


def _date(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def main() -> None:
    with open(ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    prices = pd.read_parquet(ROOT / "dataset" / "raw" / "prices_raw.parquet")
    volumes = pd.read_parquet(ROOT / "dataset" / "raw" / "volumes_raw.parquet")
    fred = pd.read_parquet(ROOT / "dataset" / "raw" / "fred_context.parquet")
    benchmark = pd.read_parquet(ROOT / "dataset" / "raw" / "benchmark_raw.parquet")

    fold_results = pd.read_csv(ROOT / "results" / "final_v2" / "ppo_gated_full_fold_results.csv")
    global_metrics = pd.read_csv(ROOT / "results" / "final_v2" / "global_metrics_all_strategies.csv")
    series = pd.read_csv(ROOT / "results" / "final_v2" / "portfolio_series_all_strategies.csv")

    first_train_end = prices.index[
        prices.index.searchsorted(prices.index.min() + pd.DateOffset(years=int(config["walk_forward"]["train_years"])))
    ]
    bundle = DataPreprocessor(config).prepare_bundle(
        prices=prices,
        context=fred,
        train_end_date=str(first_train_end.date()),
        volumes=volumes,
        benchmark_prices=benchmark,
    )

    fold_columns = [
        "fold_id",
        "test_start",
        "test_end",
        "sharpe",
        "sortino",
        "cagr",
        "max_drawdown",
        "calmar",
        "cvar_95",
        "final_value",
        "total_return",
        "n_days",
        "avg_turnover",
        "total_cost",
        "gross_return",
        "turnover",
        "transaction_cost",
        "slippage_cost",
        "net_return",
        "drawdown_incremental",
        "portfolio_value",
        "to_raw",
        "to_shield",
        "to_exec",
        "alpha_t",
        "non_trade_flag",
        "dist_raw_exec",
        "trade_budget_used",
        "factor_exposure_beta",
        "factor_exposure_momentum",
        "factor_exposure_low_vol",
        "stress_signal",
        "vol_signal",
        "corr_signal",
        "regime_stress_prob",
        "tau_min_t",
        "tau_max_t",
        "market_avg_corr",
        "market_realized_vol",
        "benchmark_return_1d",
        "selected_seed",
        "validation_selection_score",
        "context_source",
        "stress_fraction",
        "shield_reduction_mean",
        "dominant_regime",
        "gate_regime_mean",
        "gate_asset_mean",
        "gate_mean",
        "gate_std",
    ]
    global_columns = [
        "model",
        "protocol",
        "model_line",
        "n_folds",
        "sharpe_mean",
        "cagr_mean",
        "max_drawdown_mean",
        "calmar_mean",
        "cvar_95_mean",
        "avg_turnover_mean",
        "total_cost_mean",
    ]

    series_by_fold: dict[str, list[dict[str, Any]]] = {}
    for fold_id, group in series.groupby("fold_id"):
        group = group.sort_values("date")
        points = []
        for _, row in group.iterrows():
            points.append(
                {
                    "date": _date(row["date"]),
                    "portfolio_value": _num(row["portfolio_value"], 3),
                    "net_return": _num(row["net_return"], 7),
                    "turnover": _num(row["turnover"], 7),
                    "gross_return": _num(row["gross_return"], 7),
                    "drawdown_incremental": _num(row["drawdown_incremental"], 7),
                }
            )
        series_by_fold[str(int(fold_id))] = points

    folds_payload = []
    for _, row in fold_results.iterrows():
        folds_payload.append(
            {
                "fold_id": int(row["fold_id"]),
                "train_years": int(config["walk_forward"]["train_years"]),
                "val_years": int(config["walk_forward"]["val_years"]),
                "test_years": int(config["walk_forward"]["test_years"]),
                "slide_months": int(config["walk_forward"]["slide_months"]),
                "test_start": _date(row["test_start"]),
                "test_end": _date(row["test_end"]),
            }
        )

    dimensions = {
        "n_assets": int(prices.shape[1]),
        "market_dim": int(bundle.market_features.shape[1]),
        "context_dim": int(bundle.context_features.shape[1]),
        "weights_dim": int(prices.shape[1]),
        "factor_dim": len(bundle.factor_loadings),
        "observation_dim": int(bundle.market_features.shape[1] + prices.shape[1] + len(bundle.factor_loadings) + bundle.context_features.shape[1]),
        "action_dim": int(prices.shape[1]),
        "portfolio_series_rows": int(len(series)),
        "folds": int(len(fold_results)),
    }

    market_state_features = bundle.metadata["market_state_features"]
    regime_features = bundle.metadata["regime_features"]
    context_base_count = int(bundle.context_features.shape[1] - len(regime_features))

    payload = {
        "meta": {
            "title": "TFG RL Portfolio - Demo visual",
            "model": "PPO_gated_full",
            "experiment_version": config["experiment"]["version"],
            "protocol": "version_2",
            "source_note": "Datos reales del repositorio; no incluye entrenamiento en vivo ni reconstruye pesos diarios no versionados.",
        },
        "sources": [
            "config/config.yaml",
            "models/policy.py",
            "models/market_encoder.py",
            "models/context_encoder.py",
            "environment/portfolio_env.py",
            "environment/safety.py",
            "data/preprocessor.py",
            "results/final_v2/ppo_gated_full_fold_results.csv",
            "results/final_v2/portfolio_series_all_strategies.csv",
            "results/final_v2/global_metrics_all_strategies.csv",
            "dataset/raw/prices_raw.parquet",
            "dataset/raw/volumes_raw.parquet",
            "dataset/raw/benchmark_raw.parquet",
            "dataset/raw/fred_context.parquet",
        ],
        "config": {
            "date_range": {"start": config["dates"]["start"], "end": config["dates"]["end"]},
            "walk_forward": config["walk_forward"],
            "market_feature_families": config["market_features"]["features"],
            "fred_series": config["context"]["fred_series"],
            "factor_features": config["factor_features"],
            "environment": {
                "initial_capital": config["environment"]["initial_capital"],
                "transaction_cost": config["environment"]["transaction_cost"],
                "slippage": config["environment"]["slippage"],
                "reward_type": config["environment"]["reward_type"],
                "action_type": config["environment"]["action_type"],
                "delta_scale": config["environment"]["delta_scale"],
                "reward": config["environment"]["reward"],
            },
            "safety": config["safety"],
            "model": config["model"],
            "training": {
                "algorithm": "PPO",
                "total_timesteps": config["training"]["total_timesteps"],
                "seed": config["training"]["seed"],
                "net_arch": config["model"]["policy"]["net_arch"],
                "activation": config["model"]["policy"]["activation"],
            },
        },
        "data_facts": {
            "prices_shape": list(prices.shape),
            "volumes_shape": list(volumes.shape),
            "fred_shape": list(fred.shape),
            "benchmark_shape": list(benchmark.shape),
            "raw_date_min": _date(prices.index.min()),
            "raw_date_max": _date(prices.index.max()),
            "first_tickers": list(prices.columns[:12]),
            "market_state_features": market_state_features,
            "regime_features": regime_features,
            "factor_names": list(bundle.factor_loadings.keys()),
        },
        "architecture": {
            "dimensions": dimensions,
            "stages": [
                {
                    "id": "data",
                    "title": "Datos reales",
                    "label": "Precios, volumen, SPY y FRED",
                    "metric": "100 activos + 9 series macro",
                    "source": "dataset/raw/*",
                    "description": "Entrada usada por el experimento final: precios ajustados y volumen del universo, benchmark SPY y contexto macrofinanciero FRED.",
                },
                {
                    "id": "features",
                    "title": "Features",
                    "label": "Mercado, contexto, factores",
                    "metric": f"{dimensions['market_dim']} market + {dimensions['context_dim']} contexto",
                    "source": "data/preprocessor.py",
                    "description": f"Se construyen {dimensions['market_dim']} features de mercado normalizadas, {context_base_count} features de contexto base, {len(regime_features)} features de regimen y 3 exposiciones factoriales.",
                },
                {
                    "id": "observation",
                    "title": "Estado RL",
                    "label": "Vector observado por el agente",
                    "metric": f"{dimensions['observation_dim']} dimensiones",
                    "source": "environment/portfolio_env.py",
                    "description": "La observacion concatena features de mercado, pesos actuales, exposiciones factoriales y contexto. Esa es la fotografia que recibe la politica en cada paso.",
                },
                {
                    "id": "extractor",
                    "title": "FullGatedExtractor",
                    "label": "Market encoder + context encoder + gates",
                    "metric": "embedding 64",
                    "source": "models/policy.py",
                    "description": "La rama de mercado y la rama de contexto generan embeddings de 64 dimensiones. Las compuertas de regimen y asset gate regulan cuanta informacion contextual entra en la representacion final.",
                },
                {
                    "id": "policy",
                    "title": "Politica PPO",
                    "label": "Actor-critic",
                    "metric": "accion de 100 componentes",
                    "source": "training/trainer.py",
                    "description": "PPO aprende una politica actor-critic con capas [256, 128]. En la configuracion final la accion representa deltas de cartera, no pesos absolutos.",
                },
                {
                    "id": "safety",
                    "title": "Turnover shield + safety",
                    "label": "Presupuesto adaptativo y proyeccion factible",
                    "metric": "0 <= w_i <= 10%, suma=1",
                    "source": "environment/safety.py",
                    "description": "Antes de ejecutar, el sistema limita el turnover con tau adaptativo y proyecta la cartera al conjunto factible: long-only, fully invested y maximo 10% por activo.",
                },
                {
                    "id": "reward",
                    "title": "Entorno y recompensa",
                    "label": "Retorno neto, costes, downside y drawdown",
                    "metric": config["environment"]["reward_type"],
                    "source": "environment/portfolio_env.py",
                    "description": "El entorno aplica retornos del dia siguiente, costes de transaccion, slippage y penalizaciones por turnover, downside y drawdown incremental.",
                },
                {
                    "id": "evaluation",
                    "title": "Evaluacion walk-forward",
                    "label": "13 folds fuera de muestra",
                    "metric": "train 4y · val 1y · test 1y",
                    "source": "results/final_v2/*",
                    "description": "La comparativa final se resume en folds walk-forward y baselines clasicos. La demo lee esos resultados ya curados.",
                },
            ],
        },
        "folds": [_record(row, fold_columns) for _, row in fold_results.iterrows()],
        "fold_windows": folds_payload,
        "series_by_fold": series_by_fold,
        "global_metrics": [_record(row, global_columns) for _, row in global_metrics.iterrows()],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "window.DEMO_DATA = "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
