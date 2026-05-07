"""Punto de entrada principal para ejecutar baselines y modelo final."""

from __future__ import annotations

import argparse
import copy
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)
    config["_meta"] = {"config_path": path}
    return config


def _clear_raw_cache(config: dict) -> None:
    raw_dir = Path(config["paths"]["data_raw"])
    for filename in [
        "prices_raw.parquet",
        "volumes_raw.parquet",
        "benchmark_raw.parquet",
        "market_bundle_metadata.yaml",
        "fred_context.parquet",
        "fred_context_metadata.yaml",
    ]:
        path = raw_dir / filename
        if path.exists():
            path.unlink()
            logger.info("Eliminado cache raw: %s", path)


def load_data(config: dict, force_refresh: bool = False) -> tuple[dict, pd.DataFrame, dict]:
    from data.downloader import download_market_bundle
    from data.fred_downloader import download_fred_series
    from data.universe import get_tickers_from_config

    tickers = get_tickers_from_config(config)
    experiment_cfg = config.get("experiment", {})

    if force_refresh:
        _clear_raw_cache(config)

    market_bundle = download_market_bundle(
        tickers=tickers,
        start=config["dates"]["start"],
        end=config["dates"]["end"],
        output_dir=config["paths"]["data_raw"],
        frequency=config["dates"]["frequency"],
        benchmark_ticker=config.get("baselines", {}).get("benchmark_ticker", "SPY"),
        require_real_volume=bool(experiment_cfg.get("require_real_volume", False)),
        require_real_benchmark=bool(experiment_cfg.get("require_real_benchmark", False)),
        force_refresh=force_refresh,
    )
    context, context_meta = download_fred_series(
        series_ids=config["context"]["fred_series"],
        start=config["dates"]["start"],
        end=config["dates"]["end"],
        output_dir=config["paths"]["data_raw"],
        lag_days=config["context"]["lag_days"],
        strict_real=bool(experiment_cfg.get("strict_real_context", False)),
        return_metadata=True,
    )

    run_metadata = {
        "context_source": context_meta.get("source", "unknown"),
        "context_metadata": context_meta,
        "market_metadata": market_bundle.get("metadata", {}),
        "config_path": config.get("_meta", {}).get("config_path", "config/config.yaml"),
        "experiment_version": config.get("experiment", {}).get("version", "full_gated_v2"),
        "expected_raw_files": [
            "prices_raw.parquet",
            "volumes_raw.parquet",
            "benchmark_raw.parquet",
            "market_bundle_metadata.yaml",
            "fred_context.parquet",
            "fred_context_metadata.yaml",
        ],
    }
    return market_bundle, context, run_metadata


def _save_final_summary(df: pd.DataFrame, config: dict, subdir: str) -> str:
    out_dir = Path(config["paths"]["summaries"]) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "final_summary.csv"
    df.to_csv(out_path)
    logger.info("Resumen final guardado en %s", out_path)
    return str(out_path)


def _safe_git_commit(repo_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def _save_run_manifest(config: dict, run_metadata: dict, cli_args: dict) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifests_dir = Path(config["paths"]["manifests"])
    manifests_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = Path(__file__).resolve().parent.parent
    config_snapshot = copy.deepcopy({key: value for key, value in config.items() if key != "_meta"})
    manifest = {
        "model_line": "PPO_gated_full",
        "experiment_version": config.get("experiment", {}).get("version", "full_gated_v2"),
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "config_path": config.get("_meta", {}).get("config_path", "config/config.yaml"),
        "git_commit": _safe_git_commit(repo_dir),
        "cli_args": cli_args,
        "training": {
            "seed": config.get("training", {}).get("seed"),
            "seeds": config.get("training", {}).get("seeds", []),
            "benchmark_seeds": config.get("training", {}).get("benchmark_seeds", []),
            "use_benchmark_seeds": config.get("training", {}).get("use_benchmark_seeds", False),
        },
        "paths": config.get("paths", {}),
        "expected_raw_files": run_metadata.get("expected_raw_files", []),
        "run_metadata": run_metadata,
    }

    manifest_path = manifests_dir / f"run_manifest_{timestamp}.yaml"
    latest_manifest = manifests_dir / "run_manifest_latest.yaml"
    config_snapshot_path = manifests_dir / f"config_snapshot_{timestamp}.yaml"
    latest_config = manifests_dir / "config_snapshot_latest.yaml"

    for path, payload in [
        (manifest_path, manifest),
        (latest_manifest, manifest),
        (config_snapshot_path, config_snapshot),
        (latest_config, config_snapshot),
    ]:
        with open(path, "w", encoding="utf-8") as file_obj:
            yaml.safe_dump(payload, file_obj, sort_keys=False)

    logger.info("Run manifest guardado en %s", manifest_path)
    return str(manifest_path)


def run_baselines(config: dict, market_bundle: dict, context_raw: pd.DataFrame, max_folds: int | None = None) -> pd.DataFrame:
    from evaluation.baselines import run_walk_forward_baselines
    from training.walk_forward import generate_walk_forward_folds

    logger.info("=== BASELINES WALK-FORWARD ===")
    prices = market_bundle["prices"]
    benchmark = market_bundle.get("benchmark")
    common_idx = prices.index.intersection(context_raw.index)
    prices = prices.loc[common_idx].dropna(axis=1, how="all").ffill().dropna()
    returns = prices.pct_change().dropna()
    folds = generate_walk_forward_folds(
        dates=returns.index,
        train_years=config["walk_forward"]["train_years"],
        val_years=config["walk_forward"]["val_years"],
        test_years=config["walk_forward"]["test_years"],
        slide_months=config["walk_forward"]["slide_months"],
    )
    if max_folds is not None:
        folds = folds[: int(max_folds)]

    summary_df, fold_df = run_walk_forward_baselines(returns, folds, config, benchmark_prices=benchmark)
    _save_final_summary(summary_df, config, "baselines")
    out_dir = Path(config["paths"]["summaries"]) / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(out_dir / "fold_results.csv", index=False)
    return summary_df


def run_final_model(
    config: dict,
    market_bundle: dict,
    context_raw: pd.DataFrame,
    run_metadata: dict,
    max_folds: int | None = None,
    fold_ids: list[int] | None = None,
) -> pd.DataFrame:
    from evaluation.metrics import summarize_walk_forward
    from training.walk_forward import WalkForwardRunner

    logger.info("=== MODELO FINAL PPO_GATED_FULL ===")
    runner = WalkForwardRunner(config, output_dir=str(Path(config["paths"]["models"]) / "walk_forward"))
    fold_metrics = runner.run(
        prices=market_bundle["prices"],
        context_raw=context_raw,
        model_name="PPO_gated_full",
        volumes=market_bundle.get("volumes"),
        benchmark_prices=market_bundle.get("benchmark"),
        run_metadata=run_metadata,
        max_folds=max_folds,
        include_fold_ids=fold_ids,
    )

    summary_df = pd.DataFrame([summarize_walk_forward(fold_metrics)]).set_index(pd.Index(["PPO_gated_full"], name="model"))
    _save_final_summary(summary_df, config, "final_model")

    runner_output = Path(runner.output_dir)
    fold_results_path = runner_output / "PPO_gated_full_fold_results.csv"
    sanity_path = runner_output / "PPO_gated_full_sanity_by_fold.csv"
    summaries_dir = Path(config["paths"]["summaries"]) / "final_model"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    if fold_results_path.exists():
        pd.read_csv(fold_results_path).to_csv(summaries_dir / "fold_results.csv", index=False)
    if sanity_path.exists():
        pd.read_csv(sanity_path).to_csv(summaries_dir / "sanity_by_fold.csv", index=False)
    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="full", choices=["baselines_only", "model_only", "full"])
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--fold-ids", type=str, default=None)
    parser.add_argument("--timesteps-override", type=int, default=None)
    parser.add_argument("--force-refresh-data", action="store_true")
    args = parser.parse_args()

    setup_logging(args.log_level)
    config = load_config(args.config)
    config.setdefault("experiment", {})
    config["experiment"]["mode"] = "experiment"
    if args.timesteps_override is not None:
        config["training"]["total_timesteps"] = int(args.timesteps_override)

    fold_ids = [int(part.strip()) for part in args.fold_ids.split(",")] if args.fold_ids else None
    for path_value in config["paths"].values():
        os.makedirs(path_value, exist_ok=True)

    t0 = time.time()
    market_bundle, context, run_metadata = load_data(config, force_refresh=args.force_refresh_data)
    _save_run_manifest(
        config=config,
        run_metadata=run_metadata,
        cli_args={
            "mode": args.mode,
            "config": args.config,
            "log_level": args.log_level,
            "max_folds": args.max_folds,
            "fold_ids": fold_ids,
            "timesteps_override": args.timesteps_override,
            "force_refresh_data": bool(args.force_refresh_data),
        },
    )

    if args.mode == "baselines_only":
        run_baselines(config, market_bundle, context, max_folds=args.max_folds)
    elif args.mode == "model_only":
        run_final_model(
            config,
            market_bundle,
            context,
            run_metadata,
            max_folds=args.max_folds,
            fold_ids=fold_ids,
        )
    else:
        baseline_summary = run_baselines(config, market_bundle, context, max_folds=args.max_folds)
        model_summary = run_final_model(
            config,
            market_bundle,
            context,
            run_metadata,
            max_folds=args.max_folds,
            fold_ids=fold_ids,
        )
        combined = pd.concat([baseline_summary, model_summary], axis=0)
        _save_final_summary(combined, config, "combined")

    logger.info("Completado en %.1f minutos", (time.time() - t0) / 60.0)


if __name__ == "__main__":
    main()
