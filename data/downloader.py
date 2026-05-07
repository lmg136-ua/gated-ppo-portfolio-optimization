"""
downloader.py
-------------
Descarga de mercado para modelo:
  - precios ajustados
  - volumen
  - benchmark (SPY por defecto)


"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
import yaml

from data.universe import get_tickers

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_if_exists(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def _metadata_path(output_dir: str) -> Path:
    return Path(output_dir) / "market_bundle_metadata.yaml"


def _save_metadata(output_dir: str, metadata: dict[str, Any]) -> None:
    with open(_metadata_path(output_dir), "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)


def _load_metadata(output_dir: str) -> dict[str, Any]:
    path = _metadata_path(output_dir)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _download_batch(batch: list[str], start: str, end: str, frequency: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = yf.download(
        batch,
        start=start,
        end=end,
        interval=frequency,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if isinstance(data.columns, pd.MultiIndex):
        close_data = data["Close"]
        volume_data = data["Volume"] if "Volume" in data.columns.get_level_values(0) else None
    else:
        close_data = data[["Close"]].rename(columns={"Close": batch[0]})
        volume_data = data[["Volume"]].rename(columns={"Volume": batch[0]}) if "Volume" in data.columns else None
    if volume_data is None:
        volume_data = pd.DataFrame(index=close_data.index, columns=close_data.columns, dtype=float)
    return close_data, volume_data


def _needs_refresh(
    cached_prices: pd.DataFrame | None,
    cached_volumes: pd.DataFrame | None,
    cached_benchmark: pd.DataFrame | None,
    require_volume: bool,
    require_benchmark: bool,
    force_refresh: bool,
) -> bool:
    if force_refresh or cached_prices is None:
        return True
    if require_volume and (cached_volumes is None or cached_volumes.empty):
        return True
    if require_benchmark and (cached_benchmark is None or cached_benchmark.empty):
        return True
    return False


def download_market_bundle(
    tickers: list[str],
    start: str,
    end: str,
    output_dir: str,
    frequency: str = "1d",
    benchmark_ticker: str = "SPY",
    require_real_volume: bool = False,
    require_real_benchmark: bool = False,
    force_refresh: bool = False,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    output_dir_p = Path(output_dir)

    prices_path = output_dir_p / "prices_raw.parquet"
    volumes_path = output_dir_p / "volumes_raw.parquet"
    benchmark_path = output_dir_p / "benchmark_raw.parquet"

    cached_prices = _load_if_exists(prices_path)
    cached_volumes = _load_if_exists(volumes_path)
    cached_benchmark = _load_if_exists(benchmark_path)
    metadata = _load_metadata(output_dir)

    if not _needs_refresh(
        cached_prices=cached_prices,
        cached_volumes=cached_volumes,
        cached_benchmark=cached_benchmark,
        require_volume=require_real_volume,
        require_benchmark=require_real_benchmark or bool(benchmark_ticker),
        force_refresh=force_refresh,
    ):
        logger.info("Bundle de mercado cargado desde %s", output_dir_p)
        return {
            "prices": cached_prices,
            "volumes": cached_volumes,
            "benchmark": cached_benchmark,
            "metadata": metadata
            or {
                "prices_path": str(prices_path),
                "volumes_path": str(volumes_path) if cached_volumes is not None else None,
                "benchmark_path": str(benchmark_path) if cached_benchmark is not None else None,
                "has_volume": cached_volumes is not None,
                "has_benchmark": cached_benchmark is not None,
                "source": "cache",
            },
        }

    logger.info("Descargando bundle de mercado para %d activos desde %s hasta %s", len(tickers), start, end)
    batch_size = 20
    all_prices = []
    all_volumes = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        logger.info("  Descargando lote %d: %s...", i // batch_size + 1, batch[:3])
        try:
            close_data, volume_data = _download_batch(batch, start, end, frequency)
            all_prices.append(close_data)
            all_volumes.append(volume_data)
        except Exception as exc:
            logger.warning("Error descargando lote %s: %s", batch, exc)

    if not all_prices:
        raise RuntimeError("No se pudo descargar ningun precio del universo.")

    prices = pd.concat(all_prices, axis=1)
    volumes = pd.concat(all_volumes, axis=1) if all_volumes else None
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"
    if volumes is not None:
        volumes.index = pd.to_datetime(volumes.index)
        volumes.index.name = "date"

    available = [t for t in tickers if t in prices.columns]
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        logger.warning("Tickers no disponibles (%d): %s", len(missing), missing)
    prices = prices[available].ffill(limit=5).bfill(limit=5)
    threshold = 0.8 * len(available)
    prices = prices.dropna(thresh=int(threshold))

    if volumes is not None:
        volumes = volumes.reindex(columns=available).reindex(index=prices.index).ffill(limit=5).bfill(limit=5)
        if volumes.empty:
            volumes = None

    benchmark = None
    if benchmark_ticker:
        try:
            bench_raw = yf.download(
                benchmark_ticker,
                start=start,
                end=end,
                interval=frequency,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            benchmark = bench_raw[["Close"]].rename(columns={"Close": benchmark_ticker})
            benchmark.index = pd.to_datetime(benchmark.index)
            benchmark.index.name = "date"
            benchmark = benchmark.reindex(prices.index).ffill().bfill()
        except Exception as exc:
            logger.warning("No se pudo descargar benchmark %s: %s", benchmark_ticker, exc)

    if require_real_volume and (volumes is None or volumes.empty):
        raise RuntimeError("La V2 requiere volumen real y no se ha podido descargar volumes_raw.parquet.")
    if require_real_benchmark and (benchmark is None or benchmark.empty):
        raise RuntimeError(f"La V2 requiere benchmark real ({benchmark_ticker}) y no se ha podido descargar.")

    prices.to_parquet(prices_path)
    if volumes is not None and not volumes.empty:
        volumes.to_parquet(volumes_path)
    if benchmark is not None and not benchmark.empty:
        benchmark.to_parquet(benchmark_path)

    metadata = {
        "source": "yfinance",
        "downloaded_at": datetime.utcnow().isoformat() + "Z",
        "prices_path": str(prices_path),
        "volumes_path": str(volumes_path) if volumes is not None else None,
        "benchmark_path": str(benchmark_path) if benchmark is not None else None,
        "has_volume": volumes is not None and not volumes.empty,
        "has_benchmark": benchmark is not None and not benchmark.empty,
        "benchmark_ticker": benchmark_ticker,
        "start": start,
        "end": end,
        "frequency": frequency,
    }
    _save_metadata(output_dir, metadata)

    logger.info(
        "Precios descargados: %s | benchmark=%s | volumen=%s",
        prices.shape,
        benchmark is not None and not benchmark.empty,
        volumes is not None and not volumes.empty,
    )
    return {
        "prices": prices,
        "volumes": volumes,
        "benchmark": benchmark,
        "metadata": metadata,
    }


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    output_dir: str,
    frequency: str = "1d",
) -> pd.DataFrame:
    bundle = download_market_bundle(
        tickers=tickers,
        start=start,
        end=end,
        output_dir=output_dir,
        frequency=frequency,
    )
    return bundle["prices"]


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


def get_data_summary(prices: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> dict:
    returns = compute_returns(prices)
    summary = {
        "n_assets": len(prices.columns),
        "n_days": len(prices),
        "start_date": str(prices.index[0].date()),
        "end_date": str(prices.index[-1].date()),
        "missing_pct": prices.isnull().mean().mean(),
        "mean_annual_return": returns.mean().mean() * 252,
        "mean_annual_vol": returns.std().mean() * np.sqrt(252),
    }
    if benchmark is not None and not benchmark.empty:
        bench_returns = compute_returns(benchmark.squeeze().to_frame()).iloc[:, 0]
        summary["benchmark_annual_return"] = bench_returns.mean() * 252
        summary["benchmark_annual_vol"] = bench_returns.std() * np.sqrt(252)
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    config = load_config()
    tickers = get_tickers()
    exp_cfg = config.get("experiment", {})
    bundle = download_market_bundle(
        tickers=tickers,
        start=config["dates"]["start"],
        end=config["dates"]["end"],
        output_dir=config["paths"]["data_raw"],
        frequency=config["dates"]["frequency"],
        benchmark_ticker=config.get("baselines", {}).get("benchmark_ticker", "SPY"),
        require_real_volume=bool(exp_cfg.get("require_real_volume", False) and exp_cfg.get("mode", "experiment") == "experiment"),
        require_real_benchmark=bool(exp_cfg.get("require_real_benchmark", False) and exp_cfg.get("mode", "experiment") == "experiment"),
    )

    summary = get_data_summary(bundle["prices"], bundle.get("benchmark"))
    logger.info("=== RESUMEN DATOS ===")
    for key, value in summary.items():
        logger.info("  %s: %s", key, value)


if __name__ == "__main__":
    main()
