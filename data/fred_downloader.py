"""
fred_downloader.py
------------------
Descarga datos macro desde FRED y separa claramente:
  - modo experimento: requiere datos reales
  - modo demo: puede usar sinteticos
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _is_real_fred_source(source: str | None) -> bool:
    return str(source or "").startswith("fred_real")


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _metadata_path(output_dir: str) -> Path:
    return Path(output_dir) / "fred_context_metadata.yaml"


def _save_metadata(output_dir: str, metadata: dict[str, Any]) -> None:
    with open(_metadata_path(output_dir), "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)


def _load_metadata(output_dir: str) -> dict[str, Any]:
    path = _metadata_path(output_dir)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def download_fred_series(
    series_ids: list[str],
    start: str,
    end: str,
    output_dir: str,
    api_key: str | None = None,
    lag_days: int = 1,
    strict_real: bool = False,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / "fred_context.parquet"
    metadata = _load_metadata(output_dir)

    if output_path.exists():
        logger.info("Datos FRED ya descargados, cargando desde %s", output_path)
        context = pd.read_parquet(output_path)
        source = metadata.get("source", "unknown")
        if strict_real and not _is_real_fred_source(source):
            raise RuntimeError(
                "El experimento requiere contexto FRED real, pero el cache actual no es real. "
                "Borra fred_context.parquet/fred_context_metadata.yaml y vuelve a descargar con una fuente real de FRED."
            )
        return (context, metadata) if return_metadata else context

    context = None
    metadata = None
    errors: list[str] = []

    try:
        from fredapi import Fred

        if api_key is None:
            api_key = os.environ.get("FRED_API_KEY")
        if api_key is not None:
            fred = Fred(api_key=api_key)
            all_series = {}
            for series_id in series_ids:
                try:
                    logger.info("  Descargando FRED (fredapi): %s...", series_id)
                    s = fred.get_series(series_id, observation_start=start, observation_end=end)
                    all_series[series_id] = s
                except Exception as exc:
                    logger.warning("  Error descargando %s con fredapi: %s", series_id, exc)

            if all_series:
                context = pd.DataFrame(all_series)
                context.index = pd.to_datetime(context.index)
                context.index.name = "date"
                daily_idx = pd.date_range(start=start, end=end, freq="B")
                context = context.reindex(daily_idx).ffill().bfill()
                context = context.shift(lag_days).dropna()
                metadata = {
                    "source": "fred_real_api",
                    "downloaded_at": datetime.utcnow().isoformat() + "Z",
                    "series_ids": list(context.columns),
                    "lag_days": lag_days,
                    "start": start,
                    "end": end,
                }
        else:
            errors.append("FRED_API_KEY no encontrada para fredapi")
    except ImportError:
        errors.append("fredapi no instalado")
    except Exception as exc:
        errors.append(f"fredapi fallo: {exc}")

    if context is None:
        try:
            from pandas_datareader import data as web

            all_series = {}
            for series_id in series_ids:
                try:
                    logger.info("  Descargando FRED (pandas-datareader): %s...", series_id)
                    s = web.DataReader(series_id, "fred", start, end)
                    if s is not None and not s.empty:
                        all_series[series_id] = s.iloc[:, 0]
                except Exception as exc:
                    logger.warning("  Error descargando %s con pandas-datareader: %s", series_id, exc)

            if all_series:
                context = pd.DataFrame(all_series)
                context.index = pd.to_datetime(context.index)
                context.index.name = "date"
                daily_idx = pd.date_range(start=start, end=end, freq="B")
                context = context.reindex(daily_idx).ffill().bfill()
                context = context.shift(lag_days).dropna()
                metadata = {
                    "source": "fred_real_pdr",
                    "downloaded_at": datetime.utcnow().isoformat() + "Z",
                    "series_ids": list(context.columns),
                    "lag_days": lag_days,
                    "start": start,
                    "end": end,
                }
            else:
                errors.append("pandas-datareader no devolvio series FRED")
        except Exception as exc:
            errors.append(f"pandas-datareader fallo: {exc}")

    if context is None or metadata is None:
        if strict_real:
            raise RuntimeError(
                "No se pudo descargar contexto FRED real en modo experimento. "
                f"Intentos realizados: {errors}"
            )
        logger.warning("No se pudo descargar ninguna serie FRED real. Usando sinteticos solo para demo.")
        context, metadata = _generate_synthetic_macro(series_ids, start, end, output_dir, lag_days)
        return (context, metadata) if return_metadata else context

    context.to_parquet(output_path)
    _save_metadata(output_dir, metadata)
    logger.info("Datos macro FRED guardados en %s", output_path)
    return (context, metadata) if return_metadata else context


def _generate_synthetic_macro(
    series_ids: list[str],
    start: str,
    end: str,
    output_dir: str,
    lag_days: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / "fred_context.parquet"
    daily_idx = pd.date_range(start=start, end=end, freq="B")
    n = len(daily_idx)
    np.random.seed(42)

    def ar1_process(n_points: int, mean: float, std: float, phi: float = 0.99) -> np.ndarray:
        x = np.zeros(n_points)
        x[0] = mean
        for i in range(1, n_points):
            x[i] = phi * x[i - 1] + (1 - phi) * mean + std * np.random.randn()
        return x

    synthetic = pd.DataFrame(index=daily_idx)
    synthetic["DGS10"] = ar1_process(n, mean=3.5, std=0.02)
    synthetic["DGS2"] = ar1_process(n, mean=2.8, std=0.02)
    synthetic["T10Y2Y"] = synthetic["DGS10"] - synthetic["DGS2"]
    synthetic["UNRATE"] = ar1_process(n, mean=5.0, std=0.05)
    synthetic["CPIAUCSL"] = ar1_process(n, mean=100.0, std=0.3)
    synthetic["FEDFUNDS"] = ar1_process(n, mean=2.5, std=0.01)
    synthetic["VIXCLS"] = np.abs(ar1_process(n, mean=20.0, std=1.0))
    synthetic["DCOILWTICO"] = np.abs(ar1_process(n, mean=70.0, std=1.0))
    synthetic["DEXUSEU"] = ar1_process(n, mean=1.10, std=0.005)
    synthetic = synthetic[[s for s in series_ids if s in synthetic.columns]]
    synthetic.index.name = "date"
    synthetic = synthetic.shift(lag_days).dropna()

    metadata = {
        "source": "synthetic_demo",
        "downloaded_at": datetime.utcnow().isoformat() + "Z",
        "series_ids": list(synthetic.columns),
        "lag_days": lag_days,
        "start": start,
        "end": end,
    }
    synthetic.to_parquet(output_path)
    _save_metadata(output_dir, metadata)
    logger.info("Datos macro sinteticos guardados en %s", output_path)
    return synthetic, metadata


def normalize_context(
    context: pd.DataFrame,
    train_mask: pd.Series,
) -> pd.DataFrame:
    train_mean = context[train_mask].mean()
    train_std = context[train_mask].std().replace(0, 1)
    normalized = (context - train_mean) / train_std
    return normalized


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    config = load_config()
    strict_real = bool(config.get("experiment", {}).get("strict_real_context", False))
    context, metadata = download_fred_series(
        series_ids=config["context"]["fred_series"],
        start=config["dates"]["start"],
        end=config["dates"]["end"],
        output_dir=config["paths"]["data_raw"],
        lag_days=config["context"]["lag_days"],
        strict_real=strict_real,
        return_metadata=True,
    )

    logger.info("Contexto macro descargado: %s | fuente=%s", context.shape, metadata.get("source"))


if __name__ == "__main__":
    main()
