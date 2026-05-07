# Reproducibility

## Configuracion

- `config/config.yaml`: configuracion activa del proyecto

## Requisitos

- Python con dependencias de `requirements.txt`
- `pyarrow` para lectura/escritura parquet
- `FRED_API_KEY` para modo experimento

## Datos raw esperados localmente

En `runtime/data/raw/` deben existir:

- `prices_raw.parquet`
- `volumes_raw.parquet`
- `benchmark_raw.parquet`
- `market_bundle_metadata.yaml`
- `fred_context.parquet`
- `fred_context_metadata.yaml`

## Regenerar datos

```bash
python experiments/run_experiment.py --mode baselines_only --force-refresh-data --max-folds 1
```

Ese comando regenera localmente el bundle de mercado y el contexto FRED.

## Ejecucion principal

```bash
python experiments/run_experiment.py --mode full
```

## Salidas

- resultados finales incluidos en el repo: `results/final_v2/`
- datos raw, logs, manifests, checkpoints y resúmenes intermedios se generan en `runtime/` y no se versionan
