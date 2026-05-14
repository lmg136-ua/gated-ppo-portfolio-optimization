# Dataset del experimento final

Esta carpeta recoge los datos utilizados por la version final del proyecto, centrada en el modelo `PPO_gated_full` y en la evaluacion `walk-forward` descrita en la memoria.

## Estructura

- `raw/prices_raw.parquet`: precios ajustados diarios de los 100 activos del universo final.
- `raw/volumes_raw.parquet`: volumen diario asociado a esos mismos 100 activos.
- `raw/benchmark_raw.parquet`: serie diaria del benchmark `SPY`.
- `raw/fred_context.parquet`: series macrofinancieras descargadas de FRED.
- `raw/universe_tickers.csv`: listado de los 100 tickers del universo final.
- `raw/market_bundle_metadata.yaml`: metadatos de descarga del bloque de mercado.
- `raw/fred_context_metadata.yaml`: metadatos de descarga del bloque de contexto.

## Cobertura temporal

- Mercado y benchmark: desde `2010-01-04` hasta `2024-12-30`
- Contexto FRED: desde `2010-01-04` hasta `2024-12-31`


### Contexto macrofinanciero

- `DGS10`
- `DGS2`
- `T10Y2Y`
- `UNRATE`
- `CPIAUCSL`
- `FEDFUNDS`
- `VIXCLS`
- `DCOILWTICO`
- `DEXUSEU`

## Fuentes 

- Yahoo Finance, descargado mediante `yfinance`
- FRED (Federal Reserve Economic Data), descargado mediante `pandas_datareader`

