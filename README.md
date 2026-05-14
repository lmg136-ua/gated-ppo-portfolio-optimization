# TFG RL Portfolio

Repositorio del TFG centrado en el modelo final `PPO_gated_full` para optimizacion de carteras con aprendizaje por refuerzo, contexto exogeno y evaluacion `walk-forward`.

## Que contiene este repo

- codigo fuente del pipeline completo
- configuracion final del experimento
- dataset usado en la evaluacion final
- baselines clasicos usados en la comparativa
- resultados finales ya procesados y listos para consulta

## Estructura

```text
config/        configuracion activa del experimento
data/          descarga, validacion y preprocesado
dataset/       datos utilizados en la evaluacion final
features/      features de mercado, regimen y factores
models/        policy, encoders y gating
environment/   entorno de cartera y restricciones
training/      entrenamiento y walk-forward
evaluation/    metricas y estrategias baseline
experiments/   puntos de entrada para ejecutar el proyecto
results/       resultados finales curados del estudio
docs/          reproducibilidad y limitaciones
```

## Configuracion activa

- `config/config.yaml`


## Ejecucion minima

### 1. Regenerar datos raw localmente

```bash
python experiments/run_experiment.py --mode baselines_only --force-refresh-data --max-folds 1
```

### 2. Ejecutar el experimento completo

```bash
python experiments/run_experiment.py --mode full
```

## Resultados incluidos

En `results/final_v2/` se incluyen:

- resumen global de estrategias
- metricas por ventana temporal
- series temporales de valor de cartera
- resumen final del modelo `PPO_gated_full`
- resultados por fold del modelo final

## Dataset incluido

En `dataset/raw/` se incluyen los datos de entrada usados en el experimento final:

- precios ajustados diarios de los 100 activos del universo
- volumen diario de esos 100 activos
- benchmark `SPY`
- bloque de contexto macrofinanciero procedente de FRED
- metadatos de descarga y listado final de tickers

## Notas

- el proyecto esta orientado a la configuracion final usada en el TFG
- las salidas generadas al ejecutar experimentos se escriben en `runtime/` y no se versionan
- el dataset incluido corresponde al bloque de datos usado en la evaluacion final
- los detalles minimos de reproducibilidad se recogen en `docs/REPRODUCIBILITY.md`
