# Reproducibility Guide

This guide describes how to reproduce the experiments, regenerate datasets, and run the benchmark evaluations for the **Gated PPO Portfolio Optimization** framework.

---

## 1. Environment & Dependencies

* Python 3.10 or higher
* Core dependencies listed in `requirements.txt`
* `pyarrow` for reading/writing parquet files
* `FRED_API_KEY` environment variable (required only if downloading new macroeconomic data from FRED)

```bash
# Setup virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Active Configuration

The canonical experiment parameters are defined in:
* `config/config.yaml`: Controls the universe (100 assets), walk-forward windows (4y train, 1y val, 1y test, 6m slide), feature dimensions, neural network architecture, reward functions, safety layer parameters, and execution frictions.

---

## 3. Data Structure

Pre-curated datasets used in the final evaluation are provided in `dataset/raw/`:
* `prices_raw.parquet`: Daily adjusted closing prices for 100 S&P equities.
* `volumes_raw.parquet`: Daily volume series for all 100 universe equities.
* `benchmark_raw.parquet`: Daily adjusted price and return series for S&P 500 ETF (`SPY`).
* `fred_context.parquet`: Macroeconomic time series fetched from FRED (`DGS10`, `DGS2`, `T10Y2Y`, `UNRATE`, `CPIAUCSL`, `FEDFUNDS`, `VIXCLS`, `DCOILWTICO`, `DEXUSEU`).
* Associated YAML metadata files.

### Regenerating Raw Data Locally

To rebuild the market bundle and FRED context from scratch:

```bash
python experiments/run_experiment.py --mode baselines_only --force-refresh-data --max-folds 1
```

---

## 4. Running Experiments

### A. Classical Baselines Only
Runs Equal Weight, Inverse Volatility, Markowitz Mean-Variance, and Minimum Variance against the S&P 500 benchmark:

```bash
python experiments/run_experiment.py --mode baselines_only
```

### B. Full Walk-Forward Experiment (`PPO_gated_full`)
Runs the complete 13-fold out-of-sample walk-forward evaluation:

```bash
python experiments/run_experiment.py --mode full
```

---

## 5. Experiment Outputs & Artifacts

* **Curated Final Results**: Located in `results/final_v2/` (version-controlled and documented in the repository).
* **Runtime Data & Checkpoints**: Intermediate training logs, TensorBoard events, checkpoints, and execution manifests are written to `runtime/` (git-ignored to keep the repository lightweight).
