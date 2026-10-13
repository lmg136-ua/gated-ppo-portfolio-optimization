# Final V2 Curated Results

This directory contains the finalized, publication-grade results for the definitive model: **`PPO_gated_full`** (`full_gated_v2`).

---

## Files Included

* `global_metrics_all_strategies.csv`: Global aggregated performance metrics (mean and standard deviation across all 13 walk-forward folds) comparing `PPO_gated_full` against all classical benchmarks in `version_2`.
* `window_metrics_all_strategies.csv`: Window-by-window (fold-by-fold) performance metrics for every evaluated strategy across all 13 walk-forward testing periods.
* `window_metrics_ppo_gated_full.csv`: Window metrics specifically for the `PPO_gated_full` model.
* `portfolio_series_all_strategies.csv`: Step-by-step daily portfolio values, net returns, turnover, and incremental drawdown time series across all folds.
* `ppo_gated_full_summary.csv`: Comprehensive diagnostic summary for `PPO_gated_full`, including factor loadings, gating weights, stress probabilities, turnover bounds, and execution statistics.
* `ppo_gated_full_fold_results.csv`: Complete fold-level diagnostic records (57 columns) for the final model.

---

## Evaluation Standard

* **Universe**: 100 S&P Equities (2010–2024).
* **Out-of-Sample Folds**: 13 Walk-Forward windows with 4-year training, 1-year validation, 1-year test, and 6-month sliding step.
* **Execution Frictions**: 10 bps transaction fee + 5 bps slippage per trade.
* **Safety**: CVXPY/OSQP convex quadratic programming layer ensuring 100% operational validity ($w_i \ge 0$, $\sum w_i = 1$, $w_i \le 10\%$).
