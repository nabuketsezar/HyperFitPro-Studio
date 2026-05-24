# HyperFitPro Studio — Optimization System Upgrade

This release addresses the optimization weaknesses identified after .

## Implemented items

1. **Smarter adaptive bounds**
   - `recommended_adaptive_bounds` now shrinks bounds based on parameter count, model complexity, data points per parameter and whether the current candidate is close to a bound.
   - The adaptive bound history is saved in `logs/optimization_diagnostics.json`.

2. **Regularization**
   - Supported regularization types: `none`, `l2`, `l1`, `elastic_net`, `magnitude`.
   - Optional near-bound penalty helps flag solutions that are artificially limited by search bounds.

3. **Overfitting checks**
   - Information criteria are computed: AIC, AICc, BIC.
   - Adjusted R² is computed.
   - Optional train/validation split is supported.
   - Warnings are generated for high parameter-to-data ratio, single-mode high-order fits, poor validation behavior and weakly identified parameters.

4. **Parameter confidence and correlation**
   - Finite-difference Jacobian is computed around the optimum.
   - Approximate covariance, standard errors, 95% confidence intervals and parameter correlation matrix are saved.

5. **Parallel optimization**
   - Threaded parallel multi-start least-squares is available through `--workers` or the GUI worker count field.
   - Differential evolution can also use a thread-map backend where supported.

6. **Stop / pause / resume support**
   - Long runs can be paused, resumed or stopped safely from the GUI.
   - The optimizer stops at the next residual evaluation rather than killing the Python thread.

7. **Checkpoint / resume**
   - Checkpoints are saved to `logs/optimizer_checkpoint.json` during runs.
   - CLI supports `--resume-from` and `--checkpoint`.

8. **Multi-model comparison**
   - `core/model_comparison.py` can fit several models and rank them by AICc, validation RMSE and RMSE.
   - CLI option: `--compare-models 6,7,8,11`.

## Main output files

- `logs/optimization_trace.csv`
- `logs/optimization_diagnostics.json`
- `logs/parameter_confidence_intervals.csv`
- `logs/parameter_correlation_matrix.csv`
- `logs/optimizer_checkpoint.json`
- `logs/model_comparison.csv` when model comparison is used

## Example CLI

```bat
python -m hyperfitpro.cli --model 8 --method recommended_adaptive_bounds ^
  --data uniaxial sample_data/neo_hookean_uniaxial.csv 1 ^
  --data biaxial sample_data/neo_hookean_biaxial.csv 1 ^
  --regularization 1e-4 --regularization-type elastic_net ^
  --validation-fraction 0.15 --workers 4
```

Model comparison:

```bat
python -m hyperfitpro.cli --method least_squares_trf --max-evals 1200 ^
  --compare-models 6,7,8,11 ^
  --data uniaxial sample_data/neo_hookean_uniaxial.csv 1 ^
  --data biaxial sample_data/neo_hookean_biaxial.csv 1
```
