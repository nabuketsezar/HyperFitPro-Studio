# HyperFitPro Studio — Model Library Completion

This release closes the **Model Library** gap from the issue list.

## What changed

1. **Model 32 was implemented and activated** as the Van Der Waals model supplied by the user image.
2. All 42 table rows are now active models in the registry.
3. Every model carries extended metadata:
   - category
   - family
   - recommended tests
   - material class tags
   - application tags
   - calibration notes
   - parameterization notes
   - limitation notes
   - preferred optimizer
   - complexity level
4. A model catalog exporter was added:
   - `docs/model_catalog_.json`
   - `docs/model_catalog_.csv`
5. A model-library validation and recommendation module was added:
   - `hyperfitpro/core/model_library.py`
   - `verify_model_library.py`

## Model 32: Van Der Waals

Implemented equation:

```latex
W=\mu\left[-(\lambda_m^2-3)\left(\ln(1-\eta)+\eta\right)-\frac{2a}{3}\left(\frac{I_1-3}{2}\right)^{3/2}\right]
```

with

```latex
\eta=\sqrt{\frac{(1-\beta)I_1+\beta I_2-3}{\lambda_m^2-3}}
```

Parameters:

| Parameter | Type | Bound policy |
|---|---|---|
| `mu` | stress-like positive parameter | scaled by imported stress data |
| `lambda_m` | limiting stretch parameter | domain checked; must keep denominator positive and eta < 1 |
| `a` | dimensionless interaction coefficient | broad bounded search |
| `beta` | invariant mixing coefficient | constrained to `[0, 1]` |

## Bound policy

The bounds in HyperFitPro are **optimization search bounds**, not universal material constants. Stress-like parameters are scaled with the imported data stress magnitude. Limit parameters are additionally checked by the model domain and the stability scan.

## Verification

Run:

```bash
python verify_model_library.py
```

The verification checks no duplicate model numbers, all 42 active rows, Van Der Waals activation, finite reference/moderate-stretch energies, catalog export and recommendation-engine execution.
