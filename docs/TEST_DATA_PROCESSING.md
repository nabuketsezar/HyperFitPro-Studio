# HyperFitPro Studio Test Data Processing

This release closes the test-data processing gap that existed in the earlier prototype builds.
The optimizer no longer assumes that every imported file is already clean two-column nominal stress-strain data.

## Supported import families

| Import kind | Typical columns | Required user inputs | Internal output |
|---|---|---|---|
| `stress_strain` | `strain`, `stress` | stress unit, optional percent flag | nominal strain and nominal stress in MPa |
| `true_stress_strain` | `true_strain`, `true_stress` or stretch + Cauchy stress | stress unit | engineering strain and nominal stress |
| `force_displacement` | `force`, `displacement` | force unit, length unit, gauge length, area or width+thickness or diameter | engineering strain and nominal stress |
| `stretch_stress` | `lambda`, `stress` | stress unit | `lambda-1` and stress |
| `volumetric` | volumetric strain / `J-1`, pressure | stress unit | volumetric input and pressure |

## Geometry conversion

Force-displacement data are converted using

```text
engineering strain = displacement / gauge length
nominal stress = force / initial area
```

The initial area can be supplied directly, or calculated from `width * thickness`, or from a round specimen diameter.
All geometry values are interpreted using the selected length unit.

## Cleaning and branch handling

The GUI and CLI now support:

- zero-offset correction,
- loading / unloading / first monotonic branch extraction for cyclic data,
- toe-region removal by fraction or strain threshold,
- moving-average smoothing,
- Savitzky-Golay smoothing,
- z-score or MAD outlier rejection,
- downsampling for heavy optimizations,
- low-strain, high-strain, and balanced-bin region weighting.

## Multi-dataset weighting

Large data files can dominate smaller but more informative modes. `auto_balance_dataset_weights` balances each dataset using point count and stress scale while preserving the user's relative weight.
Point-wise region weights are also passed into the optimizer residual vector.

## Run outputs

Each run now saves processed data files in `input_data/processed_XX_<mode>_<file>.csv` containing:

- `x`,
- `stress`,
- `point_weight`.

The run summary also includes preprocessing diagnostics for every imported dataset.
