# Developer Guide

## Repository layout

```text
hyperfitpro/               Main Python package
hyperfitpro/core/          Mechanics, optimization, export, reporting and project logic
hyperfitpro/models/        Built-in hyperelastic model definitions
hyperfitpro/plugins/       Embedded model and tool plugins
hyperfitpro/verification/  Diagnostic scripts used by GUI and CI
sample_data/               Non-confidential sample datasets
docs/                      User guide, theory notes and release documentation
scripts/                   Developer automation scripts
tests/                     Lightweight pytest checks
```

## Launching the GUI

```bash
python run_hyperfit_pro.py
```

or:

```bash
python -m hyperfitpro
```

## Running checks

Fast checks:

```bash
python scripts/dev_check.py --fast
```

Full checks:

```bash
python scripts/dev_check.py --full
```

## Adding a model

1. Add a model file under `hyperfitpro/models/` or as a plugin under `hyperfitpro/plugins/embedded_models/`.
2. Define parameter specs with min, max and initial values.
3. Add equation LaTeX and model metadata.
4. Verify the registry:

```bash
python -m hyperfitpro.verification.verify_model_library
```

## Adding a tool plugin

1. Add a `pXXX_plugin_name.py` file under `hyperfitpro/plugins/embedded_tools/`.
2. Provide plugin metadata, category and run function.
3. Run:

```bash
python -m hyperfitpro.verification.verify_50_plugins
```

## Generated outputs

Run outputs, reports, exports, plots and plugin outputs should be generated in user-selected working directories, not committed to source control.
