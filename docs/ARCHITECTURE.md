# HyperFitPro Studio Software Architecture Layer

This release adds the architecture layer that was missing from earlier prototypes.

## Added modules

| Module | Purpose |
|---|---|
| `hyperfitpro.core.config` | Persistent user/application settings. Stores JSON config under `~/.hyperfitpro_studio/settings.json`. |
| `hyperfitpro.core.plugin_manager` | External user model plugin discovery through configured plugin folders or `HYPERFITPRO_PLUGIN_PATH`. |
| `hyperfitpro.core.project` | `.hfp` project file save/load format containing model, datasets, optimizer settings, bounds, report sections and last run metadata. |
| `hyperfitpro.core.run_database` | SQLite run-history database for indexing, listing and reopening past runs. |
| `hyperfitpro.core.undo_redo` | Generic undo/redo snapshot stack used by the GUI and available for future editors. |
| `hyperfitpro.core.architecture` | One-call architecture status report for CLI/debugging. |

## Plugin model workflow

Create a template:

```bat
python -m hyperfitpro.cli --export-plugin-template "%USERPROFILE%\.hyperfitpro_studio\plugins\my_model.py"
```

Then edit the generated file and keep `MODEL_CLASS = MyCustomHyperelasticModel` at the bottom.
The registry will load it automatically on the next application start. Plugin collisions with built-in model numbers are ignored; built-ins win.

## Project workflow

Save a project from CLI after a run:

```bat
python -m hyperfitpro.cli --model 6 --method least_squares_trf ^
  --data uniaxial sample_data/neo_hookean_uniaxial.csv 1 ^
  --save-project my_fit_project.hfp
```

Load the project later:

```bat
python -m hyperfitpro.cli --load-project my_fit_project.hfp
```

The GUI also has toolbar buttons for Save Project and Load Project.

## Run database

Every completed `save_run()` now indexes the run into SQLite. List previous runs:

```bat
python -m hyperfitpro.cli --list-runs
```

Show complete architecture status:

```bat
python -m hyperfitpro.cli --architecture-status
```

## Verification

Run:

```bat
python verify_architecture.py
```

This checks config persistence, plugin discovery, project save/load, undo/redo and run database indexing.
