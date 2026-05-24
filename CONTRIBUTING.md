# Contributing

## Local development setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## Run the application

```bash
python run_hyperfit_pro.py
```

## Run repository checks

```bash
python scripts/dev_check.py
```

## Branch naming

Use short descriptive branch names:

- `feature/plugin-name`
- `fix/gui-model-panel`
- `docs/help-example-update`
- `validation/neo-hookean-benchmark`

## Pull request checklist

- The GUI still launches from `run_hyperfit_pro.py`.
- `python scripts/dev_check.py` passes.
- New plugin files are discoverable from the Plugin Workbench.
- Any new generated output is ignored by `.gitignore` unless it is intentional sample data.
- Engineering assumptions are documented in `docs/`.
