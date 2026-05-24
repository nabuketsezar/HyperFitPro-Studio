# HyperFitPro Studio Creative Embedded Tool Plugins

adds non-model embedded plugins. These are not hyperelastic material models; they are assistant tools that use the selected model, imported datasets and/or fitted result to generate engineering outputs.

## Embedded tool plugin categories

| Category | Purpose |
|---|---|
| Plotter | Residual diagnostics, parameter sensitivity, extrapolation envelope and dashboard plots |
| Data | Data quality audit and raw/processed preview |
| Advisor | Optimizer recipe and model-complexity radar |
| Report | Equation cards and report summary assets |
| FEA | Single-element verification target curves |
| Diagnostics | Bound health and parameter-correlation network |

## Shipped tool plugins

1. `plot.residual_diagnostics`
2. `plot.parameter_sensitivity`
3. `data.quality_inspector`
4. `plot.extrapolation_envelope`
5. `advisor.optimizer_recipe`
6. `report.asset_builder`
7. `report.latex_equation_card`
8. `fea.single_element_target_plotter`
9. `plot.parameter_correlation_network`
10. `diagnostics.bounds_health_auditor`
11. `plot.dashboard_pack`
12. `advisor.model_complexity_radar`

## GUI usage

Open **7 Plugins** after importing data or finishing a fit. Tick desired tools in the **Run** column and click **Run selected tools**. Output files are written to:

```text
<run_folder>/tools/<plugin_name>/
```

If no fit has been run yet, data-only and advisor tools can still run; fit-dependent tools are skipped with a clear message.

## CLI usage

List tool plugins:

```bash
python -m hyperfitpro.cli --list-tool-plugins
```

Run all compatible plugins after a fit:

```bash
python -m hyperfitpro.cli --model 6 --method least_squares_trf \
  --data uniaxial sample_data/neo_hookean_uniaxial.csv 1 \
  --run-tool-plugins all
```

Run selected plugins:

```bash
python -m hyperfitpro.cli --model 6 --method least_squares_trf \
  --data uniaxial sample_data/neo_hookean_uniaxial.csv 1 \
  --run-tool-plugins plot.residual_diagnostics,report.latex_equation_card
```

## User tool plugin template

```bash
python -m hyperfitpro.cli --export-tool-plugin-template "%USERPROFILE%/.hyperfitpro_studio/plugins/tools/my_tool.py"
```

A user tool plugin exposes `TOOL_CLASS` and inherits from `hyperfitpro.core.tool_plugins.ToolPlugin`.
