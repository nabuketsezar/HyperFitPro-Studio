# Plugin Authoring Guide

HyperFitPro supports two plugin types:

1. Material model plugins
2. Tool plugins

## Material model plugins

Material model plugins define an energy function, parameters, model metadata and calibration notes. They are loaded together with the built-in model registry.

Recommended metadata:

- model number
- model name
- category and family
- equation LaTeX
- parameter specifications
- recommended tests
- known limitations
- preferred optimizer

## Tool plugins

Tool plugins run after a calibration result exists. They can create plots, reports, diagnostics, FEA checks, uncertainty estimates or package outputs.

Recommended outputs:

- `*.md` or `*.html` summary
- `*.csv` machine-readable tables
- `*.png` visual diagnostics
- optional `*.xlsx` calculator or matrix

## Safety rules

- Do not assume private customer paths.
- Do not write outside the active run or selected project folder.
- Do not silently overwrite user files.
- Put expensive calculations behind clear user confirmation or a plugin setting.
