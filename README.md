# HyperFitPro Studio

HyperFitPro Studio is a desktop engineering workbench for hyperelastic material-model calibration, optimization, reporting, plugin-based diagnostics and FEA export.

## Main capabilities

- 42 built-in hyperelastic material models.
- 13 embedded model plugins.
- 50 embedded tool plugins for diagnostics, plotting, reporting, FEA checks and project packaging.
- Multi-test data import for uniaxial, biaxial, planar, simple shear and volumetric data.
- Force-displacement conversion, unit conversion, preprocessing and dataset weighting.
- Multiple optimization methods with live iteration plots.
- PDF reports, Excel calculator export and FEA export bundle.
- `.hyp2fit` project files and project library.
- Five-language GUI: Turkish, English, German, French and Spanish.
- Overleaf-ready theory manual.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python run_hyperfit_pro.py
```

or:

```bash
python -m hyperfitpro
```

## Theory manual

The Overleaf-ready LaTeX source is available at:

```text
docs/theory/HyperFitPro_Hyperelastic_Theory_Manual.tex
```

## Help

Offline user guide files are included in:

```text
docs/help/HyperFitPro_User_Guide.pdf
docs/help/HyperFitPro_User_Guide.html
```

## License

See `LICENSE`.
