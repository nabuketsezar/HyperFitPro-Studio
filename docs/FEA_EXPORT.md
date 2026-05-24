# HyperFitPro FEA Export Package

This document describes the FEA export layer.

## Internal unit convention

HyperFitPro test-data processing converts stress-like fitted values to MPa. The
exporter explicitly scales stress-like model parameters to the selected export
stress unit. The default is:

- unit basis: `MPa-mm-N`
- stress-like coefficient unit: `MPa`

The generated `unit_consistency_report.md` records all scaled and unscaled
parameter values.

## Exported files

Each completed run now writes the following files in `exports/`:

- `material_fit.json` – complete model metadata and fitted coefficients.
- `abaqus_material.inp` – Abaqus material include.
- `calculix_material.inp` – CalculiX/Abaqus-style material include.
- `ansys_material.mac` – ANSYS APDL material macro.
- `lsdyna_material.k` – LS-DYNA keyword coefficient include.
- `marc_material.txt` – Marc/Mentat coefficient listing.
- `unit_consistency_report.md` – unit and scaling audit.
- `single_element_verification_targets.csv` – HyperFitPro one-element target responses.
- `abaqus_single_element_probe_template.inp` – single-element probe template.
- `ansys_single_element_probe_template.mac` – single-element probe template.
- `fea_export_manifest.json` – export manifest with status and notes.

## Direct built-in mappings

Direct solver keyword generation is implemented for the common hyperelastic
families that have broadly available built-in forms:

- Neo-Hookean
- 2-term Mooney-Rivlin
- Polynomial / reduced polynomial variants
- Yeoh reduced polynomial
- Ogden
- Arruda-Boyce
- Gent
- Van der Waals

Other source-table models are exported as coefficient/user-material packages.
This avoids pretending that a solver built-in card exists when the source-table
model is a paper-specific or non-standard strain-energy function.

## Volumetric term handling

If a fitted bulk modulus `K` exists, the Abaqus/CalculiX-style D coefficient is
written using the small-strain relation:

```text
D1 = 2 / K
```

If `K` does not exist, D terms are written as `0.0` where supported to indicate
the incompressible form. The manifest and unit report explicitly record this.

## Verification workflow

Before production FEA use:

1. Import the generated solver material card.
2. Build a single-element cube model.
3. Prescribe the deformation modes listed in `single_element_verification_targets.csv`.
4. Compare solver nominal/shear stress against the target values.
5. Only then use the material in a full model.

The Abaqus and ANSYS probe files are templates to accelerate this process.
They still require project-specific solver settings, element formulation checks,
and output extraction setup.
