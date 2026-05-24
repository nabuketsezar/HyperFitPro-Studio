# Validation Notes

HyperFitPro includes diagnostic scripts and smoke tests, but any commercial engineering use still requires project-specific validation.

## Included verification coverage

- Built-in model registry loading
- Embedded model plugin loading
- Tool plugin discovery
- Neo-Hookean response-kernel verification
- Data preprocessing verification
- Optimization diagnostics verification
- FEA export bundle generation
- Project format and architecture checks

## Required project validation before engineering release

- Confirm the material model is appropriate for the measured strain range.
- Use at least two independent deformation modes where possible.
- Review parameter confidence intervals and correlation.
- Run single-element solver verification after FEA export.
- Store source test data, preprocessing settings and fitting logs with the released material card.
