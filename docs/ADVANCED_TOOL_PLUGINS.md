# HyperFitPro Studio Surprise Plugin Lab

This release adds a second wave of embedded tool plugins that go beyond ordinary plots.
They turn a fitted model into a reviewable material package, a digital test lab, a risk card,
an identifiability audit and an interactive dossier.

## New embedded tools

| Plugin ID | Name | Main output |
|---|---|---|
| `studio.material_passport` | Material passport generator | PDF/PNG/CSV material passport with risk badge |
| `lab.virtual_test_lab` | Virtual test lab | Standard response curves for uniaxial/biaxial/planar/shear |
| `diagnostics.identifiability_svd` | Identifiability SVD inspector | Singular values, weak parameter direction, recommendations |
| `advisor.next_test_designer` | Next-test DOE designer | Next test matrix + Excel workbook + HTML guide |
| `plot.optimization_cinema` | Optimization cinema | Standalone HTML slider showing fit evolution |
| `report.interactive_html_dossier` | Interactive HTML calibration dossier | Self-contained engineering dossier |
| `advisor.model_risk_scorecard` | Model risk scorecard | Green/Amber/Red scorecard |
| `plot.response_surface_3d` | 3D response surface explorer | 3D sensitivity surfaces |
| `story.calibration_storyboard` | Calibration storyboard | One-page engineering story board |
| `fea.export_auditor` | FEA export auditor | Solver-readiness audit checklist |

Outputs are written to `<run_folder>/tools/`.
