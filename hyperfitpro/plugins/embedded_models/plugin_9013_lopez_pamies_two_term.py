from __future__ import annotations
from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_pow, BIG
import numpy as np

class EmbeddedLopezPamiesTwoTerm(HyperelasticModel):
    number = 9013
    name = "Embedded Plugin - Lopez-Pamies two-term"
    category = "Embedded plugin / I1 power series"
    family = "Lopez-Pamies I1 powers"
    active = True
    recommended_tests = ["uniaxial", "biaxial", "planar"]
    material_classes = ["rubber-like elastomer", "filled rubber"]
    application_tags = ["plugin", "I1-only", "power-law", "large strain"]
    complexity_level = 4
    minimum_recommended_points_per_parameter = 12
    preferred_optimizer = "recommended_adaptive_bounds"
    calibration_notes = "Prefer at least two deformation modes. alpha parameters are weakly identifiable from narrow strain ranges."
    parameterization_notes = "mu1, mu2 are stress-like coefficients; alpha1, alpha2 are dimensionless exponents."
    limitation_notes = "I1-only model cannot independently control I2 sensitivity."
    source_equation_status = "embedded plugin extension"
    equation_latex = r"W=\sum_{r=1}^{2}\frac{\mu_r}{\alpha_r}\left(I_1^{\alpha_r/2}-3^{\alpha_r/2}\right)"
    parameter_specs = [
        ParameterSpec("mu1", 0.0, 20.0, 0.1, "positive_stress", "stress", "First stress-like coefficient."),
        ParameterSpec("alpha1", 0.2, 12.0, 2.0, "dimensionless", "-", "First I1 exponent."),
        ParameterSpec("mu2", -20.0, 20.0, 0.0, "stress", "stress", "Second stress-like coefficient."),
        ParameterSpec("alpha2", 0.2, 16.0, 6.0, "dimensionless", "-", "Second I1 exponent."),
    ]
    def W(self, stretches, p):
        I1,_ = invariants(stretches)
        val = 0.0
        for k in (1,2):
            mu = p[f"mu{k}"]; a = p[f"alpha{k}"]
            if abs(a) < 1e-12:
                return BIG
            val += mu/a*(safe_pow(I1,0.5*a) - safe_pow(3.0,0.5*a))
        return val if np.isfinite(val) else BIG
MODEL_CLASS = EmbeddedLopezPamiesTwoTerm
