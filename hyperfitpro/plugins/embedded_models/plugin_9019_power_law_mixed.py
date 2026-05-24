from __future__ import annotations
from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_pow, BIG
import numpy as np

class EmbeddedPowerLawMixed(HyperelasticModel):
    number = 9019
    name = "Embedded Plugin - Shifted power-law mixed"
    category = "Embedded plugin / Power-law mixed"
    family = "Shifted invariant power-law"
    active = True
    recommended_tests = ["uniaxial", "biaxial", "planar", "simple_shear"]
    material_classes = ["rubber-like elastomer", "foam-like first-pass"]
    application_tags = ["plugin", "power-law", "I1-I2"]
    complexity_level = 3
    minimum_recommended_points_per_parameter = 12
    preferred_optimizer = "recommended_adaptive_bounds"
    calibration_notes = "Useful experimental plugin when a shifted power-law provides better curvature than low-order polynomial terms."
    parameterization_notes = "c shifts the I1 invariant to avoid singular behavior; n controls power-law curvature."
    limitation_notes = "Not a replacement for a compressible foam model; use only for incompressible scalar response fitting."
    source_equation_status = "embedded plugin extension"
    equation_latex = r"W=\frac{A}{n}\left[(I_1-3+c)^n-c^n\right]+C_{01}(I_2-3)"
    parameter_specs = [
        ParameterSpec("A", 0.0, 20.0, 0.1, "positive_stress", "stress", "Power-law stress scale."),
        ParameterSpec("n", 0.2, 6.0, 1.5, "dimensionless", "-", "Power-law exponent."),
        ParameterSpec("c", 0.05, 20.0, 1.0, "dimensionless", "-", "Invariant shift parameter."),
        ParameterSpec("C01", -20.0, 20.0, 0.0, "stress", "stress", "Linear I2 coefficient."),
    ]
    def W(self, stretches, p):
        I1,I2 = invariants(stretches); n=p["n"]; c=p["c"]
        if abs(n) < 1e-12 or c <= 0:
            return BIG
        base = I1-3.0+c
        if base <= 0:
            return BIG
        val = p["A"]/n*(safe_pow(base,n)-safe_pow(c,n)) + p["C01"]*(I2-3.0)
        return val if np.isfinite(val) else BIG
MODEL_CLASS = EmbeddedPowerLawMixed
