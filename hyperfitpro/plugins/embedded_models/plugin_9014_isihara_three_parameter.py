from __future__ import annotations
from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants
import numpy as np

class EmbeddedIsiharaThreeParameter(HyperelasticModel):
    number = 9014
    name = "Embedded Plugin - Isihara three-parameter"
    category = "Embedded plugin / Polynomial I1-I2"
    family = "Mooney-polynomial extension"
    active = True
    recommended_tests = ["uniaxial", "biaxial", "planar"]
    material_classes = ["rubber-like elastomer"]
    application_tags = ["plugin", "polynomial", "I1-I2", "moderate strain"]
    complexity_level = 2
    minimum_recommended_points_per_parameter = 10
    preferred_optimizer = "recommended_adaptive_bounds"
    calibration_notes = "Compact polynomial extension; use biaxial/planar data to separate C10 and C01."
    parameterization_notes = "C20 controls the first nonlinear I1 correction."
    limitation_notes = "Polynomial extrapolation should be checked with the stability scan."
    source_equation_status = "embedded plugin extension"
    equation_latex = r"W=C_{10}(I_1-3)+C_{01}(I_2-3)+C_{20}(I_1-3)^2"
    parameter_specs = [
        ParameterSpec("C10", 0.0, 20.0, 0.1, "positive_stress", "stress", "Linear I1 coefficient."),
        ParameterSpec("C01", -20.0, 20.0, 0.0, "stress", "stress", "Linear I2 coefficient."),
        ParameterSpec("C20", -20.0, 20.0, 0.0, "stress", "stress", "Quadratic I1 coefficient."),
    ]
    def W(self, stretches, p):
        I1,I2 = invariants(stretches)
        x = I1-3.0; y = I2-3.0
        val = p["C10"]*x + p["C01"]*y + p["C20"]*x*x
        return val if np.isfinite(val) else 1e60
MODEL_CLASS = EmbeddedIsiharaThreeParameter
