from __future__ import annotations
from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_pow
import numpy as np

class EmbeddedTreloarI1Series(HyperelasticModel):
    number = 9022
    name = "Embedded Plugin - Treloar I1 series"
    category = "Embedded plugin / I1 series"
    family = "Treloar invariant series"
    active = True
    recommended_tests = ["uniaxial", "biaxial", "planar"]
    material_classes = ["rubber-like elastomer"]
    application_tags = ["plugin", "I1-only", "series"]
    complexity_level = 3
    minimum_recommended_points_per_parameter = 10
    preferred_optimizer = "recommended_adaptive_bounds"
    calibration_notes = "Use as a compact I1-only series baseline for rubber-like data."
    parameterization_notes = "Each term is zero at I1=3 through subtraction of 3^n."
    limitation_notes = "I1-only form cannot fully distinguish biaxial and shear behavior if calibrated from uniaxial only."
    source_equation_status = "embedded plugin extension"
    equation_latex = r"W=A_1(I_1-3)+A_2(I_1^2-9)+A_3(I_1^3-27)"
    parameter_specs = [
        ParameterSpec("A1", 0.0, 20.0, 0.1, "positive_stress", "stress", "First I1-series coefficient."),
        ParameterSpec("A2", -5.0, 5.0, 0.0, "stress", "stress", "Second I1-series coefficient."),
        ParameterSpec("A3", -2.0, 2.0, 0.0, "stress", "stress", "Third I1-series coefficient."),
    ]
    def W(self, stretches, p):
        I1,_=invariants(stretches)
        val = p["A1"]*(I1-3.0) + p["A2"]*(safe_pow(I1,2.0)-9.0) + p["A3"]*(safe_pow(I1,3.0)-27.0)
        return val if np.isfinite(val) else 1e60
MODEL_CLASS = EmbeddedTreloarI1Series
