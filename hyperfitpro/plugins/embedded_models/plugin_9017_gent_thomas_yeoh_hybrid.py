from __future__ import annotations
from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, BIG
import numpy as np

class EmbeddedGentThomasYeohHybrid(HyperelasticModel):
    number = 9017
    name = "Embedded Plugin - Gent-Thomas-Yeoh hybrid"
    category = "Embedded plugin / Hybrid I1-I2"
    family = "Gent-Thomas + Yeoh polynomial"
    active = True
    recommended_tests = ["uniaxial", "biaxial", "planar", "simple_shear"]
    material_classes = ["rubber-like elastomer", "filled rubber"]
    application_tags = ["plugin", "hybrid", "I2 sensitivity", "moderate strain"]
    complexity_level = 3
    minimum_recommended_points_per_parameter = 12
    preferred_optimizer = "recommended_adaptive_bounds"
    calibration_notes = "Use multi-mode data because the logarithmic I2 contribution is not well separated by uniaxial-only data."
    parameterization_notes = "C2 multiplies ln(I2/3), which is zero at the reference state."
    limitation_notes = "I2 logarithmic term requires positive I2; this is normally satisfied for physical stretches."
    source_equation_status = "embedded plugin extension"
    equation_latex = r"W=C_{10}(I_1-3)+C_{20}(I_1-3)^2+3C_2\ln\left(\frac{I_2}{3}\right)"
    parameter_specs = [
        ParameterSpec("C10", 0.0, 20.0, 0.1, "positive_stress", "stress", "Linear I1 coefficient."),
        ParameterSpec("C20", -20.0, 20.0, 0.0, "stress", "stress", "Quadratic I1 coefficient."),
        ParameterSpec("C2", -20.0, 20.0, 0.0, "stress", "stress", "Gent-Thomas logarithmic I2 coefficient."),
    ]
    def W(self, stretches, p):
        I1,I2 = invariants(stretches)
        if I2 <= 0:
            return BIG
        x = I1-3.0
        val = p["C10"]*x + p["C20"]*x*x + 3.0*p["C2"]*safe_log(I2/3.0)
        return val if np.isfinite(val) else BIG
MODEL_CLASS = EmbeddedGentThomasYeohHybrid
