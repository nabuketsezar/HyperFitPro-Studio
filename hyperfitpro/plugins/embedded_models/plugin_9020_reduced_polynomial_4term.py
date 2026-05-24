from __future__ import annotations
from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants
import numpy as np

class EmbeddedReducedPolynomial4Term(HyperelasticModel):
    number = 9020
    name = "Embedded Plugin - Reduced polynomial 4-term"
    category = "Embedded plugin / Reduced polynomial"
    family = "Yeoh-style reduced polynomial"
    active = True
    recommended_tests = ["uniaxial", "biaxial", "planar"]
    material_classes = ["rubber-like elastomer", "filled rubber"]
    application_tags = ["plugin", "Yeoh-style", "I1-only", "large strain"]
    complexity_level = 4
    minimum_recommended_points_per_parameter = 12
    preferred_optimizer = "recommended_adaptive_bounds"
    calibration_notes = "I1-only model; recommended when reliable biaxial data is unavailable but large uniaxial strain coverage exists."
    parameterization_notes = "C40 adds additional high-strain curvature beyond the built-in 3-term Yeoh."
    limitation_notes = "I1-only response may underperform in shear or biaxial prediction if calibrated only to uniaxial data."
    source_equation_status = "embedded plugin extension"
    equation_latex = r"W=C_{10}x+C_{20}x^2+C_{30}x^3+C_{40}x^4,\quad x=I_1-3"
    parameter_specs = [
        ParameterSpec("C10", 0.0, 20.0, 0.1, "positive_stress", "stress", "Linear I1 coefficient."),
        ParameterSpec("C20", -20.0, 20.0, 0.0, "stress", "stress", "Quadratic I1 coefficient."),
        ParameterSpec("C30", -10.0, 10.0, 0.0, "stress", "stress", "Cubic I1 coefficient."),
        ParameterSpec("C40", -5.0, 5.0, 0.0, "stress", "stress", "Quartic I1 coefficient."),
    ]
    def W(self, stretches, p):
        I1,_ = invariants(stretches); x=I1-3.0
        val = p["C10"]*x + p["C20"]*x**2 + p["C30"]*x**3 + p["C40"]*x**4
        return val if np.isfinite(val) else 1e60
MODEL_CLASS = EmbeddedReducedPolynomial4Term
