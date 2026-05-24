from __future__ import annotations
from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants
import numpy as np

class EmbeddedHainesWilsonFiveParameter(HyperelasticModel):
    number = 9015
    name = "Embedded Plugin - Haines-Wilson five-parameter"
    category = "Embedded plugin / Polynomial I1-I2"
    family = "Haines-Wilson polynomial"
    active = True
    recommended_tests = ["uniaxial", "biaxial", "planar", "simple_shear"]
    material_classes = ["rubber-like elastomer", "filled rubber"]
    application_tags = ["plugin", "polynomial", "multi-mode calibration"]
    complexity_level = 4
    minimum_recommended_points_per_parameter = 12
    preferred_optimizer = "recommended_adaptive_bounds"
    calibration_notes = "Use multi-mode data. This five-parameter polynomial is not recommended for sparse uniaxial-only fitting."
    parameterization_notes = "C11 couples I1 and I2 nonlinear response."
    limitation_notes = "Higher-order terms can create nonphysical extrapolation if bounds are too wide."
    source_equation_status = "embedded plugin extension"
    equation_latex = r"W=C_{10}x+C_{01}y+C_{20}x^2+C_{30}x^3+C_{11}xy,\quad x=I_1-3,\ y=I_2-3"
    parameter_specs = [
        ParameterSpec("C10", 0.0, 20.0, 0.1, "positive_stress", "stress", "Linear I1 coefficient."),
        ParameterSpec("C01", -20.0, 20.0, 0.0, "stress", "stress", "Linear I2 coefficient."),
        ParameterSpec("C20", -20.0, 20.0, 0.0, "stress", "stress", "Quadratic I1 coefficient."),
        ParameterSpec("C30", -10.0, 10.0, 0.0, "stress", "stress", "Cubic I1 coefficient."),
        ParameterSpec("C11", -20.0, 20.0, 0.0, "stress", "stress", "I1-I2 coupling coefficient."),
    ]
    def W(self, stretches, p):
        I1,I2 = invariants(stretches); x=I1-3.0; y=I2-3.0
        val = p["C10"]*x + p["C01"]*y + p["C20"]*x*x + p["C30"]*x**3 + p["C11"]*x*y
        return val if np.isfinite(val) else 1e60
MODEL_CLASS = EmbeddedHainesWilsonFiveParameter
