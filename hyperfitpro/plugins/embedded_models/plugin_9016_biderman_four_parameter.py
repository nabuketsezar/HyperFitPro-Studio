from __future__ import annotations
from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants
import numpy as np

class EmbeddedBidermanFourParameter(HyperelasticModel):
    number = 9016
    name = "Embedded Plugin - Biderman four-parameter"
    category = "Embedded plugin / Polynomial I1-I2"
    family = "Biderman polynomial"
    active = True
    recommended_tests = ["uniaxial", "biaxial", "planar"]
    material_classes = ["rubber-like elastomer"]
    application_tags = ["plugin", "polynomial", "large strain"]
    complexity_level = 3
    minimum_recommended_points_per_parameter = 10
    preferred_optimizer = "recommended_adaptive_bounds"
    calibration_notes = "Polynomial model with I1 cubic term. Use it when Yeoh-like curvature plus I2 influence are both needed."
    parameterization_notes = "C30 controls high-stretch curvature and should be bounded carefully."
    limitation_notes = "Do not extrapolate far outside fitted strain range without stability scan."
    source_equation_status = "embedded plugin extension"
    equation_latex = r"W=C_{10}(I_1-3)+C_{01}(I_2-3)+C_{20}(I_1-3)^2+C_{30}(I_1-3)^3"
    parameter_specs = [
        ParameterSpec("C10", 0.0, 20.0, 0.1, "positive_stress", "stress", "Linear I1 coefficient."),
        ParameterSpec("C01", -20.0, 20.0, 0.0, "stress", "stress", "Linear I2 coefficient."),
        ParameterSpec("C20", -20.0, 20.0, 0.0, "stress", "stress", "Quadratic I1 coefficient."),
        ParameterSpec("C30", -10.0, 10.0, 0.0, "stress", "stress", "Cubic I1 coefficient."),
    ]
    def W(self, stretches, p):
        I1,I2 = invariants(stretches); x=I1-3.0; y=I2-3.0
        val = p["C10"]*x + p["C01"]*y + p["C20"]*x*x + p["C30"]*x**3
        return val if np.isfinite(val) else 1e60
MODEL_CLASS = EmbeddedBidermanFourParameter
