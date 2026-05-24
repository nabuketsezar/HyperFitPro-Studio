from __future__ import annotations
from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_pow, BIG
import numpy as np

class EmbeddedCarrollHighStrain(HyperelasticModel):
    number = 9012
    name = "Embedded Plugin - Carroll high-strain"
    category = "Embedded plugin / High-strain invariant"
    family = "Carroll invariant model"
    active = True
    recommended_tests = ["uniaxial", "biaxial", "planar", "simple_shear"]
    material_classes = ["rubber-like elastomer", "filled rubber"]
    application_tags = ["plugin", "high strain", "I1-I2"]
    complexity_level = 3
    minimum_recommended_points_per_parameter = 12
    preferred_optimizer = "recommended_adaptive_bounds"
    calibration_notes = "Useful when I1 high-power stiffening and I2 response are both needed. Avoid fitting all parameters from uniaxial-only data."
    parameterization_notes = "A and C are primary stress-like terms; B controls the high-strain I1^4 correction."
    limitation_notes = "The I1^4 term can extrapolate aggressively at very high stretch."
    source_equation_status = "embedded plugin extension"
    equation_latex = r"W=A(I_1-3)+B(I_1^4-81)+C(\sqrt{I_2}-\sqrt{3})"
    parameter_specs = [
        ParameterSpec("A", 0.0, 20.0, 0.1, "positive_stress", "stress", "Linear I1 stress-scale coefficient."),
        ParameterSpec("B", -2.0, 2.0, 0.0, "stress", "stress", "High-strain I1^4 coefficient."),
        ParameterSpec("C", -20.0, 20.0, 0.0, "stress", "stress", "I2 square-root coefficient."),
    ]
    def W(self, stretches, p):
        I1,I2 = invariants(stretches)
        if I2 <= 0:
            return BIG
        val = p["A"]*(I1-3.0) + p["B"]*(safe_pow(I1,4.0)-81.0) + p["C"]*(np.sqrt(I2)-np.sqrt(3.0))
        return val if np.isfinite(val) else BIG
MODEL_CLASS = EmbeddedCarrollHighStrain
