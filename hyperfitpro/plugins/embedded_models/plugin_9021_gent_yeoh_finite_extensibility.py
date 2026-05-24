from __future__ import annotations
from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, BIG
import numpy as np

class EmbeddedGentYeohFiniteExtensibility(HyperelasticModel):
    number = 9021
    name = "Embedded Plugin - Gent-Yeoh finite extensibility"
    category = "Embedded plugin / Limiting-chain hybrid"
    family = "Gent + Yeoh hybrid"
    active = True
    recommended_tests = ["uniaxial", "biaxial", "planar", "simple_shear"]
    material_classes = ["filled rubber", "rubber-like elastomer", "finite extensibility"]
    application_tags = ["plugin", "finite extensibility", "large strain", "hybrid"]
    complexity_level = 4
    minimum_recommended_points_per_parameter = 12
    preferred_optimizer = "recommended_adaptive_bounds"
    calibration_notes = "Use when finite chain locking is expected but low/mid-strain curvature also needs Yeoh correction."
    parameterization_notes = "Jm must remain larger than max(I1-3) in the calibration and verification domain."
    limitation_notes = "Singular as I1-3 approaches Jm; wide high-strain validation is required."
    source_equation_status = "embedded plugin extension"
    equation_latex = r"W=-\frac{\mu J_m}{2}\ln\left(1-\frac{I_1-3}{J_m}\right)+C_{20}(I_1-3)^2+C_{30}(I_1-3)^3"
    parameter_specs = [
        ParameterSpec("mu", 0.0, 20.0, 0.1, "positive_stress", "stress", "Gent stress scale."),
        ParameterSpec("Jm", 0.1, 500.0, 50.0, "invariant_limit", "-", "Limiting invariant parameter."),
        ParameterSpec("C20", -20.0, 20.0, 0.0, "stress", "stress", "Quadratic Yeoh correction."),
        ParameterSpec("C30", -10.0, 10.0, 0.0, "stress", "stress", "Cubic Yeoh correction."),
    ]
    def W(self, stretches, p):
        I1,_ = invariants(stretches); x=I1-3.0; Jm=p["Jm"]
        if Jm <= 0 or x >= Jm:
            return BIG
        val = -0.5*p["mu"]*Jm*safe_log(1.0 - x/Jm) + p["C20"]*x*x + p["C30"]*x**3
        return val if np.isfinite(val) else BIG
MODEL_CLASS = EmbeddedGentYeohFiniteExtensibility
