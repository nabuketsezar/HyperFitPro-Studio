from __future__ import annotations
from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_exp, BIG
import numpy as np

class EmbeddedExponentialPolynomialI1I2(HyperelasticModel):
    number = 9018
    name = "Embedded Plugin - Exponential-polynomial I1/I2"
    category = "Embedded plugin / Exponential-polynomial"
    family = "Hybrid exponential polynomial"
    active = True
    recommended_tests = ["uniaxial", "biaxial", "planar"]
    material_classes = ["soft tissue", "filled rubber", "rubber-like elastomer"]
    application_tags = ["plugin", "exponential", "polynomial", "stiffening"]
    complexity_level = 4
    minimum_recommended_points_per_parameter = 12
    preferred_optimizer = "recommended_adaptive_bounds"
    calibration_notes = "Use for data with initial polynomial behavior and strong high-strain stiffening."
    parameterization_notes = "A/B controls exponential I1 contribution; C01 and C20 add I2 and low-order curvature."
    limitation_notes = "The exponential term can dominate extrapolation; use adaptive bounds and validation data."
    source_equation_status = "embedded plugin extension"
    equation_latex = r"W=\frac{A}{B}\left(e^{B(I_1-3)}-1\right)+C_{01}(I_2-3)+C_{20}(I_1-3)^2"
    parameter_specs = [
        ParameterSpec("A", 0.0, 20.0, 0.1, "positive_stress", "stress", "Exponential stress scale."),
        ParameterSpec("B", 1.0e-4, 60.0, 1.0, "dimensionless", "-", "Exponential stiffening coefficient."),
        ParameterSpec("C01", -20.0, 20.0, 0.0, "stress", "stress", "Linear I2 coefficient."),
        ParameterSpec("C20", -20.0, 20.0, 0.0, "stress", "stress", "Quadratic I1 coefficient."),
    ]
    def W(self, stretches, p):
        I1,I2 = invariants(stretches); B=p["B"]
        if B <= 0:
            return BIG
        x=I1-3.0; y=I2-3.0
        val = p["A"]/B*(safe_exp(B*x)-1.0) + p["C01"]*y + p["C20"]*x*x
        return val if np.isfinite(val) else BIG
MODEL_CLASS = EmbeddedExponentialPolynomialI1I2
