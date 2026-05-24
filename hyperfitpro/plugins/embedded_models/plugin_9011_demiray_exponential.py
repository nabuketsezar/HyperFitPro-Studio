from __future__ import annotations
from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_exp, BIG
import numpy as np

class EmbeddedDemirayExponential(HyperelasticModel):
    number = 9011
    name = "Embedded Plugin - Demiray exponential"
    category = "Embedded plugin / Exponential I1"
    family = "Exponential soft-tissue / elastomer"
    active = True
    recommended_tests = ["uniaxial", "biaxial", "planar"]
    material_classes = ["soft tissue", "rubber-like elastomer"]
    application_tags = ["plugin", "exponential", "stiffening"]
    complexity_level = 2
    minimum_recommended_points_per_parameter = 10
    preferred_optimizer = "recommended_adaptive_bounds"
    calibration_notes = "Use when stress stiffening is clearly exponential. Biaxial or planar data improves identifiability of b."
    parameterization_notes = "a is stress-like; b is dimensionless and positive. For very small b the model approaches Neo-Hookean behavior."
    limitation_notes = "Can over-stiffen outside the calibration range; use stability and extrapolation plots before export."
    source_equation_status = "embedded plugin extension"
    equation_latex = r"W=\frac{a}{2b}\left(e^{b(I_1-3)}-1\right)"
    parameter_specs = [
        ParameterSpec("a", 0.0, 20.0, 0.1, "positive_stress", "stress", "Initial stress-scale coefficient."),
        ParameterSpec("b", 1.0e-4, 80.0, 1.0, "dimensionless", "-", "Dimensionless exponential stiffening coefficient."),
    ]
    def W(self, stretches, p):
        I1,_ = invariants(stretches)
        b = p["b"]
        if b <= 0:
            return BIG
        val = p["a"]/(2.0*b)*(safe_exp(b*(I1-3.0))-1.0)
        return val if np.isfinite(val) else BIG
MODEL_CLASS = EmbeddedDemirayExponential
