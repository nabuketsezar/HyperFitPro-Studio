from __future__ import annotations
from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, BIG
import numpy as np

class EmbeddedOneTermOgden(HyperelasticModel):
    number = 9010
    name = "Embedded Plugin - One-term Ogden"
    category = "Embedded plugin / Ogden family"
    family = "Ogden power-law"
    active = True
    recommended_tests = ["uniaxial", "biaxial", "planar", "simple_shear"]
    material_classes = ["rubber-like elastomer", "soft tissue first-pass"]
    application_tags = ["plugin", "baseline", "moderate strain", "power-law"]
    complexity_level = 2
    minimum_recommended_points_per_parameter = 10
    preferred_optimizer = "recommended_adaptive_bounds"
    calibration_notes = "Good compact alternative to Neo-Hookean when one curvature parameter is needed. Use more than one deformation mode if possible."
    parameterization_notes = "alpha controls the nonlinearity; alpha near zero is excluded to avoid the singular Ogden limit."
    limitation_notes = "A single Ogden term may not capture filled rubber S-shape behavior over very large strain ranges."
    source_equation_status = "embedded plugin extension"
    equation_latex = r"W=\frac{\mu}{\alpha}\left(\lambda_1^{\alpha}+\lambda_2^{\alpha}+\lambda_3^{\alpha}-3\right)"
    parameter_specs = [
        ParameterSpec("mu", 0.0, 20.0, 0.1, "positive_stress", "stress", "Ogden stress-scale coefficient."),
        ParameterSpec("alpha", 0.15, 12.0, 2.0, "dimensionless", "-", "Ogden exponent; alpha=2 gives Neo-Hookean-like response."),
    ]
    def W(self, stretches, p):
        mu = p["mu"]; alpha = p["alpha"]
        if abs(alpha) < 1e-10:
            return BIG
        l1,l2,l3 = [float(x) for x in stretches]
        val = mu/alpha*(l1**alpha + l2**alpha + l3**alpha - 3.0)
        return val if np.isfinite(val) else BIG
MODEL_CLASS = EmbeddedOneTermOgden
