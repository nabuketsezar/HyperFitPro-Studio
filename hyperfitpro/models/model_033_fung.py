from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model033(HyperelasticModel):
    number = 33
    name = 'Fung'
    category = 'Exponential / Fung-type'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'Classic exponential stiffening model; monitor overflow and tangent stiffness.'
    application_tags = ['exponential stiffening']
    material_classes = ['soft tissue', 'rubber-like elastomer']
    complexity_level = 2
    family = 'Fung'
    equation_latex = 'W=\\frac{\\mu}{2b}(e^{b(I_1-3)}-1)'
    recommended_tests = ['uniaxial', 'biaxial', 'planar']
    notes = 'Classic exponential stiffening model; monitor overflow and tangent stiffness.'
    active = True
    parameter_specs = [
        ParameterSpec('mu', 0, 20, 0.1, 'positive_stress', '-', ''),
        ParameterSpec('b', 1e-08, 100, 1, 'dimensionless', '-', '')
    ]

    def W(self, stretches, p):
        I1,_=invariants(stretches); return p['mu']/(2*p['b'])*(safe_exp(p['b']*(I1-3))-1)

MODEL_CLASS = Model033
