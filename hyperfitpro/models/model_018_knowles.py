from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model018(HyperelasticModel):
    number = 18
    name = 'Knowles'
    category = 'Network / generalized finite extensibility'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'n and b may be correlated; use adaptive bounds.'
    application_tags = ['power law', 'finite extensibility']
    material_classes = ['rubber-like elastomer']
    complexity_level = 3
    family = 'Knowles'
    equation_latex = 'W=\\frac{\\mu}{2b}[(1+\\frac{b(I_1-3)}{n})^n-1]'
    recommended_tests = ['uniaxial', 'biaxial', 'planar']
    notes = 'n and b may be correlated; use adaptive bounds.'
    active = True
    parameter_specs = [
        ParameterSpec('mu', 0, 20, 0.1, 'positive_stress', '-', ''),
        ParameterSpec('b', 1e-08, 100, 1, 'dimensionless', '-', ''),
        ParameterSpec('n', 0.05, 20, 2, 'dimensionless', '-', '')
    ]

    def W(self, stretches, p):
        I1,_=invariants(stretches); return p['mu']/(2*p['b'])*(safe_pow(1+p['b']*(I1-3)/p['n'], p['n'])-1)

MODEL_CLASS = Model018
