from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model037(HyperelasticModel):
    number = 37
    name = 'Low strain Hoss-Marczak'
    category = 'Hybrid / low-strain Hoss-Marczak'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'Low-strain hybrid; parameters can be correlated.'
    application_tags = ['low strain', 'hybrid']
    material_classes = ['rubber-like elastomer', 'filled rubber']
    complexity_level = 5
    family = 'Hoss-Marczak'
    equation_latex = 'W=\\frac{\\alpha}{\\beta}(1-e^{-\\beta(I_1-3)})+\\frac{\\mu}{2b}[(1+\\frac{b(I_1-3)}{n})^n-1]'
    recommended_tests = ['uniaxial', 'biaxial', 'planar']
    notes = 'Low-strain hybrid; parameters can be correlated.'
    active = True
    parameter_specs = [
        ParameterSpec('alpha', 0, 20, 0.05, 'positive_stress', '-', ''),
        ParameterSpec('beta', 1e-08, 100, 1, 'dimensionless', '-', ''),
        ParameterSpec('mu', 0, 20, 0.1, 'positive_stress', '-', ''),
        ParameterSpec('b', 1e-08, 100, 1, 'dimensionless', '-', ''),
        ParameterSpec('n', 0.05, 20, 2, 'dimensionless', '-', '')
    ]

    def W(self, stretches, p):
        I1,_=invariants(stretches); x=I1-3
        return p['alpha']/p['beta']*(1-safe_exp(-p['beta']*x))+p['mu']/(2*p['b'])*(safe_pow(1+p['b']*x/p['n'],p['n'])-1)

MODEL_CLASS = Model037
