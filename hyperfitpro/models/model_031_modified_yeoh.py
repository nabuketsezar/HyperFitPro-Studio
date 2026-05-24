from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model031(HyperelasticModel):
    number = 31
    name = 'Modified Yeoh'
    category = 'Reduced polynomial / modified Yeoh'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'Modified Yeoh combines polynomial and exponential saturation.'
    application_tags = ['Yeoh + exponential saturation']
    material_classes = ['filled rubber', 'rubber-like elastomer']
    complexity_level = 4
    family = 'Modified Yeoh'
    equation_latex = 'W=C_{10}(I_1-3)+C_{20}(I_1-3)^2+C_{30}(I_1-3)^3+\\frac{\\alpha}{\\beta}(1-e^{-\\beta(I_1-3)})'
    recommended_tests = ['uniaxial', 'biaxial', 'planar']
    notes = 'Modified Yeoh combines polynomial and exponential saturation.'
    active = True
    parameter_specs = [
        ParameterSpec('C10', 0, 20, 0.05, 'positive_stress', '-', ''),
        ParameterSpec('C20', -20, 20, 0, 'stress', '-', ''),
        ParameterSpec('C30', -20, 20, 0, 'stress', '-', ''),
        ParameterSpec('alpha', 0, 20, 0.05, 'positive_stress', '-', ''),
        ParameterSpec('beta', 1e-08, 100, 1, 'dimensionless', '-', '')
    ]

    def W(self, stretches, p):
        I1,_=invariants(stretches); x=I1-3
        return p['C10']*x+p['C20']*x**2+p['C30']*x**3+p['alpha']/p['beta']*(1-safe_exp(-p['beta']*x))

MODEL_CLASS = Model031
