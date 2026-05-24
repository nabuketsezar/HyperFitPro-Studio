from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model023(HyperelasticModel):
    number = 23
    name = 'David De-Thomas'
    category = 'Power-law / phenomenological'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'Use multi-start optimization due to nonlinearity in n and C.'
    application_tags = ['power-law', 'quadratic correction']
    material_classes = ['rubber-like elastomer']
    complexity_level = 3
    family = 'David De-Thomas'
    equation_latex = 'W=\\frac{A}{2-n}(I_1-3+C^2)^{1-n/2}+k(I_1-3)^2'
    recommended_tests = ['uniaxial', 'biaxial', 'planar', 'simple_shear']
    notes = 'Use multi-start optimization due to nonlinearity in n and C.'
    active = True
    parameter_specs = [
        ParameterSpec('A', -20, 20, 0.05, 'stress', '-', ''),
        ParameterSpec('C', 1e-06, 100, 1, 'dimensionless', '-', ''),
        ParameterSpec('n', -20, 1.95, 0, 'dimensionless', '-', ''),
        ParameterSpec('k', -20, 20, 0, 'stress', '-', '')
    ]

    def W(self, stretches, p):
        I1,_=invariants(stretches); x=I1-3; z=x+p['C']**2
        return p['A']/(2-p['n'])*safe_pow(z,1-p['n']/2)+p['k']*x*x

MODEL_CLASS = Model023
