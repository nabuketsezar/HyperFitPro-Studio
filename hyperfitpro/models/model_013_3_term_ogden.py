from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model013(HyperelasticModel):
    number = 13
    name = '3 term Ogden'
    category = 'Principal-stretch / Ogden family'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'Three-term Ogden needs broad multi-mode data and regularization/selection checks.'
    application_tags = ['spectral', 'many parameters', 'high flexibility']
    material_classes = ['rubber-like elastomer']
    complexity_level = 5
    family = 'Ogden spectral'
    equation_latex = 'W=\\sum_{i=1}^{3}\\frac{\\mu_i}{\\alpha_i}(\\lambda_1^{\\alpha_i}+\\lambda_2^{\\alpha_i}+\\lambda_3^{\\alpha_i}-3)'
    recommended_tests = ['uniaxial', 'biaxial', 'planar', 'simple_shear']
    notes = 'Three-term Ogden needs broad multi-mode data and regularization/selection checks.'
    active = True
    parameter_specs = [
        ParameterSpec('mu1', -20, 20, 0.1, 'stress', '-', ''),
        ParameterSpec('alpha1', -20, 20, 2, 'dimensionless', '-', ''),
        ParameterSpec('mu2', -20, 20, 0, 'stress', '-', ''),
        ParameterSpec('alpha2', -20, 20, -2, 'dimensionless', '-', ''),
        ParameterSpec('mu3', -20, 20, 0, 'stress', '-', ''),
        ParameterSpec('alpha3', -20, 20, 4, 'dimensionless', '-', '')
    ]

    def W(self, stretches, p):
        l1,l2,l3=stretches
        val=0.0
        for i in (1,2,3):
            a=p[f'alpha{i}']
            if abs(a)<1e-8: return BIG
            val += p[f'mu{i}']/a*(safe_pow(l1,a)+safe_pow(l2,a)+safe_pow(l3,a)-3)
        return val

MODEL_CLASS = Model013
