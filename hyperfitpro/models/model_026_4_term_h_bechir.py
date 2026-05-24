from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model026(HyperelasticModel):
    number = 26
    name = '4 term H. Bechir et al.'
    category = 'Principal-stretch / Bechir spectral polynomial'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'Four coefficients; multi-mode calibration recommended.'
    application_tags = ['spectral polynomial', 'overfit risk']
    material_classes = ['rubber-like elastomer']
    complexity_level = 4
    family = 'H. Bechir et al. spectral polynomial'
    equation_latex = 'W=\\sum_{n=1}^{2}\\sum_{r=1}^{2}C_n^r(\\lambda_1^{2n}+\\lambda_2^{2n}+\\lambda_3^{2n}-3)^r'
    recommended_tests = ['uniaxial', 'biaxial', 'planar', 'simple_shear']
    notes = 'Four coefficients; multi-mode calibration recommended.'
    active = True
    parameter_specs = [
        ParameterSpec('C11', -20, 20, 0.05, 'stress', '-', ''),
        ParameterSpec('C12', -20, 20, 0, 'stress', '-', ''),
        ParameterSpec('C21', -20, 20, 0, 'stress', '-', ''),
        ParameterSpec('C22', -20, 20, 0, 'stress', '-', '')
    ]

    def W(self, stretches, p):
        l1,l2,l3=stretches; val=0.0
        for n in (1,2):
            z=l1**(2*n)+l2**(2*n)+l3**(2*n)-3
            for r in (1,2): val += p[f'C{n}{r}']*(z**r)
        return val

MODEL_CLASS = Model026
