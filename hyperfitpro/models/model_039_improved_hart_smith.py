from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model039(HyperelasticModel):
    number = 39
    name = 'Improved Hart-Smith'
    category = 'Exponential / improved Hart-Smith'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'n controls exponential power; use bounded global-local fitting.'
    application_tags = ['exponential power', 'I2 log']
    material_classes = ['soft tissue', 'rubber-like elastomer']
    complexity_level = 4
    family = 'Improved Hart-Smith'
    equation_latex = 'W=\\frac{C_1e^{C_3(I_1-3)^n}}{n}+3C_2\\ln(I_2)'
    recommended_tests = ['uniaxial', 'biaxial', 'planar']
    notes = 'n controls exponential power; use bounded global-local fitting.'
    active = True
    parameter_specs = [
        ParameterSpec('C1', 0, 20, 0.05, 'positive_stress', '-', ''),
        ParameterSpec('C2', -20, 20, 0, 'stress', '-', ''),
        ParameterSpec('C3', 1e-08, 100, 1, 'dimensionless', '-', ''),
        ParameterSpec('n', 0.1, 10, 2, 'dimensionless', '-', '')
    ]

    def W(self, stretches, p):
        I1,I2=invariants(stretches); return p['C1']*safe_exp(p['C3']*safe_pow(I1-3,p['n']))/p['n']+3*p['C2']*safe_log(I2)

MODEL_CLASS = Model039
