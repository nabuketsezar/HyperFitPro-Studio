from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model014(HyperelasticModel):
    number = 14
    name = 'Veronda-Westmann'
    category = 'Exponential / soft-tissue type'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'Exponential models can overflow; keep parameter bounds physically modest.'
    application_tags = ['exponential stiffening']
    material_classes = ['soft tissue', 'rubber-like elastomer']
    complexity_level = 2
    family = 'Veronda-Westmann'
    equation_latex = 'W=C_1(e^{\\alpha(I_1-3)}-1)-C_2(I_2-3)'
    recommended_tests = ['uniaxial', 'biaxial', 'planar']
    notes = 'Exponential models can overflow; keep parameter bounds physically modest.'
    active = True
    parameter_specs = [
        ParameterSpec('C1', 0, 20, 0.05, 'positive_stress', '-', ''),
        ParameterSpec('alpha', 1e-08, 100, 1, 'dimensionless', '-', ''),
        ParameterSpec('C2', -20, 20, 0, 'stress', '-', '')
    ]

    def W(self, stretches, p):
        I1,I2=invariants(stretches); x=I1-3
        return p['C1']*(safe_exp(p['alpha']*x)-1)-p['C2']*(I2-3)

MODEL_CLASS = Model014
