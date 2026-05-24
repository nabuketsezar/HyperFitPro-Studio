from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model001(HyperelasticModel):
    number = 1
    name = '2 term Mooney-Rivlin'
    category = 'Invariant polynomial / Mooney-Rivlin family'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'Good low-order baseline. Use at least two deformation modes when possible.'
    application_tags = ['baseline', 'low-order', 'moderate strain']
    material_classes = ['rubber-like elastomer']
    complexity_level = 1
    family = 'Mooney-Rivlin polynomial'
    equation_latex = 'W=C_{10}(I_1-3)+C_{01}(I_2-3)'
    recommended_tests = ['uniaxial', 'biaxial', 'planar']
    notes = 'Good low-order baseline. Use at least two deformation modes when possible.'
    active = True
    parameter_specs = [
        ParameterSpec('C10', -10, 10, 0.05, 'stress', '-', ''),
        ParameterSpec('C01', -10, 10, 0.05, 'stress', '-', '')
    ]

    def W(self, stretches, p):
        I1,I2=invariants(stretches)
        return p['C10']*(I1-3)+p['C01']*(I2-3)

MODEL_CLASS = Model001
