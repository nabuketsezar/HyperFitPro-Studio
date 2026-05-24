from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model004(HyperelasticModel):
    number = 4
    name = '9 term Mooney-Rivlin'
    category = 'Invariant polynomial / Mooney-Rivlin family'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'Image row named 9-term; implemented exactly as supplied active-coefficient table variant.'
    application_tags = ['source-table variant', 'higher order']
    material_classes = ['rubber-like elastomer']
    complexity_level = 3
    family = 'Mooney-Rivlin polynomial'
    equation_latex = 'W=C_{10}(I_1-3)+C_{01}(I_2-3)+C_{20}(I_1-3)^2+C_{30}(I_1-3)^3+C_{11}(I_1-3)(I_2-3)'
    recommended_tests = ['uniaxial', 'biaxial', 'planar', 'simple_shear']
    notes = 'Image row named 9-term; implemented exactly as supplied active-coefficient table variant.'
    active = True
    parameter_specs = [
        ParameterSpec('C10', -10, 10, 0.05, 'stress', '-', ''),
        ParameterSpec('C01', -10, 10, 0.05, 'stress', '-', ''),
        ParameterSpec('C20', -10, 10, 0, 'stress', '-', ''),
        ParameterSpec('C30', -10, 10, 0, 'stress', '-', ''),
        ParameterSpec('C11', -10, 10, 0, 'stress', '-', '')
    ]

    def W(self, stretches, p):
        I1,I2=invariants(stretches); x=I1-3; y=I2-3
        return p['C10']*x+p['C01']*y+p['C20']*x**2+p['C30']*x**3+p['C11']*x*y

MODEL_CLASS = Model004
