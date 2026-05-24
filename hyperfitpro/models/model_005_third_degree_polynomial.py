from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model005(HyperelasticModel):
    number = 5
    name = 'Third degree polynomial'
    category = 'Invariant polynomial / general polynomial'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'Many-parameter polynomial; use multi-mode data and overfitting checks.'
    application_tags = ['many parameters', 'overfit risk']
    material_classes = ['rubber-like elastomer']
    complexity_level = 4
    family = 'Third-degree polynomial'
    equation_latex = 'W=C_{10}(I_1-3)+C_{01}(I_2-3)+C_{11}(I_1-3)(I_2-3)+C_{20}(I_1-3)^2+C_{02}(I_2-3)^2+C_{21}(I_1-3)^2(I_2-3)+C_{12}(I_1-3)(I_2-3)^2'
    recommended_tests = ['uniaxial', 'biaxial', 'planar', 'simple_shear']
    notes = 'Many-parameter polynomial; use multi-mode data and overfitting checks.'
    active = True
    parameter_specs = [
        ParameterSpec('C10', -10, 10, 0.05, 'stress', '-', ''),
        ParameterSpec('C01', -10, 10, 0.05, 'stress', '-', ''),
        ParameterSpec('C11', -10, 10, 0, 'stress', '-', ''),
        ParameterSpec('C20', -10, 10, 0, 'stress', '-', ''),
        ParameterSpec('C02', -10, 10, 0, 'stress', '-', ''),
        ParameterSpec('C21', -10, 10, 0, 'stress', '-', ''),
        ParameterSpec('C12', -10, 10, 0, 'stress', '-', '')
    ]

    def W(self, stretches, p):
        I1,I2=invariants(stretches); x=I1-3; y=I2-3
        return p['C10']*x+p['C01']*y+p['C11']*x*y+p['C20']*x**2+p['C02']*y**2+p['C21']*x**2*y+p['C12']*x*y**2

MODEL_CLASS = Model005
