from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model024(HyperelasticModel):
    number = 24
    name = 'Gent-Thomas'
    category = 'Hybrid invariant / Gent-Thomas'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'I2 log term benefits from planar/shear or biaxial data.'
    application_tags = ['I1 + log I2']
    material_classes = ['rubber-like elastomer']
    complexity_level = 2
    family = 'Gent-Thomas'
    equation_latex = 'W=C_1(I_1-3)+3C_2\\ln I_2'
    recommended_tests = ['uniaxial', 'biaxial', 'planar', 'simple_shear']
    notes = 'I2 log term benefits from planar/shear or biaxial data.'
    active = True
    parameter_specs = [
        ParameterSpec('C1', 0, 20, 0.05, 'positive_stress', '-', ''),
        ParameterSpec('C2', -20, 20, 0, 'stress', '-', '')
    ]

    def W(self, stretches, p):
        I1,I2=invariants(stretches); return p['C1']*(I1-3)+3*p['C2']*safe_log(I2)

MODEL_CLASS = Model024
