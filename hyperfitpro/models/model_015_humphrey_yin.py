from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model015(HyperelasticModel):
    number = 15
    name = 'Humphrey-Yin'
    category = 'Exponential / soft-tissue type'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'Best used for strain-stiffening soft tissue style data.'
    application_tags = ['I1 exponential']
    material_classes = ['soft tissue']
    complexity_level = 2
    family = 'Humphrey-Yin'
    equation_latex = 'W=C_1(e^{C_2(I_1-3)}-1)'
    recommended_tests = ['uniaxial', 'biaxial', 'planar']
    notes = 'Best used for strain-stiffening soft tissue style data.'
    active = True
    parameter_specs = [
        ParameterSpec('C1', 0, 20, 0.05, 'positive_stress', '-', ''),
        ParameterSpec('C2', 1e-08, 100, 1, 'dimensionless', '-', '')
    ]

    def W(self, stretches, p):
        I1,_=invariants(stretches); return p['C1']*(safe_exp(p['C2']*(I1-3))-1)

MODEL_CLASS = Model015
