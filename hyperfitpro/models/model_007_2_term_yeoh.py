from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model007(HyperelasticModel):
    number = 7
    name = '2 term Yeoh'
    category = 'Reduced polynomial / Yeoh family'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'Can fit uniaxial well, but biaxial/planar are needed for transferability.'
    application_tags = ['I1-only', 'filled rubber candidate']
    material_classes = ['filled rubber', 'rubber-like elastomer']
    complexity_level = 2
    family = 'Yeoh reduced polynomial'
    equation_latex = 'W=C_{10}(I_1-3)+C_{20}(I_1-3)^2'
    recommended_tests = ['uniaxial', 'biaxial', 'planar']
    notes = 'Can fit uniaxial well, but biaxial/planar are needed for transferability.'
    active = True
    parameter_specs = [
        ParameterSpec('C10', 0, 20, 0.05, 'positive_stress', '-', ''),
        ParameterSpec('C20', -10, 10, 0, 'stress', '-', '')
    ]

    def W(self, stretches, p):
        I1,_=invariants(stretches); x=I1-3
        return p['C10']*x+p['C20']*x**2

MODEL_CLASS = Model007
