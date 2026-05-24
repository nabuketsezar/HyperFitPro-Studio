from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model006(HyperelasticModel):
    number = 6
    name = 'Neo-Hookean'
    category = 'Neo-Hookean / one-parameter network'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'Best first baseline and sanity check; one parameter only.'
    application_tags = ['baseline', 'small/moderate strain']
    material_classes = ['rubber-like elastomer', 'soft tissue first-pass']
    complexity_level = 1
    family = 'Neo-Hookean baseline'
    equation_latex = 'W=\\frac{\\mu}{2}(I_1-3)'
    recommended_tests = ['uniaxial', 'biaxial', 'planar']
    notes = 'Best first baseline and sanity check; one parameter only.'
    active = True
    parameter_specs = [
        ParameterSpec('mu', 0, 20, 0.1, 'positive_stress', '-', '')
    ]

    def W(self, stretches, p):
        I1,_=invariants(stretches)
        return 0.5*p['mu']*(I1-3)

MODEL_CLASS = Model006
