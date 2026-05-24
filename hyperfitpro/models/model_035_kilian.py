from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model035(HyperelasticModel):
    number = 35
    name = 'Kilian'
    category = 'Network / finite extensibility'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'Logarithmic finite-extensibility form; J_L must exceed sampled domain.'
    application_tags = ['limiting I1']
    material_classes = ['rubber-like elastomer']
    complexity_level = 2
    family = 'Kilian'
    equation_latex = 'W=-\\mu J_L[\\ln(1-\\sqrt{\\frac{I_1-3}{J_L}})+\\sqrt{\\frac{I_1-3}{J_L}}]'
    recommended_tests = ['uniaxial', 'biaxial', 'planar']
    notes = 'Logarithmic finite-extensibility form; J_L must exceed sampled domain.'
    active = True
    parameter_specs = [
        ParameterSpec('mu', 0, 20, 0.1, 'positive_stress', '-', ''),
        ParameterSpec('J_L', 3.01, 1000, 50, 'invariant_limit', '-', '')
    ]

    def W(self, stretches, p):
        I1,_=invariants(stretches); JL=p['J_L']; q=(I1-3)/JL
        if q < 0: return BIG
        s=np.sqrt(q)
        return -p['mu']*JL*(safe_log(1-s)+s)

MODEL_CLASS = Model035
