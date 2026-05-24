from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model022(HyperelasticModel):
    number = 22
    name = 'Edwards-Vilgis'
    category = 'Network / finite extensibility'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'J_L must stay above maximum I1-3 domain requirement.'
    application_tags = ['limiting I1']
    material_classes = ['rubber-like elastomer', 'network elastomer']
    complexity_level = 2
    family = 'Edwards-Vilgis simplified'
    equation_latex = 'W=\\frac{\\mu}{2}[\\frac{(J_L+2)(J_L-3)(I_1-3)}{J_L(J_L-I_1+3)}+\\ln(1-\\frac{I_1-3}{J_L})]'
    recommended_tests = ['uniaxial', 'biaxial', 'planar']
    notes = 'J_L must stay above maximum I1-3 domain requirement.'
    active = True
    parameter_specs = [
        ParameterSpec('mu', 0, 20, 0.1, 'positive_stress', '-', ''),
        ParameterSpec('J_L', 3.01, 1000, 50, 'invariant_limit', '-', '')
    ]

    def W(self, stretches, p):
        I1,_=invariants(stretches); JL=p['J_L']; den=JL*(JL-I1+3)
        if abs(den)<1e-14: return BIG
        return 0.5*p['mu']*(((JL+2)*(JL-3)*(I1-3))/den + safe_log(1-(I1-3)/JL))

MODEL_CLASS = Model022
