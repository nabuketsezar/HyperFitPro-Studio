from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model040(HyperelasticModel):
    number = 40
    name = 'Takamizawa-Hayashi'
    category = 'Network / finite extensibility'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'J_L must be large enough to keep log argument positive.'
    application_tags = ['limiting I1']
    material_classes = ['soft tissue', 'rubber-like elastomer']
    complexity_level = 2
    family = 'Takamizawa-Hayashi'
    equation_latex = 'W=-c\\ln[1-(\\frac{I_1-2}{J_L})^2]'
    recommended_tests = ['uniaxial', 'biaxial', 'planar']
    notes = 'J_L must be large enough to keep log argument positive.'
    active = True
    parameter_specs = [
        ParameterSpec('c', 0, 20, 0.05, 'positive_stress', '-', ''),
        ParameterSpec('J_L', 3.01, 1000, 50, 'invariant_limit', '-', '')
    ]

    def W(self, stretches, p):
        I1,_=invariants(stretches); return -p['c']*safe_log(1-((I1-2)/p['J_L'])**2)

MODEL_CLASS = Model040
