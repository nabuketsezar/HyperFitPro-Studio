from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model011(HyperelasticModel):
    number = 11
    name = 'Gent'
    category = 'Network / finite chain extensibility'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'I_L must exceed the maximum sampled I1. Strongly penalized near the singularity.'
    application_tags = ['limiting I1', 'strain stiffening']
    material_classes = ['rubber-like elastomer']
    complexity_level = 2
    family = 'Gent'
    equation_latex = 'W=-\\frac{\\mu}{2}(I_L-3)\\ln\\left(1-\\frac{I_1-3}{I_L-3}\\right)'
    recommended_tests = ['uniaxial', 'biaxial', 'planar']
    notes = 'I_L must exceed the maximum sampled I1. Strongly penalized near the singularity.'
    active = True
    parameter_specs = [
        ParameterSpec('mu', 0, 20, 0.1, 'positive_stress', '-', ''),
        ParameterSpec('I_L', 3.01, 1000, 50, 'invariant_limit', '-', '')
    ]

    def W(self, stretches, p):
        I1,_=invariants(stretches); IL=p['I_L']; z=1-(I1-3)/(IL-3)
        return -0.5*p['mu']*(IL-3)*safe_log(z)

MODEL_CLASS = Model011
