from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model034(HyperelasticModel):
    number = 34
    name = 'Horgan-Saccomandi'
    category = 'Network / finite extensibility'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'Requires admissible logarithm argument; domain controlled in stability scan.'
    application_tags = ['limiting chain', 'I1/I2 coupled']
    material_classes = ['rubber-like elastomer']
    complexity_level = 2
    family = 'Horgan-Saccomandi'
    equation_latex = 'W=-\\frac{\\mu}{2}J_L\\ln\\left(\\frac{J_L^3-J_L^2I_1+J_LI_2-1}{(J_L-1)^3}\\right)'
    recommended_tests = ['uniaxial', 'biaxial', 'planar']
    notes = 'Requires admissible logarithm argument; domain controlled in stability scan.'
    active = True
    parameter_specs = [
        ParameterSpec('mu', 0, 20, 0.1, 'positive_stress', '-', ''),
        ParameterSpec('J_L', 3.01, 1000, 50, 'invariant_limit', '-', '')
    ]

    def W(self, stretches, p):
        I1,I2=invariants(stretches); JL=p['J_L']; num=JL**3-JL**2*I1+JL*I2-1; den=(JL-1)**3
        return -0.5*p['mu']*JL*safe_log(num/den)

MODEL_CLASS = Model034
