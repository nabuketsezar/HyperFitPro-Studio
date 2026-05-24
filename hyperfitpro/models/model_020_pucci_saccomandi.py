from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model020(HyperelasticModel):
    number = 20
    name = 'Pucci-Saccomandi'
    category = 'Network / finite extensibility with I2 correction'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'Needs I2-sensitive data such as planar/shear for C2 identification.'
    application_tags = ['Gent-type', 'I2 correction']
    material_classes = ['rubber-like elastomer']
    complexity_level = 3
    family = 'Pucci-Saccomandi'
    equation_latex = 'W=-\\frac12\\mu J_L\\ln(1-\\frac{I_1-3}{J_L})+C_2\\ln(\\frac13 I_2)'
    recommended_tests = ['uniaxial', 'biaxial', 'planar', 'simple_shear']
    notes = 'Needs I2-sensitive data such as planar/shear for C2 identification.'
    active = True
    parameter_specs = [
        ParameterSpec('mu', 0, 20, 0.1, 'positive_stress', '-', ''),
        ParameterSpec('J_L', 3.01, 1000, 50, 'invariant_limit', '-', ''),
        ParameterSpec('C2', -20, 20, 0, 'stress', '-', '')
    ]

    def W(self, stretches, p):
        I1,I2=invariants(stretches); JL=p['J_L']
        return -0.5*p['mu']*JL*safe_log(1-(I1-3)/JL)+p['C2']*safe_log(I2/3)

MODEL_CLASS = Model020
