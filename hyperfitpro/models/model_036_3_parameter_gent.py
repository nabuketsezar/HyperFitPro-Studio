from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model036(HyperelasticModel):
    number = 36
    name = '3 parameter Gent'
    category = 'Network / finite extensibility with I2 blend'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'alpha blends limiting-chain and I2 behavior.'
    application_tags = ['Gent + I2 blend']
    material_classes = ['rubber-like elastomer']
    complexity_level = 3
    family = 'Three-parameter Gent'
    equation_latex = 'W=\\frac{\\mu}{2}[-\\alpha(I_L-3)\\ln(1-\\frac{I_1-3}{I_L-3})+(1-\\alpha)(I_2-3)]'
    recommended_tests = ['uniaxial', 'biaxial', 'planar', 'simple_shear']
    notes = 'alpha blends limiting-chain and I2 behavior.'
    active = True
    parameter_specs = [
        ParameterSpec('mu', 0, 20, 0.1, 'positive_stress', '-', ''),
        ParameterSpec('alpha', 0, 1, 0.8, 'dimensionless', '-', ''),
        ParameterSpec('I_L', 3.01, 1000, 50, 'invariant_limit', '-', '')
    ]

    def W(self, stretches, p):
        I1,I2=invariants(stretches); IL=p['I_L']; a=p['alpha']
        return 0.5*p['mu']*(-a*(IL-3)*safe_log(1-(I1-3)/(IL-3))+(1-a)*(I2-3))

MODEL_CLASS = Model036
