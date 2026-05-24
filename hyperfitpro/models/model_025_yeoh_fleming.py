from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model025(HyperelasticModel):
    number = 25
    name = 'Yeoh-Fleming'
    category = 'Hybrid finite extensibility / Yeoh-Fleming'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'Combine exponential and limiting-chain terms; use wide-strain data.'
    application_tags = ['exponential + limiting chain']
    material_classes = ['rubber-like elastomer', 'filled rubber']
    complexity_level = 4
    family = 'Yeoh-Fleming'
    equation_latex = 'W=\\frac{A}{B}(1-e^{-B(I_1-3)})-C_{10}(I_L-3)\\ln(1-\\frac{I_1-3}{I_L-3})'
    recommended_tests = ['uniaxial', 'biaxial', 'planar']
    notes = 'Combine exponential and limiting-chain terms; use wide-strain data.'
    active = True
    parameter_specs = [
        ParameterSpec('A', 0, 20, 0.05, 'positive_stress', '-', ''),
        ParameterSpec('B', 1e-08, 100, 1, 'dimensionless', '-', ''),
        ParameterSpec('C10', 0, 20, 0.05, 'positive_stress', '-', ''),
        ParameterSpec('I_L', 3.01, 1000, 50, 'invariant_limit', '-', '')
    ]

    def W(self, stretches, p):
        I1,_=invariants(stretches); x=I1-3; IL=p['I_L']
        return p['A']/p['B']*(1-safe_exp(-p['B']*x))-p['C10']*(IL-3)*safe_log(1-x/(IL-3))

MODEL_CLASS = Model025
