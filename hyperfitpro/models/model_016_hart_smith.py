from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model016(HyperelasticModel):
    number = 16
    name = 'Hart-Smith'
    category = 'Exponential / Hart-Smith family'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'C3 controls exponential stiffening; use stability scan after fit.'
    application_tags = ['exponential', 'I2 log']
    material_classes = ['rubber-like elastomer', 'soft tissue']
    complexity_level = 3
    family = 'Hart-Smith'
    equation_latex = 'W=\\frac{C_1e^{C_3(I_1-3)^2}}{2}+3C_2\\ln I_2'
    recommended_tests = ['uniaxial', 'biaxial', 'planar']
    notes = 'C3 controls exponential stiffening; use stability scan after fit.'
    active = True
    parameter_specs = [
        ParameterSpec('C1', 0, 20, 0.05, 'positive_stress', '-', ''),
        ParameterSpec('C2', -20, 20, 0.05, 'stress', '-', ''),
        ParameterSpec('C3', 1e-08, 100, 1, 'dimensionless', '-', '')
    ]

    def W(self, stretches, p):
        I1,I2=invariants(stretches); return 0.5*p['C1']*safe_exp(p['C3']*(I1-3)**2)+3*p['C2']*safe_log(I2)

MODEL_CLASS = Model016
