from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model029(HyperelasticModel):
    number = 29
    name = '5 term Hartmann-Neff'
    category = 'Invariant polynomial / Hartmann-Neff family'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'Five coefficients; use stability scan after fitting.'
    application_tags = ['higher order', 'I1/I2']
    material_classes = ['rubber-like elastomer']
    complexity_level = 4
    family = 'Hartmann-Neff'
    equation_latex = 'W=\\alpha(I_1^3-3^3)+C_{10}(I_1-3)+C_{01}(I_2^{3/2}-3\\sqrt3)+C_{20}(I_1-3)^2+C_{02}(I_2^{3/2}-3\\sqrt3)^2'
    recommended_tests = ['uniaxial', 'biaxial', 'planar', 'simple_shear']
    notes = 'Five coefficients; use stability scan after fitting.'
    active = True
    parameter_specs = [
        ParameterSpec('alpha', -20, 20, 0, 'stress', '-', ''),
        ParameterSpec('C10', -20, 20, 0.05, 'stress', '-', ''),
        ParameterSpec('C01', -20, 20, 0.05, 'stress', '-', ''),
        ParameterSpec('C20', -20, 20, 0, 'stress', '-', ''),
        ParameterSpec('C02', -20, 20, 0, 'stress', '-', '')
    ]

    def W(self, stretches, p):
        I1,I2=invariants(stretches); x=I1-3; y=safe_pow(I2,1.5)-3*np.sqrt(3)
        return p['alpha']*(I1**3-27)+p['C10']*x+p['C01']*y+p['C20']*x*x+p['C02']*y*y

MODEL_CLASS = Model029
