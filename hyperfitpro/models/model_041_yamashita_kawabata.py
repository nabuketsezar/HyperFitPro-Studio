from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model041(HyperelasticModel):
    number = 41
    name = 'Yamashita-Kawabata'
    category = 'Reduced polynomial / power-law'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'N controls nonlinear order; multi-start recommended.'
    application_tags = ['power law', 'reduced polynomial']
    material_classes = ['rubber-like elastomer', 'filled rubber']
    complexity_level = 3
    family = 'Yamashita-Kawabata'
    equation_latex = 'W=C_{10}(I_1-3)+\\frac{C_3}{N+1}(I_1-3)^{N+1}'
    recommended_tests = ['uniaxial', 'biaxial', 'planar']
    notes = 'N controls nonlinear order; multi-start recommended.'
    active = True
    parameter_specs = [
        ParameterSpec('C10', 0, 20, 0.05, 'positive_stress', '-', ''),
        ParameterSpec('C3', -20, 20, 0, 'stress', '-', ''),
        ParameterSpec('N', 0.05, 10, 2, 'dimensionless', '-', '')
    ]

    def W(self, stretches, p):
        I1,_=invariants(stretches); x=I1-3
        return p['C10']*x+p['C3']/(p['N']+1)*safe_pow(x,p['N']+1)

MODEL_CLASS = Model041
