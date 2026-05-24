from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model042(HyperelasticModel):
    number = 42
    name = 'Amin'
    category = 'Reduced polynomial / power-law'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'M and N can be correlated; use adaptive bounds and model comparison.'
    application_tags = ['two power-law terms']
    material_classes = ['rubber-like elastomer', 'filled rubber']
    complexity_level = 4
    family = 'Amin'
    equation_latex = 'W=C_{10}(I_1-3)+\\frac{C_3}{N+1}(I_1-3)^{N+1}+\\frac{C_4}{M+1}(I_1-3)^{M+1}'
    recommended_tests = ['uniaxial', 'biaxial', 'planar', 'simple_shear']
    notes = 'M and N can be correlated; use adaptive bounds and model comparison.'
    active = True
    parameter_specs = [
        ParameterSpec('C10', 0, 20, 0.05, 'positive_stress', '-', ''),
        ParameterSpec('C3', -20, 20, 0, 'stress', '-', ''),
        ParameterSpec('N', 0.05, 10, 2, 'dimensionless', '-', ''),
        ParameterSpec('C4', -20, 20, 0, 'stress', '-', ''),
        ParameterSpec('M', 0.05, 10, 3, 'dimensionless', '-', '')
    ]

    def W(self, stretches, p):
        I1,_=invariants(stretches); x=I1-3
        return p['C10']*x+p['C3']/(p['N']+1)*safe_pow(x,p['N']+1)+p['C4']/(p['M']+1)*safe_pow(x,p['M']+1)

MODEL_CLASS = Model042
