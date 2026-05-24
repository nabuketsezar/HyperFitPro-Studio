from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model019(HyperelasticModel):
    number = 19
    name = 'Martins'
    category = 'Fiber-influenced / exponential anisotropic approximation'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'lambda_f is approximated internally as max principal stretch; true anisotropic fiber direction is not yet modeled.'
    application_tags = ['fiber stretch proxy', 'exponential']
    material_classes = ['soft tissue', 'fiber-reinforced approximation']
    complexity_level = 4
    family = 'Martins'
    equation_latex = 'W=C_1(e^{C_2(I_1-3)}-1)+C_3(e^{C_4(\\lambda_f-1)^2}-1)'
    recommended_tests = ['uniaxial', 'biaxial', 'planar']
    notes = 'lambda_f is approximated internally as max principal stretch; true anisotropic fiber direction is not yet modeled.'
    active = True
    parameter_specs = [
        ParameterSpec('C1', 0, 20, 0.05, 'positive_stress', '-', ''),
        ParameterSpec('C2', 1e-08, 100, 1, 'dimensionless', '-', ''),
        ParameterSpec('C3', 0, 20, 0.05, 'positive_stress', '-', ''),
        ParameterSpec('C4', 1e-08, 100, 1, 'dimensionless', '-', '')
    ]

    def W(self, stretches, p):
        I1,_=invariants(stretches); lf=max(stretches)
        return p['C1']*(safe_exp(p['C2']*(I1-3))-1)+p['C3']*(safe_exp(p['C4']*(lf-1)**2)-1)

MODEL_CLASS = Model019
