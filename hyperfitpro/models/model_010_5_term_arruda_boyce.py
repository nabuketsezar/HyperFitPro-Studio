from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model010(HyperelasticModel):
    number = 10
    name = '5 term Arruda-Boyce'
    category = 'Network / finite chain extensibility'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'lambda_L should be constrained by maximum test stretch; good physical finite-chain model.'
    application_tags = ['limiting chain stretch', 'physical network']
    material_classes = ['rubber-like elastomer', 'network elastomer']
    complexity_level = 2
    family = 'Arruda-Boyce'
    equation_latex = 'W=\\mu\\sum_{i=1}^{5}\\frac{c_i}{\\lambda_L^{2i-2}}(I_1^i-3^i)'
    recommended_tests = ['uniaxial', 'biaxial', 'planar']
    notes = 'lambda_L should be constrained by maximum test stretch; good physical finite-chain model.'
    active = True
    parameter_specs = [
        ParameterSpec('mu', 0, 20, 0.1, 'positive_stress', '-', ''),
        ParameterSpec('lambda_L', 1.01, 20, 5, 'stretch_limit', '-', '')
    ]

    def W(self, stretches, p):
        I1,_=invariants(stretches); mu=p['mu']; L=p['lambda_L']; c=[0.5,1/20,11/1050,19/7050,519/673750]
        return mu*sum(c[i-1]/(L**(2*i-2))*(I1**i-3**i) for i in range(1,6))

MODEL_CLASS = Model010
