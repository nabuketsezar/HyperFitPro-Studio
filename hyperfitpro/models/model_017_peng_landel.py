from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_exp, safe_pow, BIG
import numpy as np

class Model017(HyperelasticModel):
    number = 17
    name = 'Peng-Landel'
    category = 'Principal-stretch / logarithmic spectral'
    preferred_optimizer = 'recommended_adaptive_bounds'
    reference_keys = ['user-supplied hyperelastic model table', 'standard hyperelastic calibration practice']
    source_equation_status = 'digitized from user supplied hyperelastic model table; metadata reviewed'
    limitation_notes = 'Validate extrapolation with the stability scan and independent test modes before FEA use.'
    parameterization_notes = 'Parameter bounds are optimizer search bounds. Stress-scaled parameters are multiplied by the detected test stress scale; limit parameters are domain-checked during evaluation.'
    calibration_notes = 'Single scale parameter; good spectral comparison model.'
    application_tags = ['principal stretch', 'log strain series']
    material_classes = ['rubber-like elastomer']
    complexity_level = 2
    family = 'Peng-Landel'
    equation_latex = 'W=C_1\\sum_{i=1}^{3}[\\lambda_i^2-1-\\ln\\lambda_i^2-\\frac16(\\ln\\lambda_i^2)^2+\\frac1{18}(\\ln\\lambda_i^2)^3-\\frac1{216}(\\ln\\lambda_i^2)^4]'
    recommended_tests = ['uniaxial', 'biaxial', 'planar', 'simple_shear']
    notes = 'Single scale parameter; good spectral comparison model.'
    active = True
    parameter_specs = [
        ParameterSpec('C1', 0, 20, 0.05, 'positive_stress', '-', '')
    ]

    def W(self, stretches, p):
        val=0.0
        for lam in stretches:
            q=safe_log(lam*lam)
            val += lam*lam-1-q-(1/6)*q**2+(1/18)*q**3-(1/216)*q**4
        return p['C1']*val

MODEL_CLASS = Model017
