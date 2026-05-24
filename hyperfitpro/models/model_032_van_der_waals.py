from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants, safe_log, safe_pow, BIG
import numpy as np

class Model032(HyperelasticModel):
    number = 32
    name = 'Van Der Waals'
    category = 'Van der Waals / limiting-chain interaction'
    family = 'Limiting-chain interaction model'
    equation_latex = r'W=\mu\left[-(\lambda_m^2-3)\left(\ln(1-\eta)+\eta\right)-\frac{2a}{3}\left(\frac{I_1-3}{2}\right)^{3/2}\right],\quad \eta=\sqrt{\frac{(1-\beta)I_1+\beta I_2-3}{\lambda_m^2-3}}'
    recommended_tests = ['uniaxial', 'biaxial', 'planar', 'simple_shear']
    material_classes = ['filled rubber', 'rubber-like elastomer', 'finite extensibility']
    application_tags = ['large strain', 'limiting chain stretch', 'I1/I2 interaction', 'filled elastomer candidate']
    calibration_notes = 'Prefer at least uniaxial + biaxial/planar data over a wide strain range. The beta parameter is weakly identifiable from uniaxial-only data.'
    parameterization_notes = 'lambda_m must keep eta < 1 over the complete calibration domain. beta is constrained to [0, 1]. a is dimensionless.'
    limitation_notes = 'Logarithmic singularity as eta approaches 1; the kernel returns a large penalty outside the admissible domain.'
    source_equation_status = 'row 32 supplied by user image and implemented'
    reference_keys = ['user-supplied row 32 equation', 'standard finite-extensibility calibration practice']
    complexity_level = 4
    minimum_recommended_points_per_parameter = 12
    preferred_optimizer = 'recommended_adaptive_bounds'
    active = True
    parameter_specs = [
        ParameterSpec('mu', 0, 20, 0.1, 'positive_stress', '-', 'Small-strain stress scale / shear-like modulus parameter.'),
        ParameterSpec('lambda_m', 1.74, 30, 5.0, 'stretch_limit', '-', 'Limiting chain stretch parameter; keep lambda_m^2-3 positive and eta<1.'),
        ParameterSpec('a', -10, 10, 0.0, 'dimensionless', '-', 'Dimensionless interaction/non-Gaussian correction coefficient.'),
        ParameterSpec('beta', 0.0, 1.0, 0.5, 'dimensionless', '-', 'Invariant mixing coefficient between I1 and I2.'),
    ]

    def W(self, stretches, p):
        I1, I2 = invariants(stretches)
        mu = p['mu']; lm = p['lambda_m']; a = p['a']; beta = p['beta']
        den = lm*lm - 3.0
        if den <= 1e-12:
            return BIG
        rad = ((1.0-beta)*I1 + beta*I2 - 3.0) / den
        if (not np.isfinite(rad)) or rad < -1e-12:
            return BIG
        eta = np.sqrt(max(rad, 0.0))
        if eta >= 1.0 or not np.isfinite(eta):
            return BIG
        z = max((I1 - 3.0) / 2.0, 0.0)
        term1 = -den * (safe_log(1.0 - eta) + eta)
        term2 = -(2.0*a/3.0) * safe_pow(z, 1.5)
        val = mu * (term1 + term2)
        return val if np.isfinite(val) else BIG

MODEL_CLASS = Model032
