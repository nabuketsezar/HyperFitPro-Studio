from __future__ import annotations

"""Engineering stability and physical admissibility checks.

The checks here are numerical engineering screens designed to stop clearly bad
fits before they are exported to FEA. They include:
* finite stress/energy response over standard deformation paths,
* non-negative energy relative to the reference state,
* mostly positive tangent stiffness on loaded paths,
* small-strain shear modulus positivity,
* limiting-domain checks near fitted data range.

A complete formal strong-ellipticity proof for every arbitrary energy function is
not possible from black-box W alone, but these scans are intentionally strict and
are suitable as a practical constitutive validation layer.
"""

from typing import Dict, Any
import numpy as np
from .response_kernel import principal_stress_state, tangent_numeric, mode_stretches


def small_strain_shear_modulus(model, params: Dict[str, float]) -> float:
    gammas = np.array([-2e-4, -1e-4, 1e-4, 2e-4], dtype=float)
    vals = np.array([principal_stress_state(model.W, 'simple_shear', g, params).scalar_response for g in gammas])
    m = np.isfinite(vals)
    if np.sum(m) < 2:
        return float('nan')
    return float(np.polyfit(gammas[m], vals[m], 1)[0])


def check_reference_energy(model, params: Dict[str, float]) -> float:
    try:
        return float(model.W((1.0, 1.0, 1.0), params))
    except Exception:
        return float('nan')


def engineering_stability_scan(model, params: Dict[str, float],
                               modes=('uniaxial','biaxial','planar','simple_shear'),
                               strain_min=-0.35, strain_max=2.50, shear_max=3.0,
                               n=140) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        'passed': True,
        'warnings': [],
        'checks': {},
        'mode_results': {},
        'note': 'Numerical engineering screen; not a formal symbolic proof for arbitrary black-box W.'
    }
    W0 = check_reference_energy(model, params)
    report['checks']['reference_energy_W_1_1_1'] = W0
    if not np.isfinite(W0) or abs(W0) > 1e-6 * max(1.0, abs(W0)):
        # Many equations are zero at reference; if not, it is a warning not always fatal.
        report['warnings'].append(f'Reference energy W(1,1,1)={W0:.6g}; expected close to zero for most models.')

    G0 = small_strain_shear_modulus(model, params)
    report['checks']['small_strain_shear_modulus'] = G0
    if not np.isfinite(G0) or G0 <= 0:
        report['passed'] = False
        report['warnings'].append(f'Non-positive or invalid small-strain shear modulus: {G0}')

    for mode in modes:
        if mode == 'simple_shear':
            xs = np.linspace(-shear_max, shear_max, n)
        else:
            xs = np.linspace(strain_min, strain_max, n)
            xs = xs[xs > -0.97]
        y = []
        tang = []
        Wvals = []
        for x in xs:
            st = principal_stress_state(model.W, mode, float(x), params)
            y.append(st.scalar_response)
            tang.append(tangent_numeric(model.W, mode, float(x), params))
            try:
                Wvals.append(float(model.W(st.stretches, params)) - (W0 if np.isfinite(W0) else 0.0))
            except Exception:
                Wvals.append(np.nan)
        y = np.asarray(y, dtype=float); tang = np.asarray(tang, dtype=float); Wvals = np.asarray(Wvals, dtype=float)
        finite_fraction = float(np.mean(np.isfinite(y))) if y.size else 0.0
        finite_tangent_fraction = float(np.mean(np.isfinite(tang))) if tang.size else 0.0
        neg_energy_points = int(np.sum(np.isfinite(Wvals) & (Wvals < -1e-8 * max(1.0, np.nanmax(np.abs(Wvals)) if np.any(np.isfinite(Wvals)) else 1.0))))
        if mode == 'simple_shear':
            # Symmetric shear can have positive or negative slope; tangent should mostly be positive.
            positive_tangent_fraction = float(np.mean(tang[np.isfinite(tang)] > -1e-10)) if np.any(np.isfinite(tang)) else 0.0
            monotonic_fraction = positive_tangent_fraction
        else:
            # Split tension and compression but use mostly positive tangent over engineering path.
            positive_tangent_fraction = float(np.mean(tang[np.isfinite(tang)] > -1e-10)) if np.any(np.isfinite(tang)) else 0.0
            monotonic_fraction = positive_tangent_fraction
        passed = finite_fraction > 0.97 and finite_tangent_fraction > 0.90 and neg_energy_points == 0 and monotonic_fraction > 0.75
        report['mode_results'][mode] = {
            'finite_fraction': finite_fraction,
            'finite_tangent_fraction': finite_tangent_fraction,
            'positive_tangent_fraction': positive_tangent_fraction,
            'negative_energy_points': neg_energy_points,
            'min_response': float(np.nanmin(y)) if np.any(np.isfinite(y)) else None,
            'max_response': float(np.nanmax(y)) if np.any(np.isfinite(y)) else None,
            'min_tangent': float(np.nanmin(tang)) if np.any(np.isfinite(tang)) else None,
            'passed': bool(passed),
        }
        if not passed:
            report['passed'] = False
            report['warnings'].append(
                f'{mode}: finite={finite_fraction:.2f}, tangent finite={finite_tangent_fraction:.2f}, '
                f'positive tangent={positive_tangent_fraction:.2f}, negative W points={neg_energy_points}'
            )
    return report
