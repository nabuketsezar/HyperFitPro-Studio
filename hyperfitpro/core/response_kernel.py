from __future__ import annotations

"""
HyperFitPro constitutive response kernel.

This module converts an incompressible isotropic strain-energy density W(lambda1,
lambda2,lambda3) into stress responses used for parameter identification.

Important conventions
---------------------
* Normal test x-values are engineering strain, lambda = 1 + x.
* simple_shear x-values are gamma.
* Returned normal stresses are nominal/engineering stresses P11 in the loaded
  direction, with traction-free out-of-plane directions handled by the
  incompressibility pressure term p.
* Returned simple-shear stress is the physical shear stress sigma12 for the
  incompressible simple-shear deformation.

The kernel is model-agnostic: each model only needs to supply W(stretches, p).
For robust commercial fitting, analytical closed-form expressions can still be
registered per model later, but this kernel is already mechanically correct for
all current isotropic incompressible model files.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List, Any
import math
import numpy as np

BIG = 1.0e60


@dataclass
class StressState:
    mode: str
    x: float
    stretches: Tuple[float, float, float]
    dW_dlambda: Tuple[float, float, float]
    pressure: float
    nominal: Tuple[float, float, float]
    cauchy: Tuple[float, float, float]
    scalar_response: float
    response_label: str


def mode_stretches(mode: str, x: float) -> Tuple[float, float, float]:
    mode = mode.lower().strip()
    if mode in ("simple_shear", "shear"):
        gamma = float(x)
        root = math.sqrt(1.0 + 0.25 * gamma * gamma)
        # Principal stretches of simple shear. Product is exactly one.
        return (root + 0.5 * gamma, root - 0.5 * gamma, 1.0)
    lam = 1.0 + float(x)
    if lam <= 0.0 or not np.isfinite(lam):
        return (np.nan, np.nan, np.nan)
    if mode in ("uniaxial", "uniaxial_tension", "uniaxial_compression"):
        lat = lam ** -0.5
        return (lam, lat, lat)
    if mode in ("biaxial", "equibiaxial", "biaxial_tension"):
        return (lam, lam, lam ** -2.0)
    if mode in ("planar", "pure_shear", "planar_tension"):
        return (lam, 1.0, lam ** -1.0)
    raise ValueError(f"Unknown test mode: {mode}")


def _is_valid_energy(y: float) -> bool:
    try:
        return bool(np.isfinite(y) and abs(float(y)) < BIG / 1000.0)
    except Exception:
        return False


def dW_dlambda_numeric(W_func: Callable[[Tuple[float, float, float], Dict[str, float]], float],
                       stretches: Tuple[float, float, float],
                       params: Dict[str, float],
                       rel_step: float = 2.5e-6) -> Tuple[float, float, float]:
    """Robust partial derivative of W with respect to each principal stretch.

    The derivative is taken with the other two principal stretches held fixed,
    which is the correct derivative used in the principal-stress expression
    sigma_i = lambda_i * dW/dlambda_i - p for incompressible isotropic materials.
    """
    lams = np.asarray(stretches, dtype=float)
    if not np.all(np.isfinite(lams)) or np.any(lams <= 0.0):
        return (np.nan, np.nan, np.nan)

    out: List[float] = []
    for i in range(3):
        li = float(lams[i])
        h = rel_step * max(1.0, abs(li))
        h = min(h, 0.20 * li) if li > 0 else h
        if h <= 0 or not np.isfinite(h):
            out.append(np.nan); continue

        lp = lams.copy(); lm = lams.copy()
        lp[i] = li + h
        lm[i] = li - h
        # Keep stretches positive. If central is impossible, use one-sided.
        if lm[i] <= 0:
            f0 = W_func(tuple(lams), params)
            fp = W_func(tuple(lp), params)
            if _is_valid_energy(f0) and _is_valid_energy(fp):
                out.append(float((fp - f0) / h))
            else:
                out.append(np.nan)
            continue

        fp = W_func(tuple(lp), params)
        fm = W_func(tuple(lm), params)
        if _is_valid_energy(fp) and _is_valid_energy(fm):
            out.append(float((fp - fm) / (2.0 * h)))
            continue

        # Adaptive fallback: reduce step and retry.
        val = np.nan
        for factor in (0.2, 0.05, 0.01):
            hh = h * factor
            lp = lams.copy(); lm = lams.copy()
            lp[i] = li + hh; lm[i] = max(li - hh, li * 0.5)
            fp = W_func(tuple(lp), params); fm = W_func(tuple(lm), params)
            if _is_valid_energy(fp) and _is_valid_energy(fm) and lp[i] != lm[i]:
                val = float((fp - fm) / (lp[i] - lm[i]))
                break
        out.append(val)
    return tuple(out)  # type: ignore[return-value]


def principal_stress_state(W_func: Callable[[Tuple[float, float, float], Dict[str, float]], float],
                           mode: str,
                           x: float,
                           params: Dict[str, float]) -> StressState:
    """Return mechanically consistent scalar response and full stress state."""
    m = mode.lower().strip()
    l1, l2, l3 = mode_stretches(m, x)
    d1, d2, d3 = dW_dlambda_numeric(W_func, (l1, l2, l3), params)
    if not np.all(np.isfinite([l1, l2, l3, d1, d2, d3])):
        nan3 = (np.nan, np.nan, np.nan)
        return StressState(m, float(x), (l1, l2, l3), (d1, d2, d3), np.nan, nan3, nan3, np.nan, 'invalid')

    # Incompressibility pressure from the traction-free principal direction.
    # Nominal stress P_i = dW/dlambda_i - p/lambda_i.
    # Cauchy stress sigma_i = lambda_i*dW/dlambda_i - p.
    if m in ("uniaxial", "uniaxial_tension", "uniaxial_compression"):
        pressure = l2 * d2  # lateral directions are traction free
        nominal = (d1 - pressure / l1, d2 - pressure / l2, d3 - pressure / l3)
        cauchy = (l1 * d1 - pressure, l2 * d2 - pressure, l3 * d3 - pressure)
        scalar = nominal[0]
        label = 'P11_uniaxial_nominal'
    elif m in ("biaxial", "equibiaxial", "biaxial_tension"):
        pressure = l3 * d3  # thickness direction is traction free
        nominal = (d1 - pressure / l1, d2 - pressure / l2, d3 - pressure / l3)
        cauchy = (l1 * d1 - pressure, l2 * d2 - pressure, l3 * d3 - pressure)
        scalar = 0.5 * (nominal[0] + nominal[1])
        label = 'P11_equal_biaxial_nominal'
    elif m in ("planar", "pure_shear", "planar_tension"):
        pressure = l3 * d3  # out-of-plane is traction free; width direction constrained
        nominal = (d1 - pressure / l1, d2 - pressure / l2, d3 - pressure / l3)
        cauchy = (l1 * d1 - pressure, l2 * d2 - pressure, l3 * d3 - pressure)
        scalar = nominal[0]
        label = 'P11_planar_nominal'
    elif m in ("simple_shear", "shear"):
        # In principal axes, p cancels out from sigma1-sigma2. Transform back to
        # lab coordinates. For simple shear, sin(2theta)=2/sqrt(gamma^2+4), so
        # sigma12=(sigma1-sigma2)/sqrt(gamma^2+4).
        gamma = float(x)
        pressure = 0.0  # arbitrary for scalar shear response because it cancels
        cauchy_principal = (l1*d1 - pressure, l2*d2 - pressure, l3*d3 - pressure)
        shear = (cauchy_principal[0] - cauchy_principal[1]) / math.sqrt(gamma*gamma + 4.0)
        nominal = (np.nan, np.nan, np.nan)
        cauchy = cauchy_principal
        scalar = shear
        label = 'sigma12_simple_shear'
    else:
        raise ValueError(f"Unknown test mode: {mode}")

    if not np.isfinite(scalar) or abs(float(scalar)) > BIG / 1000.0:
        scalar = np.nan
    return StressState(m, float(x), (l1, l2, l3), (d1, d2, d3), float(pressure),
                       tuple(float(v) for v in nominal), tuple(float(v) for v in cauchy),
                       float(scalar), label)


def predict_response(W_func, mode: str, xs: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    return np.asarray([principal_stress_state(W_func, mode, float(x), params).scalar_response for x in xs], dtype=float)


def tangent_numeric(W_func, mode: str, x: float, params: Dict[str, float]) -> float:
    """Numerical tangent d(response)/dx along the selected test path."""
    h = 1e-5 * max(1.0, abs(float(x)))
    if mode.lower() not in ('simple_shear', 'shear') and x - h <= -0.999:
        h = max(1e-7, 0.5*(x+0.999))
    yp = principal_stress_state(W_func, mode, x+h, params).scalar_response
    ym = principal_stress_state(W_func, mode, x-h, params).scalar_response
    if np.isfinite(yp) and np.isfinite(ym):
        return float((yp-ym)/(2*h))
    y0 = principal_stress_state(W_func, mode, x, params).scalar_response
    yp = principal_stress_state(W_func, mode, x+h, params).scalar_response
    if np.isfinite(y0) and np.isfinite(yp):
        return float((yp-y0)/h)
    return float('nan')
