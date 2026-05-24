from __future__ import annotations

"""Volumetric/compressible response helpers.

Most source equations in HyperFitPro are incompressible deviatoric energies.
This module supplies a consistent optional volumetric penalty that can be fitted
when volumetric data are supplied.

Supported volumetric laws:
* quadratic: U(J) = 0.5*K*(J-1)^2
* abaqus_D1: U(J) = (1/D1)*(J-1)^2, so K = 2/D1 for small strain

For GUI/optimizer integration, volumetric test data x can be either volumetric
engineering strain e_v=J-1 or J directly if values are mostly around 1. Stress is
interpreted as hydrostatic pressure positive in compression unless the user data
uses the opposite sign; the optimizer handles sign through the residual.
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass(frozen=True)
class VolumetricSpec:
    name: str = 'K'
    lower_factor: float = 0.1
    upper_factor: float = 1000.0
    initial_factor: float = 50.0
    unit: str = 'stress'
    description: str = 'Bulk modulus for optional compressible/volumetric response.'


def J_from_x(x):
    arr = np.asarray(x, dtype=float)
    # If data are around 1, interpret as J. Otherwise use J=1+e_v.
    if arr.size and np.nanmin(arr) > 0.2 and np.nanmax(arr) < 5.0 and np.nanmedian(arr) > 0.5:
        # Ambiguous; values between 0.8 and 1.2 are more likely J.
        if np.nanmin(arr) > 0.5 and np.nanmax(arr) < 1.5:
            return arr
    return 1.0 + arr


def volumetric_energy(J, K: float) -> np.ndarray:
    J = np.asarray(J, dtype=float)
    return 0.5 * float(K) * (J - 1.0) ** 2


def hydrostatic_pressure_from_K(J, K: float) -> np.ndarray:
    """Hydrostatic pressure p = dU/dJ for the quadratic volumetric law."""
    J = np.asarray(J, dtype=float)
    return float(K) * (J - 1.0)


def estimate_bulk_initial(volumetric_dataset, stress_scale: float) -> float:
    try:
        J = J_from_x(volumetric_dataset.x)
        y = np.asarray(volumetric_dataset.stress, dtype=float)
        dx = J - 1.0
        m = np.isfinite(dx) & np.isfinite(y) & (np.abs(dx) > 1e-9)
        if np.sum(m) >= 2:
            K = float(np.median(np.abs(y[m] / dx[m])))
            if np.isfinite(K) and K > 0:
                return K
    except Exception:
        pass
    return max(float(stress_scale) * 50.0, 1e-12)


def volumetric_bounds(stress_scale: float, volumetric_dataset=None) -> tuple[float, float, float]:
    base = estimate_bulk_initial(volumetric_dataset, stress_scale) if volumetric_dataset is not None else max(stress_scale*50.0, 1e-12)
    lo = max(base / 1.0e4, stress_scale * 1e-6, 1e-12)
    hi = max(base * 1.0e4, stress_scale * 1e6, lo*10)
    return float(lo), float(hi), float(min(max(base, lo), hi))
