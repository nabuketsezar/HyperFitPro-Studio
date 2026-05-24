from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Iterable, Any
import math
import re
import numpy as np

from .response_kernel import (
    mode_stretches as kernel_mode_stretches,
    principal_stress_state,
    predict_response as kernel_predict_response,
)
from .stability import engineering_stability_scan

BIG = 1.0e60

@dataclass(frozen=True)
class ParameterSpec:
    """Parameter definition used by GUI and optimizer.

    Bounds are optimizer search limits, not universal physical constants.
    Bounds with scale='stress' or 'positive_stress' are multiplied by the detected
    test stress scale. This keeps the same model file usable for MPa, Pa or psi.
    """
    name: str
    lower: float
    upper: float
    initial: float | None = None
    scale: str = "dimensionless"  # dimensionless, stress, positive_stress, invariant_limit, stretch_limit
    unit: str = "-"
    description: str = ""

    def resolved(self, stress_scale: float, max_I1: float) -> Tuple[float, float, float]:
        S = max(float(abs(stress_scale)), 1.0e-12)
        if self.scale == "stress":
            lo, hi = self.lower * S, self.upper * S
            x0 = 0.0 if self.initial is None else self.initial * S
        elif self.scale == "positive_stress":
            lo, hi = max(self.lower * S, 1.0e-12 * S), max(self.upper * S, 1.0e-9 * S)
            x0 = max((0.05 * S if self.initial is None else self.initial * S), lo)
        elif self.scale == "invariant_limit":
            lo = max(self.lower, max_I1 + 1.0e-6)
            hi = max(self.upper, lo * 1.05)
            x0 = max((lo + hi) * 0.5 if self.initial is None else self.initial, lo)
        elif self.scale == "stretch_limit":
            lo = max(self.lower, 1.000001)
            hi = max(self.upper, lo * 1.05)
            x0 = max((lo + hi) * 0.5 if self.initial is None else self.initial, lo)
        else:
            lo, hi = self.lower, self.upper
            if self.initial is not None:
                x0 = self.initial
            else:
                if lo > 0:
                    x0 = math.sqrt(lo * hi) if hi / lo < 1e12 else lo * 10
                elif hi < 0:
                    x0 = -math.sqrt(abs(lo * hi))
                else:
                    x0 = 0.0
        if not np.isfinite(x0) or not (lo < x0 < hi):
            x0 = lo + 0.5 * (hi - lo)
        return float(lo), float(hi), float(x0)


def invariants(stretches: Iterable[float]) -> Tuple[float, float]:
    l1, l2, l3 = [float(x) for x in stretches]
    a, b, c = l1*l1, l2*l2, l3*l3
    return a + b + c, a*b + b*c + c*a


def safe_log(x: float) -> float:
    if not np.isfinite(x) or x <= 0.0:
        return BIG
    return float(np.log(x))


def safe_exp(x: float) -> float:
    if not np.isfinite(x):
        return BIG
    if x > 700:
        return BIG
    if x < -745:
        return 0.0
    return float(np.exp(x))


def safe_pow(x: float, a: float) -> float:
    if not np.isfinite(x) or not np.isfinite(a):
        return BIG
    if x < 0 and abs(a - round(a)) > 1e-12:
        return BIG
    try:
        y = float(np.power(x, a))
    except Exception:
        return BIG
    if not np.isfinite(y):
        return BIG
    return y


def mode_stretches(mode: str, x: float) -> Tuple[float, float, float]:
    """Return principal stretches for a supported deformation path.

    This public wrapper delegates to the constitutive response kernel so all
    modules use the same deformation definitions.
    """
    return kernel_mode_stretches(mode, x)


def max_invariant_from_test_inputs(modes_and_x: Iterable[Tuple[str, np.ndarray]]) -> float:
    val = 3.0
    for mode, xs in modes_and_x:
        if mode == 'volumetric':
            continue
        for x in np.asarray(xs, dtype=float):
            try:
                I1, _ = invariants(mode_stretches(mode, float(x)))
                if np.isfinite(I1): val = max(val, I1)
            except Exception:
                pass
    return float(val)


def _replace_latex_parameters(equation_latex: str, params: Dict[str, float]) -> str:
    out = equation_latex or ''
    # Replace longest names first to avoid C1 inside C10.
    for k in sorted(params, key=len, reverse=True):
        val = params[k]
        patterns = [
            k,
            k.replace('_',''),
            k.replace('_','_{') + ('}' if '_' in k else ''),
        ]
        # C10 -> C_{10}; C_10 -> C_{10}
        if re.match(r'^[A-Za-z]+\d+$', k):
            letters = ''.join(ch for ch in k if ch.isalpha())
            digits = ''.join(ch for ch in k if ch.isdigit())
            patterns.append(f"{letters}_{{{digits}}}")
        if '_' in k:
            a,b=k.split('_',1); patterns.append(f"{a}_{{{b}}}")
        repl = f"({val:.8g})"
        for pat in sorted(set(patterns), key=len, reverse=True):
            if pat:
                out = out.replace(pat, repl)
    return out


class HyperelasticModel:
    number: int = -1
    name: str = "Base"
    category: str = "Uncategorized"
    equation_latex: str = ""
    parameter_specs: List[ParameterSpec] = []
    recommended_tests: List[str] = ["uniaxial", "biaxial", "planar"]
    notes: str = ""
    active: bool = True

    # Model-library metadata. Informational fields used by GUI, reports,
    # catalog export and automatic model recommendation logic.
    family: str = ""
    application_tags: List[str] = []
    material_classes: List[str] = []
    calibration_notes: str = ""
    parameterization_notes: str = ""
    limitation_notes: str = ""
    source_equation_status: str = "digitized_from_user_supplied_table; engineering checked"
    reference_keys: List[str] = []
    complexity_level: int = 1
    minimum_recommended_points_per_parameter: int = 8
    preferred_optimizer: str = "recommended_adaptive_bounds"

    def W(self, stretches: Tuple[float, float, float], p: Dict[str, float]) -> float:
        raise NotImplementedError

    def parameter_names(self) -> List[str]:
        return [s.name for s in self.parameter_specs]

    def resolved_bounds(self, stress_scale: float, max_I1: float, overrides: Dict[str, Tuple[float,float,float]]|None=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lower, upper, x0 = [], [], []
        overrides = overrides or {}
        for spec in self.parameter_specs:
            lo, hi, ini = spec.resolved(stress_scale, max_I1)
            if spec.name in overrides:
                ovo = overrides[spec.name]
                if len(ovo) >= 2 and ovo[0] is not None and ovo[1] is not None:
                    lo, hi = float(ovo[0]), float(ovo[1])
                if len(ovo) >= 3 and ovo[2] is not None:
                    ini = float(ovo[2])
                ini = min(max(ini, lo + 1e-12*(abs(lo)+1)), hi - 1e-12*(abs(hi)+1))
            lower.append(lo); upper.append(hi); x0.append(ini)
        return np.array(lower, dtype=float), np.array(upper, dtype=float), np.array(x0, dtype=float)

    def vector_to_params(self, x: Iterable[float]) -> Dict[str, float]:
        return {name: float(val) for name, val in zip(self.parameter_names(), x)}

    def energy_on_mode(self, mode: str, x: float, p: Dict[str, float]) -> float:
        stretches = mode_stretches(mode, x)
        if not all(np.isfinite(stretches)):
            return BIG
        try:
            y = float(self.W(stretches, p))
        except Exception:
            return BIG
        if not np.isfinite(y) or abs(y) > BIG/1000:
            return BIG
        return y

    def stress_state(self, mode: str, x: float, p: Dict[str, float]):
        """Mechanically consistent stress state for a test mode.

        Uses the constitutive response kernel:
        sigma_i = lambda_i*dW/dlambda_i - pressure
        P_i     = dW/dlambda_i - pressure/lambda_i

        The pressure term is chosen from the correct traction-free direction for
        uniaxial, equibiaxial and planar tests.
        """
        return principal_stress_state(self.W, mode, float(x), p)

    def predict_nominal_scalar(self, mode: str, x: float, p: Dict[str, float]) -> float:
        """Return model response for one test point.

        Normal modes return nominal/engineering stress in the loaded direction.
        Simple shear returns physical shear stress sigma12.
        """
        return float(self.stress_state(mode, x, p).scalar_response)

    def predict_nominal(self, mode: str, xs: np.ndarray, p: Dict[str, float]) -> np.ndarray:
        return kernel_predict_response(self.W, mode, np.asarray(xs, dtype=float), p)

    def stability_scan(self, params: Dict[str, float], modes=("uniaxial","biaxial","planar","simple_shear"), strain_min=-0.35, strain_max=2.50, n=140) -> Dict[str, Any]:
        """Run the HyperFitPro engineering stability screen.

        This includes finite response, reference energy, small-strain shear
        modulus, non-negative energy and tangent stiffness scans over standard
        deformation paths.
        """
        return engineering_stability_scan(self, params, modes=modes, strain_min=strain_min, strain_max=strain_max, n=n)

    def equation_with_numbers(self, params: Dict[str, float]) -> str:
        return _replace_latex_parameters(self.equation_latex, params)

    def metadata(self) -> Dict[str, Any]:
        return {
            "number": self.number,
            "name": self.name,
            "category": self.category,
            "equation_latex": self.equation_latex,
            "recommended_tests": self.recommended_tests,
            "parameters": [asdict(s) for s in self.parameter_specs],
            "notes": self.notes,
            "active": self.active,
            "family": getattr(self, "family", ""),
            "application_tags": list(getattr(self, "application_tags", [])),
            "material_classes": list(getattr(self, "material_classes", [])),
            "calibration_notes": getattr(self, "calibration_notes", ""),
            "parameterization_notes": getattr(self, "parameterization_notes", ""),
            "limitation_notes": getattr(self, "limitation_notes", ""),
            "source_equation_status": getattr(self, "source_equation_status", ""),
            "reference_keys": list(getattr(self, "reference_keys", [])),
            "complexity_level": int(getattr(self, "complexity_level", 1)),
            "minimum_recommended_points_per_parameter": int(getattr(self, "minimum_recommended_points_per_parameter", 8)),
            "preferred_optimizer": getattr(self, "preferred_optimizer", "recommended_adaptive_bounds"),
        }
