from __future__ import annotations

"""FEA material export utilities for HyperFitPro.

Internal fit stress unit in HyperFitPro is MPa because the data-processing layer
normalises stress to MPa. Export functions therefore explicitly write the unit
basis and can scale stress-like coefficients to other user-selected systems.

The exporter distinguishes between:
- exact/direct solver material cards for common built-in families,
- conservative coefficient include files for custom/non-built-in families,
- verification datasets that let the user compare the exported material card in
  a solver single-element test against HyperFitPro's constitutive kernel.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import json
import math
import re
from typing import Dict, Iterable, List, Tuple, Any

import numpy as np

# HyperFitPro stores stress-like parameters in MPa.
STRESS_SCALE_FROM_MPA = {
    "MPa": 1.0,
    "N/mm^2": 1.0,
    "Pa": 1.0e6,
    "kPa": 1.0e3,
    "GPa": 1.0e-3,
    "psi": 145.03773773020923,
    "ksi": 0.14503773773020923,
}


@dataclass
class ExportedFile:
    path: str
    kind: str
    status: str
    notes: str = ""


@dataclass
class SolverExportSpec:
    solver: str
    unit_system: str = "MPa-mm-N"
    stress_unit: str = "MPa"
    material_name: str = "HYPERFIT_MATERIAL"
    density: float | None = None
    include_comments: bool = True


def _safe_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(name).strip())
    if not s:
        s = "HYPERFIT_MATERIAL"
    if s[0].isdigit():
        s = "M_" + s
    return s[:80]


def _stress_scale(stress_unit: str) -> float:
    return float(STRESS_SCALE_FROM_MPA.get(stress_unit, 1.0))


def _scaled_params(model, params: Dict[str, float], stress_unit: str = "MPa") -> Dict[str, float]:
    """Scale stress-like model parameters from internal MPa to export stress unit."""
    scale = _stress_scale(stress_unit)
    out = {}
    specs = {s.name: s for s in getattr(model, "parameter_specs", [])}
    for k, v in params.items():
        spec = specs.get(k)
        if k == "K" or (spec and spec.scale in {"stress", "positive_stress"}):
            out[k] = float(v) * scale
        else:
            out[k] = float(v)
    return out


def _val(params: Dict[str, float], key: str, default: float = 0.0) -> float:
    return float(params.get(key, default))


def _bulk_Ds(params: Dict[str, float], n: int = 1) -> List[float]:
    """Return Abaqus/CalculiX D coefficients using K0 = 2 / D1 convention.

    If K is not available, D terms are written as 0.0 to request the
    incompressible form in solvers that accept zero D values. The run manifest
    also records this assumption explicitly.
    """
    K = params.get("K")
    if K is None or not np.isfinite(K) or K <= 0:
        return [0.0 for _ in range(n)]
    return [2.0 / float(K)] + [0.0 for _ in range(max(0, n - 1))]


def _split_lines(values: Iterable[float], per_line: int = 8) -> List[str]:
    vals = list(values)
    lines = []
    for i in range(0, len(vals), per_line):
        lines.append(", ".join(f"{float(v):.12g}" for v in vals[i:i+per_line]) + "\n")
    return lines


def _family(model) -> str:
    name = f"{getattr(model, 'name', '')} {getattr(model, 'family', '')} {getattr(model, 'category', '')}".lower()
    return name


def _yeoh_order(model, params: Dict[str, float]) -> int:
    order = 0
    for i in range(1, 6):
        if f"C{i}0" in params or (i == 1 and "C10" in params):
            order = max(order, i)
    # Models 7/8/9 are exactly 2/3/5 term Yeoh in the source table.
    if getattr(model, "number", None) == 7: return 2
    if getattr(model, "number", None) == 8: return 3
    if getattr(model, "number", None) == 9: return 5
    return max(order, 1)


def _polynomial_order(model, params: Dict[str, float]) -> int:
    if any(k in params for k in ("C30", "C21", "C12", "C03")):
        return 3
    if any(k in params for k in ("C20", "C11", "C02")):
        return 2
    return 1


def _ogden_order(params: Dict[str, float]) -> int:
    n = 0
    for i in range(1, 8):
        if f"mu{i}" in params and f"alpha{i}" in params:
            n = i
        elif f"mu_{i}" in params and f"alpha_{i}" in params:
            n = i
    return max(n, 1)


def _is_supported_direct(model, params: Dict[str, float]) -> Tuple[str, str]:
    fam = _family(model)
    no = getattr(model, "number", -1)
    if no == 6 or "neo-hookean" in fam or "neo hookean" in fam:
        return "neo_hookean", "direct"
    if "mooney" in fam:
        return "mooney_rivlin_polynomial", "direct"
    if no in {3, 4, 5} or ("polynomial" in fam and not "yeoh" in fam and not "reduced" in fam):
        return "polynomial", "direct"
    if "yeoh" in fam and "modified" not in fam and "fleming" not in fam:
        return "yeoh", "direct"
    if "ogden" in fam:
        return "ogden", "direct"
    if "arruda" in fam:
        return "arruda_boyce", "direct"
    if getattr(model, "number", -1) == 11 or " gent" in fam or fam.startswith("gent"):
        if getattr(model, "number", -1) == 36:
            return "custom", "coefficient_only"
        return "gent", "direct"
    if getattr(model, "number", -1) == 32 or "van der waals" in fam:
        return "van_der_waals", "direct"
    return "custom", "coefficient_only"


def abaqus_material_lines(model, params: Dict[str, float], spec: SolverExportSpec | None = None) -> Tuple[List[str], Dict[str, Any]]:
    spec = spec or SolverExportSpec("abaqus")
    p = _scaled_params(model, params, spec.stress_unit)
    mat = _safe_name(spec.material_name)
    family, status = _is_supported_direct(model, p)
    n = getattr(model, "number", -1)
    lines = [
        f"** HyperFitPro generated Abaqus material include\n",
        f"** Model {n:02d}: {getattr(model, 'name', '')}\n",
        f"** Unit basis: {spec.unit_system}; stress-like coefficients exported in {spec.stress_unit}\n",
        f"** Internal HyperFitPro fit unit is MPa; stress scale applied = {_stress_scale(spec.stress_unit):.12g}\n",
        f"*MATERIAL, NAME={mat}\n",
    ]
    if spec.density is not None:
        lines += ["*DENSITY\n", f"{spec.density:.12g}\n"]

    meta = {"solver": "Abaqus", "family": family, "status": status, "stress_unit": spec.stress_unit}

    if family == "neo_hookean":
        C10 = _val(p, "mu") / 2.0
        D = _bulk_Ds(p, 1)
        lines += ["*HYPERELASTIC, NEO HOOKE\n"] + _split_lines([C10] + D)
        meta["parameters_written"] = {"C10": C10, "D1": D[0]}
    elif family == "mooney_rivlin_polynomial":
        # Abaqus Mooney-Rivlin is polynomial N=1. Higher Mooney-Rivlin rows are exported as POLYNOMIAL.
        order = _polynomial_order(model, p)
        if order == 1:
            D = _bulk_Ds(p, 1)
            vals = [_val(p, "C10"), _val(p, "C01")] + D
            lines += ["*HYPERELASTIC, MOONEY-RIVLIN\n"] + _split_lines(vals)
            meta["parameters_written"] = {"C10": vals[0], "C01": vals[1], "D1": vals[2]}
        else:
            D = _bulk_Ds(p, order)
            coeffs = [_val(p, k) for k in ["C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03"]]
            vals = coeffs[:2 if order == 1 else 5 if order == 2 else 9] + D
            lines += [f"*HYPERELASTIC, POLYNOMIAL, N={order}\n"] + _split_lines(vals)
            meta["parameters_written"] = dict(zip(["C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03"][:len(vals)-len(D)] + [f"D{i+1}" for i in range(len(D))], vals))
    elif family == "polynomial":
        order = _polynomial_order(model, p)
        D = _bulk_Ds(p, order)
        coeffs = [_val(p, k) for k in ["C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03"]]
        vals = coeffs[:2 if order == 1 else 5 if order == 2 else 9] + D
        lines += [f"*HYPERELASTIC, POLYNOMIAL, N={order}\n"] + _split_lines(vals)
        meta["parameters_written"] = dict(zip(["C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03"][:len(vals)-len(D)] + [f"D{i+1}" for i in range(len(D))], vals))
    elif family == "yeoh":
        order = _yeoh_order(model, p)
        order = max(1, min(order, 5))
        coeff_names = [f"C{i}0" for i in range(1, order+1)]
        coeff_names[0] = "C10"
        coeffs = [_val(p, k) for k in coeff_names]
        D = _bulk_Ds(p, order)
        lines += [f"*HYPERELASTIC, REDUCED POLYNOMIAL, N={order}\n"] + _split_lines(coeffs + D)
        meta["parameters_written"] = dict(zip(coeff_names + [f"D{i+1}" for i in range(order)], coeffs + D))
    elif family == "ogden":
        order = _ogden_order(p)
        coeffs = []
        names = []
        for i in range(1, order+1):
            mu = p.get(f"mu{i}", p.get(f"mu_{i}", 0.0))
            alpha = p.get(f"alpha{i}", p.get(f"alpha_{i}", 0.0))
            coeffs += [mu, alpha]
            names += [f"mu{i}", f"alpha{i}"]
        D = _bulk_Ds(p, order)
        lines += [f"*HYPERELASTIC, OGDEN, N={order}\n"] + _split_lines(coeffs + D)
        meta["parameters_written"] = dict(zip(names + [f"D{i+1}" for i in range(order)], coeffs + D))
    elif family == "arruda_boyce":
        D = _bulk_Ds(p, 1)
        vals = [_val(p, "mu"), _val(p, "lambda_L", _val(p, "lambdaL", 0.0))] + D
        lines += ["*HYPERELASTIC, ARRUDA-BOYCE\n"] + _split_lines(vals)
        meta["parameters_written"] = {"mu": vals[0], "lambda_L": vals[1], "D1": vals[2]}
    elif family == "gent":
        # Abaqus Gent normally uses mu and Jm. In the source table I_L-3 is the limiting invariant offset.
        Jm = _val(p, "I_L", _val(p, "J_L", 0.0)) - 3.0 if "I_L" in p else _val(p, "J_L", _val(p, "Jm", 0.0))
        D = _bulk_Ds(p, 1)
        vals = [_val(p, "mu"), Jm] + D
        lines += ["*HYPERELASTIC, GENT\n"] + _split_lines(vals)
        meta["parameters_written"] = {"mu": vals[0], "Jm": vals[1], "D1": vals[2]}
        lines += ["** Note: source-table Gent parameter I_L is converted to Jm = I_L - 3 when I_L is present.\n"]
    elif family == "van_der_waals":
        D = _bulk_Ds(p, 1)
        vals = [_val(p, "mu"), _val(p, "lambda_m"), _val(p, "a"), _val(p, "beta")] + D
        lines += ["*HYPERELASTIC, VAN DER WAALS\n"] + _split_lines(vals)
        meta["parameters_written"] = {"mu": vals[0], "lambda_m": vals[1], "a": vals[2], "beta": vals[3], "D1": vals[4]}
    else:
        lines += [
            "** This source-table model is not a direct Abaqus built-in mapping in HyperFitPro.\n",
            "** Coefficients are listed below for a user-material or manual solver implementation.\n",
            f"** Equation LaTeX: {getattr(model, 'equation_latex', '')}\n",
        ]
        for k, v in p.items():
            lines.append(f"** PARAM {k} = {v:.12g}\n")
        meta["parameters_written"] = dict(p)
    return lines, meta


def export_abaqus_material(model, params, path, material_name='HYPERFIT_MATERIAL', stress_unit: str = "MPa", unit_system: str = "MPa-mm-N"):
    path = Path(path)
    lines, _ = abaqus_material_lines(model, params, SolverExportSpec("abaqus", unit_system, stress_unit, material_name))
    path.write_text("".join(lines), encoding="utf-8")
    return str(path)


def export_calculix_material(model, params, path, material_name='HYPERFIT_MATERIAL', stress_unit: str = "MPa", unit_system: str = "MPa-mm-N"):
    """CalculiX uses Abaqus-like input syntax for many hyperelastic cards."""
    path = Path(path)
    lines, meta = abaqus_material_lines(model, params, SolverExportSpec("calculix", unit_system, stress_unit, material_name))
    lines.insert(0, "** CalculiX-compatible Abaqus-style material include generated by HyperFitPro\n")
    lines.append("** Verify exact keyword support in your CalculiX version before production solve.\n")
    path.write_text("".join(lines), encoding="utf-8")
    return str(path)


def ansys_material_lines(model, params: Dict[str, float], spec: SolverExportSpec | None = None) -> Tuple[List[str], Dict[str, Any]]:
    spec = spec or SolverExportSpec("ansys")
    p = _scaled_params(model, params, spec.stress_unit)
    family, status = _is_supported_direct(model, p)
    mat_id = 1
    lines = [
        "! HyperFitPro generated ANSYS APDL material macro\n",
        f"! Model {getattr(model, 'number', -1):02d}: {getattr(model, 'name', '')}\n",
        f"! Unit basis: {spec.unit_system}; stress-like coefficients exported in {spec.stress_unit}\n",
        f"! Internal HyperFitPro fit unit is MPa; stress scale applied = {_stress_scale(spec.stress_unit):.12g}\n",
        f"MP,EX,{mat_id},1e-9\n",
        f"MP,NUXY,{mat_id},0.4999\n",
    ]
    meta = {"solver": "ANSYS", "family": family, "status": status, "stress_unit": spec.stress_unit}

    def tbdata(vals):
        out = []
        for i in range(0, len(vals), 6):
            out.append(f"TBDATA,{i+1}," + ",".join(f"{float(v):.12g}" for v in vals[i:i+6]) + "\n")
        return out

    if family == "neo_hookean":
        lines += [f"TB,HYPER,{mat_id},,1,NEO\n"] + tbdata([_val(p, "mu") / 2.0])
        meta["parameters_written"] = {"C10": _val(p, "mu")/2.0}
    elif family in {"mooney_rivlin_polynomial", "polynomial"}:
        order = _polynomial_order(model, p)
        coeffs = [_val(p, k) for k in ["C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03"]]
        vals = coeffs[:2 if order == 1 else 5 if order == 2 else 9]
        tbopt = "MOONEY" if order == 1 else "POLY"
        lines += [f"TB,HYPER,{mat_id},,{order},{tbopt}\n"] + tbdata(vals)
        meta["parameters_written"] = dict(zip(["C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03"][:len(vals)], vals))
    elif family == "yeoh":
        order = _yeoh_order(model, p)
        coeff_names = [f"C{i}0" for i in range(1, order+1)]
        coeff_names[0] = "C10"
        vals = [_val(p, k) for k in coeff_names]
        lines += [f"TB,HYPER,{mat_id},,{order},YEOH\n"] + tbdata(vals)
        meta["parameters_written"] = dict(zip(coeff_names, vals))
    elif family == "ogden":
        order = _ogden_order(p)
        vals = []
        names = []
        for i in range(1, order+1):
            vals += [p.get(f"mu{i}", p.get(f"mu_{i}", 0.0)), p.get(f"alpha{i}", p.get(f"alpha_{i}", 0.0))]
            names += [f"mu{i}", f"alpha{i}"]
        lines += [f"TB,HYPER,{mat_id},,{order},OGDEN\n"] + tbdata(vals)
        meta["parameters_written"] = dict(zip(names, vals))
    elif family == "arruda_boyce":
        vals = [_val(p, "mu"), _val(p, "lambda_L")]
        lines += [f"TB,HYPER,{mat_id},,1,BOYCE\n"] + tbdata(vals)
        meta["parameters_written"] = {"mu": vals[0], "lambda_L": vals[1]}
    elif family == "gent":
        Jm = _val(p, "I_L", _val(p, "J_L", 0.0)) - 3.0 if "I_L" in p else _val(p, "J_L", _val(p, "Jm", 0.0))
        vals = [_val(p, "mu"), Jm]
        lines += [f"TB,HYPER,{mat_id},,1,GENT\n"] + tbdata(vals)
        meta["parameters_written"] = {"mu": vals[0], "Jm": vals[1]}
    else:
        lines += [
            "! No direct ANSYS built-in mapping is available in HyperFitPro for this source-table model.\n",
            "! Use the coefficient list below in USERMAT / USERHYPER or manual TB,HYPER mapping.\n",
            f"! Equation LaTeX: {getattr(model, 'equation_latex', '')}\n",
        ]
        for k, v in p.items():
            lines.append(f"! PARAM {k} = {v:.12g}\n")
        meta["parameters_written"] = dict(p)
    return lines, meta


def export_ansys_text(model, params, path, material_name='HYPERFIT_MATERIAL', stress_unit: str = "MPa", unit_system: str = "MPa-mm-N"):
    path = Path(path)
    lines, _ = ansys_material_lines(model, params, SolverExportSpec("ansys", unit_system, stress_unit, material_name))
    path.write_text("".join(lines), encoding="utf-8")
    return str(path)


def export_lsdyna_keyword(model, params, path, material_name='HYPERFIT_MATERIAL', stress_unit: str = "MPa", unit_system: str = "MPa-mm-N"):
    """Create an LS-DYNA keyword include.

    LS-DYNA hyperelastic cards vary significantly by version and material card.
    HyperFitPro writes a conservative keyword include with direct coefficients for
    common families and clear comments. It is intended as a solver input starting
    point plus the verification CSV; final MAT card choice remains project- and
    solver-version-specific.
    """
    path = Path(path)
    p = _scaled_params(model, params, stress_unit)
    family, status = _is_supported_direct(model, p)
    mid = 1
    lines = [
        "$ HyperFitPro generated LS-DYNA keyword coefficient include\n",
        f"$ Model {getattr(model, 'number', -1):02d}: {getattr(model, 'name', '')}\n",
        f"$ Unit basis: {unit_system}; stress-like coefficients exported in {stress_unit}\n",
        "$ Select the final *MAT card according to the LS-DYNA version and analysis type.\n",
        "*KEYWORD\n",
    ]
    if family == "neo_hookean":
        lines += [
            "$ Neo-Hookean coefficient listing: W=C10(I1-3); C10=mu/2\n",
            "*DEFINE_TABLE_TITLE\n",
            "HyperFitPro_NeoHookean_Coefficients\n",
            "$ mid, C10, mu, K(optional)\n",
            f"{mid}, {_val(p,'mu')/2.0:.12g}, {_val(p,'mu'):.12g}, {_val(p,'K',0.0):.12g}\n",
        ]
    elif family in {"mooney_rivlin_polynomial", "polynomial", "yeoh", "ogden", "arruda_boyce", "gent", "van_der_waals"}:
        lines += ["$ Direct coefficient listing for supported hyperelastic family.\n", "*DEFINE_TABLE_TITLE\n", f"HyperFitPro_{family}_Coefficients\n", "$ parameter, value\n"]
        for k, v in p.items():
            lines.append(f"$ {k}, {v:.12g}\n")
    else:
        lines += ["$ Custom source-table model; implement through user material or manually selected MAT card.\n"]
        for k, v in p.items():
            lines.append(f"$ {k} = {v:.12g}\n")
    lines.append("*END\n")
    path.write_text("".join(lines), encoding="utf-8")
    return str(path)


def export_marc_text(model, params, path, material_name='HYPERFIT_MATERIAL', stress_unit: str = "MPa", unit_system: str = "MPa-mm-N"):
    path = Path(path)
    p = _scaled_params(model, params, stress_unit)
    family, _ = _is_supported_direct(model, p)
    lines = [
        "# HyperFitPro Marc/Mentat coefficient listing\n",
        f"# Model {getattr(model, 'number', -1):02d}: {getattr(model, 'name', '')}\n",
        f"# Unit basis: {unit_system}; stress-like coefficients exported in {stress_unit}\n",
        f"# Suggested family mapping: {family}\n",
        "# Enter these coefficients through the Marc hyperelastic material dialog or user material interface.\n",
    ]
    for k, v in p.items():
        lines.append(f"{k} = {v:.12g}\n")
    path.write_text("".join(lines), encoding="utf-8")
    return str(path)


def export_json_material(model, params, path, stress_unit: str = "MPa", unit_system: str = "MPa-mm-N"):
    path = Path(path)
    payload = {
        'model': model.metadata(),
        'parameters_internal_MPa': {k: float(v) for k, v in params.items()},
        'parameters_export_units': _scaled_params(model, params, stress_unit),
        'unit_system': unit_system,
        'stress_unit': stress_unit,
    }
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return str(path)


def export_unit_consistency_report(model, params, path, stress_unit: str = "MPa", unit_system: str = "MPa-mm-N"):
    path = Path(path)
    p = _scaled_params(model, params, stress_unit)
    lines = [
        "# HyperFitPro FEA Unit Consistency Report\n\n",
        f"Model: {getattr(model, 'number', -1):02d} - {getattr(model, 'name', '')}\n\n",
        f"Internal fit stress unit: MPa\n\n",
        f"Export unit basis: {unit_system}\n\n",
        f"Stress-like export unit: {stress_unit}\n\n",
        f"Stress scale from MPa: {_stress_scale(stress_unit):.12g}\n\n",
        "## Exported parameters\n\n",
        "| Parameter | Internal value [MPa basis] | Export value | Note |\n",
        "|---|---:|---:|---|\n",
    ]
    specs = {s.name: s for s in getattr(model, "parameter_specs", [])}
    for k, v in params.items():
        spec = specs.get(k)
        scale_note = "stress-like scaled" if k == "K" or (spec and spec.scale in {"stress", "positive_stress"}) else "dimensionless / unchanged"
        lines.append(f"| {k} | {float(v):.12g} | {p[k]:.12g} | {scale_note} |\n")
    lines += [
        "\n## Important checks before production FEA\n\n",
        "1. Solver unit system must be internally consistent. HyperFitPro does not know your model length/mass/time convention beyond the selected stress unit.\n",
        "2. If a fitted bulk modulus `K` is absent, exported D terms are written as incompressible/zero where supported.\n",
        "3. Run the generated single-element verification target data and compare solver stresses with `single_element_verification_targets.csv`.\n",
        "4. For custom/non-built-in models, use coefficient exports with a user-material implementation rather than assuming a direct built-in card exists.\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")
    return str(path)


def generate_single_element_verification_targets(model, params, path, stress_unit: str = "MPa"):
    """Write canonical one-element verification targets from HyperFitPro kernel."""
    path = Path(path)
    scale = _stress_scale(stress_unit)
    modes = {
        "uniaxial": [0.80, 0.90, 1.00, 1.10, 1.25, 1.50, 2.00],
        "biaxial": [0.90, 1.00, 1.05, 1.10, 1.25, 1.50],
        "planar": [0.90, 1.00, 1.10, 1.25, 1.50, 2.00],
        "simple_shear": [0.00, 0.05, 0.10, 0.20, 0.50, 1.00],
    }
    rows = []
    for mode, xs in modes.items():
        for x in xs:
            try:
                y = float(model.predict_nominal_scalar(mode, float(x), params)) * scale
            except Exception:
                y = float("nan")
            rows.append({"mode": mode, "input_x": x, "target_response": y, "response_unit": stress_unit})
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["mode", "input_x", "target_response", "response_unit"])
        wr.writeheader(); wr.writerows(rows)
    return str(path)


def export_abaqus_single_element_probe(model, params, path, material_include: str = "abaqus_material.inp", material_name: str = "HYPERFIT_MATERIAL"):
    path = Path(path)
    mat = _safe_name(material_name)
    lines = [
        "** HyperFitPro Abaqus single-element probe template\n",
        "** Purpose: run small prescribed deformation cases and compare reaction stress with single_element_verification_targets.csv\n",
        "** This is a template, not a finished production model. Adjust element type, step controls and output requests for your solver version.\n",
        f"*INCLUDE, INPUT={material_include}\n",
        "*PART, NAME=CUBE\n",
        "*NODE\n",
        "1,0,0,0\n2,1,0,0\n3,1,1,0\n4,0,1,0\n5,0,0,1\n6,1,0,1\n7,1,1,1\n8,0,1,1\n",
        "*ELEMENT, TYPE=C3D8H, ELSET=EALL\n1,1,2,3,4,5,6,7,8\n",
        f"*SOLID SECTION, ELSET=EALL, MATERIAL={mat}\n",
        "*END PART\n",
        "** Create assemblies/steps per mode using prescribed displacements.\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")
    return str(path)


def export_ansys_single_element_probe(model, params, path, material_macro: str = "ansys_material.mac"):
    path = Path(path)
    lines = [
        "! HyperFitPro ANSYS single-element probe template\n",
        "! Compare reaction stress with single_element_verification_targets.csv.\n",
        f"/INPUT,{Path(material_macro).stem},mac\n",
        "/PREP7\n",
        "ET,1,185\n",
        "KEYOPT,1,2,3 ! mixed u-P formulation option may need adjustment by version\n",
        "BLOCK,0,1,0,1,0,1\n",
        "VMESH,ALL\n",
        "! Add displacement boundary conditions for uniaxial/biaxial/planar/shear target rows.\n",
        "FINISH\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")
    return str(path)


def export_fea_bundle(model, params, out_dir, material_name: str = "HYPERFIT_MATERIAL", stress_unit: str = "MPa", unit_system: str = "MPa-mm-N") -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    exported: List[ExportedFile] = []

    def add(path: str, kind: str, status: str, notes: str = ""):
        exported.append(ExportedFile(str(path), kind, status, notes))

    family, status = _is_supported_direct(model, params)
    add(export_json_material(model, params, out / "material_fit.json", stress_unit, unit_system), "JSON material archive", "ok")
    add(export_abaqus_material(model, params, out / "abaqus_material.inp", material_name, stress_unit, unit_system), "Abaqus material include", status)
    add(export_calculix_material(model, params, out / "calculix_material.inp", material_name, stress_unit, unit_system), "CalculiX material include", status)
    add(export_ansys_text(model, params, out / "ansys_material.mac", material_name, stress_unit, unit_system), "ANSYS APDL material macro", status)
    add(export_lsdyna_keyword(model, params, out / "lsdyna_material.k", material_name, stress_unit, unit_system), "LS-DYNA keyword coefficient include", "coefficient_listing")
    add(export_marc_text(model, params, out / "marc_material.txt", material_name, stress_unit, unit_system), "Marc/Mentat coefficient listing", "coefficient_listing")
    add(export_unit_consistency_report(model, params, out / "unit_consistency_report.md", stress_unit, unit_system), "Unit consistency report", "ok")
    add(generate_single_element_verification_targets(model, params, out / "single_element_verification_targets.csv", stress_unit), "Single-element verification target CSV", "ok")
    add(export_abaqus_single_element_probe(model, params, out / "abaqus_single_element_probe_template.inp", "abaqus_material.inp", material_name), "Abaqus single-element probe template", "template")
    add(export_ansys_single_element_probe(model, params, out / "ansys_single_element_probe_template.mac", "ansys_material.mac"), "ANSYS single-element probe template", "template")

    manifest = {
        "model": model.metadata(),
        "material_name": _safe_name(material_name),
        "family_mapping": family,
        "direct_mapping_status": status,
        "unit_system": unit_system,
        "stress_unit": stress_unit,
        "stress_scale_from_internal_MPa": _stress_scale(stress_unit),
        "files": [asdict(x) for x in exported],
        "production_use_notes": [
            "Abaqus/CalculiX/ANSYS direct cards are generated for common built-in families only.",
            "LS-DYNA and Marc outputs are conservative coefficient/template exports because exact MAT syntax is version- and card-dependent.",
            "Always run single-element verification before using the exported material in production FEA.",
            "If no volumetric K was fitted, exported D coefficients use the incompressible convention where supported.",
        ],
    }
    manifest_path = out / "fea_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    add(str(manifest_path), "FEA export manifest", "ok")
    return manifest
