from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List
import csv
import json
from pathlib import Path

from .registry import load_models

TEST_DESCRIPTIONS: Dict[str, str] = {
    'uniaxial': 'Uniaxial tension/compression nominal stress vs stretch. Minimum baseline for all models.',
    'biaxial': 'Equibiaxial tension nominal stress vs stretch. Separates I1/I2 behavior and constrains high-strain response.',
    'planar': 'Planar tension / pure shear nominal stress vs stretch. Useful for I2-sensitive and FEA-transferable fits.',
    'simple_shear': 'Simple shear stress vs shear strain. Useful for I2-sensitive, spectral and high-order polynomial models.',
    'volumetric': 'Volumetric pressure/bulk response. Required for compressible FEA material cards.',
}

CATEGORY_GUIDE: Dict[str, str] = {
    'Neo-Hookean / one-parameter network': 'First-pass baseline and sanity check.',
    'Invariant polynomial / Mooney-Rivlin family': 'Low-to-high order invariant polynomial rubber models.',
    'Invariant polynomial / general polynomial': 'Flexible I1/I2 polynomial forms; high overfitting risk.',
    'Reduced polynomial / Yeoh family': 'I1-only reduced polynomial models, often used for filled rubbers.',
    'Reduced polynomial / modified Yeoh': 'Yeoh polynomial with exponential correction.',
    'Reduced polynomial / power-law': 'Reduced polynomial plus power-law terms.',
    'Invariant polynomial / Hartmann-Neff family': 'Hartmann-Neff invariant family with I1^3 and I2^(3/2) terms.',
    'Network / finite chain extensibility': 'Physically motivated limiting-chain models with finite extensibility.',
    'Network / generalized finite extensibility': 'Finite extensibility with additional power-law flexibility.',
    'Network / finite extensibility': 'Finite extensibility / limiting-chain model family.',
    'Network / finite extensibility with I2 correction': 'Limiting-chain models with additional I2 sensitivity.',
    'Network / finite extensibility with I2 blend': 'Gent-type finite extensibility blended with I2 behavior.',
    'Principal-stretch / Ogden family': 'Spectral principal-stretch models; powerful but non-unique without multi-mode data.',
    'Principal-stretch / logarithmic spectral': 'Principal-stretch logarithmic series model.',
    'Principal-stretch / Bechir spectral polynomial': 'Spectral polynomial forms in powers of principal stretches.',
    'Exponential / soft-tissue type': 'Exponential strain-stiffening models, common for soft-tissue style behavior.',
    'Exponential / Hart-Smith family': 'Hart-Smith exponential/log-I2 family.',
    'Exponential / Fung-type': 'Fung-type exponential stiffening.',
    'Exponential / improved Hart-Smith': 'Power-enhanced Hart-Smith style exponential model.',
    'Fiber-influenced / exponential anisotropic approximation': 'Fiber-like exponential model approximated in this isotropic workflow.',
    'Van der Waals / limiting-chain interaction': 'Finite-extensibility model with I1/I2 invariant mixing.',
    'Power-law / phenomenological': 'Phenomenological power-law models; require robust fitting and validation.',
    'Hybrid finite extensibility / Yeoh-Fleming': 'Mixed exponential and finite-chain stiffening model.',
    'Hybrid invariant / Gent-Thomas': 'I1 + log I2 hybrid invariant form.',
    'Hybrid / low-strain Hoss-Marczak': 'Low-strain hybrid model.',
    'Hybrid / high-strain Hoss-Marczak': 'High-strain hybrid with I2 correction.',
}

BOUND_POLICY = (
    "HyperFitPro stores parameter ranges as calibration search bounds, not universal material constants. "
    "Stress-like parameters use scale='stress' or scale='positive_stress' and are multiplied by the imported data stress scale. "
    "Limit parameters such as I_L, J_L, lambda_L and lambda_m are additionally checked against the deformation domain by the model kernel and stability scan."
)

@dataclass
class ModelAdvice:
    model_number: int
    model_name: str
    category: str
    score: float
    warnings: List[str]
    recommended_tests: List[str]
    preferred_optimizer: str


def model_catalog(include_inactive: bool = False) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for m in load_models(include_inactive=include_inactive):
        md = m.metadata()
        md['n_parameters'] = len(m.parameter_specs)
        md['parameter_names'] = m.parameter_names()
        md['category_description'] = CATEGORY_GUIDE.get(m.category, '')
        rows.append(md)
    return rows


def export_model_catalog(out_dir: str | Path) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = model_catalog(include_inactive=True)
    json_path = out / 'model_catalog.json'
    csv_path = out / 'model_catalog.csv'
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    fieldnames = [
        'number','name','active','category','family','n_parameters','parameter_names',
        'recommended_tests','material_classes','application_tags','complexity_level',
        'preferred_optimizer','notes','calibration_notes','limitation_notes','source_equation_status'
    ]
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            rr = {k: r.get(k, '') for k in fieldnames}
            for k in ['parameter_names','recommended_tests','material_classes','application_tags']:
                rr[k] = ', '.join(map(str, rr.get(k, []) or []))
            w.writerow(rr)
    return {'json': str(json_path), 'csv': str(csv_path)}


def recommend_models(dataset_modes: Iterable[str], n_points: int, material_hint: str = '') -> List[ModelAdvice]:
    modes = set(dataset_modes)
    hint = (material_hint or '').lower()
    out: List[ModelAdvice] = []
    for m in load_models(include_inactive=False):
        npar = len(m.parameter_specs)
        score = 0.0
        warnings: List[str] = []
        rec = set(m.recommended_tests)
        coverage = len(modes & rec) / max(len(rec), 1)
        score += 60.0 * coverage
        if modes:
            score += 10.0 * min(len(modes), 4)
        min_pts = npar * int(getattr(m, 'minimum_recommended_points_per_parameter', 8))
        if n_points >= min_pts:
            score += 15.0
        else:
            warnings.append(f'Data may be sparse for {npar} parameters; target at least {min_pts} points.')
            score -= 10.0
        if npar <= 2 and len(modes) <= 1:
            score += 12.0
        if npar >= 5 and len(modes) <= 1:
            warnings.append('High-parameter model with single-mode data has overfitting/non-uniqueness risk.')
            score -= 18.0
        tags = ' '.join(getattr(m, 'material_classes', []) + getattr(m, 'application_tags', []) + [m.category, getattr(m, 'family', '')]).lower()
        if hint and any(tok in tags for tok in hint.replace('/', ' ').split()):
            score += 10.0
        if 'simple_shear' not in modes and any(('I2' in t or 'shear' in t.lower()) for t in (getattr(m, 'application_tags', []) + [m.notes])):
            warnings.append('Planar/simple-shear data is recommended to identify I2-sensitive terms.')
        out.append(ModelAdvice(m.number, m.name, m.category, round(score, 2), warnings, m.recommended_tests, getattr(m, 'preferred_optimizer', 'recommended_adaptive_bounds')))
    out.sort(key=lambda a: a.score, reverse=True)
    return out


def validate_model_library() -> Dict[str, Any]:
    rows = model_catalog(include_inactive=True)
    numbers = [r['number'] for r in rows]
    duplicates = sorted({n for n in numbers if numbers.count(n) > 1})
    inactive = [r for r in rows if not r.get('active', True)]
    missing_eq = [r['number'] for r in rows if not r.get('equation_latex')]
    active_without_params = [r['number'] for r in rows if r.get('active', True) and len(r.get('parameters', [])) == 0]
    return {
        'n_models_total': len(rows),
        'n_active': sum(1 for r in rows if r.get('active', True)),
        'duplicates': duplicates,
        'inactive_models': [(r['number'], r['name']) for r in inactive],
        'missing_equations': missing_eq,
        'active_models_without_parameters': active_without_params,
        'categories': sorted(set(r.get('category', '') for r in rows)),
        'bound_policy': BOUND_POLICY,
    }
