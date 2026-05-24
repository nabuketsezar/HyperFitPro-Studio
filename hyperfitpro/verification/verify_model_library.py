from pathlib import Path
import json
import numpy as np

from hyperfitpro.core.registry import load_models
from hyperfitpro.core.model_library import validate_model_library, export_model_catalog, recommend_models


def main():
    report = validate_model_library()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    assert report['duplicates'] == [], 'Duplicate model numbers found'
    assert report['n_active'] >= 42, 'Expected all 42 user-table models to be active after model-library verification'
    assert 32 in [m.number for m in load_models(include_inactive=False)], 'Van Der Waals model 32 is not active'
    failures = []
    for m in load_models(include_inactive=False):
        p = {s.name: (s.initial if s.initial is not None else (s.lower + s.upper) / 2.0) for s in m.parameter_specs}
        try:
            w0 = m.W((1.0, 1.0, 1.0), p)
            w1 = m.W((1.2, 1.0/np.sqrt(1.2), 1.0/np.sqrt(1.2)), p)
            if not np.isfinite(w0) or not np.isfinite(w1):
                failures.append((m.number, m.name, w0, w1))
        except Exception as e:
            failures.append((m.number, m.name, str(e)))
    if failures:
        print('Energy smoke check failures:')
        for f in failures:
            print(f)
        raise SystemExit(2)
    paths = export_model_catalog(Path('hyperfit_test_catalog'))
    print('Catalog exported:', paths)
    advice = recommend_models(['uniaxial', 'biaxial'], n_points=80, material_hint='rubber')[:5]
    print('Top recommendations for uniaxial + biaxial rubber data:')
    for a in advice:
        print(f'{a.model_number:02d} {a.model_name:32s} score={a.score} optimizer={a.preferred_optimizer}')
    print('Model library verification passed.')


if __name__ == '__main__':
    main()
