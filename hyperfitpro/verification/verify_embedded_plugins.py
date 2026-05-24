from __future__ import annotations

import numpy as np

from hyperfitpro.core.registry import load_models
from hyperfitpro.core.plugin_manager import PluginRecord, discover_plugin_model_classes


def main():
    records: list[PluginRecord] = []
    classes = discover_plugin_model_classes(diagnostics=records)
    failed = [r for r in records if r.status != 'loaded']
    if failed:
        for r in failed:
            print(r)
        raise SystemExit('One or more embedded/user plugins failed to load')

    models = load_models(include_inactive=True)
    embedded = [m for m in models if 9010 <= m.number <= 9022]
    assert len(embedded) == 13, f'Expected 13 embedded plugin models, got {len(embedded)}'

    failures = []
    for m in embedded:
        p = {s.name: (s.initial if s.initial is not None else (s.lower + s.upper) / 2.0) for s in m.parameter_specs}
        try:
            w0 = m.W((1.0, 1.0, 1.0), p)
            w1 = m.W((1.25, 1.0/np.sqrt(1.25), 1.0/np.sqrt(1.25)), p)
            y = m.predict_nominal('uniaxial', np.array([1.0, 1.1, 1.2]), p)
            if not np.isfinite(w0) or not np.isfinite(w1) or not np.all(np.isfinite(y)):
                failures.append((m.number, m.name, w0, w1, y.tolist()))
        except Exception as exc:
            failures.append((m.number, m.name, str(exc)))
    if failures:
        print('Embedded plugin failures:')
        for item in failures:
            print(item)
        raise SystemExit(2)

    print('Embedded plugin verification passed.')
    print('Loaded embedded plugins:')
    for m in embedded:
        print(f'  {m.number}: {m.name} | {m.category}')


if __name__ == '__main__':
    main()
