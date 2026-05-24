from __future__ import annotations

import math
from hyperfitpro.models.model_006_neo_hookean import MODEL_CLASS


def main():
    m = MODEL_CLASS()
    p = {'mu': 2.0}
    checks = [
        ('uniaxial', 0.2, p['mu']*((1.2) - (1.2)**-2)),
        ('biaxial', 0.2, p['mu']*((1.2) - (1.2)**-5)),
        ('planar', 0.2, p['mu']*((1.2) - (1.2)**-3)),
        ('simple_shear', 0.3, p['mu']*0.3),
    ]
    for mode, x, expected in checks:
        got = m.predict_nominal_scalar(mode, x, p)
        err = abs(got - expected)
        print(f'{mode:14s} got={got:.12g} expected={expected:.12g} abs_err={err:.3e}')
        assert err < 1e-6
    stab = m.stability_scan(p)
    print('stability passed:', stab['passed'])
    assert stab['passed']

if __name__ == '__main__':
    main()
