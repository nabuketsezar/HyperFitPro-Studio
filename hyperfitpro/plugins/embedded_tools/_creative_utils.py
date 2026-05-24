
from __future__ import annotations

from pathlib import Path
import csv, json, zipfile, html
import numpy as np


def out_dir(ctx, plugin_id: str) -> Path:
    p = ctx.output_root() / plugin_id.replace('.', '_')
    p.mkdir(parents=True, exist_ok=True)
    return p


def params(ctx) -> dict:
    out = {}
    try:
        out.update(dict(getattr(ctx.fit_result, 'parameters', {}) or {}))
    except Exception:
        pass
    try:
        out.update(dict(ctx.parameters or {}))
    except Exception:
        pass
    return out


def metrics(ctx) -> dict:
    r = getattr(ctx, 'fit_result', None)
    keys = ['rmse', 'mae', 'r2', 'adjusted_r2', 'aic', 'aicc', 'bic', 'validation_rmse']
    d = {}
    for k in keys:
        try:
            v = getattr(r, k, None)
            if v is not None:
                d[k] = float(v)
        except Exception:
            pass
    return d


def datasets(ctx):
    return list(getattr(ctx, 'datasets', []) or [])


def dataset_arrays(d):
    try:
        x = np.asarray(getattr(d, 'x', []), dtype=float)
        y = np.asarray(getattr(d, 'y', []), dtype=float)
    except Exception:
        x = np.asarray([], dtype=float); y = np.asarray([], dtype=float)
    if x.size and y.size and x.shape == y.shape:
        m = np.isfinite(x) & np.isfinite(y)
        return x[m], y[m]
    return np.asarray([], dtype=float), np.asarray([], dtype=float)


def dataset_summary(ctx) -> list[dict]:
    rows=[]
    for i,d in enumerate(datasets(ctx), 1):
        x,y = dataset_arrays(d)
        rows.append({
            'index': i,
            'mode': getattr(d, 'mode', '-'),
            'points': int(len(x)),
            'x_min': float(np.min(x)) if len(x) else None,
            'x_max': float(np.max(x)) if len(x) else None,
            'y_min': float(np.min(y)) if len(y) else None,
            'y_max': float(np.max(y)) if len(y) else None,
            'weight': getattr(d, 'weight', None),
            'name': getattr(d, 'name', '') or getattr(d, 'source', '') or '',
        })
    return rows


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
    return path


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text('', encoding='utf-8')
        return path
    keys=[]
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    return path


def write_md(path: Path, title: str, body: str):
    path.write_text(f'# {title}\n\n{body}\n', encoding='utf-8')
    return path


def safe_model_label(ctx):
    m=getattr(ctx, 'model', None)
    if m is None:
        return 'No model'
    return f'{getattr(m, "number", "-")} - {getattr(m, "name", "-")}'


def equation(ctx):
    m=getattr(ctx,'model',None)
    for k in ['equation_latex', 'latex', 'equation']:
        try:
            v=getattr(m,k)
            if v:
                return str(v)
        except Exception:
            pass
    return ''


def zip_dir(folder: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in folder.rglob('*'):
            if p.is_file() and p != zip_path:
                z.write(p, p.relative_to(folder))
    return zip_path


def simple_html(title: str, body: str) -> str:
    return '<!doctype html><html><head><meta charset="utf-8"><title>{}</title><style>body{{font-family:Arial,sans-serif;margin:32px;color:#1f2937}}.card{{border:1px solid #d7dee8;border-radius:12px;padding:18px;margin:12px 0;background:#fff}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #e5e7eb;padding:8px}}h1{{color:#0f172a}}</style></head><body><h1>{}</h1>{}</body></html>'.format(html.escape(title), html.escape(title), body)


def escape_pre(text: str) -> str:
    return '<pre>' + html.escape(text) + '</pre>'
