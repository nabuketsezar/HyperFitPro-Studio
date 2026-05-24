from __future__ import annotations

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hyperfitpro.core.tool_plugins import ToolPlugin, ToolResult


def _metric(v):
    try:
        if v is None or not np.isfinite(float(v)):
            return '-'
        return f'{float(v):.6g}'
    except Exception:
        return '-'


def _safe_len(a):
    try:
        return len(a)
    except Exception:
        return 0


def _risk_badge(result, datasets, model):
    risks=[]
    if len(datasets) < 2:
        risks.append('Only one test mode loaded')
    npar=len(getattr(model, 'parameter_specs', []) or [])
    npts=sum(_safe_len(getattr(d, 'x', [])) for d in datasets)
    if npar and npts/max(npar,1) < 10:
        risks.append('Low data/parameter ratio')
    r2=getattr(result, 'r2', float('nan'))
    if np.isfinite(r2) and r2 < 0.97:
        risks.append('R² below 0.97')
    for w in getattr(result, 'overfitting_warnings', []) or []:
        risks.append(str(w))
    passed = bool(getattr(result, 'stability', {}) and getattr(result, 'stability', {}).get('passed', False))
    if not passed:
        risks.append('Stability scan not fully green')
    if not risks:
        return 'GREEN', ['Fit is clean enough for engineering review.']
    if len(risks) <= 2:
        return 'AMBER', risks
    return 'RED', risks


class MaterialPassportGenerator(ToolPlugin):
    plugin_id = 'studio.material_passport'
    name = 'Material passport generator'
    category = 'Studio'
    description = 'Creates a premium one-page material passport PDF/PNG/CSV with parameters, metrics, dataset coverage, and risk badges.'
    requires_model = True
    requires_data = True
    requires_fit = True
    output_types = ['pdf', 'png', 'csv', 'md']

    def run(self, ctx):
        out = ctx.output_root() / 'studio_material_passport'
        out.mkdir(parents=True, exist_ok=True)
        res = ctx.fit_result
        params = dict(ctx.parameters or getattr(res, 'parameters', {}) or {})
        badge, risks = _risk_badge(res, ctx.datasets, ctx.model)
        modes = sorted({getattr(d, 'mode', '-') for d in ctx.datasets})
        npts = sum(_safe_len(getattr(d, 'x', [])) for d in ctx.datasets)

        fig = plt.figure(figsize=(11.2, 7.2), dpi=160)
        ax = fig.add_subplot(111); ax.axis('off')
        ax.text(0.03, 0.94, 'HYPERFITPRO MATERIAL PASSPORT', fontsize=18, fontweight='bold', va='top')
        ax.text(0.03, 0.885, f'{ctx.model.number}  |  {ctx.model.name}', fontsize=15, fontweight='bold')
        ax.text(0.03, 0.845, f'Family: {getattr(ctx.model, "family", "-") or "-"}     Category: {getattr(ctx.model, "category", "-")}', fontsize=10)
        ax.text(0.76, 0.925, badge, fontsize=17, fontweight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.45', fc='white', ec='black', lw=1.3))

        left_metrics = [('RMSE', getattr(res,'rmse',None)), ('MAE', getattr(res,'mae',None)), ('R²', getattr(res,'r2',None)), ('Adj. R²', getattr(res,'adjusted_r2',None)), ('AICc', getattr(res,'aicc',None)), ('BIC', getattr(res,'bic',None))]
        ax.text(0.03, 0.76, 'Fit metrics', fontsize=12, fontweight='bold')
        y=0.72
        for k,v in left_metrics:
            ax.text(0.055, y, f'{k:<8} {_metric(v)}', fontsize=10, family='monospace'); y -= 0.04

        ax.text(0.31, 0.76, 'Data coverage', fontsize=12, fontweight='bold')
        y=0.72
        ax.text(0.335, y, f'Modes: {", ".join(modes)}', fontsize=10); y-=0.04
        ax.text(0.335, y, f'Datasets: {len(ctx.datasets)}', fontsize=10); y-=0.04
        ax.text(0.335, y, f'Points: {npts}', fontsize=10); y-=0.04
        for d in ctx.datasets[:6]:
            try:
                ax.text(0.335, y, f'{d.mode}: x=[{np.nanmin(d.x):.3g}, {np.nanmax(d.x):.3g}], n={len(d.x)}', fontsize=8.7)
            except Exception:
                ax.text(0.335, y, f'{getattr(d,"mode","-")}: n={_safe_len(getattr(d,"x",[]))}', fontsize=8.7)
            y-=0.032

        ax.text(0.03, 0.45, 'Fitted parameters', fontsize=12, fontweight='bold')
        xcols=[0.055, 0.36, 0.66]
        for i,(k,v) in enumerate(params.items()):
            col=i//8; row=i%8
            if col>2: break
            ax.text(xcols[col], 0.41-row*0.04, f'{k} = {v:.8g}', fontsize=9.2, family='monospace')
        if len(params)>24:
            ax.text(0.66, 0.09, f'+ {len(params)-24} more parameters', fontsize=9, family='monospace')

        ax.text(0.03, 0.085, 'Review notes', fontsize=11, fontweight='bold')
        for i, r in enumerate(risks[:4]):
            ax.text(0.055, 0.055 - i*0.027, f'• {r}', fontsize=8.7)
        fig.tight_layout()
        png = out / 'material_passport.png'
        fig.savefig(png)
        plt.close(fig)

        csv_path = out / 'material_passport_parameters.csv'
        with csv_path.open('w', newline='', encoding='utf-8') as f:
            wr = csv.writer(f)
            wr.writerow(['parameter','value'])
            for k,v in params.items(): wr.writerow([k, v])
            wr.writerow([]); wr.writerow(['metric','value'])
            for k,v in left_metrics: wr.writerow([k, v])
            wr.writerow(['risk_badge', badge])
            for r in risks: wr.writerow(['risk_note', r])

        pdf_path = out / 'material_passport.pdf'
        pdf_ok = False
        try:
            from reportlab.lib.pagesizes import landscape, A4
            from reportlab.pdfgen import canvas
            from reportlab.lib.utils import ImageReader
            c = canvas.Canvas(str(pdf_path), pagesize=landscape(A4))
            w,h = landscape(A4)
            c.drawImage(ImageReader(str(png)), 18, 18, width=w-36, height=h-36, preserveAspectRatio=True, anchor='c')
            c.save()
            pdf_ok = True
        except Exception as exc:
            (out/'pdf_generation_error.txt').write_text(str(exc), encoding='utf-8')
        md = out / 'material_passport.md'
        md.write_text(f'# Material Passport\n\nBadge: **{badge}**\n\nFiles: `{png.name}`, `{csv_path.name}`' + (f', `{pdf_path.name}`' if pdf_ok else '') + '\n', encoding='utf-8')
        files=[str(png), str(csv_path), str(md)]
        if pdf_ok: files.append(str(pdf_path))
        return ToolResult(self.plugin_id, self.name, True, f'Material passport generated with {badge} review badge.', files, payload={'badge': badge, 'risks': risks})


TOOL_CLASS = MaterialPassportGenerator
