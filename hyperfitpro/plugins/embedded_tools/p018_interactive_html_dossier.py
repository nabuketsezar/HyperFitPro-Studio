from __future__ import annotations

import base64
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hyperfitpro.core.tool_plugins import ToolPlugin, ToolResult


def _embed(path):
    return base64.b64encode(Path(path).read_bytes()).decode('ascii')


def _fmt(v):
    try: return f'{float(v):.6g}'
    except Exception: return '-'


class InteractiveHTMLDossier(ToolPlugin):
    plugin_id = 'report.interactive_html_dossier'
    name = 'Interactive HTML calibration dossier'
    category = 'Report'
    description = 'Builds a polished standalone HTML dossier with embedded plots, equation, parameters, acceptance checklist, and expandable engineering sections.'
    requires_model = True
    requires_data = True
    requires_fit = True
    output_types = ['html', 'png']

    def run(self, ctx):
        out=ctx.output_root()/'report_interactive_html_dossier'; out.mkdir(parents=True, exist_ok=True)
        res=ctx.fit_result; params=dict(ctx.parameters or getattr(res,'parameters',{}) or {})
        files=[]
        fig,ax=plt.subplots(figsize=(8.5,5.2), dpi=150)
        for d in ctx.datasets:
            if getattr(d,'mode','')=='volumetric': continue
            xs=np.asarray(d.x,float)
            ax.scatter(xs, d.stress, s=16, alpha=0.50, label=f'{d.mode} data')
            try:
                xp=np.linspace(float(np.nanmin(xs)), float(np.nanmax(xs)), 220)
                yp=ctx.model.predict_nominal(d.mode, xp, params)
                ax.plot(xp, yp, label=f'{d.mode} fit')
            except Exception: pass
        ax.set_title('Measured data vs fitted model'); ax.set_xlabel('input'); ax.set_ylabel('response'); ax.grid(True, alpha=0.25); ax.legend(fontsize=8,ncol=2)
        fig.tight_layout(); fit_png=out/'dossier_fit_plot.png'; fig.savefig(fit_png); plt.close(fig); files.append(str(fit_png))
        trace=list(getattr(res,'trace',[]) or [])
        trace_png=None
        if trace:
            fig,ax=plt.subplots(figsize=(8.5,3.8), dpi=150)
            ax.semilogy([r.get('eval',i) for i,r in enumerate(trace)], [r.get('best_objective', r.get('objective', np.nan)) for r in trace])
            ax.set_title('Convergence history'); ax.set_xlabel('evaluation'); ax.set_ylabel('best objective'); ax.grid(True, which='both', alpha=0.25)
            fig.tight_layout(); trace_png=out/'dossier_convergence.png'; fig.savefig(trace_png); plt.close(fig); files.append(str(trace_png))
        rows=''.join(f'<tr><td>{k}</td><td>{v:.10g}</td></tr>' for k,v in params.items())
        warnings=''.join(f'<li>{w}</li>' for w in (getattr(res,'overfitting_warnings',[]) or [])) or '<li>No optimizer warning recorded.</li>'
        stability=getattr(res,'stability',{}) or {}
        stability_text='PASSED' if stability.get('passed') else 'REVIEW REQUIRED'
        html=out/'interactive_calibration_dossier.html'
        conv_html = f'<div class="card"><h2>Convergence</h2><img src="data:image/png;base64,{_embed(trace_png)}"></div>' if trace_png else ''
        html.write_text(f'''<!doctype html><html><head><meta charset="utf-8"><title>HyperFitPro Interactive Dossier</title>
<style>body{{font-family:Segoe UI,Arial;margin:0;background:#f3f4f6;color:#111827}} header{{background:#0f172a;color:white;padding:24px 36px}} main{{max-width:1120px;margin:24px auto;padding:0 22px}} .card{{background:white;border-radius:16px;padding:22px;margin:16px 0;box-shadow:0 8px 26px rgba(0,0,0,.08);border:1px solid #e5e7eb}} details{{margin:12px 0}} summary{{cursor:pointer;font-weight:700}} table{{border-collapse:collapse;width:100%}} td,th{{border-bottom:1px solid #e5e7eb;padding:9px;text-align:left}} img{{max-width:100%;border:1px solid #e5e7eb;border-radius:12px}} .badge{{display:inline-block;padding:8px 12px;border-radius:999px;background:#eef2ff;font-weight:700}}</style></head>
<body><header><h1>HyperFitPro Interactive Calibration Dossier</h1><p>{ctx.model.number} - {ctx.model.name}</p></header><main>
<div class="card"><h2>Executive summary</h2><p><span class="badge">RMSE {_fmt(getattr(res,'rmse',None))}</span> <span class="badge">R² {_fmt(getattr(res,'r2',None))}</span> <span class="badge">AICc {_fmt(getattr(res,'aicc',None))}</span> <span class="badge">Stability: {stability_text}</span></p></div>
<div class="card"><h2>Fit plot</h2><img src="data:image/png;base64,{_embed(fit_png)}"></div>
{conv_html}
<div class="card"><details open><summary>Model equation</summary><pre>{ctx.model.equation_latex}</pre></details><details><summary>Equation with fitted parameters</summary><pre>{ctx.model.equation_with_numbers(params)}</pre></details></div>
<div class="card"><h2>Parameters</h2><table><tr><th>Parameter</th><th>Value</th></tr>{rows}</table></div>
<div class="card"><h2>Acceptance checklist</h2><ul><li>Check stability result: {stability_text}</li><li>Check missing modes against recommended tests: {', '.join(getattr(ctx.model,'recommended_tests',[]) or [])}</li>{warnings}</ul></div>
</main></body></html>''', encoding='utf-8')
        files.append(str(html))
        return ToolResult(self.plugin_id,self.name,True,'Interactive HTML dossier generated.',files)


TOOL_CLASS = InteractiveHTMLDossier
