from __future__ import annotations

import base64
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hyperfitpro.core.tool_plugins import ToolPlugin, ToolResult


def _b64(p): return base64.b64encode(Path(p).read_bytes()).decode('ascii')


class CalibrationStoryboard(ToolPlugin):
    plugin_id = 'story.calibration_storyboard'
    name = 'Calibration storyboard'
    category = 'Story'
    description = 'Creates a compact engineering storyboard: data ingestion, fit, residuals, parameters, and next actions in one HTML page.'
    requires_model = True
    requires_data = True
    requires_fit = True
    output_types = ['html', 'png']

    def run(self, ctx):
        out=ctx.output_root()/'story_calibration_storyboard'; out.mkdir(parents=True, exist_ok=True)
        files=[]
        fig,ax=plt.subplots(figsize=(8.5,3.5), dpi=150)
        for i,d in enumerate(ctx.datasets):
            try:
                ax.hlines(i, np.nanmin(d.x), np.nanmax(d.x), lw=8, label=d.mode)
                ax.scatter(d.x, np.full_like(d.x,i,dtype=float), s=8)
            except Exception: pass
        ax.set_yticks(range(len(ctx.datasets))); ax.set_yticklabels([getattr(d,'mode','-') for d in ctx.datasets])
        ax.set_xlabel('input coverage'); ax.set_title('Dataset coverage map'); ax.grid(True, axis='x', alpha=0.25)
        fig.tight_layout(); cov=out/'story_coverage_map.png'; fig.savefig(cov); plt.close(fig); files.append(str(cov))
        params=dict(ctx.parameters or getattr(ctx.fit_result,'parameters',{}) or {})
        fig,ax=plt.subplots(figsize=(8.5,3.7), dpi=150)
        names=list(params.keys()); vals=[abs(float(params[n])) for n in names]
        if vals:
            vals=np.asarray(vals,float); vals=vals/np.nanmax(vals) if np.nanmax(vals)>0 else vals
            ax.bar(range(len(names)), vals)
            ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha='right')
            ax.set_ylabel('normalized |value|')
        ax.set_title('Fitted parameter magnitude fingerprint'); ax.grid(True, axis='y', alpha=0.25)
        fig.tight_layout(); par=out/'story_parameter_fingerprint.png'; fig.savefig(par); plt.close(fig); files.append(str(par))
        res=ctx.fit_result
        html=out/'calibration_storyboard.html'
        actions=[]
        rec=set(getattr(ctx.model,'recommended_tests',[]) or [])
        present={getattr(d,'mode','') for d in ctx.datasets}
        missing=sorted(rec-present)
        if missing: actions.append('Add missing recommended test modes: ' + ', '.join(missing))
        if (getattr(res,'stability',{}) or {}).get('passed',False) is False: actions.append('Review stability scan before FEA export')
        if not actions: actions.append('Proceed to FEA single-element verification and material card review')
        action_html=''.join(f'<li>{a}</li>' for a in actions)
        html.write_text(f'''<!doctype html><html><head><meta charset="utf-8"><style>body{{font-family:Segoe UI,Arial;margin:32px;background:#fcfcfd;color:#111827}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}} .card{{background:white;border:1px solid #e5e7eb;border-radius:16px;padding:18px;box-shadow:0 4px 18px rgba(0,0,0,.06)}} img{{max-width:100%;border-radius:10px;border:1px solid #eee}} code{{background:#f3f4f6;padding:3px 6px;border-radius:5px}}</style></head><body><h1>Calibration Storyboard</h1><h2>{ctx.model.number} - {ctx.model.name}</h2><div class="grid"><div class="card"><h3>1. Dataset coverage</h3><img src="data:image/png;base64,{_b64(cov)}"></div><div class="card"><h3>2. Parameter fingerprint</h3><img src="data:image/png;base64,{_b64(par)}"></div><div class="card"><h3>3. Fit outcome</h3><p>RMSE: <code>{getattr(res,'rmse',None):.6g}</code></p><p>R²: <code>{getattr(res,'r2',None):.6g}</code></p><p>Method: <code>{getattr(res,'method','')}</code></p></div><div class="card"><h3>4. Next actions</h3><ol>{action_html}</ol></div></div></body></html>''', encoding='utf-8')
        files.append(str(html))
        return ToolResult(self.plugin_id,self.name,True,'Calibration storyboard generated.',files,payload={'next_actions':actions})


TOOL_CLASS = CalibrationStoryboard
