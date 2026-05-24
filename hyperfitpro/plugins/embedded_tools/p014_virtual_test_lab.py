from __future__ import annotations

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hyperfitpro.core.tool_plugins import ToolPlugin, ToolResult


class VirtualTestLab(ToolPlugin):
    plugin_id = 'lab.virtual_test_lab'
    name = 'Virtual test lab'
    category = 'Lab'
    description = 'Generates virtual uniaxial/biaxial/planar/shear material response curves from the fitted model, like a mini digital test lab.'
    requires_model = True
    requires_data = True
    requires_fit = True
    output_types = ['png', 'csv', 'html']

    def _range_for(self, mode, datasets):
        xs=[]
        for d in datasets:
            if getattr(d, 'mode', '') == mode:
                xs.extend(np.asarray(d.x, float).tolist())
        if xs:
            lo, hi = float(np.nanmin(xs)), float(np.nanmax(xs))
            span = max(hi-lo, 0.05)
            return max(-0.45, lo-0.15*span), min(3.0, hi+0.25*span)
        defaults = {'uniaxial':(-0.35,1.75), 'biaxial':(-0.20,1.20), 'planar':(-0.25,1.50), 'simple_shear':(-1.50,1.50)}
        return defaults.get(mode, (0,1))

    def run(self, ctx):
        out = ctx.output_root() / 'lab_virtual_test_lab'
        out.mkdir(parents=True, exist_ok=True)
        params = dict(ctx.parameters or getattr(ctx.fit_result, 'parameters', {}) or {})
        modes = ['uniaxial','biaxial','planar','simple_shear']
        rows=[]; files=[]
        fig, ax = plt.subplots(figsize=(9.0, 5.8), dpi=150)
        for mode in modes:
            lo,hi = self._range_for(mode, ctx.datasets)
            xs=np.linspace(lo, hi, 220)
            try:
                ys=ctx.model.predict_nominal(mode, xs, params)
                ax.plot(xs, ys, label=mode)
                for x,y in zip(xs,ys): rows.append({'mode':mode,'x':float(x),'predicted_response':float(y)})
            except Exception as exc:
                rows.append({'mode':mode,'x':float('nan'),'predicted_response':float('nan'),'error':str(exc)})
        for d in ctx.datasets:
            if getattr(d,'mode','') in modes:
                ax.scatter(d.x, d.stress, s=16, alpha=0.50, label=f'{d.mode} data')
        ax.set_title(f'Virtual test lab - {ctx.model.name}')
        ax.set_xlabel('nominal strain / shear gamma')
        ax.set_ylabel('nominal stress / shear stress')
        ax.grid(True, alpha=0.30); ax.legend(fontsize=8, ncol=2)
        fig.tight_layout(); png=out/'virtual_test_lab_curves.png'; fig.savefig(png); plt.close(fig); files.append(str(png))
        csv_path=out/'virtual_test_lab_curves.csv'
        keys=sorted({k for r in rows for k in r})
        with csv_path.open('w', newline='', encoding='utf-8') as f:
            wr=csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(rows)
        files.append(str(csv_path))
        html=out/'virtual_test_lab.html'
        html.write_text(f'''<!doctype html><html><head><meta charset="utf-8"><title>Virtual Test Lab</title>
<style>body{{font-family:Segoe UI,Arial;margin:32px;color:#111827}} .card{{border:1px solid #ddd;border-radius:14px;padding:18px;max-width:1050px}} img{{max-width:100%;border:1px solid #eee;border-radius:10px}}</style></head>
<body><div class="card"><h1>Virtual Test Lab</h1><h2>{ctx.model.number} - {ctx.model.name}</h2><p>Generated standard deformation-path responses from the fitted parameters.</p><img src="{png.name}"><p><b>CSV:</b> {csv_path.name}</p></div></body></html>''', encoding='utf-8')
        files.append(str(html))
        return ToolResult(self.plugin_id, self.name, True, 'Virtual test lab generated.', files)


TOOL_CLASS = VirtualTestLab
