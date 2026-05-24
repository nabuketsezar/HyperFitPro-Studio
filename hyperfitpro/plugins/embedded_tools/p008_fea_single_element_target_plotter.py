from __future__ import annotations

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hyperfitpro.core.tool_plugins import ToolPlugin, ToolResult


class FEASingleElementTargetPlotter(ToolPlugin):
    plugin_id = 'fea.single_element_target_plotter'
    name = 'FEA single-element target plotter'
    category = 'FEA'
    description = 'Builds target curves for single-element verification in uniaxial, biaxial, planar and simple-shear modes.'
    requires_model = True
    requires_data = False
    requires_fit = True
    output_types = ['png', 'csv']

    def run(self, ctx):
        params=dict(ctx.parameters or getattr(ctx.fit_result,'parameters',{}) or {})
        out=ctx.output_root()/'fea_single_element_target_plotter'; out.mkdir(parents=True, exist_ok=True)
        files=[]; rows=[]
        ranges={'uniaxial':(-0.30,1.50),'biaxial':(0.0,1.00),'planar':(0.0,1.50),'simple_shear':(0.0,2.00)}
        for mode,(lo,hi) in ranges.items():
            xs=np.linspace(lo,hi,140)
            try: yp=ctx.model.predict_nominal(mode,xs,params)
            except Exception: continue
            for x,y in zip(xs,yp): rows.append({'mode':mode,'input':float(x),'target_response':float(y)})
            fig, ax=plt.subplots(figsize=(6.8,4.4), dpi=140)
            ax.plot(xs, yp)
            ax.set_title(f'Single-element target - {mode}')
            ax.set_xlabel('engineering strain / shear gamma'); ax.set_ylabel('nominal stress or shear stress')
            ax.grid(True, alpha=0.3)
            fig.tight_layout(); fp=out/f'fea_target_{mode}.png'; fig.savefig(fp); plt.close(fig); files.append(str(fp))
        csv_path=out/'single_element_target_curves.csv'
        with csv_path.open('w', newline='', encoding='utf-8') as f:
            wr=csv.DictWriter(f, fieldnames=['mode','input','target_response']); wr.writeheader(); wr.writerows(rows)
        files.append(str(csv_path))
        return ToolResult(self.plugin_id, self.name, True, 'FEA verification target plots generated.', files)


TOOL_CLASS = FEASingleElementTargetPlotter
