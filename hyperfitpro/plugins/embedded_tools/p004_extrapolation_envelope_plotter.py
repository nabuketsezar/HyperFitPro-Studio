from __future__ import annotations

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hyperfitpro.core.tool_plugins import ToolPlugin, ToolResult


class ExtrapolationEnvelopePlotter(ToolPlugin):
    plugin_id = 'plot.extrapolation_envelope'
    name = 'Extrapolation envelope plotter'
    category = 'Plotter'
    description = 'Plots fitted model response beyond measured range to reveal dangerous extrapolation behavior.'
    requires_model = True
    requires_data = True
    requires_fit = True
    output_types = ['png', 'csv']

    def run(self, ctx):
        params=dict(ctx.parameters or getattr(ctx.fit_result,'parameters',{}) or {})
        out=ctx.output_root()/'plot_extrapolation_envelope'; out.mkdir(parents=True, exist_ok=True)
        files=[]; rows=[]
        modes=sorted({d.mode for d in ctx.datasets if d.mode!='volumetric'}) or ['uniaxial','biaxial','planar','simple_shear']
        for mode in modes:
            xs_meas=np.concatenate([np.asarray(d.x,float) for d in ctx.datasets if d.mode==mode]) if any(d.mode==mode for d in ctx.datasets) else np.array([0.0,1.0])
            lo=float(np.nanmin(xs_meas)); hi=float(np.nanmax(xs_meas))
            span=max(hi-lo, 0.1)
            xlo=max(-0.8, lo-0.75*span)
            xhi=min(4.0, hi+1.50*span)
            if mode in ('biaxial','planar','uniaxial'):
                xlo=max(-0.75, xlo)
            xs=np.linspace(xlo, xhi, 260)
            yp=ctx.model.predict_nominal(mode, xs, params)
            for x,y in zip(xs,yp): rows.append({'mode':mode,'x':float(x),'predicted':float(y),'inside_measured_range':bool(lo<=x<=hi)})
            fig, ax=plt.subplots(figsize=(7.4,4.7), dpi=140)
            ax.plot(xs, yp, label='model prediction')
            ax.axvspan(lo, hi, alpha=0.18, label='measured range')
            for d in ctx.datasets:
                if d.mode==mode:
                    ax.scatter(d.x, d.stress, s=16, label=f'data {mode}')
            ax.set_title(f'Extrapolation envelope - {mode}')
            ax.set_xlabel('input strain/stretch parameter'); ax.set_ylabel('stress response')
            ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
            fig.tight_layout(); fp=out/f'extrapolation_envelope_{mode}.png'; fig.savefig(fp); plt.close(fig); files.append(str(fp))
        csv_path=out/'extrapolation_envelope_predictions.csv'
        with csv_path.open('w', newline='', encoding='utf-8') as f:
            wr=csv.DictWriter(f, fieldnames=['mode','x','predicted','inside_measured_range']); wr.writeheader(); wr.writerows(rows)
        files.append(str(csv_path))
        return ToolResult(self.plugin_id, self.name, True, 'Extrapolation envelope generated.', files)


TOOL_CLASS = ExtrapolationEnvelopePlotter
