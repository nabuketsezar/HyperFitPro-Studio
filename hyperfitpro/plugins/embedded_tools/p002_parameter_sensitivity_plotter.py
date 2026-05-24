from __future__ import annotations

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hyperfitpro.core.tool_plugins import ToolPlugin, ToolResult


class ParameterSensitivityPlotter(ToolPlugin):
    plugin_id = 'plot.parameter_sensitivity'
    name = 'Parameter sensitivity plotter'
    category = 'Plotter'
    description = 'Perturbs each fitted parameter and plots normalized stress-response sensitivity by test mode.'
    requires_model = True
    requires_data = True
    requires_fit = True
    output_types = ['png', 'csv']

    def run(self, ctx):
        params = dict(ctx.parameters or getattr(ctx.fit_result, 'parameters', {}) or {})
        out = ctx.output_root() / 'plot_parameter_sensitivity'
        out.mkdir(parents=True, exist_ok=True)
        files=[]; rows=[]
        names=list(params.keys())
        if not names:
            return ToolResult(self.plugin_id, self.name, False, 'No fitted parameters were available.')
        for ds in ctx.datasets:
            xs=np.linspace(float(np.nanmin(ds.x)), float(np.nanmax(ds.x)), 160)
            base=ctx.model.predict_nominal(ds.mode, xs, params)
            fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=140)
            for nm in names:
                val=float(params[nm])
                delta=max(abs(val)*0.02, 1e-8)
                p2=dict(params); p2[nm]=val+delta
                try:
                    yp=ctx.model.predict_nominal(ds.mode, xs, p2)
                    sens=(yp-base)/delta
                    scale=max(np.nanmax(np.abs(sens)), 1e-30)
                    ax.plot(xs, sens/scale, label=nm)
                    rows.append({'mode':ds.mode,'parameter':nm,'max_abs_sensitivity':float(np.nanmax(np.abs(sens))),'delta':delta})
                except Exception:
                    rows.append({'mode':ds.mode,'parameter':nm,'max_abs_sensitivity':float('nan'),'delta':delta})
            ax.set_title(f'Normalized parameter sensitivity - {ds.mode}')
            ax.set_xlabel(ds.x_label or 'input')
            ax.set_ylabel('normalized d(response)/d(parameter)')
            ax.grid(True, alpha=0.3); ax.legend(fontsize=8, ncol=2)
            fig.tight_layout(); fp=out/f'parameter_sensitivity_{ds.mode}.png'; fig.savefig(fp); plt.close(fig); files.append(str(fp))
        csv_path=out/'parameter_sensitivity_summary.csv'
        with csv_path.open('w', newline='', encoding='utf-8') as f:
            wr=csv.DictWriter(f, fieldnames=['mode','parameter','max_abs_sensitivity','delta'])
            wr.writeheader(); wr.writerows(rows)
        files.append(str(csv_path))
        return ToolResult(self.plugin_id, self.name, True, 'Parameter sensitivity plots generated.', files)


TOOL_CLASS = ParameterSensitivityPlotter
