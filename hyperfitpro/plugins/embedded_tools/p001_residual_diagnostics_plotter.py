from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hyperfitpro.core.tool_plugins import ToolPlugin, ToolResult


class ResidualDiagnosticsPlotter(ToolPlugin):
    plugin_id = 'plot.residual_diagnostics'
    name = 'Residual diagnostics plotter'
    category = 'Plotter'
    description = 'Creates residual-vs-stretch, residual histogram and parity plots for every imported dataset.'
    requires_model = True
    requires_data = True
    requires_fit = True
    output_types = ['png', 'csv']
    autorun_after_fit = False

    def run(self, ctx):
        p = ctx.parameters or getattr(ctx.fit_result, 'parameters', {}) or {}
        out = ctx.output_root() / 'plot_residual_diagnostics'
        out.mkdir(parents=True, exist_ok=True)
        files=[]
        rows=[]
        all_true=[]; all_pred=[]; all_res=[]
        for ds in ctx.datasets:
            y_pred = ctx.model.predict_nominal(ds.mode, np.asarray(ds.x, dtype=float), p)
            res = y_pred - np.asarray(ds.stress, dtype=float)
            for x, y, yp, r in zip(ds.x, ds.stress, y_pred, res):
                rows.append({'mode': ds.mode, 'x': float(x), 'measured': float(y), 'predicted': float(yp), 'residual': float(r)})
            all_true.extend(np.asarray(ds.stress, dtype=float).tolist())
            all_pred.extend(np.asarray(y_pred, dtype=float).tolist())
            all_res.extend(np.asarray(res, dtype=float).tolist())
            fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=140)
            ax.axhline(0.0, linewidth=1)
            ax.plot(ds.x, res, marker='o', linestyle='-', markersize=3)
            ax.set_title(f'Residual vs input - {ds.mode}')
            ax.set_xlabel(ds.x_label or 'input')
            ax.set_ylabel('predicted - measured')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fp = out / f'residual_vs_x_{ds.mode}.png'
            fig.savefig(fp); plt.close(fig); files.append(str(fp))
        csv_path = out / 'residual_table.csv'
        with csv_path.open('w', newline='', encoding='utf-8') as f:
            wr=csv.DictWriter(f, fieldnames=['mode','x','measured','predicted','residual'])
            wr.writeheader(); wr.writerows(rows)
        files.append(str(csv_path))
        all_true=np.asarray(all_true,float); all_pred=np.asarray(all_pred,float); all_res=np.asarray(all_res,float)
        fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=150)
        ax.scatter(all_true, all_pred, s=14)
        mn=float(np.nanmin([all_true.min(), all_pred.min()])); mx=float(np.nanmax([all_true.max(), all_pred.max()]))
        ax.plot([mn,mx],[mn,mx], linewidth=1)
        ax.set_title('Parity plot')
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        ax.grid(True, alpha=0.3)
        fig.tight_layout(); fp=out/'parity_plot.png'; fig.savefig(fp); plt.close(fig); files.append(str(fp))
        fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=150)
        ax.hist(all_res[np.isfinite(all_res)], bins=35)
        ax.set_title('Residual histogram')
        ax.set_xlabel('Residual')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.25)
        fig.tight_layout(); fp=out/'residual_histogram.png'; fig.savefig(fp); plt.close(fig); files.append(str(fp))
        return ToolResult(self.plugin_id, self.name, True, f'Residual diagnostics generated: {len(files)} files.', files)


TOOL_CLASS = ResidualDiagnosticsPlotter
