from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hyperfitpro.core.tool_plugins import ToolPlugin, ToolResult


class LiveDashboardPack(ToolPlugin):
    plugin_id = 'plot.dashboard_pack'
    name = 'Dashboard plot pack'
    category = 'Plotter'
    description = 'Creates a compact multi-chart dashboard image: data-vs-fit, residuals, objective history and parameter values.'
    requires_model = True
    requires_data = True
    requires_fit = True
    output_types = ['png']

    def run(self, ctx):
        out=ctx.output_root()/'plot_dashboard_pack'; out.mkdir(parents=True, exist_ok=True)
        params=dict(ctx.parameters or getattr(ctx.fit_result,'parameters',{}) or {})
        fig=plt.figure(figsize=(12.5,8.2), dpi=135)
        ax1=fig.add_subplot(221); ax2=fig.add_subplot(222); ax3=fig.add_subplot(223); ax4=fig.add_subplot(224)
        for ds in ctx.datasets:
            yp=ctx.model.predict_nominal(ds.mode, np.asarray(ds.x,float), params)
            ax1.scatter(ds.x, ds.stress, s=12, label=f'{ds.mode} data')
            ax1.plot(ds.x, yp, linewidth=1.4, label=f'{ds.mode} fit')
            ax2.scatter(ds.x, yp-np.asarray(ds.stress,float), s=12, label=ds.mode)
        ax1.set_title('Data vs fit'); ax1.grid(True,alpha=0.3); ax1.legend(fontsize=7,ncol=2)
        ax2.axhline(0,linewidth=1); ax2.set_title('Residuals'); ax2.grid(True,alpha=0.3)
        trace=getattr(ctx.fit_result,'trace',[]) or []
        if trace:
            ev=[r.get('eval',i) for i,r in enumerate(trace)]
            best=[r.get('best_objective',r.get('objective',np.nan)) for r in trace]
            ax3.plot(ev,best); ax3.set_yscale('log')
        ax3.set_title('Objective history'); ax3.grid(True,alpha=0.3)
        names=list(params.keys()); vals=[params[k] for k in names]
        ax4.bar(np.arange(len(names)), vals); ax4.set_xticks(np.arange(len(names))); ax4.set_xticklabels(names,rotation=45,ha='right')
        ax4.set_title('Fitted parameters'); ax4.grid(True,axis='y',alpha=0.3)
        fig.suptitle(f'HyperFitPro dashboard - {ctx.model.number} {ctx.model.name}', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0,0,1,0.96]); fp=out/'calibration_dashboard_pack.png'; fig.savefig(fp); plt.close(fig)
        return ToolResult(self.plugin_id,self.name,True,'Dashboard plot pack generated.',[str(fp)])


TOOL_CLASS = LiveDashboardPack
