from __future__ import annotations

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hyperfitpro.core.tool_plugins import ToolPlugin, ToolResult


class ParameterCorrelationNetwork(ToolPlugin):
    plugin_id = 'plot.parameter_correlation_network'
    name = 'Parameter correlation network'
    category = 'Diagnostics'
    description = 'Visualizes fitted parameter correlation strength as a network diagram.'
    requires_model = True
    requires_data = False
    requires_fit = True
    output_types = ['png', 'md']

    def run(self, ctx):
        out=ctx.output_root()/'plot_parameter_correlation_network'; out.mkdir(parents=True, exist_ok=True)
        pc=getattr(ctx.fit_result,'parameter_correlation',{}) or {}
        names=pc.get('parameter_names') or list((ctx.parameters or getattr(ctx.fit_result,'parameters',{}) or {}).keys())
        mat=np.asarray(pc.get('matrix') or np.eye(len(names)), dtype=float)
        if len(names)==0:
            return ToolResult(self.plugin_id,self.name,False,'No parameters available.')
        n=len(names); theta=np.linspace(0,2*math.pi,n,endpoint=False)
        xy=np.c_[np.cos(theta),np.sin(theta)]
        fig, ax=plt.subplots(figsize=(6.6,6.2), dpi=150)
        ax.axis('off'); ax.set_aspect('equal')
        for i in range(n):
            for j in range(i+1,n):
                c=float(mat[i,j]) if mat.shape[0]>i and mat.shape[1]>j and np.isfinite(mat[i,j]) else 0.0
                if abs(c)>=0.55:
                    ax.plot([xy[i,0],xy[j,0]],[xy[i,1],xy[j,1]], linewidth=1+3*abs(c), alpha=min(0.85,0.25+0.6*abs(c)))
        ax.scatter(xy[:,0],xy[:,1],s=650,zorder=3)
        for (x,y),nm in zip(xy,names): ax.text(x,y,nm,ha='center',va='center',fontsize=8,color='white',fontweight='bold')
        ax.set_title('Parameter correlation network | edges shown for |corr| ≥ 0.55')
        fig.tight_layout(); fp=out/'parameter_correlation_network.png'; fig.savefig(fp); plt.close(fig)
        md=out/'parameter_correlation_network.md'
        md.write_text('High correlation indicates parameter non-uniqueness or overfitting risk. Inspect pairs with |corr| >= 0.55.\n', encoding='utf-8')
        return ToolResult(self.plugin_id,self.name,True,'Parameter correlation network generated.',[str(fp),str(md)])


TOOL_CLASS = ParameterCorrelationNetwork
