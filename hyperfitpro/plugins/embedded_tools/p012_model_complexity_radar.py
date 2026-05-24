from __future__ import annotations

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hyperfitpro.core.tool_plugins import ToolPlugin, ToolResult


class ModelComplexityRadar(ToolPlugin):
    plugin_id = 'advisor.model_complexity_radar'
    name = 'Model complexity radar'
    category = 'Advisor'
    description = 'Creates a radar-style visual of complexity, data coverage, parameter count, stability and overfitting risk.'
    requires_model = True
    requires_data = False
    requires_fit = False
    output_types = ['png', 'md']

    def run(self, ctx):
        out=ctx.output_root()/'advisor_model_complexity_radar'; out.mkdir(parents=True, exist_ok=True)
        model=ctx.model; npar=len(getattr(model,'parameter_specs',[]) or [])
        npts=sum(len(d.x) for d in ctx.datasets) if ctx.datasets else 0
        nmodes=len(set(d.mode for d in ctx.datasets)) if ctx.datasets else 0
        complexity=min(1.0, max(0.05, float(getattr(model,'complexity_level',npar))/10.0))
        param_score=min(1.0,npar/10.0)
        data_score=1.0-min(1.0, npar/max(npts,1)*8) if npts else 0.0
        mode_score=min(1.0,nmodes/4.0)
        stable_score=0.5
        if ctx.fit_result is not None and getattr(ctx.fit_result,'stability',None):
            stable_score=1.0 if ctx.fit_result.stability.get('overall_pass',False) else 0.2
        overfit_score=1.0-data_score
        labels=['Complexity','Parameter count','Data adequacy','Mode coverage','Stability','Overfit risk']
        vals=[complexity,param_score,data_score,mode_score,stable_score,overfit_score]
        angles=np.linspace(0,2*math.pi,len(labels),endpoint=False).tolist(); vals2=vals+[vals[0]]; angles2=angles+[angles[0]]
        fig=plt.figure(figsize=(6.4,6.0),dpi=150); ax=fig.add_subplot(111,polar=True)
        ax.plot(angles2,vals2,linewidth=2); ax.fill(angles2,vals2,alpha=0.18)
        ax.set_xticks(angles); ax.set_xticklabels(labels,fontsize=8); ax.set_ylim(0,1)
        ax.set_title(f'Model complexity radar: {model.number} {model.name}', pad=18)
        fig.tight_layout(); fp=out/'model_complexity_radar.png'; fig.savefig(fp); plt.close(fig)
        md=out/'model_complexity_radar.md'; md.write_text('\n'.join([f'- {l}: {v:.3f}' for l,v in zip(labels,vals)]),encoding='utf-8')
        return ToolResult(self.plugin_id,self.name,True,'Model complexity radar generated.',[str(fp),str(md)])


TOOL_CLASS = ModelComplexityRadar
