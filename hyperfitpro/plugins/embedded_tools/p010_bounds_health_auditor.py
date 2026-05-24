from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hyperfitpro.core.tool_plugins import ToolPlugin, ToolResult


class BoundsHealthAuditor(ToolPlugin):
    plugin_id = 'diagnostics.bounds_health_auditor'
    name = 'Bounds health auditor'
    category = 'Diagnostics'
    description = 'Checks whether fitted parameters are stuck near lower/upper bounds and plots relative bound positions.'
    requires_model = True
    requires_data = False
    requires_fit = True
    output_types = ['png', 'json']

    def run(self, ctx):
        out=ctx.output_root()/'diagnostics_bounds_health_auditor'; out.mkdir(parents=True, exist_ok=True)
        res=ctx.fit_result; bounds=getattr(res,'bounds',{}) or {}; params=getattr(res,'parameters',{}) or ctx.parameters or {}
        names=list(params.keys()); rel=[]; records=[]
        for nm in names:
            b=bounds.get(nm) or {}
            lo=b.get('lower') if isinstance(b,dict) else None; hi=b.get('upper') if isinstance(b,dict) else None
            val=float(params[nm])
            r=float('nan')
            flag='bounds_missing'
            if lo is not None and hi is not None and float(hi)>float(lo):
                r=(val-float(lo))/(float(hi)-float(lo))
                flag='near_lower_bound' if r<0.03 else ('near_upper_bound' if r>0.97 else 'ok')
            rel.append(r); records.append({'parameter':nm,'value':val,'relative_position':r,'flag':flag,'lower':lo,'upper':hi})
        fig, ax=plt.subplots(figsize=(8.0, max(3.5,0.35*len(names)+1.8)), dpi=140)
        y=np.arange(len(names)); ax.barh(y, np.nan_to_num(rel, nan=0.0)); ax.set_yticks(y); ax.set_yticklabels(names)
        ax.axvline(0.03, linestyle='--', linewidth=1); ax.axvline(0.97, linestyle='--', linewidth=1)
        ax.set_xlim(0,1); ax.set_xlabel('relative location inside optimizer bounds'); ax.set_title('Fitted parameter bound health')
        ax.grid(True, axis='x', alpha=0.3); fig.tight_layout(); fp=out/'bounds_health.png'; fig.savefig(fp); plt.close(fig)
        js=out/'bounds_health.json'; js.write_text(json.dumps(records,indent=2,default=str),encoding='utf-8')
        nbad=sum(1 for r in records if r['flag']!='ok')
        return ToolResult(self.plugin_id,self.name,True,f'Bounds health audit finished. Non-OK flags: {nbad}.',[str(fp),str(js)],payload={'records':records})


TOOL_CLASS = BoundsHealthAuditor
