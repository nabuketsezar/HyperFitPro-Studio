from __future__ import annotations

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hyperfitpro.core.tool_plugins import ToolPlugin, ToolResult


class IdentifiabilitySVD(ToolPlugin):
    plugin_id = 'diagnostics.identifiability_svd'
    name = 'Identifiability SVD inspector'
    category = 'Diagnostics'
    description = 'Computes a finite-difference Jacobian, singular values, condition number, and weak parameter directions.'
    requires_model = True
    requires_data = True
    requires_fit = True
    output_types = ['png', 'csv', 'md']

    def run(self, ctx):
        out=ctx.output_root()/'diagnostics_identifiability_svd'; out.mkdir(parents=True, exist_ok=True)
        params=dict(ctx.parameters or getattr(ctx.fit_result,'parameters',{}) or {})
        names=list(params.keys())
        if not names:
            return ToolResult(self.plugin_id,self.name,False,'No parameters available.')
        x_blocks=[]
        for d in ctx.datasets:
            if getattr(d,'mode','') == 'volumetric':
                continue
            xs=np.asarray(d.x,float)
            try:
                base=ctx.model.predict_nominal(d.mode, xs, params)
            except Exception:
                continue
            scale=max(float(np.nanpercentile(np.abs(getattr(d,'stress',base)),90)), 1e-9)
            x_blocks.append((d.mode,xs,base,scale))
        if not x_blocks:
            return ToolResult(self.plugin_id,self.name,False,'No non-volumetric datasets could be evaluated.')
        cols=[]
        for nm in names:
            p2=dict(params)
            v=float(params[nm])
            delta=max(abs(v)*1e-4, 1e-7)
            p2[nm]=v+delta
            col=[]
            for mode,xs,base,scale in x_blocks:
                try:
                    yp=ctx.model.predict_nominal(mode, xs, p2)
                    col.extend(((yp-base)/delta/scale).tolist())
                except Exception:
                    col.extend([0.0]*len(xs))
            cols.append(col)
        J=np.asarray(cols,float).T
        J[~np.isfinite(J)] = 0.0
        try:
            U,S,Vt=np.linalg.svd(J, full_matrices=False)
        except Exception as exc:
            return ToolResult(self.plugin_id,self.name,False,f'SVD failed: {exc}')
        cond=float(S[0]/S[-1]) if len(S) and S[-1] > 1e-14 else float('inf')
        weak_vec=Vt[-1,:] if Vt.size else np.zeros(len(names))
        contrib={n:float(abs(v)) for n,v in zip(names,weak_vec)}
        ranked=sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
        files=[]
        fig,ax=plt.subplots(figsize=(7.2,4.5), dpi=150)
        ax.semilogy(range(1,len(S)+1), S, marker='o')
        ax.set_title(f'Identifiability singular values | cond={cond:.3g}')
        ax.set_xlabel('singular value index'); ax.set_ylabel('singular value')
        ax.grid(True, which='both', alpha=0.3)
        fig.tight_layout(); fp=out/'svd_singular_values.png'; fig.savefig(fp); plt.close(fig); files.append(str(fp))
        fig,ax=plt.subplots(figsize=(max(6,0.38*len(names)+2),4.6), dpi=150)
        ax.bar(range(len(names)), [contrib[n] for n in names])
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_title('Weakest identifiable direction contribution')
        ax.set_ylabel('|component| in weakest singular vector')
        ax.grid(True, axis='y', alpha=0.25)
        fig.tight_layout(); fp2=out/'weak_parameter_direction.png'; fig.savefig(fp2); plt.close(fig); files.append(str(fp2))
        csv_path=out/'identifiability_svd.csv'
        with csv_path.open('w', newline='', encoding='utf-8') as f:
            wr=csv.writer(f); wr.writerow(['singular_index','singular_value'])
            for i,s in enumerate(S,1): wr.writerow([i,float(s)])
            wr.writerow([]); wr.writerow(['condition_number',cond])
            wr.writerow([]); wr.writerow(['parameter','weak_direction_abs_component'])
            for n,v in ranked: wr.writerow([n,v])
        files.append(str(csv_path))
        md=out/'identifiability_recommendations.md'
        verdict = 'Excellent' if cond < 1e4 else ('Usable but correlated' if cond < 1e8 else 'Poor / ill-conditioned')
        top=', '.join(n for n,_ in ranked[:3])
        md.write_text(f'# Identifiability SVD\n\nVerdict: **{verdict}**\n\nCondition number: `{cond:.6g}`\n\nWeakest direction mostly involves: `{top}`\n\nRecommended action: add missing deformation modes, widen strain range, or freeze/re-bound the weakest parameters if confidence intervals are large.\n', encoding='utf-8')
        files.append(str(md))
        return ToolResult(self.plugin_id,self.name,True,f'Identifiability scan complete. Condition number={cond:.3g}',files,payload={'condition_number':cond,'weak_parameters':[n for n,_ in ranked[:5]]})


TOOL_CLASS = IdentifiabilitySVD
