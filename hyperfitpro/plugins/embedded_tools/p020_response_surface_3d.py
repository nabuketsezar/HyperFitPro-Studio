from __future__ import annotations

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from hyperfitpro.core.tool_plugins import ToolPlugin, ToolResult


class ResponseSurface3D(ToolPlugin):
    plugin_id = 'plot.response_surface_3d'
    name = '3D response surface explorer'
    category = 'Plotter'
    description = 'Builds 3D surfaces: response as a function of deformation input and a fitted parameter multiplier.'
    requires_model = True
    requires_data = True
    requires_fit = True
    output_types = ['png', 'csv']

    def run(self, ctx):
        out=ctx.output_root()/'plot_response_surface_3d'; out.mkdir(parents=True, exist_ok=True)
        params=dict(ctx.parameters or getattr(ctx.fit_result,'parameters',{}) or {})
        names=list(params.keys())[:3]
        if not names:
            return ToolResult(self.plugin_id,self.name,False,'No parameters available.')
        mode = 'uniaxial'
        for d in ctx.datasets:
            if getattr(d,'mode','') != 'volumetric':
                mode=d.mode; break
        xs_all=[]
        for d in ctx.datasets:
            if getattr(d,'mode','') == mode: xs_all.extend(np.asarray(d.x,float).tolist())
        lo,hi=(float(np.nanmin(xs_all)),float(np.nanmax(xs_all))) if xs_all else (-0.25,1.5)
        X=np.linspace(lo,hi,80); M=np.linspace(0.55,1.45,46)
        files=[]; all_rows=[]
        for nm in names:
            Z=np.zeros((len(M),len(X)))
            for i,mult in enumerate(M):
                p2=dict(params); p2[nm]=float(params[nm])*float(mult)
                try: Z[i,:]=ctx.model.predict_nominal(mode, X, p2)
                except Exception: Z[i,:]=np.nan
            XX,MM=np.meshgrid(X,M)
            fig=plt.figure(figsize=(8.2,5.8), dpi=140)
            ax=fig.add_subplot(111, projection='3d')
            ax.plot_surface(XX,MM,Z, linewidth=0, antialiased=True, alpha=0.90)
            ax.set_title(f'3D response surface | {mode} | {nm} multiplier')
            ax.set_xlabel('input'); ax.set_ylabel(f'{nm} multiplier'); ax.set_zlabel('response')
            fig.tight_layout(); fp=out/f'response_surface_{mode}_{nm}.png'; fig.savefig(fp); plt.close(fig); files.append(str(fp))
            for i,mult in enumerate(M):
                for x,z in zip(X,Z[i,:]): all_rows.append({'mode':mode,'parameter':nm,'multiplier':float(mult),'x':float(x),'response':float(z) if np.isfinite(z) else ''})
        csv_path=out/'response_surface_3d_data.csv'
        with csv_path.open('w', newline='', encoding='utf-8') as f:
            wr=csv.DictWriter(f, fieldnames=['mode','parameter','multiplier','x','response']); wr.writeheader(); wr.writerows(all_rows)
        files.append(str(csv_path))
        return ToolResult(self.plugin_id,self.name,True,f'3D response surfaces generated for {len(names)} parameters.',files)


TOOL_CLASS = ResponseSurface3D
