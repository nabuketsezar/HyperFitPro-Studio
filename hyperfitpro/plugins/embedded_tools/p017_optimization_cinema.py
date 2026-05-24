from __future__ import annotations

import base64
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hyperfitpro.core.tool_plugins import ToolPlugin, ToolResult


def _b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode('ascii')


class OptimizationCinema(ToolPlugin):
    plugin_id = 'plot.optimization_cinema'
    name = 'Optimization cinema'
    category = 'Plotter'
    description = 'Creates a self-contained HTML animation/slider showing how the fit evolved across optimization iterations.'
    requires_model = True
    requires_data = True
    requires_fit = True
    output_types = ['html', 'png']

    def run(self, ctx):
        out=ctx.output_root()/'plot_optimization_cinema'; out.mkdir(parents=True, exist_ok=True)
        trace=list(getattr(ctx.fit_result,'trace',[]) or [])
        if len(trace) < 2:
            return ToolResult(self.plugin_id,self.name,False,'Optimization trace is too short for cinema.')
        idx=np.linspace(0, len(trace)-1, min(36, len(trace))).astype(int)
        frames=[]; files=[]
        for j,i in enumerate(idx):
            rec=trace[int(i)]
            params={k[2:]:v for k,v in rec.items() if k.startswith('p_')}
            if not params: continue
            fig, ax = plt.subplots(figsize=(7.8,4.8), dpi=120)
            for d in ctx.datasets:
                if getattr(d,'mode','') == 'volumetric': continue
                xs=np.asarray(d.x,float)
                try:
                    xp=np.linspace(float(np.nanmin(xs)), float(np.nanmax(xs)), 150)
                    yp=ctx.model.predict_nominal(d.mode, xp, params)
                    ax.plot(xp, yp, label=f'{d.mode} fit')
                    ax.scatter(d.x, d.stress, s=10, alpha=0.45, label=f'{d.mode} data')
                except Exception:
                    pass
            best=rec.get('best_objective', rec.get('objective', np.nan))
            ax.set_title(f'Optimization frame {j+1}/{len(idx)} | eval={rec.get("eval")} | best={best:.3e}')
            ax.set_xlabel('input'); ax.set_ylabel('response')
            ax.grid(True, alpha=0.25); ax.legend(fontsize=7, ncol=2)
            fig.tight_layout(); fp=out/f'frame_{j:03d}.png'; fig.savefig(fp); plt.close(fig)
            files.append(str(fp)); frames.append((fp.name, _b64(fp), rec.get('eval'), best))
        if not frames:
            return ToolResult(self.plugin_id,self.name,False,'No frames could be produced from trace parameters.')
        imgs_js=',\n'.join(["{name:%r,src:%r,eval:%r,obj:%r}"%(n,s,e,o) for n,s,e,o in frames])
        html=out/'optimization_cinema.html'
        html.write_text(f'''<!doctype html><html><head><meta charset="utf-8"><title>Optimization Cinema</title>
<style>body{{font-family:Segoe UI,Arial;margin:26px;background:#fafafa;color:#111827}} .card{{background:white;border:1px solid #ddd;border-radius:16px;padding:20px;max-width:980px;box-shadow:0 8px 28px rgba(0,0,0,.08)}} img{{width:100%;border-radius:10px;border:1px solid #e5e7eb}} input{{width:100%}}</style></head>
<body><div class="card"><h1>Optimization Cinema</h1><p>Move the slider to see how the calibration evolved.</p><input id="s" type="range" min="0" max="{len(frames)-1}" value="0"><p id="cap"></p><img id="im"></div>
<script>const frames=[{imgs_js}]; const im=document.getElementById('im'), s=document.getElementById('s'), cap=document.getElementById('cap'); function draw(){{let f=frames[+s.value]; im.src='data:image/png;base64,'+f.src; cap.innerHTML='Frame '+(+s.value+1)+' / '+frames.length+' &nbsp; | &nbsp; eval='+f.eval+' &nbsp; | &nbsp; best objective='+Number(f.obj).toExponential(3);}} s.oninput=draw; draw();</script></body></html>''', encoding='utf-8')
        files.append(str(html))
        return ToolResult(self.plugin_id,self.name,True,f'Optimization cinema with {len(frames)} frames generated.',files)


TOOL_CLASS = OptimizationCinema
