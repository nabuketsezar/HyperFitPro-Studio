from __future__ import annotations
from pathlib import Path
import csv
import numpy as np
from .volumetric import J_from_x, hydrostatic_pressure_from_K


def _setup_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_fit(model, datasets, params, out_dir):
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    plt=_setup_matplotlib()
    files=[]
    # fit per mode
    for d in datasets:
        if d.mode=='volumetric':
            if 'K' not in params:
                continue
            yhat=hydrostatic_pressure_from_K(J_from_x(d.x), params['K'])
        else:
            yhat=model.predict_nominal(d.mode,d.x,params)
        fig=plt.figure(figsize=(8.2,5.2)); ax=fig.add_subplot(111)
        ax.plot(d.x,d.stress,'o',label='test data',markersize=4)
        order=np.argsort(d.x); ax.plot(d.x[order],yhat[order],'-',label='fitted model',linewidth=2)
        ax.set_title(f'{model.number} - {model.name} | {d.mode}')
        ax.set_xlabel(d.x_label); ax.set_ylabel(d.stress_label); ax.grid(True,alpha=0.3); ax.legend()
        p=out_dir/f'fit_{d.mode}_{Path(d.source_file).stem}.png'; fig.tight_layout(); fig.savefig(p,dpi=180); plt.close(fig); files.append(str(p))
        fig=plt.figure(figsize=(8.2,4.2)); ax=fig.add_subplot(111)
        ax.axhline(0,linewidth=1); ax.plot(d.x,yhat-d.stress,'o-',markersize=4)
        ax.set_title(f'Residuals | {d.mode}'); ax.set_xlabel(d.x_label); ax.set_ylabel('prediction - test'); ax.grid(True,alpha=0.3)
        p=out_dir/f'residuals_{d.mode}_{Path(d.source_file).stem}.png'; fig.tight_layout(); fig.savefig(p,dpi=180); plt.close(fig); files.append(str(p))
    # parity plot
    all_y=[]; all_hat=[]
    for d in datasets:
        if d.mode=='volumetric':
            if 'K' not in params:
                continue
            yh=hydrostatic_pressure_from_K(J_from_x(d.x), params['K'])
        else:
            yh=model.predict_nominal(d.mode,d.x,params)
        all_y.extend(list(d.stress)); all_hat.extend(list(yh))
    if all_y:
        all_y=np.asarray(all_y); all_hat=np.asarray(all_hat)
        fig=plt.figure(figsize=(5.8,5.8)); ax=fig.add_subplot(111)
        ax.plot(all_y,all_hat,'o',markersize=4)
        lo=float(np.nanmin([np.nanmin(all_y),np.nanmin(all_hat)])); hi=float(np.nanmax([np.nanmax(all_y),np.nanmax(all_hat)]))
        ax.plot([lo,hi],[lo,hi],'-',linewidth=1.5); ax.set_title('Parity plot'); ax.set_xlabel('test stress'); ax.set_ylabel('predicted stress'); ax.grid(True,alpha=0.3)
        p=out_dir/'parity_plot.png'; fig.tight_layout(); fig.savefig(p,dpi=180); plt.close(fig); files.append(str(p))
    # parameter bars
    if params:
        fig=plt.figure(figsize=(max(7,0.6*len(params)),4.8)); ax=fig.add_subplot(111)
        names=list(params.keys()); vals=[params[k] for k in names]
        ax.bar(range(len(vals)),vals); ax.set_xticks(range(len(vals))); ax.set_xticklabels(names,rotation=45,ha='right'); ax.set_title('Fitted parameters'); ax.grid(True,axis='y',alpha=0.3)
        p=out_dir/'fitted_parameters_bar.png'; fig.tight_layout(); fig.savefig(p,dpi=180); plt.close(fig); files.append(str(p))
    return files


def plot_trace(trace, out_dir):
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    if not trace: return []
    plt=_setup_matplotlib(); files=[]
    evals=np.array([r.get('eval',i) for i,r in enumerate(trace)],dtype=float)
    best=np.array([r.get('best_objective',np.nan) for r in trace],dtype=float)
    obj=np.array([r.get('objective',np.nan) for r in trace],dtype=float)
    fig=plt.figure(figsize=(8.2,4.8)); ax=fig.add_subplot(111)
    ax.semilogy(evals, np.maximum(obj,1e-30), '.', label='current')
    ax.semilogy(evals, np.maximum(best,1e-30), '-', label='best')
    ax.set_xlabel('function evaluation'); ax.set_ylabel('objective'); ax.set_title('Optimization trace'); ax.grid(True,alpha=0.3); ax.legend()
    p=out_dir/'optimization_trace.png'; fig.tight_layout(); fig.savefig(p,dpi=180); plt.close(fig); files.append(str(p))
    # parameter trace, if not too many
    param_keys=[k for k in trace[0].keys() if k.startswith('p_')]
    if param_keys:
        fig=plt.figure(figsize=(9.5,5.8)); ax=fig.add_subplot(111)
        for k in param_keys[:12]:
            vals=np.array([r.get(k,np.nan) for r in trace],dtype=float)
            ax.plot(evals,vals,label=k[2:])
        ax.set_xlabel('function evaluation'); ax.set_ylabel('parameter value'); ax.set_title('Parameter trace'); ax.grid(True,alpha=0.3); ax.legend(ncol=2,fontsize=8)
        p=out_dir/'parameter_trace.png'; fig.tight_layout(); fig.savefig(p,dpi=180); plt.close(fig); files.append(str(p))
    return files


def save_trace_csv(trace, path):
    path=Path(path)
    if not trace:
        path.write_text('',encoding='utf-8'); return
    keys=[]
    for r in trace:
        for k in r.keys():
            if k not in keys: keys.append(k)
    with path.open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=keys); wr.writeheader(); wr.writerows(trace)
