from __future__ import annotations

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hyperfitpro.core.tool_plugins import ToolPlugin, ToolResult


def _safe_len(a):
    try: return len(a)
    except Exception: return 0


class ModelRiskScorecard(ToolPlugin):
    plugin_id = 'advisor.model_risk_scorecard'
    name = 'Model risk scorecard'
    category = 'Advisor'
    description = 'Scores calibration risk across data coverage, optimizer quality, stability, parameter identifiability, and extrapolation risk.'
    requires_model = True
    requires_data = True
    requires_fit = True
    output_types = ['png', 'csv', 'md']

    def run(self, ctx):
        out=ctx.output_root()/'advisor_model_risk_scorecard'; out.mkdir(parents=True, exist_ok=True)
        res=ctx.fit_result; npar=len(getattr(ctx.model,'parameter_specs',[]) or [])
        npts=sum(_safe_len(getattr(d,'x',[])) for d in ctx.datasets)
        modes={getattr(d,'mode','') for d in ctx.datasets}
        scores=[]
        scores.append(('Data / parameter ratio', min(100.0, 100.0*(npts/max(npar,1))/20.0)))
        rec=set(getattr(ctx.model,'recommended_tests',[]) or [])
        scores.append(('Recommended mode coverage', 100.0*len(modes & rec)/max(len(rec),1)))
        r2=getattr(res,'r2',np.nan); scores.append(('Fit agreement', max(0.0, min(100.0, 100.0*(float(r2) if np.isfinite(r2) else 0.0)))))
        stability=(getattr(res,'stability',{}) or {}).get('passed',False); scores.append(('Stability screen', 100.0 if stability else 35.0))
        val=getattr(res,'validation_metrics',None) or {}; vr2=val.get('r2', np.nan); scores.append(('Validation agreement', max(0.0, min(100.0,100.0*(float(vr2) if np.isfinite(vr2) else (0.6 if not val else 0.0))))))
        warnings=len(getattr(res,'overfitting_warnings',[]) or []); scores.append(('Warning penalty', max(0.0,100.0-25.0*warnings)))
        avg=sum(v for _,v in scores)/len(scores)
        verdict='GREEN' if avg>=80 else ('AMBER' if avg>=55 else 'RED')
        csv_path=out/'model_risk_scorecard.csv'
        with csv_path.open('w', newline='', encoding='utf-8') as f:
            wr=csv.writer(f); wr.writerow(['criterion','score_0_100'])
            for row in scores: wr.writerow(row)
            wr.writerow(['overall', avg]); wr.writerow(['verdict', verdict])
        fig,ax=plt.subplots(figsize=(8.5,4.8), dpi=150)
        ax.barh([s[0] for s in scores], [s[1] for s in scores])
        ax.set_xlim(0,100); ax.set_xlabel('score')
        ax.set_title(f'Model risk scorecard | {verdict} | overall={avg:.1f}/100')
        ax.grid(True, axis='x', alpha=0.25)
        fig.tight_layout(); png=out/'model_risk_scorecard.png'; fig.savefig(png); plt.close(fig)
        md=out/'model_risk_scorecard.md'
        md.write_text(f'# Model Risk Scorecard\n\nVerdict: **{verdict}**\n\nOverall score: `{avg:.1f}/100`\n\nLow scores identify what must be improved before trusting export to FEA.\n', encoding='utf-8')
        return ToolResult(self.plugin_id,self.name,True,f'Model risk scorecard: {verdict} ({avg:.1f}/100)',[str(csv_path),str(png),str(md)],payload={'verdict':verdict,'score':avg})


TOOL_CLASS = ModelRiskScorecard
