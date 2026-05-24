from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hyperfitpro.core.tool_plugins import ToolPlugin, ToolResult


class DataQualityInspector(ToolPlugin):
    plugin_id = 'data.quality_inspector'
    name = 'Data quality inspector'
    category = 'Data'
    description = 'Audits imported data for monotonicity, duplicates, outliers, jumps, raw/processed mismatch and test-range coverage.'
    requires_model = False
    requires_data = True
    requires_fit = False
    output_types = ['png', 'json', 'md']

    def run(self, ctx):
        out=ctx.output_root()/'data_quality_inspector'; out.mkdir(parents=True, exist_ok=True)
        files=[]; summary=[]
        for ds in ctx.datasets:
            x=np.asarray(ds.x,float); y=np.asarray(ds.stress,float)
            finite=np.isfinite(x)&np.isfinite(y)
            dx=np.diff(x[finite]) if finite.sum()>1 else np.array([])
            dy=np.diff(y[finite]) if finite.sum()>1 else np.array([])
            duplicate_count=int(len(x)-len(np.unique(np.round(x[finite],12)))) if finite.any() else 0
            monotonic_x=bool(np.all(dx>=-1e-12)) if dx.size else True
            jump_metric=float(np.nanmax(np.abs(dy))/(np.nanstd(y[finite])+1e-12)) if finite.sum()>3 else 0.0
            rec={
                'mode':ds.mode,'source_file':ds.source_file,'n_points':int(len(x)),'finite_points':int(finite.sum()),
                'x_min':float(np.nanmin(x)) if len(x) else None,'x_max':float(np.nanmax(x)) if len(x) else None,
                'stress_min':float(np.nanmin(y)) if len(y) else None,'stress_max':float(np.nanmax(y)) if len(y) else None,
                'duplicate_x_count':duplicate_count,'x_monotonic_non_decreasing':monotonic_x,'jump_metric':jump_metric,
                'preprocessing':ds.preprocessing or {}, 'diagnostics':ds.diagnostics or {},
            }
            flags=[]
            if finite.sum()<max(10, len(x)*0.9): flags.append('many_non_finite_values')
            if not monotonic_x: flags.append('x_not_monotonic')
            if duplicate_count>0: flags.append('duplicate_x_values')
            if jump_metric>12: flags.append('large_stress_jump_possible_outlier')
            rec['flags']=flags; summary.append(rec)
            fig, ax=plt.subplots(figsize=(7.2,4.6), dpi=140)
            if getattr(ds,'raw_x',None) is not None and getattr(ds,'raw_stress',None) is not None:
                ax.plot(ds.raw_x, ds.raw_stress, linestyle='--', marker='.', markersize=2, label='raw')
            ax.plot(x, y, marker='o', markersize=3, label='processed')
            ax.set_title(f'Data quality preview - {ds.mode}')
            ax.set_xlabel(ds.x_label or 'input'); ax.set_ylabel(ds.stress_label or 'stress')
            ax.grid(True, alpha=0.3); ax.legend()
            fig.tight_layout(); fp=out/f'data_quality_{ds.mode}.png'; fig.savefig(fp); plt.close(fig); files.append(str(fp))
        (out/'data_quality_summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8'); files.append(str(out/'data_quality_summary.json'))
        md=['# HyperFitPro data quality inspection\n']
        for rec in summary:
            md.append(f"\n## {rec['mode']}\n- points: {rec['n_points']}\n- x range: {rec['x_min']} to {rec['x_max']}\n- stress range: {rec['stress_min']} to {rec['stress_max']}\n- flags: {', '.join(rec['flags']) if rec['flags'] else 'none'}\n")
        (out/'data_quality_report.md').write_text('\n'.join(md), encoding='utf-8'); files.append(str(out/'data_quality_report.md'))
        nflags=sum(len(r['flags']) for r in summary)
        return ToolResult(self.plugin_id, self.name, True, f'Data quality inspection finished. Flags found: {nflags}.', files, payload={'summary':summary})


TOOL_CLASS = DataQualityInspector
