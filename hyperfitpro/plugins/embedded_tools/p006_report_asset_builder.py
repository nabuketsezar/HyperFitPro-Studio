from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hyperfitpro.core.tool_plugins import ToolPlugin, ToolResult


class ReportAssetBuilder(ToolPlugin):
    plugin_id = 'report.asset_builder'
    name = 'Report asset builder'
    category = 'Report'
    description = 'Creates a clean one-page visual summary asset that can be dropped into reports or presentations.'
    requires_model = True
    requires_data = False
    requires_fit = True
    output_types = ['png', 'md']

    def run(self, ctx):
        out=ctx.output_root()/'report_asset_builder'; out.mkdir(parents=True, exist_ok=True)
        res=ctx.fit_result; params=getattr(res,'parameters',{}) or {}
        fig=plt.figure(figsize=(9.5,6.2), dpi=150)
        ax=fig.add_subplot(111); ax.axis('off')
        title=f'HyperFitPro Material Calibration Summary\n{ctx.model.number} - {ctx.model.name}'
        ax.text(0.02,0.94,title,fontsize=17,fontweight='bold',va='top')
        metrics=[('RMSE',getattr(res,'rmse',None)),('MAE',getattr(res,'mae',None)),('R²',getattr(res,'r2',None)),('AICc',getattr(res,'aicc',None)),('BIC',getattr(res,'bic',None))]
        y=0.76
        ax.text(0.02,y,'Fit metrics',fontsize=13,fontweight='bold'); y-=0.06
        for k,v in metrics:
            ax.text(0.04,y,f'{k}: {v:.6g}' if isinstance(v,(int,float)) and v==v else f'{k}: -',fontsize=11); y-=0.045
        ax.text(0.47,0.76,'Fitted parameters',fontsize=13,fontweight='bold')
        y2=0.70
        for k,v in list(params.items())[:14]:
            ax.text(0.49,y2,f'{k} = {v:.8g}',fontsize=10, family='monospace'); y2-=0.04
        if len(params)>14: ax.text(0.49,y2,f'... +{len(params)-14} more',fontsize=10)
        modes=', '.join(sorted({d.mode for d in ctx.datasets})) if ctx.datasets else '-'
        ax.text(0.02,0.18,f'Datasets: {len(ctx.datasets)}   Modes: {modes}',fontsize=10)
        ax.text(0.02,0.12,'Use this asset as report cover visual, engineering review slide, or calibration dossier summary.',fontsize=10,color='#555555')
        fig.tight_layout(); fp=out/'calibration_summary_card.png'; fig.savefig(fp); plt.close(fig)
        md=out/'calibration_summary_card.md'; md.write_text(f'# Calibration summary card\n\nGenerated image: {fp.name}\n', encoding='utf-8')
        return ToolResult(self.plugin_id, self.name, True, 'Report summary asset created.', [str(fp), str(md)])


TOOL_CLASS = ReportAssetBuilder
