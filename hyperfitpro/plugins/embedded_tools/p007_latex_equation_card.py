from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hyperfitpro.core.tool_plugins import ToolPlugin, ToolResult
from hyperfitpro.core.base_model import _replace_latex_parameters


class LatexEquationCard(ToolPlugin):
    plugin_id = 'report.latex_equation_card'
    name = 'LaTeX equation card generator'
    category = 'Report'
    description = 'Renders the symbolic model equation and, after a fit, the parameter-substituted equation as high-resolution PNG cards.'
    requires_model = True
    requires_data = False
    requires_fit = False
    output_types = ['png', 'tex']

    @staticmethod
    def _render(text, path, title, math=True):
        text=(text or '').replace('\\dfrac','\\frac')
        width=max(8.0, min(30.0, 4.0+0.055*len(text)))
        fig, ax=plt.subplots(figsize=(width,2.4), dpi=160)
        ax.axis('off')
        ax.text(0.01,0.86,title,fontsize=12,fontweight='bold',va='top')
        rendered = False
        if math:
            try:
                ax.text(0.01,0.42,f'${text}$',fontsize=14,va='center')
                fig.tight_layout(); fig.savefig(path); rendered = True
            except Exception:
                ax.clear(); ax.axis('off'); ax.text(0.01,0.86,title,fontsize=12,fontweight='bold',va='top')
        if not rendered:
            ax.text(0.01,0.42,text,fontsize=10,va='center',family='monospace')
            fig.tight_layout(); fig.savefig(path)
        plt.close(fig)

    def run(self, ctx):
        out=ctx.output_root()/'report_latex_equation_card'; out.mkdir(parents=True, exist_ok=True)
        eq=ctx.model.equation_latex or ''
        files=[]
        sym=out/'symbolic_equation_card.png'; self._render(eq, sym, f'{ctx.model.number} - {ctx.model.name}: symbolic equation'); files.append(str(sym))
        (out/'symbolic_equation.tex').write_text(eq, encoding='utf-8'); files.append(str(out/'symbolic_equation.tex'))
        params=ctx.parameters or getattr(ctx.fit_result,'parameters',{}) if ctx.fit_result else {}
        if params:
            num_eq=_replace_latex_parameters(eq, params)
            num=out/'numeric_equation_card.png'; self._render(num_eq, num, 'Parameter-substituted equation', math=False); files.append(str(num))
            (out/'numeric_equation.tex').write_text(num_eq, encoding='utf-8'); files.append(str(out/'numeric_equation.tex'))
        return ToolResult(self.plugin_id, self.name, True, 'Equation cards generated.', files)


TOOL_CLASS = LatexEquationCard
