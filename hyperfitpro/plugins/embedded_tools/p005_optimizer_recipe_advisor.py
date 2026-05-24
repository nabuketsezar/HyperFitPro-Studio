from __future__ import annotations

from hyperfitpro.core.tool_plugins import ToolPlugin, ToolResult


class OptimizerRecipeAdvisor(ToolPlugin):
    plugin_id = 'advisor.optimizer_recipe'
    name = 'Optimizer recipe advisor'
    category = 'Advisor'
    description = 'Generates a model/data-aware optimizer recommendation before long runs.'
    requires_model = True
    requires_data = False
    requires_fit = False
    output_types = ['md']

    def run(self, ctx):
        out=ctx.output_root()/'advisor_optimizer_recipe'; out.mkdir(parents=True, exist_ok=True)
        model=ctx.model
        npar=len(getattr(model,'parameter_specs',[]) or [])
        npts=sum(len(d.x) for d in ctx.datasets) if ctx.datasets else 0
        modes=sorted({d.mode for d in ctx.datasets}) if ctx.datasets else []
        complexity=getattr(model,'complexity_level', max(1,npar))
        advice=[]
        if npar<=2 and npts>0:
            advice.append('Start with fast_local_refine or least_squares_trf; global search is usually unnecessary.')
        elif npar<=5:
            advice.append('Use recommended_adaptive_bounds. If residuals remain structured, try hybrid_global_local.')
        else:
            advice.append('Use recommended_adaptive_bounds first. For rugged landscapes use regularized_multi_start with workers >= 4.')
        if len(modes)<2 and npar>=5:
            advice.append('Warning: high-parameter model with one test mode can overfit. Add biaxial/planar data or use regularization.')
        if any('Ogden' in getattr(model,'name','') for _ in [0]) or 'ogden' in getattr(model,'family','').lower():
            advice.append('Ogden-family model: use global-to-local optimization and inspect parameter correlation/equivalence.')
        if npts and npar and npts/npar < 8:
            advice.append(f'Data/parameter ratio is low ({npts}/{npar}). Increase data or reduce model order.')
        advice.append('Recommended defaults: validation split 0.10-0.20, elastic_net regularization 1e-6 to 1e-4 for high-order models, near-bound penalty 1e-5 if parameters hit bounds.')
        md=f"""# Optimizer recipe advisor\n\nModel: **{model.number} - {model.name}**  \nCategory: {getattr(model,'category','-')}  \nParameters: {npar}  \nComplexity level: {complexity}  \nDatasets: {len(ctx.datasets)}  \nModes: {', '.join(modes) if modes else 'not imported yet'}  \nData points: {npts}\n\n## Recommended workflow\n\n"""
        for i,a in enumerate(advice,1): md += f'{i}. {a}\n'
        md += "\n## Suggested method order\n\n1. recommended_adaptive_bounds\n2. hybrid_global_local\n3. regularized_multi_start\n4. least_squares_trf / fast_local_refine for final polish\n"
        path=out/'optimizer_recipe.md'; path.write_text(md, encoding='utf-8')
        return ToolResult(self.plugin_id, self.name, True, 'Optimizer recipe created.', [str(path)], payload={'advice':advice})


TOOL_CLASS = OptimizerRecipeAdvisor
