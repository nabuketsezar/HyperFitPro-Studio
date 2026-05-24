from __future__ import annotations

from pathlib import Path
import tempfile

from hyperfitpro.core.registry import get_model
from hyperfitpro.core.tests import load_test_data
from hyperfitpro.core.optimizer import fit_model
from hyperfitpro.core.tool_plugins import discover_tool_plugins, ToolContext, run_tool_plugins


def main():
    tools = discover_tool_plugins()
    print('n_tool_plugins =', len(tools))
    if len(tools) < 10:
        raise SystemExit('Expected at least 10 embedded tool plugins')
    for t in tools:
        print(f'- {t.plugin_id} | {t.category} | {t.name}')

    model = get_model(6)
    root = Path(__file__).resolve().parents[2]
    ds = load_test_data(root / 'sample_data' / 'neo_hookean_uniaxial.csv', 'uniaxial', 1.0)
    res = fit_model(model, [ds], method='least_squares_trf', max_evals=300)
    with tempfile.TemporaryDirectory() as td:
        ctx = ToolContext(model=model, datasets=[ds], fit_result=res, run_folder=Path(td), parameters=res.parameters)
        selected = {
            'data.quality_inspector',
            'advisor.optimizer_recipe',
            'report.latex_equation_card',
            'plot.residual_diagnostics',
            'diagnostics.bounds_health_auditor',
        }
        results = run_tool_plugins(ctx, selected_ids=selected)
        failed = [r for r in results if not r.ok]
        for r in results:
            print(('OK' if r.ok else 'FAIL'), r.plugin_id, r.message)
        if failed:
            raise SystemExit('Some tool plugins failed: ' + ', '.join(r.plugin_id for r in failed))
        produced = sum(len(r.files) for r in results)
        if produced < 5:
            raise SystemExit('Tool plugins did not produce expected files')
    print('Tool plugin verification passed.')


if __name__ == '__main__':
    main()
