from __future__ import annotations

from pathlib import Path
import tempfile

from hyperfitpro.core.registry import get_model
from hyperfitpro.core.tests import load_test_data
from hyperfitpro.core.optimizer import fit_model
from hyperfitpro.core.run_manager import save_run
from hyperfitpro.core.tool_plugins import discover_tool_plugins, ToolContext, run_tool_plugins


def main():
    tools = discover_tool_plugins()
    ids = [getattr(t, 'plugin_id', '') for t in tools]
    print('tool_plugins_total =', len(tools))
    if len(tools) != 50:
        raise SystemExit(f'Expected exactly 50 embedded tool plugins, got {len(tools)}')
    if len(set(ids)) != len(ids):
        raise SystemExit('Duplicate tool plugin ids detected')

    model = get_model(6)
    root = Path(__file__).resolve().parents[2]
    ds1 = load_test_data(root / 'sample_data' / 'neo_hookean_uniaxial.csv', 'uniaxial', 1.0)
    ds2 = load_test_data(root / 'sample_data' / 'neo_hookean_biaxial.csv', 'biaxial', 1.0)
    result = fit_model(model, [ds1, ds2], method='least_squares_trf', max_evals=120)

    with tempfile.TemporaryDirectory() as td:
        run_folder = Path(td) / 'run'
        for sub in ['logs', 'plots', 'reports', 'exports', 'input_data', 'tools']:
            (run_folder / sub).mkdir(parents=True, exist_ok=True)
        save_run(model, [ds1, ds2], result, run_folder)
        ctx = ToolContext(model=model, datasets=[ds1, ds2], fit_result=result, run_folder=run_folder, parameters=result.parameters)
        results = run_tool_plugins(ctx, selected_ids=set(ids))
        failed = [r for r in results if not r.ok]
        print('tool_plugin_results =', len(results))
        print('files_created =', sum(len(r.files) for r in results))
        if failed:
            for r in failed:
                print('FAILED', r.plugin_id, r.message)
            raise SystemExit('One or more tool plugins failed')
    print('50-plugin verification passed.')


if __name__ == '__main__':
    main()
