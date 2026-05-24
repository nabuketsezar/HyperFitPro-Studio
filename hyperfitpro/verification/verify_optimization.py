from __future__ import annotations
from pathlib import Path

from hyperfitpro.core.registry import get_model
from hyperfitpro.core.tests import load_test_data
from hyperfitpro.core.optimizer import fit_model
from hyperfitpro.core.run_manager import create_run_folder, save_run
from hyperfitpro.core.model_comparison import compare_models, export_model_comparison_csv

root = Path(__file__).parent

def main():
    datasets = [
        load_test_data(root / 'sample_data' / 'neo_hookean_uniaxial.csv', 'uniaxial', 1.0),
        load_test_data(root / 'sample_data' / 'neo_hookean_biaxial.csv', 'biaxial', 1.0),
    ]
    m = get_model(6)
    run = create_run_folder(root / 'hyperfit_test_optimization', m)
    res = fit_model(
        m,
        datasets,
        method='recommended_adaptive_bounds',
        max_evals=900,
        validation_fraction=0.15,
        regularization=1e-6,
        regularization_type='l2',
        workers=1,
        checkpoint_path=run / 'logs' / 'optimizer_checkpoint.json',
        trace_stride=20,
    )
    save_run(m, datasets, res, run)
    assert res.success, res.message
    assert res.aicc is not None
    assert res.parameter_correlation is not None
    assert (run / 'logs' / 'optimization_diagnostics.json').exists()
    assert (run / 'logs' / 'parameter_confidence_intervals.csv').exists()
    rows, _ = compare_models([get_model(6), get_model(7)], datasets, method='least_squares_trf', max_evals=500)
    out = export_model_comparison_csv(rows, run / 'logs' / 'model_comparison.csv')
    assert Path(out).exists()
    print('Optimization verification passed.')
    print('Run folder:', run)
    print('RMSE:', res.rmse, 'AICc:', res.aicc, 'AdjR2:', res.adjusted_r2)

if __name__ == '__main__':
    main()
