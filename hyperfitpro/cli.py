from __future__ import annotations
import argparse
from pathlib import Path
from .core.registry import get_model, load_models
from .core.tests import load_test_data, auto_balance_dataset_weights
from .core.optimizer import fit_model, OPTIMIZATION_METHODS
from .core.run_manager import create_run_folder, save_run
from .core.model_comparison import compare_models, export_model_comparison_csv
from .core.config import ConfigManager
from .core.plugin_manager import write_model_plugin_template, write_tool_plugin_template
from .core.tool_plugins import discover_tool_plugins, ToolContext, run_tool_plugins
from .core.project import load_project, save_project, project_from_state
from .core.run_database import RunDatabase
from .core.architecture import architecture_status


def main(argv=None):
    ap=argparse.ArgumentParser(description='HyperFitPro Studio CLI')
    ap.add_argument('--list-models',action='store_true')
    ap.add_argument('--model',type=int,default=6)
    ap.add_argument('--method',default='hybrid_global_local',choices=OPTIMIZATION_METHODS+['hybrid','least_squares'])
    ap.add_argument('--max-evals',type=int,default=5000)
    ap.add_argument('--workdir',default=None)
    ap.add_argument('--regularization',type=float,default=0.0)
    ap.add_argument('--regularization-type', default='l2', choices=['none','l2','l1','elastic_net','magnitude'])
    ap.add_argument('--validation-fraction', type=float, default=0.0)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--early-stop-patience', type=int, default=0)
    ap.add_argument('--near-bound-penalty', type=float, default=0.0)
    ap.add_argument('--resume-from', default=None)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--compare-models', default=None, help='Comma separated model numbers, e.g. 6,7,8,11')
    ap.add_argument('--data', nargs=3, action='append', metavar=('MODE','FILE','WEIGHT'), help='Add dataset mode file weight')
    ap.add_argument('--data-kind', default='auto', choices=['auto','stress_strain','true_stress_strain','force_displacement','stretch_stress','volumetric'])
    ap.add_argument('--force-unit', default='N')
    ap.add_argument('--length-unit', default='mm')
    ap.add_argument('--stress-unit', default='MPa')
    ap.add_argument('--strain-percent', action='store_true')
    ap.add_argument('--area', type=float, default=None)
    ap.add_argument('--width', type=float, default=None)
    ap.add_argument('--thickness', type=float, default=None)
    ap.add_argument('--diameter', type=float, default=None)
    ap.add_argument('--gauge-length', type=float, default=None)
    ap.add_argument('--branch', default='all', choices=['all','loading','unloading','first_monotonic'])
    ap.add_argument('--zero-offset', action='store_true', default=True)
    ap.add_argument('--no-zero-offset', dest='zero_offset', action='store_false')
    ap.add_argument('--smooth-method', default='none', choices=['none','moving_average','savitzky_golay'])
    ap.add_argument('--smooth-window', type=int, default=0)
    ap.add_argument('--outlier-method', default='none', choices=['none','zscore','mad'])
    ap.add_argument('--downsample-max', type=int, default=0)
    ap.add_argument('--region-weighting', default='uniform', choices=['uniform','low_strain','high_strain','balanced_bins'])
    ap.add_argument('--auto-balance-weights', action='store_true')
    ap.add_argument('--save-project', default=None, help='Save a .hfp project file after setup/run')
    ap.add_argument('--load-project', default=None, help='Load model and dataset references from a .hfp project file')
    ap.add_argument('--list-runs', action='store_true', help='List recent indexed runs from the run database')
    ap.add_argument('--run-db', default=None, help='Optional run database path')
    ap.add_argument('--architecture-status', action='store_true', help='Print configuration, plugin and run-history status')
    ap.add_argument('--export-plugin-template', default=None, help='Write a user model plugin template to this path/directory')
    ap.add_argument('--export-tool-plugin-template', default=None, help='Write a user tool plugin template to this path/directory')
    ap.add_argument('--list-tool-plugins', action='store_true', help='List embedded and user non-model tool plugins')
    ap.add_argument('--run-tool-plugins', default=None, help='After fitting, run comma-separated tool plugin IDs, or all')
    ns=ap.parse_args(argv)
    if ns.export_plugin_template:
        path = write_model_plugin_template(ns.export_plugin_template)
        print(f'Model plugin template written: {path}')
        return 0
    if ns.export_tool_plugin_template:
        path = write_tool_plugin_template(ns.export_tool_plugin_template)
        print(f'Tool plugin template written: {path}')
        return 0
    if ns.list_tool_plugins:
        for t in discover_tool_plugins():
            needs=[]
            if getattr(t,'requires_model',False): needs.append('model')
            if getattr(t,'requires_data',False): needs.append('data')
            if getattr(t,'requires_fit',False): needs.append('fit')
            print(f"{t.plugin_id} | {t.category} | {t.name} | needs={','.join(needs) or 'none'} | outputs={','.join(getattr(t,'output_types',[]) or [])}")
        return 0
    if ns.architecture_status:
        import json
        print(json.dumps(architecture_status(), indent=2, default=str))
        return 0
    if ns.list_runs:
        db = RunDatabase(ns.run_db)
        for r in db.list_runs(limit=50):
            print(f"#{r['id']:04d} model={r.get('model_number')} {r.get('model_name')} method={r.get('method')} RMSE={r.get('rmse')} folder={r.get('run_folder')}")
        return 0
    if ns.load_project:
        project = load_project(ns.load_project)
        if project.model_number is not None:
            ns.model = int(project.model_number)
        if project.working_directory and not ns.workdir:
            ns.workdir = project.working_directory
        if project.optimizer_settings:
            ns.method = project.optimizer_settings.get('method', ns.method)
            ns.max_evals = int(project.optimizer_settings.get('max_evaluations', ns.max_evals))
            ns.regularization = float(project.optimizer_settings.get('regularization', ns.regularization))
            ns.regularization_type = project.optimizer_settings.get('regularization_type', ns.regularization_type)
            ns.validation_fraction = float(project.optimizer_settings.get('validation_fraction', ns.validation_fraction))
            ns.workers = int(project.optimizer_settings.get('workers', ns.workers))
        if not ns.data and project.datasets:
            ns.data = [(d.mode, d.source_file, str(d.weight)) for d in project.datasets]
    if ns.list_models:
        for m in load_models(include_inactive=True): print(f'{m.number:02d} | {m.category} | {m.name} | active={m.active}')
        return 0
    m=get_model(ns.model)
    if not m.active: raise SystemExit(f'Model {m.number} is disabled')
    datasets=[]
    for mode,file,weight in ns.data or []:
        datasets.append(load_test_data(
            file, mode, float(weight),
            data_kind=ns.data_kind, force_unit=ns.force_unit, length_unit=ns.length_unit, stress_unit=ns.stress_unit,
            strain_percent=ns.strain_percent, area=ns.area, width=ns.width, thickness=ns.thickness, diameter=ns.diameter,
            gauge_length=ns.gauge_length, branch=ns.branch, zero_offset=ns.zero_offset,
            smooth_method=ns.smooth_method, smooth_window=ns.smooth_window, outlier_method=ns.outlier_method,
            downsample_max_points=ns.downsample_max, region_weighting=ns.region_weighting,
        ))
    if ns.auto_balance_weights:
        auto_balance_dataset_weights(datasets)
    if not datasets: raise SystemExit('Use --data MODE FILE WEIGHT at least once')
    run_folder=create_run_folder(ns.workdir,m)
    
    def cli_progress(payload):
        if isinstance(payload, dict):
            rec = payload.get('record', {})
            if rec:
                print(f"eval={rec.get('eval')} obj={rec.get('objective'):.6e} best={rec.get('best_objective'):.6e}")
        else:
            print(payload)
    if ns.compare_models:
        nums=[int(x.strip()) for x in ns.compare_models.split(',') if x.strip()]
        models=[get_model(n) for n in nums]
        rows, _ = compare_models(models, datasets, method=ns.method, max_evals=ns.max_evals, regularization=ns.regularization, progress=cli_progress)
        out=export_model_comparison_csv(rows, run_folder/'logs'/'model_comparison.csv')
        print(f'Model comparison saved: {out}')
        for r in rows:
            print(f"rank={r['rank']} model={r['model_no']:02d} {r['model_name']} AICc={r['aicc']} RMSE={r['rmse']}")
        return 0
    res=fit_model(
        m,datasets,method=ns.method,max_evals=ns.max_evals,regularization=ns.regularization,progress=cli_progress,
        regularization_type=ns.regularization_type, validation_fraction=ns.validation_fraction, workers=ns.workers,
        early_stop_patience=ns.early_stop_patience, near_bound_penalty=ns.near_bound_penalty,
        checkpoint_path=ns.checkpoint or str(run_folder/'logs'/'optimizer_checkpoint.json'), resume_from=ns.resume_from,
    )
    summary = save_run(m,datasets,res,run_folder)
    if ns.run_tool_plugins:
        selected = None if ns.run_tool_plugins.lower() == 'all' else {x.strip() for x in ns.run_tool_plugins.split(',') if x.strip()}
        ctx = ToolContext(model=m, datasets=datasets, fit_result=res, run_folder=run_folder, workdir=ns.workdir, parameters=res.parameters, extra={'source':'cli'})
        tool_results = run_tool_plugins(ctx, selected_ids=selected)
        for tr in tool_results:
            print(('OK' if tr.ok else 'SKIP/FAIL') + f" tool={tr.plugin_id} {tr.message}")
            for fp in tr.files:
                print('  ', fp)
    if ns.save_project:
        optimizer_settings={
            'method': ns.method, 'max_evaluations': ns.max_evals, 'regularization': ns.regularization,
            'regularization_type': ns.regularization_type, 'validation_fraction': ns.validation_fraction, 'workers': ns.workers,
        }
        project = project_from_state(m, datasets, ns.workdir or '', optimizer_settings=optimizer_settings,
                                     last_run_folder=run_folder, last_fit_result=res)
        path = save_project(project, ns.save_project)
        print(f'Saved project: {path}')
    print(f'Saved run: {run_folder}')
    print(f'RMSE={res.rmse:.6g} R2={res.r2:.6g} AdjR2={res.adjusted_r2} AICc={res.aicc} BIC={res.bic}')
    if res.validation_metrics: print(f"Validation RMSE={res.validation_metrics.get('rmse')} R2={res.validation_metrics.get('r2')}")
    for w in res.overfitting_warnings: print('WARNING:', w)
    for k,v in res.parameters.items(): print(f'{k}={v:.12g}')
    return 0

if __name__=='__main__': raise SystemExit(main())
