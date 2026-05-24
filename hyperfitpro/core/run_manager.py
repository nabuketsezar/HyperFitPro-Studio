from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json, shutil, os, csv
from dataclasses import asdict
from .plotting import plot_fit, plot_trace, save_trace_csv
from .equation_logger import write_equation_report
from .exporter import export_json_material, export_abaqus_material, export_ansys_text, export_fea_bundle
from .reporting import generate_pdf_report, DEFAULT_REPORT_SECTIONS
from .excel_calculator import export_excel_calculator
from .data_processing import export_processed_dataset_csv
from .run_database import index_run

DEFAULT_ROOT = Path('C:/HyperFitPro_Runs') if os.name == 'nt' else Path.home() / 'HyperFitPro_Runs'


def create_run_folder(working_dir=None, model=None):
    base=Path(working_dir) if working_dir else DEFAULT_ROOT
    base.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    name=f'run_{stamp}'
    if model is not None:
        safe=''.join(c if c.isalnum() or c in '-_' else '_' for c in model.name)[:50]
        name += f'_{model.number:03d}_{safe}'
    path=base/name
    counter=1
    while path.exists():
        path=base/(name+f'_{counter:02d}'); counter+=1
    for sub in ['input_data','plots','exports','logs','reports']:
        (path/sub).mkdir(parents=True,exist_ok=True)
    return path


def save_run(model, datasets, fit_result, run_folder):
    run_folder=Path(run_folder)
    copied=[]
    processed=[]
    for i, d in enumerate(datasets, 1):
        src=Path(d.source_file)
        if src.exists():
            dst=run_folder/'input_data'/src.name
            try: shutil.copy2(src,dst); copied.append(str(dst))
            except Exception: pass
        try:
            safe_mode=''.join(c if c.isalnum() or c in '-_' else '_' for c in d.mode)
            proc_path=run_folder/'input_data'/f'processed_{i:02d}_{safe_mode}_{src.stem}.csv'
            export_processed_dataset_csv(d, proc_path)
            processed.append(str(proc_path))
        except Exception as e:
            (run_folder/'logs'/f'processed_dataset_{i:02d}_error.txt').write_text(str(e), encoding='utf-8')
    params=fit_result.parameters
    plot_files=plot_fit(model,datasets,params,run_folder/'plots')
    plot_files += plot_trace(fit_result.trace, run_folder/'plots')
    save_trace_csv(fit_result.trace, run_folder/'logs'/'optimization_trace.csv')
    # advanced optimization diagnostics: confidence intervals, parameter correlation,
    # overfitting flags, adaptive bounds history and information criteria.
    try:
        import json as _json, csv as _csv
        diag = {
            'adjusted_r2': fit_result.adjusted_r2,
            'aic': fit_result.aic,
            'aicc': fit_result.aicc,
            'bic': fit_result.bic,
            'train_metrics': fit_result.train_metrics,
            'validation_metrics': fit_result.validation_metrics,
            'confidence_intervals': fit_result.confidence_intervals,
            'parameter_correlation': fit_result.parameter_correlation,
            'regularization_summary': fit_result.regularization_summary,
            'overfitting_warnings': fit_result.overfitting_warnings,
            'adaptive_bounds_history': fit_result.adaptive_bounds_history,
            'checkpoint_file': fit_result.checkpoint_file,
            'stopped_by_user': fit_result.stopped_by_user,
        }
        (run_folder/'logs'/'optimization_diagnostics.json').write_text(_json.dumps(diag, indent=2), encoding='utf-8')
        if fit_result.confidence_intervals:
            with (run_folder/'logs'/'parameter_confidence_intervals.csv').open('w', newline='', encoding='utf-8') as fci:
                wr=_csv.writer(fci); wr.writerow(['parameter','value','standard_error','ci95_lower','ci95_upper','relative_se'])
                for k,v in fit_result.confidence_intervals.items():
                    wr.writerow([k, v.get('value'), v.get('standard_error'), v.get('ci95_lower'), v.get('ci95_upper'), v.get('relative_se')])
        pc = fit_result.parameter_correlation or {}
        if pc.get('available') and pc.get('matrix'):
            names = pc.get('parameter_names') or []
            with (run_folder/'logs'/'parameter_correlation_matrix.csv').open('w', newline='', encoding='utf-8') as fcorr:
                wr=_csv.writer(fcorr); wr.writerow([''] + list(names))
                for nm,row in zip(names, pc.get('matrix') or []):
                    wr.writerow([nm] + row)
    except Exception as e:
        (run_folder/'logs'/'optimization_diagnostics_error.txt').write_text(str(e), encoding='utf-8')
    write_equation_report(model,params,fit_result,run_folder/'reports'/'equation_report.md')
    try:
        export_fea_bundle(model, params, run_folder/'exports', material_name='HYPERFIT_MATERIAL', stress_unit='MPa', unit_system='MPa-mm-N')
    except Exception as e:
        (run_folder/'logs'/'fea_export_error.txt').write_text(str(e), encoding='utf-8')
    try:
        export_excel_calculator(model, params, datasets, fit_result, run_folder/'exports'/'model_calculator.xlsx')
    except Exception as e:
        (run_folder/'logs'/'excel_calculator_error.txt').write_text(str(e), encoding='utf-8')
    # fitted params csv
    with (run_folder/'fitted_parameters.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.writer(f); wr.writerow(['parameter','value'])
        for k,v in params.items(): wr.writerow([k, f'{v:.12g}'])
    summary={
        'model':model.metadata(),
        'datasets':[d.to_summary() for d in datasets],
        'fit_result':asdict(fit_result),
        'copied_input_files':copied,
        'processed_input_files':processed,
        'plot_files':plot_files,
    }
    (run_folder/'run_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    # simple HTML report
    html=['<html><head><meta charset="utf-8"><title>HyperFitPro Report</title><style>body{font-family:Segoe UI,Arial;margin:28px} table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:6px 10px} img{max-width:900px;border:1px solid #ddd;margin:10px}</style></head><body>']
    html.append(f'<h1>HyperFitPro Run Report</h1><h2>{model.number} - {model.name}</h2>')
    html.append(f'<p><b>Method:</b> {fit_result.method}<br><b>RMSE:</b> {fit_result.rmse:.6g}<br><b>MAE:</b> {fit_result.mae:.6g}<br><b>R²:</b> {fit_result.r2:.6g}<br><b>Adjusted R²:</b> {fit_result.adjusted_r2 if fit_result.adjusted_r2 is not None else "-"}<br><b>AICc:</b> {fit_result.aicc if fit_result.aicc is not None else "-"}<br><b>BIC:</b> {fit_result.bic if fit_result.bic is not None else "-"}<br><b>Objective:</b> {fit_result.objective:.6g}</p>')
    if fit_result.overfitting_warnings:
        html.append('<h3>Optimization / Overfitting Warnings</h3><ul>')
        for w in fit_result.overfitting_warnings:
            html.append(f'<li>{w}</li>')
        html.append('</ul>')
    html.append('<h3>Parameters</h3><table><tr><th>Parameter</th><th>Value</th></tr>')
    for k,v in params.items(): html.append(f'<tr><td>{k}</td><td>{v:.12g}</td></tr>')
    html.append('</table><h3>Plots</h3>')
    for pf in plot_files:
        rel=Path(pf).relative_to(run_folder)
        html.append(f'<div><img src="{rel.as_posix()}"></div>')
    html.append('</body></html>')
    (run_folder/'reports'/'run_report.html').write_text(''.join(html),encoding='utf-8')
    try:
        generate_pdf_report(run_folder, model, datasets, fit_result, plot_files=plot_files, sections=DEFAULT_REPORT_SECTIONS)
    except Exception as e:
        (run_folder/'logs'/'pdf_report_error.txt').write_text(str(e), encoding='utf-8')
    try:
        run_id = index_run(run_folder, summary)
        (run_folder/'logs'/'run_database_index.txt').write_text(f'run_id={run_id}\n', encoding='utf-8')
    except Exception as e:
        (run_folder/'logs'/'run_database_error.txt').write_text(str(e), encoding='utf-8')
    return summary
