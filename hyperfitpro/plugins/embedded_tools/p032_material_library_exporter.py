from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hyperfitpro.core.tool_plugins import ToolPlugin, ToolResult
from hyperfitpro.plugins.embedded_tools._creative_utils import out_dir, params, metrics, dataset_summary, dataset_arrays, write_json, write_csv, write_md, simple_html, escape_pre, safe_model_label, equation, zip_dir


class MaterialLibraryExporter(ToolPlugin):
    plugin_id = 'studio.material_library_exporter'
    name = 'Material library exporter'
    category = 'Studio'
    description = 'Exports fitted material data into a reusable JSON/CSV library record.'
    requires_model = True
    requires_data = False
    requires_fit = True
    output_types = ['md', 'csv', 'json', 'png', 'html']

    def run(self, ctx):
        out = out_dir(ctx, self.plugin_id)
        ps = params(ctx)
        ms = metrics(ctx)
        ds = dataset_summary(ctx)
        files = []
        payload = {'parameters': ps, 'metrics': ms, 'datasets': ds, 'model': safe_model_label(ctx)}
        files.append(str(write_json(out / 'payload.json', payload)))
        if ds:
            files.append(str(write_csv(out / 'dataset_summary.csv', ds)))
        if ps:
            files.append(str(write_csv(out / 'parameters.csv', [{'parameter': k, 'value': v} for k,v in ps.items()])))
        body = f"Model: **{safe_model_label(ctx)}**\n\nPlugin: **{self.plugin_id}**\n\nPurpose: {self.description}\n\n## Metrics\n\n"
        if ms:
            for k,v in ms.items():
                body += f'- {k}: {v:.6g}\n'
        else:
            body += '- Fit metrics are not available yet.\n'
        body += "\n## Parameters\n\n"
        if ps:
            for k,v in ps.items():
                try: body += f'- `{k}` = {float(v):.8g}\n'
                except Exception: body += f'- `{k}` = {v}\n'
        else:
            body += '- No fitted parameters available.\n'
        body += "\n## Data sets\n\n"
        if ds:
            for row in ds:
                body += f"- {row.get('mode')}: n={row.get('points')}, x=[{row.get('x_min')}, {row.get('x_max')}], y=[{row.get('y_min')}, {row.get('y_max')}]\n"
        else:
            body += '- No test data loaded.\n'
        body += "\n## Professional note\n\nThis plugin output is intended as an engineering review asset. It does not replace solver verification or material test traceability.\n"
        files.append(str(write_md(out / 'report.md', self.name, body)))
        html_path = out / 'report.html'
        html_path.write_text(simple_html(self.name, '<div class="card">' + escape_pre(body) + '</div>'), encoding='utf-8')
        files.append(str(html_path))
        fig = plt.figure(figsize=(8.8, 5.4), dpi=150)
        ax = fig.add_subplot(111)
        plotted = False
        if ds:
            for d in getattr(ctx, 'datasets', []) or []:
                x,y = dataset_arrays(d)
                if len(x) and len(y):
                    ax.plot(x, y, marker='o', linewidth=1.0, markersize=2.5, label=str(getattr(d, 'mode', '-')))
                    plotted = True
            ax.set_xlabel('Input variable')
            ax.set_ylabel('Response')
            ax.set_title(self.name + ' - data view')
        elif ps:
            labels=list(ps.keys()); vals=[float(v) for v in ps.values()]
            ax.bar(range(len(vals)), vals)
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_title(self.name + ' - parameter view')
            plotted = True
        else:
            ax.text(0.5,0.5,self.name+'\nNo data available', ha='center', va='center')
            plotted = True
        if plotted:
            ax.grid(True, alpha=0.25)
            try: ax.legend(loc='best')
            except Exception: pass
            fig.tight_layout()
            pp = out / 'overview.png'
            fig.savefig(pp)
            files.append(str(pp))
        plt.close(fig)
        files.extend(str(p) for p in self._extra(ctx, out, ps, ms, ds) if p)
        return ToolResult(self.plugin_id, self.name, True, f'{self.name} completed.', files, payload=payload)

    def _extra(self, ctx, out, ps, ms, ds):
        files=[]
        pid = self.plugin_id
        if pid == 'report.auto_report_composer':
            sections=['Cover page','Model equation','Imported data summary','Optimization setup','Fit metrics','Parameter table','Residual diagnostics','Stability scan','FEA export appendix','Engineering limitations']
            files.append(write_csv(out/'proposed_report_sections.csv', [{'order':i+1,'section':s,'active':True} for i,s in enumerate(sections)]))
        elif pid == 'studio.calibration_packager':
            root = Path(getattr(ctx, 'run_folder', '') or out)
            manifest = {'created_from': str(root), 'plugin': pid, 'files_detected': []}
            if root.exists():
                for p in list(root.rglob('*'))[:500]:
                    if p.is_file(): manifest['files_detected'].append(str(p.relative_to(root)))
            files.append(write_json(out/'package_manifest.json', manifest))
            files.append(zip_dir(out, out/'calibration_plugin_package.zip'))
        elif pid == 'diagnostics.unit_conversion_auditor':
            rows=[]
            for r in ds:
                ymax=r.get('y_max') or 0
                hint = 'MPa-like' if abs(ymax) < 1e4 else 'Pa/psi scale check needed'
                rows.append({'dataset':r.get('index'),'mode':r.get('mode'),'max_response':ymax,'scale_hint':hint})
            files.append(write_csv(out/'unit_scale_audit.csv', rows))
        elif pid == 'advisor.test_mode_gap_matrix':
            loaded={r.get('mode') for r in ds}
            required=['uniaxial','biaxial','planar','simple_shear','volumetric']
            files.append(write_csv(out/'test_mode_gap_matrix.csv', [{'mode':m,'loaded':m in loaded,'priority':'high' if m in ['uniaxial','biaxial','planar'] else 'medium'} for m in required]))
        elif pid == 'diagnostics.leave_one_dataset_out_validator':
            files.append(write_csv(out/'loo_validation_plan.csv', [{'holdout_dataset':r.get('index'),'holdout_mode':r.get('mode'),'recommended_action':'Refit without this dataset and compare validation RMSE'} for r in ds]))
        elif pid == 'diagnostics.bootstrap_parameter_sampler':
            rng=np.random.default_rng(2026); rows=[]
            for i in range(50):
                row={'sample':i+1}
                for k,v in ps.items(): row[k]=float(v)*(1+0.03*rng.normal())
                rows.append(row)
            files.append(write_csv(out/'bootstrap_parameter_samples.csv', rows))
        elif pid == 'fea.solver_keyword_comparator':
            run=Path(getattr(ctx,'run_folder','') or '')
            names=['abaqus_material.inp','calculix_material.inp','ansys_material.mac','lsdyna_material.k','marc_material.txt']
            files.append(write_csv(out/'solver_export_comparison.csv', [{'solver_file':n,'available':bool(run and (run/'exports'/n).exists())} for n in names]))
        elif pid == 'studio.material_library_exporter':
            files.append(write_json(out/'material_library_record.json', {'model':safe_model_label(ctx),'parameters':ps,'metrics':ms,'equation_latex':equation(ctx)}))
        elif pid == 'report.plot_style_sheet_generator':
            files.append(write_md(out/'plot_style_guide.md','Plot Style Guide','Use white background, clear axis labels, units in brackets, residual plots below primary fit plots, and 300 dpi images for reports.'))
        elif pid == 'advisor.decision_tree_model_selector':
            txt='Start: rubber-like material? -> Yes. Multiple modes? -> choose Ogden/Polynomial/Yeoh comparison. One mode only? -> prefer low-order Neo-Hookean/Yeoh and mark as provisional.'
            files.append(write_md(out/'model_selection_decision_tree.md','Model Selection Decision Tree',txt))
        elif pid == 'lab.parametric_sweep_batch':
            rows=[]
            for k,v in ps.items():
                for factor in [0.8,0.9,1.0,1.1,1.2]: rows.append({'parameter':k,'factor':factor,'value':float(v)*factor})
            files.append(write_csv(out/'parametric_sweep_grid.csv', rows))
        elif pid == 'report.design_allowable_table_builder':
            files.append(write_csv(out/'preliminary_allowable_grid.csv', [{'stretch':s,'engineering_note':'Use fitted model prediction and safety factor review before release'} for s in [1.05,1.1,1.2,1.3,1.5,2.0]]))
        elif pid == 'diagnostics.response_derivative_auditor':
            rows=[]
            for d in getattr(ctx,'datasets',[]) or []:
                x,y=dataset_arrays(d); bad=0
                if len(x)>2:
                    order=np.argsort(x); dy=np.diff(y[order]); bad=int(np.sum(dy < -1e-9))
                rows.append({'mode':getattr(d,'mode','-'),'negative_increments':bad,'points':len(x)})
            files.append(write_csv(out/'response_derivative_audit.csv', rows))
        elif pid == 'data.import_template_builder':
            for mode in ['uniaxial','biaxial','planar','simple_shear','volumetric','force_displacement']:
                p=out/f'{mode}_template.csv'
                p.write_text('displacement,force\n0,0\n' if mode=='force_displacement' else 'x,y\n1.0,0.0\n', encoding='utf-8')
                files.append(p)
        elif pid == 'data.time_series_synchronizer':
            files.append(write_md(out/'time_series_sync_guide.md','Time-Series Synchronization Guide','Columns: time, force, displacement, DIC_strain. Align by trigger pulse or maximum cross-correlation. Export synchronized x/y data before calibration.'))
        elif pid == 'report.quality_gate_signoff':
            gates=['Raw data reviewed','Units verified','At least two deformation modes reviewed','Overfit check reviewed','Stability scan reviewed','FEA single element probe completed','Report approved']
            files.append(write_csv(out/'quality_gate_signoff.csv',[{'gate':g,'status':'open','owner':''} for g in gates]))
        elif pid == 'studio.lims_package_builder':
            files.append(write_json(out/'lims_manifest.json',{'sample_id':'TBD','batch':'TBD','operator':'TBD','model':safe_model_label(ctx),'data_sets':ds,'parameters':ps}))
        elif pid == 'fea.versioned_material_card_builder':
            files.append(write_json(out/'versioned_material_card_metadata.json',{'version':'0.1-draft','status':'engineering review','model':safe_model_label(ctx),'parameters':ps,'reviewers':[]}))
        elif pid == 'advisor.ai_prompt_pack_generator':
            prompts=['Review this hyperelastic calibration for overfitting risks.','Generate an Abaqus single-element verification plan.','Write engineering limitations for this material card.','Compare this model against a lower-order Yeoh model.']
            files.append(write_md(out/'engineering_prompt_pack.md','Engineering AI Prompt Pack','\n'.join(f'- {p}' for p in prompts)))
        elif pid == 'report.training_asset_builder':
            cards=['What the model equation means','Why multiple test modes matter','How bounds affect optimization','How to verify FEA export']
            files.append(write_csv(out/'training_cards.csv',[{'card':i+1,'topic':c} for i,c in enumerate(cards)]))
        elif pid == 'studio.engineering_change_note':
            files.append(write_md(out/'engineering_change_note.md','Engineering Change Note',f'Material model updated to {safe_model_label(ctx)}. Parameters and fit metrics are attached. Release requires data, stability and FEA verification review.'))
        elif pid == 'studio.digital_twin_api_package':
            files.append(write_json(out/'digital_twin_material_api_payload.json',{'schema':'hyperfitpro.material.v1','model':safe_model_label(ctx),'parameters':ps,'metrics':ms}))
        elif pid == 'studio.release_readiness_board':
            score=0
            if ds: score+=25
            if len(ds)>=2: score+=20
            if ps: score+=20
            if ms.get('r2',0)>0.97: score+=20
            if ms.get('validation_rmse',0) or ms.get('rmse',0): score+=15
            status='GREEN' if score>=80 else 'AMBER' if score>=55 else 'RED'
            files.append(write_json(out/'release_readiness_score.json',{'score':score,'status':status,'model':safe_model_label(ctx)}))
        return files


TOOL_CLASS = MaterialLibraryExporter
