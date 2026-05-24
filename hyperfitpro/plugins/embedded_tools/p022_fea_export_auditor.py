from __future__ import annotations

from pathlib import Path

from hyperfitpro.core.tool_plugins import ToolPlugin, ToolResult


class FEAExportAuditor(ToolPlugin):
    plugin_id = 'fea.export_auditor'
    name = 'FEA export auditor'
    category = 'FEA'
    description = 'Audits the generated FEA export bundle and writes a practical solver-readiness checklist.'
    requires_model = True
    requires_data = False
    requires_fit = True
    output_types = ['md', 'html']

    def run(self, ctx):
        out=ctx.output_root()/'fea_export_auditor'; out.mkdir(parents=True, exist_ok=True)
        run=Path(ctx.run_folder) if ctx.run_folder else None
        exports=run/'exports' if run else out
        expected=['abaqus_material.inp','calculix_material.inp','ansys_material.mac','lsdyna_material.k','marc_material.txt','unit_consistency_report.md','single_element_verification_targets.csv']
        rows=[]
        for fn in expected:
            p=exports/fn
            rows.append({'file':fn,'status':'FOUND' if p.exists() else 'MISSING','path':str(p)})
        ready=all(r['status']=='FOUND' for r in rows)
        md=out/'fea_export_audit.md'
        md.write_text('# FEA Export Audit\n\n' + '\n'.join(f'- **{r["file"]}**: {r["status"]}' for r in rows) + f'\n\nSolver readiness: **{"READY" if ready else "INCOMPLETE"}**\n\nChecklist:\n1. Verify stress unit in solver deck.\n2. Run provided single-element target curves.\n3. Compare solver reaction force/stress with HyperFitPro targets.\n4. Lock the material card only after passing verification.\n', encoding='utf-8')
        html=out/'fea_export_audit.html'
        table=''.join(f'<tr><td>{r["file"]}</td><td>{r["status"]}</td><td>{r["path"]}</td></tr>' for r in rows)
        html.write_text(f'<!doctype html><html><head><meta charset="utf-8"><style>body{{font-family:Segoe UI,Arial;margin:32px}}td,th{{border:1px solid #ddd;padding:8px}}table{{border-collapse:collapse}}</style></head><body><h1>FEA Export Audit</h1><h2>{"READY" if ready else "INCOMPLETE"}</h2><table><tr><th>File</th><th>Status</th><th>Path</th></tr>{table}</table></body></html>', encoding='utf-8')
        return ToolResult(self.plugin_id,self.name,True if ready else False, 'FEA export audit complete.' if ready else 'FEA export bundle is incomplete; generate FEA exports first.', [str(md),str(html)], tables={'audit':rows})


TOOL_CLASS = FEAExportAuditor
