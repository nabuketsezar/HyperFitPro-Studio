from __future__ import annotations

import csv

from hyperfitpro.core.tool_plugins import ToolPlugin, ToolResult


class NextTestDesigner(ToolPlugin):
    plugin_id = 'advisor.next_test_designer'
    name = 'Next-test DOE designer'
    category = 'Advisor'
    description = 'Designs a practical next-test matrix based on model complexity, missing modes, current strain coverage, and fit risk.'
    requires_model = True
    requires_data = True
    requires_fit = False
    output_types = ['xlsx', 'csv', 'html']

    def run(self, ctx):
        out=ctx.output_root()/'advisor_next_test_designer'; out.mkdir(parents=True, exist_ok=True)
        present={getattr(d,'mode','') for d in ctx.datasets}
        recommended=list(getattr(ctx.model,'recommended_tests',[]) or ['uniaxial','biaxial','planar'])
        all_modes=['uniaxial','biaxial','planar','simple_shear','volumetric']
        rows=[]
        npar=len(getattr(ctx.model,'parameter_specs',[]) or [])
        for mode in all_modes:
            loaded = mode in present
            priority = 'High' if mode in recommended and not loaded else ('Medium' if not loaded else 'Low')
            target_n = max(45, 10*npar) if mode!='volumetric' else 18
            if mode == 'simple_shear': xr='gamma = -1.0 to +1.0, include 0 densely'
            elif mode == 'volumetric': xr='J = 0.82 to 1.12 or hydrostatic compression range'
            else: xr='engineering strain = -0.20 to 1.50; add low-strain dense points 0-0.10'
            why=[]
            if not loaded: why.append('missing mode')
            if mode in recommended: why.append('recommended for selected model')
            if npar >= 5 and mode in ('biaxial','planar'): why.append('needed to reduce parameter correlation')
            rows.append({'mode':mode,'loaded':loaded,'priority':priority,'target_points':target_n,'target_range':xr,'reason':'; '.join(why) or 'already available'})
        csv_path=out/'next_test_matrix.csv'
        with csv_path.open('w', newline='', encoding='utf-8') as f:
            wr=csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
        files=[str(csv_path)]
        xlsx=out/'next_test_matrix.xlsx'
        try:
            import xlsxwriter
            wb=xlsxwriter.Workbook(str(xlsx))
            ws=wb.add_worksheet('Test Plan')
            bold=wb.add_format({'bold':True,'bg_color':'#EAF2F8','border':1})
            wrap=wb.add_format({'text_wrap':True,'valign':'top','border':1})
            for c,h in enumerate(rows[0].keys()): ws.write(0,c,h,bold)
            for r,row in enumerate(rows,1):
                for c,h in enumerate(rows[0].keys()): ws.write(r,c,row[h],wrap)
            ws.set_column(0,0,18); ws.set_column(1,2,12); ws.set_column(3,3,14); ws.set_column(4,5,44)
            ws2=wb.add_worksheet('CSV Template')
            headers=['mode','specimen_id','x','nominal_stress','weight','notes']
            for c,h in enumerate(headers): ws2.write(0,c,h,bold)
            for r in range(1,21):
                ws2.write(r,0,'uniaxial' if r<8 else ('biaxial' if r<14 else 'planar'))
                ws2.write(r,1,f'S{r:03d}')
            ws2.set_column(0,5,18)
            wb.close(); files.append(str(xlsx))
        except Exception as exc:
            (out/'xlsx_error.txt').write_text(str(exc), encoding='utf-8')
        html=out/'next_test_plan.html'
        table=''.join('<tr>' + ''.join(f'<td>{row[k]}</td>' for k in rows[0].keys()) + '</tr>' for row in rows)
        html.write_text(f'''<!doctype html><html><head><meta charset="utf-8"><style>body{{font-family:Segoe UI,Arial;margin:32px}} table{{border-collapse:collapse}} td,th{{border:1px solid #ddd;padding:8px;vertical-align:top}} th{{background:#eef6ff}}</style></head><body><h1>Next-test DOE Designer</h1><h2>{ctx.model.name}</h2><p>Purpose: reduce parameter uncertainty and stop overfitting by adding the most valuable missing deformation modes.</p><table><tr>{''.join(f'<th>{k}</th>' for k in rows[0].keys())}</tr>{table}</table></body></html>''', encoding='utf-8')
        files.append(str(html))
        return ToolResult(self.plugin_id,self.name,True,'Next-test design matrix generated.',files,tables={'next_test_matrix':rows})


TOOL_CLASS = NextTestDesigner
