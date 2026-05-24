
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import html
import math

DEFAULT_REPORT_SECTIONS = [
    {'key':'summary','title':'Run summary','enabled':True,'order':10,'description':'Model, method, runtime and high-level result metrics.'},
    {'key':'datasets','title':'Imported test data','enabled':True,'order':20,'description':'Test modes, point counts, weights and source files.'},
    {'key':'parameters','title':'Fitted parameters','enabled':True,'order':30,'description':'Final optimized material coefficients.'},
    {'key':'metrics','title':'Fit metrics','enabled':True,'order':40,'description':'Objective, RMSE, MAE, R-squared and optimizer status.'},
    {'key':'stability','title':'Stability scan','enabled':True,'order':50,'description':'Fast path-based sanity checks before FEA export.'},
    {'key':'equation_symbolic','title':'Symbolic equation','enabled':True,'order':60,'description':'Original model equation in LaTeX notation.'},
    {'key':'equation_numeric','title':'Equation with fitted numbers','enabled':True,'order':70,'description':'Same equation with optimized parameters substituted.'},
    {'key':'plots','title':'Plots','enabled':True,'order':80,'description':'Fit, residual, parity and optimization trace figures.'},
    {'key':'exports','title':'Generated exports','enabled':True,'order':90,'description':'Abaqus, ANSYS, PDF, Excel calculator and auxiliary files.'},
]


def normalize_sections(sections=None):
    base={s['key']:dict(s) for s in DEFAULT_REPORT_SECTIONS}
    if sections:
        for s in sections:
            if not s: continue
            key=s.get('key')
            if key in base:
                base[key].update({k:v for k,v in s.items() if k in ('enabled','order','title','description')})
    return sorted(base.values(), key=lambda x: (int(x.get('order',999)), x.get('title','')))


def generate_report_preview(model, datasets, fit_result=None, sections=None) -> str:
    lines=[]
    lines.append('PDF REPORT PREVIEW / GUIDE')
    lines.append('='*72)
    lines.append(f'Model: {model.number:02d} - {model.name}')
    lines.append(f'Category: {model.category}')
    if fit_result:
        lines.append(f'Current result: RMSE={fit_result.rmse:.6g}, R²={fit_result.r2:.6g}, objective={fit_result.objective:.6g}')
    else:
        lines.append('Current result: no completed run yet. Report can be previewed, but PDF requires a completed run.')
    lines.append('')
    lines.append('Included sections in order:')
    for sec in normalize_sections(sections):
        state='ON ' if sec.get('enabled',True) else 'OFF'
        lines.append(f"  {state} | {int(sec.get('order',0)):03d} | {sec.get('title')} - {sec.get('description','')}")
    lines.append('')
    lines.append('Imported datasets:')
    if datasets:
        for i,d in enumerate(datasets,1):
            lines.append(f'  {i}. {d.mode}: {len(d.x)} points, weight={d.weight:g}, file={Path(d.source_file).name}')
    else:
        lines.append('  No dataset imported yet.')
    return '\n'.join(lines)


def _fmt(v, nd=8):
    try:
        if v is None or not math.isfinite(float(v)): return '-'
        return f'{float(v):.{nd}g}'
    except Exception:
        return str(v)


def _para(text, style):
    from reportlab.platypus import Paragraph
    # Keep LaTeX/code readable in ReportLab while avoiding markup crashes.
    text = html.escape(str(text)).replace('\n','<br/>')
    return Paragraph(text, style)


def generate_pdf_report(run_folder, model, datasets, fit_result, plot_files=None, sections=None, out_path=None):
    """Create a polished PDF run report.

    The report is intentionally selectable by section. GUI users can enable/disable
    and reorder sections before calling this function.
    """
    run_folder=Path(run_folder)
    out_path=Path(out_path) if out_path else run_folder/'reports'/'run_report.pdf'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_files=[Path(p) for p in (plot_files or []) if Path(p).exists()]

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='SmallCode', parent=styles['BodyText'], fontName='Courier', fontSize=7.2, leading=9, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name='Muted', parent=styles['BodyText'], textColor=colors.HexColor('#475569'), fontSize=8.5, leading=11))
    styles['Title'].fontSize=22
    styles['Title'].textColor=colors.HexColor('#0f172a')
    styles['Heading1'].textColor=colors.HexColor('#1d4ed8')
    styles['Heading2'].textColor=colors.HexColor('#0f172a')

    doc=SimpleDocTemplate(str(out_path), pagesize=A4, rightMargin=1.35*cm, leftMargin=1.35*cm, topMargin=1.2*cm, bottomMargin=1.2*cm)
    story=[]
    story.append(Paragraph('HyperFitPro Studio Report', styles['Title']))
    story.append(Paragraph(f'{model.number:02d} - {model.name}', styles['Heading2']))
    story.append(Paragraph(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', styles['Muted']))
    story.append(Spacer(1, 0.25*cm))

    def table(data, widths=None):
        t=Table(data, colWidths=widths, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eff6ff')),
            ('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#0f172a')),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('FONTSIZE',(0,0),(-1,-1),8),
            ('GRID',(0,0),(-1,-1),0.3,colors.HexColor('#cbd5e1')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, colors.HexColor('#f8fafc')]),
            ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
            ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
        ]))
        return t

    for sec in normalize_sections(sections):
        if not sec.get('enabled',True):
            continue
        key=sec['key']; title=sec.get('title',key)
        story.append(Paragraph(title, styles['Heading1']))
        if key=='summary':
            data=[['Item','Value'],['Model',f'{model.number:02d} - {model.name}'],['Category',model.category],['Method',fit_result.method],['Success',str(fit_result.success)],['Elapsed time',_fmt(fit_result.elapsed_s)+' s'],['Function evaluations',str(fit_result.nfev)],['Run folder',str(run_folder)]]
            story.append(table(data,[5*cm,12*cm]))
        elif key=='datasets':
            data=[['#','Mode','Points','Weight','Source file']]
            for i,d in enumerate(datasets,1): data.append([str(i),d.mode,str(len(d.x)),_fmt(d.weight),Path(d.source_file).name])
            story.append(table(data,[1*cm,3*cm,2*cm,2*cm,9*cm]))
        elif key=='parameters':
            data=[['Parameter','Fitted value']]+[[k,_fmt(v,12)] for k,v in fit_result.parameters.items()]
            story.append(table(data,[5*cm,7*cm]))
        elif key=='metrics':
            data=[['Metric','Value'],['Objective',_fmt(fit_result.objective,12)],['RMSE',_fmt(fit_result.rmse,12)],['MAE',_fmt(fit_result.mae,12)],['R-squared',_fmt(fit_result.r2,12)],['Optimizer message',fit_result.message]]
            story.append(table(data,[5*cm,12*cm]))
        elif key=='stability':
            stability=fit_result.stability or {}
            data=[['Item','Value'],['Passed',str(stability.get('passed','-'))]]
            for w in stability.get('warnings',[]) or []: data.append(['Warning',str(w)])
            mode_results=stability.get('mode_results',{}) or {}
            story.append(table(data,[4*cm,13*cm]))
            if mode_results:
                data=[['Mode','Finite fraction','Positive tangent fraction','Negative W points','Passed']]
                for mode,vals in mode_results.items(): data.append([mode,_fmt(vals.get('finite_fraction')), _fmt(vals.get('positive_tangent_fraction')), str(vals.get('negative_energy_points','-')), str(vals.get('passed','-'))])
                story.append(Spacer(1,0.15*cm)); story.append(table(data,[3*cm,3.2*cm,4.2*cm,3.4*cm,2.2*cm]))
        elif key=='equation_symbolic':
            story.append(_para(model.equation_latex, styles['SmallCode']))
        elif key=='equation_numeric':
            story.append(_para(model.equation_with_numbers(fit_result.parameters), styles['SmallCode']))
        elif key=='plots':
            if not plot_files:
                story.append(Paragraph('No plot files found for this run.', styles['BodyText']))
            for p in plot_files:
                try:
                    story.append(Paragraph(p.name, styles['Heading2']))
                    img=Image(str(p), width=16.5*cm, height=10.0*cm)
                    img.hAlign='LEFT'
                    story.append(img)
                    story.append(Spacer(1,0.2*cm))
                except Exception as e:
                    story.append(Paragraph(f'Could not embed {p.name}: {e}', styles['Muted']))
        elif key=='exports':
            export_dir=run_folder/'exports'
            report_dir=run_folder/'reports'
            files=[]
            for folder in [export_dir, report_dir, run_folder]:
                if folder.exists():
                    for f in folder.iterdir():
                        if f.is_file(): files.append([f.name, str(f.relative_to(run_folder))])
            story.append(table([['File','Relative path']]+files,[6*cm,11*cm]))
        story.append(Spacer(1,0.35*cm))
    doc.build(story)
    return str(out_path)
