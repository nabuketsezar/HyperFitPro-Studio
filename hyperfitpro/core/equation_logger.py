from __future__ import annotations
from pathlib import Path
import json


def write_equation_report(model, params, fit_result, path):
    path=Path(path)
    lines=[]
    lines.append(f"# HyperFitPro Equation Report\n\n")
    lines.append(f"## Model\n\n- No: {model.number}\n- Name: {model.name}\n- Category: {model.category}\n\n")
    lines.append("## Symbolic strain energy function\n\n")
    lines.append("```latex\n" + (model.equation_latex or '') + "\n```\n\n")
    lines.append("## Parameter-substituted strain energy function\n\n")
    lines.append("```latex\n" + model.equation_with_numbers(params) + "\n```\n\n")
    lines.append("## Fitted parameters\n\n| Parameter | Value |\n|---|---:|\n")
    for k,v in params.items(): lines.append(f"| {k} | {v:.12g} |\n")
    lines.append("\n## Fit metrics\n\n")
    lines.append(f"- Method: {fit_result.method}\n- Objective: {fit_result.objective:.12g}\n- RMSE: {fit_result.rmse:.12g}\n- MAE: {fit_result.mae:.12g}\n- R²: {fit_result.r2:.12g}\n- Function evaluations: {fit_result.nfev}\n- Elapsed: {fit_result.elapsed_s:.3f} s\n\n")
    lines.append("## Parameter bounds used\n\n| Parameter | Lower | Upper |\n|---|---:|---:|\n")
    for k,(lo,hi) in fit_result.bounds.items(): lines.append(f"| {k} | {lo:.12g} | {hi:.12g} |\n")
    lines.append("\n## Stability scan\n\n")
    stab=getattr(fit_result,'stability',{}) or {}
    lines.append("```json\n"+json.dumps(stab,indent=2)+"\n```\n")
    path.write_text(''.join(lines),encoding='utf-8')
