from __future__ import annotations

from pathlib import Path
import csv
from typing import Iterable

from .optimizer import fit_model


def compare_models(models, datasets, method="recommended_adaptive_bounds", max_evals=4000, regularization=0.0, progress=None):
    """Fit several models to the same processed datasets and rank them.

    Ranking primarily uses AICc, then validation RMSE if available, then global RMSE.
    This is intended for engineering model selection and overfitting control.
    """
    rows = []
    results = []
    for m in models:
        if not getattr(m, "active", True):
            continue
        if progress:
            progress(f"Model comparison: fitting {m.number:02d} {m.name}")
        res = fit_model(m, datasets, method=method, max_evals=max_evals, regularization=regularization, progress=progress, trace_stride=max(25, max_evals // 100), validation_fraction=0.15)
        results.append((m, res))
        val_rmse = None
        if res.validation_metrics:
            val_rmse = res.validation_metrics.get("rmse")
        rows.append({
            "model_no": m.number,
            "model_name": m.name,
            "n_parameters": len(res.parameters),
            "success": res.success,
            "method": res.method,
            "objective": res.objective,
            "rmse": res.rmse,
            "mae": res.mae,
            "r2": res.r2,
            "adjusted_r2": res.adjusted_r2,
            "aic": res.aic,
            "aicc": res.aicc,
            "bic": res.bic,
            "validation_rmse": val_rmse,
            "elapsed_s": res.elapsed_s,
            "warnings": "; ".join(res.overfitting_warnings or []),
        })
    def key(row):
        aicc = row.get("aicc")
        val = row.get("validation_rmse")
        rmse = row.get("rmse")
        return (
            float("inf") if aicc is None else aicc,
            float("inf") if val is None else val,
            float("inf") if rmse is None else rmse,
        )
    rows.sort(key=key)
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return rows, results


def export_model_comparison_csv(rows: list[dict], path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return str(path)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    return str(path)
