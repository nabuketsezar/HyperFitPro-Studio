from __future__ import annotations

from dataclasses import replace
from typing import Iterable
import math
import numpy as np

from .tests import TestDataset, stress_scale
from .volumetric import J_from_x, hydrostatic_pressure_from_K


def _copy_dataset_with_indices(d: TestDataset, idx: np.ndarray, suffix: str) -> TestDataset:
    idx = np.asarray(idx, dtype=int)
    pw = None if d.point_weight is None else np.asarray(d.point_weight)[idx]
    return replace(
        d,
        source_file=f"{d.source_file}#{suffix}",
        x=np.asarray(d.x, dtype=float)[idx],
        stress=np.asarray(d.stress, dtype=float)[idx],
        raw_x=None if d.raw_x is None else np.asarray(d.raw_x, dtype=float)[idx] if len(d.raw_x) == len(d.x) else d.raw_x,
        raw_stress=None if d.raw_stress is None else np.asarray(d.raw_stress, dtype=float)[idx] if len(d.raw_stress) == len(d.stress) else d.raw_stress,
        point_weight=pw,
        diagnostics={**(d.diagnostics or {}), "split_suffix": suffix, "split_n": int(len(idx))},
    )


def train_validation_split(datasets: list[TestDataset], validation_fraction: float = 0.0, seed: int = 1234):
    """Deterministic per-dataset train/validation split preserving all modes."""
    vf = float(validation_fraction or 0.0)
    if vf <= 0.0:
        return list(datasets), []
    vf = min(max(vf, 0.02), 0.50)
    rng = np.random.default_rng(seed)
    train, val = [], []
    for d in datasets:
        n = len(d.x)
        if n < 8:
            train.append(d)
            continue
        order = np.arange(n)
        rng.shuffle(order)
        nv = max(2, int(round(vf * n)))
        vi = np.sort(order[:nv])
        ti = np.sort(order[nv:])
        if len(ti) < max(3, len(d.x) // 10):
            train.append(d)
            continue
        train.append(_copy_dataset_with_indices(d, ti, "train"))
        val.append(_copy_dataset_with_indices(d, vi, "validation"))
    return train, val


def predict_dataset(model, d: TestDataset, params: dict[str, float]) -> np.ndarray:
    if d.mode == "volumetric":
        if "K" not in params:
            return np.full_like(np.asarray(d.x, dtype=float), np.nan, dtype=float)
        return hydrostatic_pressure_from_K(J_from_x(d.x), float(params["K"]))
    return model.predict_nominal(d.mode, np.asarray(d.x, dtype=float), params)


def collect_predictions(model, datasets: list[TestDataset], params: dict[str, float]):
    y, yh, weights = [], [], []
    for d in datasets:
        pred = predict_dataset(model, d, params)
        yy = np.asarray(d.stress, dtype=float)
        w = np.ones_like(yy) * max(float(d.weight), 0.0)
        if d.point_weight is not None and len(d.point_weight) == len(yy):
            pw = np.asarray(d.point_weight, dtype=float)
            pw[~np.isfinite(pw)] = 1.0
            pw = np.maximum(pw, 0.0)
            if np.mean(pw) > 0:
                pw = pw / np.mean(pw)
            w *= pw
        m = np.isfinite(yy) & np.isfinite(pred)
        y.extend(yy[m].tolist())
        yh.extend(np.asarray(pred, dtype=float)[m].tolist())
        weights.extend(w[m].tolist())
    return np.asarray(y, dtype=float), np.asarray(yh, dtype=float), np.asarray(weights, dtype=float)


def regression_metrics(y_true, y_pred, weights=None, n_parameters: int = 0):
    y = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y) & np.isfinite(yp)
    y = y[m]
    yp = yp[m]
    if weights is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(weights, dtype=float)[m]
        w[~np.isfinite(w)] = 1.0
        w = np.maximum(w, 0.0)
        if np.sum(w) <= 0:
            w = np.ones_like(y)
    n = int(y.size)
    if n == 0:
        return {"n": 0, "sse": None, "rmse": None, "mae": None, "r2": None, "adjusted_r2": None, "aic": None, "aicc": None, "bic": None}
    err = yp - y
    sse = float(np.sum(w * err * err))
    rmse = float(np.sqrt(sse / max(np.sum(w), 1e-30)))
    mae = float(np.sum(w * np.abs(err)) / max(np.sum(w), 1e-30))
    ybar = float(np.sum(w * y) / max(np.sum(w), 1e-30))
    sst = float(np.sum(w * (y - ybar) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 1e-30 else float("nan")
    k = int(max(n_parameters, 0))
    if n > k + 1 and np.isfinite(r2):
        adj = float(1.0 - (1.0 - r2) * (n - 1) / (n - k - 1))
    else:
        adj = float("nan")
    mse = max(sse / max(n, 1), 1e-300)
    aic = float(n * math.log(mse) + 2 * k)
    bic = float(n * math.log(mse) + k * math.log(max(n, 1)))
    aicc = float(aic + (2 * k * (k + 1)) / max(n - k - 1, 1)) if n > k + 1 else float("inf")
    return {"n": n, "sse": sse, "rmse": rmse, "mae": mae, "r2": r2, "adjusted_r2": adj, "aic": aic, "aicc": aicc, "bic": bic}


def residual_vector(model, datasets: list[TestDataset], params: dict[str, float], scale: float | None = None):
    S = float(scale or stress_scale(datasets) or 1.0)
    res = []
    for d in datasets:
        pred = predict_dataset(model, d, params)
        yy = np.asarray(d.stress, dtype=float)
        sc = max(float(np.percentile(np.abs(yy[np.isfinite(yy)]), 90)) if np.any(np.isfinite(yy)) else S, S, 1e-12)
        r = (np.asarray(pred, dtype=float) - yy) / sc
        r[~np.isfinite(r)] = 1e6
        if d.point_weight is not None and len(d.point_weight) == len(r):
            pw = np.asarray(d.point_weight, dtype=float)
            pw[~np.isfinite(pw)] = 1.0
            pw = np.maximum(pw, 0.0)
            if np.mean(pw) > 0:
                pw = pw / np.mean(pw)
            r = r * np.sqrt(pw)
        r = r * math.sqrt(max(float(d.weight), 0.0))
        res.extend(r.tolist())
    return np.asarray(res, dtype=float)


def parameter_uncertainty(model, datasets: list[TestDataset], params: dict[str, float], names: list[str], bounds: dict[str, list[float]] | None = None):
    """Finite-difference covariance/correlation approximation around optimum."""
    if not names:
        return {"available": False, "reason": "No parameters."}
    p0 = {k: float(params[k]) for k in names if k in params}
    names = [n for n in names if n in p0]
    if not names:
        return {"available": False, "reason": "No fitted parameter names found."}
    r0 = residual_vector(model, datasets, params)
    m = r0.size
    k = len(names)
    if m <= k + 1:
        return {"available": False, "reason": "Not enough residual degrees of freedom for covariance estimate.", "n_residuals": int(m), "n_parameters": int(k)}
    J = np.zeros((m, k), dtype=float)
    for j, nm in enumerate(names):
        v = p0[nm]
        lo, hi = (-np.inf, np.inf)
        if bounds and nm in bounds:
            lo, hi = float(bounds[nm][0]), float(bounds[nm][1])
        step = 1e-5 * (abs(v) + 1.0)
        xp = min(v + step, hi) if np.isfinite(hi) else v + step
        xm = max(v - step, lo) if np.isfinite(lo) else v - step
        if abs(xp - xm) < 1e-14 * (abs(v) + 1.0):
            step = 1e-4 * (abs(v) + 1.0)
            xp = v + step
            xm = v - step
        pp = dict(params); pm = dict(params)
        pp[nm] = float(xp); pm[nm] = float(xm)
        rp = residual_vector(model, datasets, pp)
        rm = residual_vector(model, datasets, pm)
        J[:, j] = (rp - rm) / max(xp - xm, 1e-30)
    dof = max(m - k, 1)
    sigma2 = float(np.sum(r0 * r0) / dof)
    try:
        cov = sigma2 * np.linalg.pinv(J.T @ J, rcond=1e-12)
    except Exception as e:
        return {"available": False, "reason": f"Covariance inversion failed: {e}"}
    diag = np.diag(cov)
    diag = np.where(diag >= 0, diag, np.nan)
    stderr = np.sqrt(diag)
    ci95 = 1.96 * stderr
    denom = np.outer(stderr, stderr)
    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
    corr = np.clip(corr, -1.0, 1.0)
    intervals = {}
    for i, nm in enumerate(names):
        se = float(stderr[i]) if np.isfinite(stderr[i]) else None
        half = float(ci95[i]) if np.isfinite(ci95[i]) else None
        val = float(params[nm])
        intervals[nm] = {
            "value": val,
            "standard_error": se,
            "ci95_lower": None if half is None else val - half,
            "ci95_upper": None if half is None else val + half,
            "relative_se": None if se is None or abs(val) < 1e-30 else abs(se / val),
        }
    return {
        "available": True,
        "n_residuals": int(m),
        "n_parameters": int(k),
        "residual_variance": sigma2,
        "parameter_names": names,
        "confidence_intervals": intervals,
        "correlation_matrix": corr.tolist(),
        "covariance_matrix": cov.tolist(),
    }


def overfitting_warnings(metrics: dict, n_parameters: int, datasets: list[TestDataset], validation_metrics: dict | None = None, uncertainty: dict | None = None):
    warnings = []
    n = int(metrics.get("n") or 0)
    if n_parameters >= max(n / 8, 1):
        warnings.append(f"High parameter-to-data ratio: {n_parameters} parameters for {n} usable points. Prefer simpler model or add more modes.")
    if validation_metrics and validation_metrics.get("rmse") and metrics.get("rmse"):
        tr = float(metrics["rmse"]); va = float(validation_metrics["rmse"])
        if tr > 0 and va / tr > 2.5:
            warnings.append(f"Validation RMSE is {va/tr:.2f}x training/global RMSE. Possible overfitting or poor extrapolation.")
    modes = {d.mode for d in datasets}
    if n_parameters >= 5 and len(modes - {"volumetric"}) < 2:
        warnings.append("High-order model fitted to a single deformation mode. Use uniaxial + biaxial/planar if possible.")
    if uncertainty and uncertainty.get("available"):
        intervals = uncertainty.get("confidence_intervals", {})
        weak = [k for k, v in intervals.items() if v.get("relative_se") is not None and v["relative_se"] > 1.0]
        if weak:
            warnings.append("Poorly identified parameters with relative standard error > 100%: " + ", ".join(weak[:8]))
        corr = np.asarray(uncertainty.get("correlation_matrix", []), dtype=float)
        names = uncertainty.get("parameter_names", [])
        if corr.size and len(names) == corr.shape[0]:
            pairs = []
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    if abs(corr[i, j]) > 0.98:
                        pairs.append(f"{names[i]}-{names[j]} ({corr[i,j]:+.3f})")
            if pairs:
                warnings.append("Near-collinear parameter pairs: " + "; ".join(pairs[:8]))
    return warnings


def build_fit_diagnostics(model, all_datasets: list[TestDataset], train_datasets: list[TestDataset], validation_datasets: list[TestDataset], params: dict[str, float], parameter_names: list[str], bounds: dict[str, list[float]]):
    y_all, yh_all, w_all = collect_predictions(model, all_datasets, params)
    y_tr, yh_tr, w_tr = collect_predictions(model, train_datasets, params)
    all_metrics = regression_metrics(y_all, yh_all, w_all, len(parameter_names))
    train_metrics = regression_metrics(y_tr, yh_tr, w_tr, len(parameter_names))
    val_metrics = None
    if validation_datasets:
        yv, yhv, wv = collect_predictions(model, validation_datasets, params)
        val_metrics = regression_metrics(yv, yhv, wv, len(parameter_names))
    unc = parameter_uncertainty(model, train_datasets or all_datasets, params, parameter_names, bounds=bounds)
    warns = overfitting_warnings(train_metrics, len(parameter_names), train_datasets or all_datasets, validation_metrics=val_metrics, uncertainty=unc)
    return {
        "all_metrics": all_metrics,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "uncertainty": unc,
        "overfitting_warnings": warns,
    }
