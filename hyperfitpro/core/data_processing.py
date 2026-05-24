from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable
import csv
import math
import numpy as np

VALID_MODES = ["uniaxial", "biaxial", "planar", "simple_shear", "volumetric"]

FORCE_TO_N = {
    "N": 1.0,
    "kN": 1000.0,
    "lbf": 4.4482216152605,
    "lb": 4.4482216152605,
}
LENGTH_TO_MM = {
    "mm": 1.0,
    "m": 1000.0,
    "cm": 10.0,
    "in": 25.4,
}
STRESS_TO_MPA = {
    "MPa": 1.0,
    "N/mm^2": 1.0,
    "N/mm2": 1.0,
    "Pa": 1.0e-6,
    "kPa": 1.0e-3,
    "GPa": 1000.0,
    "psi": 0.006894757293168361,
    "ksi": 6.894757293168361,
}


@dataclass
class SpecimenGeometry:
    """Initial test-piece geometry used for force/displacement conversion.

    Units are interpreted by ProcessingOptions.length_unit. Area is interpreted as mm^2
    after conversion from the same length unit. If area is supplied directly it is assumed
    to be in length_unit^2 and converted to mm^2.
    """
    area: float | None = None
    width: float | None = None
    thickness: float | None = None
    diameter: float | None = None
    gauge_length: float | None = None

    def area_mm2(self, length_unit: str = "mm") -> float | None:
        fac = LENGTH_TO_MM.get(length_unit, 1.0)
        if self.area is not None and self.area > 0:
            return float(self.area) * fac * fac
        if self.width is not None and self.thickness is not None and self.width > 0 and self.thickness > 0:
            return float(self.width) * float(self.thickness) * fac * fac
        if self.diameter is not None and self.diameter > 0:
            d = float(self.diameter) * fac
            return math.pi * d * d / 4.0
        return None

    def gauge_mm(self, length_unit: str = "mm") -> float | None:
        fac = LENGTH_TO_MM.get(length_unit, 1.0)
        if self.gauge_length is not None and self.gauge_length > 0:
            return float(self.gauge_length) * fac
        return None


@dataclass
class ProcessingOptions:
    """User-controlled test-data conversion and cleaning options."""
    data_kind: str = "auto"  # auto, stress_strain, true_stress_strain, force_displacement, stretch_stress, volumetric
    force_unit: str = "N"
    length_unit: str = "mm"
    stress_unit: str = "MPa"
    strain_percent: bool = False
    stress_output_unit: str = "MPa"
    branch: str = "all"  # all, loading, unloading, first_monotonic
    zero_offset: bool = True
    toe_remove_fraction: float = 0.0
    toe_remove_strain: float | None = None
    smooth_method: str = "none"  # none, moving_average, savitzky_golay
    smooth_window: int = 0
    smooth_polyorder: int = 2
    outlier_method: str = "none"  # none, zscore, mad
    outlier_threshold: float = 4.0
    downsample_max_points: int = 0
    region_weighting: str = "uniform"  # uniform, low_strain, high_strain, balanced_bins
    normalize_point_weights: bool = True
    sort_by_x: bool = True
    remove_duplicate_x: bool = False


@dataclass
class TestDataset:
    mode: str
    source_file: str
    x: np.ndarray
    stress: np.ndarray
    weight: float = 1.0
    x_label: str = "nominal_strain_or_gamma"
    stress_label: str = "nominal_stress"
    preprocessing: dict | None = None
    raw_x: np.ndarray | None = None
    raw_stress: np.ndarray | None = None
    point_weight: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)

    def to_summary(self):
        return {
            "mode": self.mode,
            "source_file": self.source_file,
            "n": int(len(self.x)),
            "weight": float(self.weight),
            "x_label": self.x_label,
            "stress_label": self.stress_label,
            "x_min": float(np.nanmin(self.x)) if len(self.x) else None,
            "x_max": float(np.nanmax(self.x)) if len(self.x) else None,
            "stress_min": float(np.nanmin(self.stress)) if len(self.stress) else None,
            "stress_max": float(np.nanmax(self.stress)) if len(self.stress) else None,
            "preprocessing": self.preprocessing or {},
            "diagnostics": self.diagnostics or {},
            "has_point_weights": bool(self.point_weight is not None),
        }


def _normalize_header(s: str) -> str:
    return ''.join(ch.lower() for ch in str(s).strip() if ch.isalnum() or ch == '_')


def _read_text_table(path: Path):
    rows = []
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=',;\t ')
        except Exception:
            dialect = csv.excel
        reader = csv.reader(f, dialect)
        for r in reader:
            if r and any(str(c).strip() for c in r):
                rows.append([str(c).strip() for c in r])
    return rows


def _read_excel_table(path: Path, sheet_name: str | None = None):
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("Excel import requires pandas and openpyxl. Install requirements.txt") from e
    df = pd.read_excel(path, sheet_name=sheet_name or 0)
    return [list(df.columns)] + df.astype(object).where(df.notna(), '').values.tolist()


def _numeric_array_from_rows(rows):
    data = []
    for r in rows:
        vals = []
        for c in r:
            s = str(c).strip().replace(',', '.')
            try:
                vals.append(float(s))
            except Exception:
                vals.append(np.nan)
        data.append(vals)
    maxlen = max((len(r) for r in data), default=0)
    if maxlen == 0:
        return np.empty((0, 0))
    arr = np.full((len(data), maxlen), np.nan)
    for i, r in enumerate(data):
        arr[i, :len(r)] = r
    return arr


def _extract_table(path: Path, sheet_name: str | None = None):
    rows = _read_excel_table(path, sheet_name) if path.suffix.lower() in ('.xlsx', '.xlsm', '.xls') else _read_text_table(path)
    if not rows:
        raise ValueError(f"No data found in {path}")
    header_candidate = [_normalize_header(c) for c in rows[0]]
    first_numeric = _numeric_array_from_rows([rows[0]])
    has_header = not np.all(np.isfinite(first_numeric[0])) if first_numeric.size else True
    if has_header:
        header = header_candidate
        arr = _numeric_array_from_rows(rows[1:])
    else:
        header = [f"col{i+1}" for i in range(len(rows[0]))]
        arr = _numeric_array_from_rows(rows)
    return header, arr


def _col_idx(name_or_idx: str | int | None, header: list[str], candidates: Iterable[str]) -> int | None:
    if isinstance(name_or_idx, int):
        return name_or_idx
    if isinstance(name_or_idx, str) and name_or_idx.strip():
        n = _normalize_header(name_or_idx)
        if n in header:
            return header.index(n)
    for cand in candidates:
        if cand in header:
            return header.index(cand)
    return None


def _safe_col(arr: np.ndarray, idx: int | None, label: str) -> np.ndarray:
    if idx is None or idx >= arr.shape[1]:
        raise ValueError(f"Could not identify {label} column")
    return np.asarray(arr[:, idx], dtype=float)


def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window < 3 or len(y) < window:
        return y
    w = int(window)
    if w % 2 == 0:
        w += 1
    pad = w // 2
    yp = np.pad(y, (pad, pad), mode='edge')
    kernel = np.ones(w) / w
    return np.convolve(yp, kernel, mode='valid')


def _savitzky_golay(y: np.ndarray, window: int, polyorder: int = 2) -> np.ndarray:
    if window < 5 or len(y) < window:
        return y
    w = int(window)
    if w % 2 == 0:
        w += 1
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(y, window_length=w, polyorder=min(polyorder, w - 2), mode='interp')
    except Exception:
        return _moving_average(y, w)


def _remove_outliers(x: np.ndarray, y: np.ndarray, method: str, threshold: float):
    if method == "none" or len(y) < 5:
        return x, y, np.ones(len(y), dtype=bool)
    resid = y - _moving_average(y, min(11, max(3, len(y)//10*2+1)))
    if method == "zscore":
        s = float(np.nanstd(resid)) or 1.0
        z = np.abs((resid - np.nanmean(resid)) / s)
        keep = z <= float(threshold)
    elif method == "mad":
        med = np.nanmedian(resid)
        mad = np.nanmedian(np.abs(resid - med)) or 1.0
        z = 0.6745 * np.abs(resid - med) / mad
        keep = z <= float(threshold)
    else:
        keep = np.ones(len(y), dtype=bool)
    if keep.sum() < max(4, int(0.25 * len(y))):
        keep = np.ones(len(y), dtype=bool)
    return x[keep], y[keep], keep


def _select_branch(x: np.ndarray, y: np.ndarray, branch: str):
    if branch == "all" or len(x) < 3:
        return x, y, np.arange(len(x))
    imax = int(np.nanargmax(x))
    imin = int(np.nanargmin(x))
    if branch == "loading":
        if imax >= 1:
            idx = np.arange(0, imax + 1)
        else:
            idx = np.arange(len(x))
    elif branch == "unloading":
        if imax < len(x) - 1:
            idx = np.arange(imax, len(x))
        elif imin < len(x) - 1:
            idx = np.arange(imin, len(x))
        else:
            idx = np.arange(len(x))
    elif branch == "first_monotonic":
        direction = np.sign(x[-1] - x[0]) or 1.0
        keep = [0]
        for i in range(1, len(x)):
            if direction * (x[i] - x[i - 1]) >= -1e-12:
                keep.append(i)
            else:
                break
        idx = np.asarray(keep, dtype=int)
    else:
        idx = np.arange(len(x))
    return x[idx], y[idx], idx


def _remove_toe_region(x: np.ndarray, y: np.ndarray, frac: float, strain_limit: float | None):
    keep = np.ones(len(x), dtype=bool)
    if len(x) == 0:
        return x, y, keep
    if frac and frac > 0:
        n = min(len(x)-1, max(0, int(math.ceil(float(frac) * len(x)))))
        keep[:n] = False
    if strain_limit is not None and strain_limit > 0:
        keep &= np.abs(x) >= float(strain_limit)
    if keep.sum() < 3:
        keep[:] = True
    return x[keep], y[keep], keep


def _downsample(x: np.ndarray, y: np.ndarray, nmax: int):
    if not nmax or nmax <= 0 or len(x) <= nmax:
        return x, y, np.arange(len(x))
    idx = np.unique(np.linspace(0, len(x)-1, int(nmax)).round().astype(int))
    return x[idx], y[idx], idx


def _point_weights(x: np.ndarray, scheme: str):
    if len(x) == 0 or scheme == "uniform":
        return None
    xn = np.abs(np.asarray(x, dtype=float))
    xmax = float(np.nanmax(xn)) or 1.0
    if scheme == "low_strain":
        w = 1.0 + 2.0 * (1.0 - xn / xmax)
    elif scheme == "high_strain":
        w = 0.75 + 2.25 * (xn / xmax)
    elif scheme == "balanced_bins":
        bins = np.linspace(float(np.nanmin(xn)), float(np.nanmax(xn)), 6)
        if np.allclose(bins[0], bins[-1]):
            return None
        ids = np.digitize(xn, bins[1:-1], right=True)
        counts = np.bincount(ids, minlength=5).astype(float)
        counts[counts == 0] = 1.0
        w = 1.0 / counts[ids]
        w *= len(w) / np.sum(w)
    else:
        return None
    m = float(np.nanmean(w)) or 1.0
    return np.asarray(w / m, dtype=float)


def _infer_kind(data_kind: str, has_force: bool, has_disp: bool, has_stress: bool, has_true_stress: bool, has_stretch: bool, mode: str):
    if data_kind and data_kind != "auto":
        return data_kind
    if mode == "volumetric":
        return "volumetric"
    if has_force or has_disp:
        return "force_displacement"
    if has_true_stress:
        return "true_stress_strain"
    if has_stretch:
        return "stretch_stress"
    return "stress_strain"


def load_test_data(path: str | Path, mode: str, weight: float = 1.0, sheet_name: str | None = None,
                   strain_col: str | int | None = None, stress_col: str | int | None = None,
                   stretch_col: str | int | None = None, force_col: str | int | None = None,
                   area: float | None = None, displacement_col: str | int | None = None,
                   gauge_length: float | None = None, smooth_window: int = 0,
                   remove_unloading: bool = False,
                   true_stress_col: str | int | None = None,
                   true_strain_col: str | int | None = None,
                   width: float | None = None, thickness: float | None = None, diameter: float | None = None,
                   force_unit: str = "N", length_unit: str = "mm", stress_unit: str = "MPa",
                   strain_percent: bool = False, data_kind: str = "auto", branch: str = "all",
                   zero_offset: bool = True, toe_remove_fraction: float = 0.0,
                   toe_remove_strain: float | None = None, smooth_method: str = "none",
                   smooth_polyorder: int = 2, outlier_method: str = "none", outlier_threshold: float = 4.0,
                   downsample_max_points: int = 0, region_weighting: str = "uniform",
                   remove_duplicate_x: bool = False) -> TestDataset:
    path = Path(path)
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}")
    header, arr = _extract_table(path, sheet_name)
    if arr.size == 0:
        raise ValueError(f"No numeric data found in {path}")

    names_x = ['strain','nominalstrain','engineeringstrain','engstrain','e','eps','epsilon','gamma']
    names_true_x = ['truestrain','logstrain','lnlambda']
    names_lam = ['stretch','lambda','lam','lmbda']
    names_stress = ['stress','nominalstress','engineeringstress','p','s','sigma','forceperarea','pressure']
    names_true_stress = ['truestress','cauchystress','sigma_true','cauchy']
    names_force = ['force','load','f','pforce','loadn','forcen']
    names_disp = ['displacement','disp','extension','elongation','u','crossheaddisplacement']

    ix = _col_idx(strain_col, header, names_x)
    itx = _col_idx(true_strain_col, header, names_true_x)
    ilam = _col_idx(stretch_col, header, names_lam)
    iy = _col_idx(stress_col, header, names_stress)
    ity = _col_idx(true_stress_col, header, names_true_stress)
    iforce = _col_idx(force_col, header, names_force)
    idisp = _col_idx(displacement_col, header, names_disp)

    # Preserve earlier behavior for simple two-column CSVs.
    if ix is None and ilam is None and idisp is None and itx is None:
        ix = 0
    if iy is None and iforce is None and ity is None and arr.shape[1] > 1:
        iy = 1

    geom = SpecimenGeometry(area=area, width=width, thickness=thickness, diameter=diameter, gauge_length=gauge_length)
    opts = ProcessingOptions(data_kind=data_kind, force_unit=force_unit, length_unit=length_unit, stress_unit=stress_unit,
                             strain_percent=strain_percent, branch='first_monotonic' if remove_unloading else branch,
                             zero_offset=zero_offset, toe_remove_fraction=toe_remove_fraction,
                             toe_remove_strain=toe_remove_strain, smooth_method=smooth_method,
                             smooth_window=smooth_window, smooth_polyorder=smooth_polyorder,
                             outlier_method=outlier_method, outlier_threshold=outlier_threshold,
                             downsample_max_points=downsample_max_points, region_weighting=region_weighting,
                             remove_duplicate_x=remove_duplicate_x)

    kind = _infer_kind(opts.data_kind, iforce is not None, idisp is not None, iy is not None, ity is not None, ilam is not None, mode)

    if kind == "volumetric":
        # x is either volumetric strain or J-1. y is pressure/stress.
        x = _safe_col(arr, ix if ix is not None else ilam, 'volumetric strain / J column')
        y = _safe_col(arr, iy if iy is not None else ity, 'pressure/stress column') * STRESS_TO_MPA.get(stress_unit, 1.0)
        x_label = 'volumetric_strain_or_Jminus1'
        y_label = f'pressure [{opts.stress_output_unit}]'
    elif kind == "force_displacement":
        A = geom.area_mm2(opts.length_unit)
        L0 = geom.gauge_mm(opts.length_unit)
        if A is None:
            raise ValueError('Force-displacement import requires area or width+thickness or diameter')
        if L0 is None:
            raise ValueError('Force-displacement import requires gauge_length')
        force = _safe_col(arr, iforce, 'force/load') * FORCE_TO_N.get(opts.force_unit, 1.0)
        disp = _safe_col(arr, idisp, 'displacement/extension') * LENGTH_TO_MM.get(opts.length_unit, 1.0)
        x = disp / L0
        y = force / A  # N/mm2 = MPa
        x_label = 'engineering_strain_from_displacement'
        y_label = 'nominal_stress_from_force_area [MPa]'
    elif kind == "true_stress_strain":
        if itx is not None:
            true_strain = _safe_col(arr, itx, 'true strain')
            lam = np.exp(true_strain)
        elif ilam is not None:
            lam = _safe_col(arr, ilam, 'stretch')
        else:
            eng = _safe_col(arr, ix, 'engineering strain')
            if opts.strain_percent:
                eng = eng / 100.0
            lam = 1.0 + eng
        true_stress = _safe_col(arr, ity if ity is not None else iy, 'true/Cauchy stress') * STRESS_TO_MPA.get(stress_unit, 1.0)
        lam = np.maximum(lam, 1e-9)
        x = lam - 1.0
        y = true_stress / lam  # nominal P = sigma / lambda for main loading direction
        x_label = 'engineering_strain_from_true_strain_or_stretch'
        y_label = 'nominal_stress_converted_from_true [MPa]'
    elif kind == "stretch_stress":
        lam = _safe_col(arr, ilam, 'stretch')
        x = lam - 1.0
        y = _safe_col(arr, iy if iy is not None else ity, 'stress') * STRESS_TO_MPA.get(stress_unit, 1.0)
        x_label = 'stretch_minus_1'
        y_label = f'nominal_stress [{opts.stress_output_unit}]'
    else:
        x = _safe_col(arr, ix, 'strain')
        if opts.strain_percent:
            x = x / 100.0
        y = _safe_col(arr, iy if iy is not None else ity, 'stress') * STRESS_TO_MPA.get(stress_unit, 1.0)
        x_label = 'gamma' if mode == 'simple_shear' else 'nominal_strain'
        y_label = f'nominal_stress [{opts.stress_output_unit}]'

    raw_x = np.asarray(x, dtype=float).copy()
    raw_y = np.asarray(y, dtype=float).copy()
    diagnostics: dict[str, Any] = {
        'source_columns': {
            'strain': ix, 'true_strain': itx, 'stretch': ilam, 'stress': iy,
            'true_stress': ity, 'force': iforce, 'displacement': idisp,
        },
        'import_kind': kind,
        'raw_points': int(len(raw_x)),
        'unit_system': {'force': force_unit, 'length': length_unit, 'stress': stress_unit, 'stress_internal': 'MPa'},
        'geometry': asdict(geom),
    }

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    diagnostics['removed_nonfinite'] = int(len(raw_x) - len(x))

    x, y, branch_idx = _select_branch(x, y, opts.branch)
    diagnostics['branch'] = opts.branch
    diagnostics['points_after_branch'] = int(len(x))

    if opts.zero_offset and len(x) > 0:
        x0 = float(x[0])
        y0 = float(y[0])
        x = x - x0
        y = y - y0
        diagnostics['zero_offset'] = {'x0': x0, 'stress0': y0}

    x, y, toe_keep = _remove_toe_region(x, y, opts.toe_remove_fraction, opts.toe_remove_strain)
    diagnostics['toe_removed_points'] = int(np.sum(~toe_keep))

    x, y, out_keep = _remove_outliers(x, y, opts.outlier_method, opts.outlier_threshold)
    diagnostics['outlier_method'] = opts.outlier_method
    diagnostics['outliers_removed'] = int(np.sum(~out_keep))

    if opts.sort_by_x:
        order = np.argsort(x)
        x, y = x[order], y[order]
    if opts.remove_duplicate_x and len(x) > 1:
        _, unique_idx = np.unique(x, return_index=True)
        unique_idx = np.sort(unique_idx)
        diagnostics['duplicates_removed'] = int(len(x) - len(unique_idx))
        x, y = x[unique_idx], y[unique_idx]

    if opts.smooth_method == 'moving_average' or (opts.smooth_window and opts.smooth_method == 'none'):
        y = _moving_average(y, int(opts.smooth_window or smooth_window))
        diagnostics['smoothing'] = {'method': 'moving_average', 'window': int(opts.smooth_window or smooth_window)}
    elif opts.smooth_method == 'savitzky_golay':
        y = _savitzky_golay(y, int(opts.smooth_window), int(opts.smooth_polyorder))
        diagnostics['smoothing'] = {'method': 'savitzky_golay', 'window': int(opts.smooth_window), 'polyorder': int(opts.smooth_polyorder)}
    else:
        diagnostics['smoothing'] = {'method': 'none'}

    x, y, down_idx = _downsample(x, y, int(opts.downsample_max_points))
    diagnostics['downsampled_to'] = int(len(x))

    pw = _point_weights(x, opts.region_weighting)
    diagnostics['region_weighting'] = opts.region_weighting

    if len(x) < 2:
        raise ValueError('Processed dataset has fewer than 2 valid points. Check columns, units and preprocessing options.')

    return TestDataset(
        mode=mode,
        source_file=str(path),
        x=np.asarray(x, dtype=float),
        stress=np.asarray(y, dtype=float),
        weight=float(weight),
        x_label=x_label,
        stress_label=y_label,
        preprocessing=asdict(opts),
        raw_x=raw_x,
        raw_stress=raw_y,
        point_weight=pw,
        diagnostics=diagnostics,
    )


def load_csv_test_data(path: str | Path, mode: str, weight: float = 1.0) -> TestDataset:
    return load_test_data(path, mode, weight)


def stress_scale(datasets: list[TestDataset]) -> float:
    vals = []
    for d in datasets:
        vals.extend(list(np.asarray(d.stress, dtype=float)))
    if not vals:
        return 1.0
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    s = np.percentile(np.abs(arr), 90)
    return float(s if s > 0 else max(np.max(np.abs(arr)), 1.0))


def auto_balance_dataset_weights(datasets: list[TestDataset]) -> list[TestDataset]:
    """Balance datasets so large files do not dominate small-but-important test modes.

    The returned datasets are modified in-place and also returned for convenience.
    """
    if not datasets:
        return datasets
    # First balance by count, then by stress scale, preserving user-entered relative weights.
    counts = np.asarray([max(1, len(d.x)) for d in datasets], dtype=float)
    scales = []
    for d in datasets:
        s = np.percentile(np.abs(d.stress), 90) if len(d.stress) else 1.0
        scales.append(float(s if s > 0 else 1.0))
    scales = np.asarray(scales, dtype=float)
    target_count = float(np.median(counts))
    target_scale = float(np.median(scales)) or 1.0
    for d, n, s in zip(datasets, counts, scales):
        user_w = float(d.weight)
        d.weight = float(user_w * (target_count / n) * (target_scale / max(s, 1e-12)))
        if d.preprocessing is None:
            d.preprocessing = {}
        d.preprocessing['auto_balanced_weight'] = True
    return datasets


def export_processed_dataset_csv(dataset: TestDataset, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['x', 'stress', 'point_weight'])
        pw = dataset.point_weight if dataset.point_weight is not None else np.ones(len(dataset.x))
        for xi, yi, wi in zip(dataset.x, dataset.stress, pw):
            wr.writerow([f'{float(xi):.12g}', f'{float(yi):.12g}', f'{float(wi):.12g}'])
