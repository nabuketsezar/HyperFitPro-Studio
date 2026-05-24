from __future__ import annotations

from dataclasses import dataclass, field
import time
import math
import numpy as np

from .tests import TestDataset, stress_scale
from .base_model import max_invariant_from_test_inputs
from .volumetric import J_from_x, hydrostatic_pressure_from_K, volumetric_bounds
from .optimization_analysis import train_validation_split, build_fit_diagnostics
from .optimizer_control import OptimizationController, OptimizationStopped, save_checkpoint, load_checkpoint

OPTIMIZATION_METHODS = [
    'least_squares_trf', 'least_squares_dogbox', 'least_squares_lm_unbounded',
    'differential_evolution', 'dual_annealing', 'basinhopping', 'shgo',
    'nelder_mead', 'powell', 'l_bfgs_b',
    'hybrid_global_local', 'recommended_adaptive_bounds', 'multi_start_lsq',
    'regularized_multi_start', 'fast_local_refine'
]

@dataclass
class FitResult:
    success: bool
    method: str
    parameters: dict
    objective: float
    rmse: float
    mae: float
    r2: float
    message: str
    nfev: int
    elapsed_s: float
    bounds: dict
    trace: list[dict] = field(default_factory=list)
    stability: dict = field(default_factory=dict)
    optimizer_settings: dict = field(default_factory=dict)
    # optimization diagnostics / model-selection fields
    adjusted_r2: float | None = None
    aic: float | None = None
    aicc: float | None = None
    bic: float | None = None
    train_metrics: dict | None = None
    validation_metrics: dict | None = None
    confidence_intervals: dict | None = None
    parameter_correlation: dict | None = None
    covariance_matrix: list | None = None
    regularization_summary: dict | None = None
    overfitting_warnings: list[str] = field(default_factory=list)
    adaptive_bounds_history: list[dict] = field(default_factory=list)
    checkpoint_file: str | None = None
    stopped_by_user: bool = False


def _metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return float('nan'), float('nan'), float('nan')
    e = y_pred[m] - y_true[m]
    rmse = float(np.sqrt(np.mean(e * e)))
    mae = float(np.mean(np.abs(e)))
    denom = float(np.sum((y_true[m] - np.mean(y_true[m])) ** 2))
    r2 = float(1.0 - np.sum(e * e) / denom) if denom > 1e-30 else float('nan')
    return rmse, mae, r2


def _threaded_de_kwargs(workers: int):
    """Return a map-like worker for scipy differential_evolution without pickle issues."""
    workers = int(workers or 1)
    if workers <= 1:
        return {}, None
    try:
        from multiprocessing.pool import ThreadPool
        pool = ThreadPool(processes=workers)
        return {'workers': pool.map, 'updating': 'deferred'}, pool
    except Exception:
        return {}, None


def _regularization_residuals(x, x0, scale_params, strength: float, regularization_type: str):
    strength = float(strength or 0.0)
    if strength <= 0:
        return []
    z = (np.asarray(x, dtype=float) - np.asarray(x0, dtype=float)) / np.maximum(scale_params, 1e-30)
    kind = (regularization_type or 'l2').lower()
    lam = math.sqrt(strength)
    if kind == 'none':
        return []
    if kind == 'l1':
        return list(lam * np.sign(z) * np.sqrt(np.abs(z) + 1e-18))
    if kind == 'elastic_net':
        l2 = math.sqrt(0.5 * strength) * z
        l1 = math.sqrt(0.5 * strength) * np.sign(z) * np.sqrt(np.abs(z) + 1e-18)
        return list(l2) + list(l1)
    if kind == 'magnitude':
        return list(lam * np.asarray(x, dtype=float) / np.maximum(scale_params, 1e-30))
    # default L2/Tikhonov
    return list(lam * z)


def _near_bound_penalty(x, lb, ub, strength: float):
    strength = float(strength or 0.0)
    if strength <= 0:
        return []
    span = np.maximum(np.asarray(ub) - np.asarray(lb), 1e-30)
    rel_lo = (np.asarray(x) - np.asarray(lb)) / span
    rel_hi = (np.asarray(ub) - np.asarray(x)) / span
    margin = np.minimum(rel_lo, rel_hi)
    # Penalize parameters sitting within 2% of a bound; useful for diagnosing bad bounds.
    p = np.maximum(0.02 - margin, 0.0) / 0.02
    return list(math.sqrt(strength) * p)


def fit_model(
    model,
    datasets: list[TestDataset],
    method='hybrid_global_local',
    max_evals=6000,
    seed=1234,
    progress=None,
    robust_loss='soft_l1',
    override_bounds=None,
    regularization=0.0,
    trace_stride=10,
    # advanced optimization controls
    regularization_type='l2',
    validation_fraction=0.0,
    cv_folds=0,
    workers=1,
    early_stop_patience=0,
    early_stop_tol=1e-10,
    control: OptimizationController | None = None,
    checkpoint_path=None,
    resume_from=None,
    near_bound_penalty=0.0,
) -> FitResult:
    t0 = time.time()
    if not datasets:
        raise ValueError('At least one test dataset is required')
    method_alias = {
        'least_squares': 'least_squares_trf',
        'hybrid': 'hybrid_global_local',
        'recommended': 'recommended_adaptive_bounds',
    }
    method = method_alias.get(method, method)
    if method == 'fast_local_refine':
        method = 'least_squares_trf'
    if method == 'regularized_multi_start':
        method = 'multi_start_lsq'
        if regularization <= 0:
            regularization = 1e-4
            regularization_type = 'elastic_net'

    train_datasets, validation_datasets = train_validation_split(datasets, validation_fraction, seed=seed)
    S = stress_scale(train_datasets or datasets)
    max_I1 = max_invariant_from_test_inputs([(d.mode, d.x) for d in train_datasets or datasets])
    lb, ub, x0 = model.resolved_bounds(S, max_I1, overrides=override_bounds)
    base_names = model.parameter_names()
    names = list(base_names)
    base_n = len(base_names)
    volumetric_sets = [d for d in datasets if d.mode == 'volumetric']
    has_volumetric = bool(volumetric_sets)
    if has_volumetric:
        klo, khi, kini = volumetric_bounds(S, volumetric_sets[0])
        if override_bounds and 'K' in override_bounds:
            ovo = override_bounds['K']
            if len(ovo) >= 2 and ovo[0] is not None and ovo[1] is not None:
                klo, khi = float(ovo[0]), float(ovo[1])
            if len(ovo) >= 3 and ovo[2] is not None:
                kini = float(ovo[2])
        names.append('K')
        lb = np.concatenate([lb, [klo]])
        ub = np.concatenate([ub, [khi]])
        x0 = np.concatenate([x0, [min(max(kini, klo), khi)]])
    bounds_dict = {n: [float(a), float(b)] for n, a, b in zip(names, lb, ub)}
    if len(names) == 0:
        raise ValueError('Selected model has no active parameters')

    nfev = 0
    trace = []
    adaptive_bounds_history: list[dict] = []
    best_obj = float('inf')
    best_x = np.clip(x0, lb, ub).astype(float)
    rng = np.random.default_rng(seed)
    scale_params = np.maximum(np.abs(x0), 1.0)
    stopped_by_user = False
    last_improve_eval = 0
    pool = None

    if resume_from:
        try:
            chk = load_checkpoint(resume_from)
            chk_names = chk.get('names', [])
            chk_x = np.asarray(chk.get('best_x', []), dtype=float)
            if list(chk_names) == list(names) and chk_x.size == len(names):
                best_x = np.clip(chk_x, lb, ub)
                best_obj = float(chk.get('best_objective', best_obj))
                trace = list(chk.get('trace', []))[-1000:]
                if progress:
                    progress(f'Resumed from checkpoint: {resume_from}')
        except Exception as e:
            if progress:
                progress(f'Could not resume checkpoint: {e}')

    def _checkpoint(force=False):
        if not checkpoint_path:
            return
        if force or nfev == 1 or nfev % max(25, int(trace_stride) * 5) == 0:
            try:
                save_checkpoint(checkpoint_path, {
                    'model_no': getattr(model, 'number', None),
                    'model_name': getattr(model, 'name', ''),
                    'method': method,
                    'names': names,
                    'best_x': best_x,
                    'best_objective': best_obj,
                    'nfev': nfev,
                    'trace': trace[-2000:],
                    'timestamp_s': time.time(),
                })
            except Exception as e:
                if progress:
                    progress(f'Checkpoint save failed: {e}')

    def residuals(x):
        nonlocal nfev, best_obj, best_x, last_improve_eval
        if control:
            control.check()
        nfev += 1
        x = np.asarray(x, dtype=float)
        if np.any(x < lb) or np.any(x > ub) or not np.all(np.isfinite(x)):
            return np.ones(sum(len(d.x) for d in train_datasets) + len(x)) * 1e8
        p = model.vector_to_params(x[:base_n])
        if has_volumetric:
            p['K'] = float(x[-1])
        res = []
        for d in train_datasets:
            if d.mode == 'volumetric':
                if not has_volumetric:
                    continue
                J = J_from_x(d.x)
                yhat = hydrostatic_pressure_from_K(J, x[-1])
            else:
                yhat = model.predict_nominal(d.mode, d.x, p)
            scale = max(float(np.percentile(np.abs(d.stress), 90)), S, 1e-12)
            r = (yhat - d.stress) / scale
            r[~np.isfinite(r)] = 1e6
            point_weight = getattr(d, 'point_weight', None)
            if point_weight is not None and len(point_weight) == len(r):
                pw = np.asarray(point_weight, dtype=float)
                pw[~np.isfinite(pw)] = 1.0
                pw = np.maximum(pw, 0.0)
                m = float(np.mean(pw)) if pw.size else 1.0
                if m > 0:
                    pw = pw / m
                r = r * np.sqrt(pw)
            res.extend(list(np.sqrt(max(d.weight, 0.0)) * r))
        reg = _regularization_residuals(x, x0, scale_params, regularization, regularization_type)
        if reg:
            res.extend(reg)
        nb = _near_bound_penalty(x, lb, ub, near_bound_penalty)
        if nb:
            res.extend(nb)
        r = np.asarray(res, dtype=float)
        obj = float(np.mean(r * r)) if r.size else 1e50
        if obj + float(early_stop_tol or 0.0) < best_obj:
            best_obj = obj
            best_x = x.copy()
            last_improve_eval = nfev
        if early_stop_patience and nfev - last_improve_eval > int(early_stop_patience):
            raise OptimizationStopped(f'Early stop: no objective improvement for {early_stop_patience} evaluations.')
        if nfev == 1 or nfev % max(1, int(trace_stride)) == 0:
            rec = {'eval': int(nfev), 'objective': float(obj), 'best_objective': float(best_obj)}
            rec.update({f'p_{nm}': float(v) for nm, v in zip(names, x)})
            trace.append(rec)
            if progress:
                progress({
                    'type': 'iteration',
                    'record': dict(rec),
                    'eval': int(nfev),
                    'objective': float(obj),
                    'best_objective': float(best_obj),
                    'parameters': {nm: float(v) for nm, v in zip(names, x)},
                    'best_parameters': {nm: float(v) for nm, v in zip(names, best_x)},
                })
                if nfev == 1 or nfev % (max(1, int(trace_stride)) * 10) == 0:
                    progress(f"eval={nfev} obj={obj:.6e} best={best_obj:.6e}")
            _checkpoint(force=False)
        return r

    def objective(x):
        r = residuals(x)
        if not np.all(np.isfinite(r)):
            return 1e50
        return float(np.mean(r * r))

    def polish_least_squares(start, opt, maxn=None, loss=None, ls_method='trf'):
        nonlocal best_x, best_obj
        try:
            ls = opt.least_squares(
                residuals,
                np.clip(start, lb, ub),
                bounds=(lb, ub),
                max_nfev=maxn or max(200, int(max_evals) // 2),
                x_scale='jac',
                loss=loss or robust_loss,
                method=ls_method,
            )
            obj = float(np.mean(ls.fun * ls.fun))
            if obj < best_obj:
                best_x, best_obj = ls.x, obj
            return bool(ls.success), str(ls.message)
        except OptimizationStopped:
            raise
        except Exception as e:
            return False, f'least_squares polish failed: {e}'

    def run_parallel_multistart(opt, starts, maxn_each):
        """Threaded multi-start LS. Falls back to sequential if workers <= 1."""
        msgs = []
        if int(workers or 1) <= 1 or len(starts) <= 1:
            for st in starts:
                ok, msg = polish_least_squares(st, opt, maxn=maxn_each)
                msgs.append(msg)
            return msgs
        from concurrent.futures import ThreadPoolExecutor, as_completed
        # Use bounded number of threads; each LS still calls the shared residuals/progress safely enough for monitoring.
        with ThreadPoolExecutor(max_workers=min(int(workers), len(starts))) as ex:
            futs = [ex.submit(polish_least_squares, st, opt, maxn_each) for st in starts]
            for fut in as_completed(futs):
                ok, msg = fut.result()
                msgs.append(msg)
        return msgs

    success = False
    message = ''
    try:
        import scipy.optimize as opt
        bounds = list(zip(lb, ub))
        dim = len(names)
        if method == 'least_squares_trf':
            ls = opt.least_squares(residuals, best_x, bounds=(lb, ub), max_nfev=int(max_evals), x_scale='jac', loss=robust_loss, method='trf')
            best_x, best_obj = ls.x, float(np.mean(ls.fun * ls.fun))
            success = bool(ls.success)
            message = str(ls.message)
        elif method == 'least_squares_dogbox':
            ls = opt.least_squares(residuals, best_x, bounds=(lb, ub), max_nfev=int(max_evals), x_scale='jac', loss=robust_loss, method='dogbox')
            best_x, best_obj = ls.x, float(np.mean(ls.fun * ls.fun))
            success = bool(ls.success)
            message = str(ls.message)
        elif method == 'least_squares_lm_unbounded':
            # Only safe when no finite bounds are required. We transform the initial point but keep a penalty through residuals.
            def uncon_res(z):
                return residuals(np.clip(z, lb, ub))
            ls = opt.least_squares(uncon_res, best_x, max_nfev=int(max_evals), x_scale='jac', loss='linear', method='lm')
            bx = np.clip(ls.x, lb, ub)
            obj = objective(bx)
            if obj < best_obj:
                best_x, best_obj = bx, obj
            success = bool(ls.success)
            message = 'LM unbounded/clipped refinement: ' + str(ls.message)
        elif method == 'multi_start_lsq':
            starts = [best_x]
            nstarts = min(32, max(4, int(max_evals) // 220))
            # Latin-hypercube-like uniform starts to cover bounds better than pure random.
            for i in range(nstarts):
                u = (i + rng.random(dim)) / max(nstarts, 1)
                rng.shuffle(u)
                starts.append(lb + u * (ub - lb))
            msgs = run_parallel_multistart(opt, starts, max(80, int(max_evals) // max(1, len(starts))))
            success = True
            message = 'parallel multi-start least squares completed; ' + (msgs[-1] if msgs else '')
        elif method == 'differential_evolution':
            maxiter = max(4, int(max_evals) // max(20, dim * 15))
            wkwargs, pool = _threaded_de_kwargs(workers)
            try:
                de = opt.differential_evolution(objective, bounds, seed=seed, polish=False, maxiter=maxiter, tol=1e-7, **({'updating': 'immediate'} if not wkwargs else wkwargs))
            finally:
                if pool:
                    pool.close(); pool.join()
            if de.fun < best_obj:
                best_x, best_obj = de.x, float(de.fun)
            ok, msg = polish_least_squares(best_x, opt)
            success = True
            message = 'differential_evolution + least_squares: ' + msg
        elif method == 'dual_annealing':
            da = opt.dual_annealing(objective, bounds, maxfun=int(max_evals), seed=seed, no_local_search=True)
            if da.fun < best_obj:
                best_x, best_obj = da.x, float(da.fun)
            ok, msg = polish_least_squares(best_x, opt)
            success = True
            message = 'dual_annealing + least_squares: ' + msg
        elif method == 'basinhopping':
            minimizer_kwargs = {'method': 'L-BFGS-B', 'bounds': bounds}
            bh = opt.basinhopping(objective, best_x, niter=max(5, int(max_evals) // max(100, dim * 30)), seed=seed, minimizer_kwargs=minimizer_kwargs)
            if bh.fun < best_obj:
                best_x, best_obj = bh.x, float(bh.fun)
            ok, msg = polish_least_squares(best_x, opt)
            success = True
            message = 'basinhopping + least_squares: ' + msg
        elif method == 'shgo':
            iters = max(1, min(4, int(max_evals) // max(250, dim * 80)))
            sg = opt.shgo(objective, bounds, n=max(32, dim * 24), iters=iters)
            if np.isfinite(sg.fun) and sg.fun < best_obj:
                best_x, best_obj = sg.x, float(sg.fun)
            ok, msg = polish_least_squares(best_x, opt)
            success = True
            message = 'shgo + least_squares: ' + msg
        elif method == 'nelder_mead':
            def penalized(z):
                z = np.clip(np.asarray(z, dtype=float), lb, ub)
                return objective(z)
            nm = opt.minimize(penalized, best_x, method='Nelder-Mead', options={'maxfev': int(max_evals), 'xatol': 1e-8, 'fatol': 1e-10})
            if nm.fun < best_obj:
                best_x, best_obj = np.clip(nm.x, lb, ub), float(nm.fun)
            ok, msg = polish_least_squares(best_x, opt)
            success = True
            message = 'nelder_mead + least_squares: ' + msg
        elif method == 'powell':
            pw = opt.minimize(objective, best_x, method='Powell', bounds=bounds, options={'maxfev': int(max_evals), 'xtol': 1e-8, 'ftol': 1e-10})
            if pw.fun < best_obj:
                best_x, best_obj = pw.x, float(pw.fun)
            ok, msg = polish_least_squares(best_x, opt)
            success = True
            message = 'powell + least_squares: ' + msg
        elif method == 'l_bfgs_b':
            lbfgs = opt.minimize(objective, best_x, method='L-BFGS-B', bounds=bounds, options={'maxfun': int(max_evals), 'ftol': 1e-12})
            if lbfgs.fun < best_obj:
                best_x, best_obj = lbfgs.x, float(lbfgs.fun)
            ok, msg = polish_least_squares(best_x, opt)
            success = True
            message = 'l_bfgs_b + least_squares: ' + msg
        elif method == 'hybrid_global_local':
            budget = int(max_evals)
            maxiter = max(4, budget // max(30, dim * 25))
            wkwargs, pool = _threaded_de_kwargs(workers)
            try:
                de = opt.differential_evolution(objective, bounds, seed=seed, polish=False, maxiter=maxiter, tol=1e-7, **({'updating': 'immediate'} if not wkwargs else wkwargs))
            finally:
                if pool:
                    pool.close(); pool.join()
            if de.fun < best_obj:
                best_x, best_obj = de.x, float(de.fun)
            if budget > 1500:
                da = opt.dual_annealing(objective, bounds, maxfun=max(200, budget // 3), seed=seed, no_local_search=True, x0=best_x)
                if da.fun < best_obj:
                    best_x, best_obj = da.x, float(da.fun)
            starts = [best_x] + [lb + rng.random(dim) * (ub - lb) for _ in range(min(8, max(2, budget // 700)))]
            run_parallel_multistart(opt, starts, max(100, budget // max(2, len(starts))))
            success = True
            message = 'hybrid_global_local completed with optional threaded multi-start polish'
        elif method == 'recommended_adaptive_bounds':
            budget = int(max_evals)
            dim = len(names)
            if progress:
                progress('recommended_adaptive_bounds: Stage 1/4 broad global exploration with differential evolution')
            de_iter = max(3, budget // max(45, dim * 32))
            wkwargs, pool = _threaded_de_kwargs(workers)
            try:
                de = opt.differential_evolution(objective, bounds, seed=seed, polish=False, maxiter=de_iter, tol=1e-6, **({'updating': 'immediate'} if not wkwargs else wkwargs))
            finally:
                if pool:
                    pool.close(); pool.join()
            if de.fun < best_obj:
                best_x, best_obj = de.x, float(de.fun)
            original_span = np.maximum(ub - lb, 1e-30)
            center = np.clip(best_x, lb, ub)
            # Smarter shrink: high-order models get tighter spans; weak data keeps more span.
            npts = sum(len(d.x) for d in train_datasets)
            ppr = npts / max(dim, 1)
            complexity = getattr(model, 'complexity_level', max(1, dim // 2))
            shrink_ratio = 0.35
            if dim >= 8 or complexity >= 4:
                shrink_ratio = 0.16
            elif dim >= 5:
                shrink_ratio = 0.22
            if ppr < 10:
                shrink_ratio = max(shrink_ratio, 0.32)  # do not over-shrink underdetermined fits
            local_span = np.maximum(shrink_ratio * original_span, 0.10 * np.maximum(np.abs(center), 1.0))
            # If best is close to a bound, expand toward interior rather than clamping too tightly.
            rel = (center - lb) / original_span
            near = (rel < 0.06) | (rel > 0.94)
            local_span[near] = np.maximum(local_span[near], 0.45 * original_span[near])
            lb2 = np.maximum(lb, center - local_span)
            ub2 = np.minimum(ub, center + local_span)
            bad = (ub2 - lb2) < 1e-12 * np.maximum(np.abs(center), 1.0)
            lb2[bad] = lb[bad]
            ub2[bad] = ub[bad]
            adaptive_bounds_history.append({
                'stage': 'adaptive_shrink',
                'n_parameters': int(dim),
                'points_per_parameter': float(ppr),
                'shrink_ratio_base': float(shrink_ratio),
                'lower': {n: float(v) for n, v in zip(names, lb2)},
                'upper': {n: float(v) for n, v in zip(names, ub2)},
                'center': {n: float(v) for n, v in zip(names, center)},
            })
            if progress:
                progress('recommended_adaptive_bounds: Stage 2/4 bounds adapted using parameter count, data support and bound proximity')
            old_lb, old_ub = lb.copy(), ub.copy()
            lb[:] = lb2
            ub[:] = ub2
            if progress:
                progress('recommended_adaptive_bounds: Stage 3/4 robust least-squares polishing inside adaptive bounds')
            ls = opt.least_squares(residuals, np.clip(center, lb, ub), bounds=(lb, ub), max_nfev=max(250, budget // 2), x_scale='jac', loss=robust_loss, method='trf')
            obj = float(np.mean(ls.fun * ls.fun))
            if obj < best_obj:
                best_x, best_obj = ls.x, obj
            if progress:
                progress('recommended_adaptive_bounds: Stage 4/4 final polish on original engineering bounds')
            lb[:] = old_lb
            ub[:] = old_ub
            polish_least_squares(best_x, opt, maxn=max(150, budget // 4), ls_method='trf')
            success = True
            message = 'recommended adaptive-bounds workflow completed: global exploration + data-aware adaptive bounds + robust local polishing'
        else:
            raise ValueError(f'Unknown optimization method: {method}')
    except OptimizationStopped as e:
        stopped_by_user = True
        success = False
        message = str(e)
    except Exception as e:
        message = f'SciPy optimizer unavailable or failed; fallback random/coordinate search used. Details: {e}'
        span = ub - lb
        for i in range(max(300, int(max_evals))):
            if control:
                control.check()
            x = lb + rng.random(len(names)) * span
            obj = objective(x)
            if obj < best_obj:
                best_x, best_obj = x, obj
        step = 0.2 * span
        for cycle in range(20):
            improved = False
            for j in range(len(names)):
                for sgn in (-1.0, 1.0):
                    x = best_x.copy()
                    x[j] = np.clip(x[j] + sgn * step[j], lb[j], ub[j])
                    obj = objective(x)
                    if obj < best_obj:
                        best_x, best_obj = x, obj
                        improved = True
            if not improved:
                step *= 0.5
        success = True
    finally:
        if pool:
            try:
                pool.close(); pool.join()
            except Exception:
                pass
        _checkpoint(force=True)

    p = model.vector_to_params(best_x[:base_n])
    if has_volumetric:
        p['K'] = float(best_x[-1])
    all_y = []
    all_hat = []
    for d in datasets:
        if d.mode == 'volumetric':
            if has_volumetric:
                yh = hydrostatic_pressure_from_K(J_from_x(d.x), best_x[-1])
            else:
                continue
        else:
            yh = model.predict_nominal(d.mode, d.x, p)
        all_y.extend(list(d.stress))
        all_hat.extend(list(yh))
    rmse, mae, r2 = _metrics(np.array(all_y), np.array(all_hat))
    try:
        stability = model.stability_scan(p)
    except Exception as e:
        stability = {'passed': False, 'warnings': [str(e)], 'mode_results': {}}
    if trace and trace[-1].get('eval') != nfev:
        rec = {'eval': int(nfev), 'objective': float(best_obj), 'best_objective': float(best_obj)}
        rec.update({f'p_{nm}': float(v) for nm, v in zip(names, best_x)})
        trace.append(rec)
    diagnostics = build_fit_diagnostics(model, datasets, train_datasets, validation_datasets, p, names, bounds_dict)
    all_metrics = diagnostics.get('all_metrics') or {}
    train_metrics = diagnostics.get('train_metrics') or {}
    validation_metrics = diagnostics.get('validation_metrics')
    uncertainty = diagnostics.get('uncertainty') or {}
    confidence_intervals = uncertainty.get('confidence_intervals') if uncertainty.get('available') else None
    parameter_correlation = {
        'available': bool(uncertainty.get('available')),
        'parameter_names': uncertainty.get('parameter_names', []),
        'matrix': uncertainty.get('correlation_matrix'),
        'reason': uncertainty.get('reason'),
    }
    covariance_matrix = uncertainty.get('covariance_matrix') if uncertainty.get('available') else None
    regularization_summary = {
        'strength': float(regularization or 0.0),
        'type': str(regularization_type),
        'near_bound_penalty': float(near_bound_penalty or 0.0),
        'applied': bool((regularization and regularization > 0) or (near_bound_penalty and near_bound_penalty > 0)),
    }
    return FitResult(
        success=success,
        method=method,
        parameters=p,
        objective=float(best_obj),
        rmse=rmse,
        mae=mae,
        r2=r2,
        message=message,
        nfev=int(nfev),
        elapsed_s=float(time.time() - t0),
        bounds=bounds_dict,
        trace=trace,
        stability=stability,
        optimizer_settings={
            'max_evals': max_evals,
            'seed': seed,
            'robust_loss': robust_loss,
            'regularization': regularization,
            'regularization_type': regularization_type,
            'validation_fraction': validation_fraction,
            'cv_folds': cv_folds,
            'workers': workers,
            'early_stop_patience': early_stop_patience,
            'early_stop_tol': early_stop_tol,
            'near_bound_penalty': near_bound_penalty,
        },
        adjusted_r2=all_metrics.get('adjusted_r2'),
        aic=all_metrics.get('aic'),
        aicc=all_metrics.get('aicc'),
        bic=all_metrics.get('bic'),
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        confidence_intervals=confidence_intervals,
        parameter_correlation=parameter_correlation,
        covariance_matrix=covariance_matrix,
        regularization_summary=regularization_summary,
        overfitting_warnings=diagnostics.get('overfitting_warnings') or [],
        adaptive_bounds_history=adaptive_bounds_history,
        checkpoint_file=str(checkpoint_path) if checkpoint_path else None,
        stopped_by_user=stopped_by_user,
    )
