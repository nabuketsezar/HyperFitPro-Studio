from __future__ import annotations
import math


def information_criteria(result, n_points: int, n_params: int):
    rss = max(result.rmse**2 * max(n_points,1), 1e-300)
    n=max(n_points,1); k=max(n_params,1)
    aic = n*math.log(rss/n) + 2*k
    bic = n*math.log(rss/n) + k*math.log(n)
    return {'AIC': float(aic), 'BIC': float(bic)}
