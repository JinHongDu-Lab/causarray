"""estimate_r: one shared initial GLM for the r grid, per-r wall time, sane JIC."""
import numpy as np
import pandas as pd

from causarray import estimate_r


def test_estimate_r_returns_time_and_finite_jic():
    rng = np.random.default_rng(0)
    n, p, a, r_true = 400, 150, 3, 2
    U = rng.normal(0, 0.6, (n, r_true)); V = rng.normal(0, 0.6, (p, r_true))
    grp = rng.integers(-1, a, n)
    A = np.zeros((n, a)); A[np.arange(n)[grp >= 0], grp[grp >= 0]] = 1
    Ba = rng.normal(0, 0.4, (p, a))
    eta = rng.uniform(-1, 2, p)[None, :] + U @ V.T + A @ Ba.T
    Y = rng.poisson(np.exp(eta)).astype(float)
    X = np.ones((n, 1))
    df = estimate_r(Y, X, A, r_max=3, family='poisson', offset=False,
                    kwargs_es_1=dict(max_iters=3), kwargs_es_2=dict(max_iters=3))
    assert list(df.columns) == ['r', 'deviance', 'nu', 'JIC', 'time_s']
    assert df['r'].tolist() == [0, 1, 2, 3]
    assert np.all(np.isfinite(df[['deviance', 'nu', 'JIC']].to_numpy()))
    assert (df['time_s'] > 0).all()
    # adding factors must not make the (unpenalised) deviance term larger than r = 0
    assert df.loc[df.r == r_true, 'deviance'].item() < df.loc[df.r == 0, 'deviance'].item()
