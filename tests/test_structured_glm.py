"""causarray's GLM path on designs ``[covariates | one-hot treatments]``.

The solver itself lives in crispyx (``crispyx.glm.StructuredGLMBatchFitter``);
what is tested here is causarray's side of it -- that the router reaches it,
that the conventions line up (NB size ``r`` against crispyx's ``alpha``,
column order, counterfactual tensors), and that the result is the same model
the gene-by-gene statsmodels path fits.
"""
import numpy as np
import pytest
import statsmodels.api as sm

import causarray.gcate_glm as g
from causarray.nb_glm_fast import fit_glm_fast


def _sim(n=600, p=30, dx=3, a=6, seed=0, nb=True):
    rng = np.random.default_rng(seed)
    X = np.c_[np.ones(n), rng.standard_normal((n, dx - 1))]
    grp = rng.integers(-1, a, n)                 # -1 = control
    G = np.zeros((n, a)); G[np.arange(n)[grp >= 0], grp[grp >= 0]] = 1.0
    off = rng.normal(0, 0.3, n)
    Bx = rng.normal(0, 0.3, (p, dx)); Bx[:, 0] = rng.uniform(-1, 2, p)
    Ba = rng.normal(0, 0.5, (p, a))
    eta = off[:, None] + X @ Bx.T + G @ Ba.T
    mu = np.exp(eta)
    r = rng.uniform(2, 10, p)
    Y = rng.poisson(mu * rng.gamma(r, 1 / r, mu.shape)) if nb else rng.poisson(mu)
    return Y.astype(float), X, G, off, r, grp


def _statsmodels(Y, X, G, off, family, r):
    B = np.empty((Y.shape[1], X.shape[1] + G.shape[1]))
    D = np.c_[X, G]
    for j in range(Y.shape[1]):
        fam = sm.families.Poisson() if family == 'poisson' else sm.families.NegativeBinomial(alpha=1 / r[j])
        B[j] = sm.GLM(Y[:, j], D, family=fam, offset=off).fit(maxiter=200, tol=1e-10).params
    return B


@pytest.mark.parametrize('family', ['poisson', 'nb'])
def test_matches_statsmodels_on_well_conditioned_genes(family):
    Y, X, G, off, r, _ = _sim(nb=(family == 'nb'))
    B, mu, _, _, dres = fit_glm_fast(
        Y, X, A=G, family=family, disp_glm=r if family == 'nb' else None, offset=off)
    B_sm = _statsmodels(Y, X, G, off, family, r)
    np.testing.assert_allclose(B, B_sm, atol=2e-5, rtol=1e-5)
    assert mu.shape == Y.shape and np.all(np.isfinite(mu))
    assert dres.shape == Y.shape and np.all(np.isfinite(dres))


def test_zero_count_group_is_finite_and_clipped():
    """A treatment arm with no counts has an unbounded MLE; it must stay finite."""
    Y, X, G, off, r, grp = _sim(nb=False)
    Y[grp == 2, 0] = 0.0                            # gene 0 has no counts in group 2
    B, *_ = fit_glm_fast(Y, X, A=G, family='poisson', offset=off)
    assert np.all(np.isfinite(B))
    assert B[0, X.shape[1] + 2] <= -9.0             # driven to the clip bound
    B_sm = _statsmodels(Y[:, :1], X, G, off, 'poisson', r[:1])
    keep = np.r_[np.arange(X.shape[1]), X.shape[1] + np.array([0, 1, 3, 4, 5])]
    np.testing.assert_allclose(B[0, keep], B_sm[0, keep], atol=5e-3)


def test_fit_glm_auto_routes_wide_onehot_design_and_matches_statsmodels():
    """GCATE hands the treatments inside X; that design must not fall to statsmodels."""
    Y, X, G, off, r, _ = _sim(n=500, p=60, dx=3, a=8, nb=True)
    B, Yhat, disp_out, offsets, dres = g.fit_glm_auto(
        Y, np.c_[X, G], None, family='nb', disp_glm=r, offset=off)
    assert B.shape == (60, 11) and Yhat.shape == Y.shape and dres.shape == Y.shape
    B_sm = _statsmodels(Y, X, G, off, 'nb', r)
    ok = np.abs(B_sm) < 8
    assert np.nanmedian(np.abs(B - B_sm)[ok]) < 1e-4


def test_estimate_disp_auto_is_close_to_the_statsmodels_estimate():
    Y, X, G, off, r, _ = _sim(n=800, p=80, dx=2, a=5, nb=True)
    d_fast = g.estimate_disp_auto(Y, np.c_[X, G], offset=off, disp_family='poisson')
    d_slow = g.estimate_disp(Y, np.c_[X, G], offset=off, disp_family='poisson')
    assert d_fast.shape == (80,) and np.all(np.isfinite(d_fast))
    # same model, same moments estimator: agree within a factor 1.5 on the median
    assert 0.67 < np.median(d_fast / d_slow) < 1.5


def test_estimate_disp_auto_returns_none_without_the_fast_path():
    """None means "no cheap estimate"; callers then let the fitter estimate per gene."""
    Y, X, G, off, r, _ = _sim(n=200, p=80, dx=2, a=4, nb=True)
    with g._backend_override('original'):
        assert g.estimate_disp_auto(Y, np.c_[X, G], offset=off) is None
    assert g.estimate_disp_auto(Y[:, :4], np.c_[X, G], offset=off) is None  # p < _FAST_MIN_P


def test_impute_path_matches_statsmodels_reference():
    """The counterfactual means equal the statsmodels backend's, gene for gene."""
    Y, X, G, off, r, _ = _sim(n=500, p=60, dx=3, a=4, nb=True)
    with g._backend_override('original'):
        B_ref, (Y0_ref, Y1_ref), *_ = g.fit_glm_auto(
            Y, X, G, family='nb', disp_glm=r, offset=off, impute=True, n_jobs=1)
    B, (Y0, Y1), disp_out, offsets, dres = g.fit_glm_auto(
        Y, X, G, family='nb', disp_glm=r, offset=off, impute=True)
    assert Y0.shape == (500, 60, 4) and Y1.shape == (500, 60, 4)
    ok = (np.abs(B_ref) < 8).all(axis=1)             # genes without divergent coefficients
    np.testing.assert_allclose(B[ok], B_ref[ok], atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(Y0[:, ok], Y0_ref[:, ok], rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(Y1[:, ok], Y1_ref[:, ok], rtol=1e-3, atol=1e-6)
