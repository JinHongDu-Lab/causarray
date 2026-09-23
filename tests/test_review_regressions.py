"""Regressions for offset inference, callbacks and dense/on-disk fitting."""
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from causarray.DR_learner import LFC, compute_causal_estimand
from causarray.nb_glm_fast import fit_glm_fast, fit_glm_ondisk


@pytest.mark.parametrize('variable', [False, True])
def test_poisson_floor_uses_arm_size_factors(variable):
    A = np.repeat([0., 1.], [60, 40])
    sf = np.resize([1., 2., 4., 8.], 100) if variable else np.ones(100)
    pred = np.broadcast_to(sf[:, None, None, None] * np.array([2., 4.]),
                           (100, 2, 1, 2)).copy()
    Y = pred[np.arange(100), :, 0, A.astype(int)]
    results = []
    for scale in [1., 100.]:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            df, _ = LFC(Y, np.ones((100, 1)), A, offset=np.log(sf * scale),
                        Y_hat=pred.copy(), pi_hat=np.full((100, 1), .4),
                        thres_min=0, thres_diff=0, ps_clip=None)
        expected = np.mean(1 / sf[A == 0]) / (60 * 2)
        expected += np.mean(1 / sf[A == 1]) / (40 * 4)
        np.testing.assert_allclose(df['std']**2, expected)
        assert df['var_floored'].all()
        results.append(df)
    np.testing.assert_allclose(results[0][['tau', 'std', 'pvalue']],
                               results[1][['tau', 'std', 'pvalue']])


def test_two_argument_estimand_callback():
    def callback(etas, A):
        eta = etas[..., 1] - etas[..., 0]
        return eta, eta.mean(axis=0), np.var(eta, axis=0) / len(A)

    rng = np.random.default_rng(13)
    Y = rng.poisson(3, (40, 2)).astype(float)
    df, _ = compute_causal_estimand(
        callback, Y, np.ones((40, 1)), np.repeat([0., 1.], 20),
        Y_hat=np.full((40, 2, 1, 2), 3.), pi_hat=np.full((40, 1), .5))
    assert np.isfinite(df['std']).all()


@pytest.mark.parametrize('embedded', [False, True])
@pytest.mark.parametrize('size', [.1, 100.])
def test_dense_glm_honors_fixed_dispersion(embedded, size):
    rng = np.random.default_rng(12)
    X = np.c_[np.ones(100), rng.normal(size=100)]
    if embedded:
        X = np.c_[X, np.repeat([0., 1.], 50)]
    off = rng.normal(0, .2, 100)
    Y = rng.negative_binomial(2, .4, (100, 2)).astype(float)
    B, _, disp, _, _ = fit_glm_fast(Y, X, family='nb',
                                  disp_glm=np.full(2, size), offset=off)
    np.testing.assert_allclose(disp, size)
    expected = [sm.GLM(Y[:, j], X, offset=off,
                      family=sm.families.NegativeBinomial(alpha=1 / size))
                .fit(tol=1e-10).params for j in range(2)]
    np.testing.assert_allclose(B, expected, atol=2e-5)


def test_ondisk_filters_covariates_with_empty_selected_cells(tmp_path):
    ad = pytest.importorskip('anndata')
    rng = np.random.default_rng(7)
    Y = rng.poisson(3, (30, 3)).astype(float)
    Y[4, :2] = 0  # Empty only among selected genes.
    obs = pd.DataFrame({'perturbation': ['control', 'target', 'other'] * 10,
                        'covariate': rng.normal(size=30)},
                       index=[str(i) for i in range(30)])
    path = tmp_path / 'counts.h5ad'
    ad.AnnData(Y, obs=obs).write_h5ad(path)
    with pytest.warns(RuntimeWarning, match='dropped'):
        actual = fit_glm_ondisk(str(path), target_label='target',
                               gene_indices=np.array([0, 1]),
                               covariate_columns=['covariate'])
    keep = (obs['perturbation'] != 'other').to_numpy() & (Y[:, :2].sum(axis=1) > 0)
    expected = fit_glm_fast(
        Y[keep, :2], np.c_[np.ones(keep.sum()), obs['covariate'].to_numpy()[keep]],
        A=(obs['perturbation'].to_numpy()[keep] == 'target')[:, None].astype(float),
        offset=True, family="nb", maxiter=25)
    for value, reference in zip(actual, expected):
        np.testing.assert_allclose(value, reference)
