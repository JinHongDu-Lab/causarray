"""The fused parallel likelihood kernels (nll_mat, grad_genes, grad_cells) must
reproduce the whole-array reference kernels (nll, grad) for every broadcast
shape the optimiser uses, and their results must not depend on the numba
thread count."""
import numpy as np
import numba
import pytest

from causarray.gcate_likelihood import nll, grad, nll_mat, grad_genes, grad_cells, type_f


def _problem(n=120, p=90, d=4, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 0.5, (n, d)); A[:, 0] = 1
    B = rng.normal(0, 0.5, (p, d)); B[:, 0] = rng.uniform(-2, 3, p)
    Y = rng.poisson(np.exp(A @ B.T)).astype(type_f)
    nuisance = rng.uniform(0.5, 200, (1, p))      # mixes genes above and below thres_disp
    return Y, A, B, nuisance


@pytest.mark.parametrize('family', ['poisson', 'nb'])
@pytest.mark.parametrize('tys_shape', [(120, 90), (120, 1), (1, 1)])
def test_nll_mat_matches_nll(family, tys_shape):
    Y, A, B, nu = _problem()
    Tys = np.random.default_rng(1).normal(size=tys_shape)
    ref = nll(Y, A, B, family, nu, Tys, 10.)
    new = nll_mat(Y, A, B, family, nu, Tys, 10.)
    assert abs(ref - new) <= 1e-12 * abs(ref)


@pytest.mark.parametrize('family', ['poisson', 'nb'])
def test_gradients_match_reference(family):
    Y, A, B, nu = _problem()
    np.testing.assert_allclose(grad_genes(Y, A, B, family, nu, 10.), grad(Y, A, B, family, nu, 10.), rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(grad_cells(Y, A, B, family, nu, 10.), grad(Y.T, B, A, family, nu.T, 10.), rtol=1e-11, atol=1e-13)


def test_full_matrix_nuisance_broadcast():
    Y, A, B, nu = _problem()
    nu_full = np.broadcast_to(nu, Y.shape).copy()
    Tys = np.zeros((1, 1))
    ref = nll(Y, A, B, 'nb', nu_full, Tys, 10.)
    assert abs(ref - nll_mat(Y, A, B, 'nb', nu_full, Tys, 10.)) <= 1e-12 * abs(ref)
    np.testing.assert_allclose(grad_genes(Y, A, B, 'nb', nu_full, 10.), grad(Y, A, B, 'nb', nu_full, 10.), rtol=1e-11, atol=1e-13)


def test_results_independent_of_thread_count():
    """Fixed-order reductions: bitwise identical across thread counts."""
    Y, A, B, nu = _problem(n=300, p=400)
    Tys = np.zeros((1, 1))
    saved = numba.get_num_threads()
    out = {}
    try:
        for k in sorted({1, 2, min(4, numba.config.NUMBA_NUM_THREADS)}):
            numba.set_num_threads(k)
            out[k] = (nll_mat(Y, A, B, 'nb', nu, Tys, 10.), grad_genes(Y, A, B, 'nb', nu, 10.), grad_cells(Y, A, B, 'nb', nu, 10.))
    finally:
        numba.set_num_threads(saved)
    ref = out[1]
    for k, (v, gg, gc) in out.items():
        assert v == ref[0]
        assert np.array_equal(gg, ref[1]) and np.array_equal(gc, ref[2])
