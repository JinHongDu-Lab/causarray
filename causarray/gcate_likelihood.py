import numpy as np
# from scipy.special import expit, xlogy, xlog1py, logsumexp, factorial
# from scipy.stats import binom, poisson, norm, nbinom

import numba as nb
from numba import njit, prange

type_f = np.float64


from scipy.special import xlogy, gammaln
from numba.extending import get_cython_function_address
import ctypes

_PTR = ctypes.POINTER
_dble = ctypes.c_double
_ptr_dble = _PTR(_dble)

addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
functype = ctypes.CFUNCTYPE(_dble, _dble)
gammaln_float64 = functype(addr)

@nb.vectorize
def gammaln_nb(x):
  return gammaln_float64(x)


def log_h(y, family, nuisance):
    if family=='nb':
        return gammaln(y + nuisance) - gammaln(nuisance) - gammaln(y+1)
    elif family=='poisson':
        return - gammaln(y+1)


@nb.vectorize
def log1mexp(a):
    '''
    A numeral stable function to compute log(1-exp(a)) for a in [-inf,0].
    '''
    if(a >= -np.log(type_f(2.))):
        return np.log(-np.expm1(a)) 
    else:
        return np.log1p(-np.exp(a))
    

@njit
def nll(Y, A, B, family, nuisance=np.ones((1,1)), Tys=np.zeros((1,1)), thres_disp=10.
    #  size_factor=np.ones((1,1))
    ):
    """
    Compute the negative log likelihood for generalized linear models with optional nuisance parameters.
    
    Parameters:
    Y : array-like of shape (n_samples, n_features)
        The response variable.
    A : array-like of shape (n_samples, n_factors)
        The input data matrix.
    B : array-like of shape (n_features, n_factors)
        The input data matrix.
    family : str, optional (default='gaussian')
        The family of the generalized linear model. Options include 'poisson', and 'nb'.
    nuisance : float or array-like of shape (n_samples,), optional (default=1)
        The nuisance parameter for the family. For the Gaussian family, this is the variance; for the Poisson
        family, this is the scaling factor; and for the negative binomial family, this is the overdispersion
        parameter.
    size_factor : float or array-like of shape (n_samples,), optional (default=1)
        The size factor for the response variable.
    
    Returns:
    nll : float
        The negative log likelihood.
    """
    
    Theta = A @ B.T
    Ty = Y
    n = Y.shape[0]
    
    if family == 'poisson':
        Theta = np.clip(Theta, -np.inf, type_f(1e2))
        b = np.exp(Theta)
    elif family == 'nb':
        Xi = np.clip(Theta, -np.inf, type_f(1e2))
        exp_Xi = np.exp(Xi)
        tmp = np.clip(1 / (type_f(1.) + exp_Xi / nuisance), 1e-6, 1-1e-6)

        Theta = np.where(nuisance > thres_disp, Xi, np.log1p(-tmp))
        b = np.where(nuisance > thres_disp, exp_Xi, - nuisance * np.log(tmp) #+ gammaln_nb(nuisance+Y) - gammaln_nb(nuisance)
        ) # ignoring a common factor - gammaln_nb(Y)
    else:
        raise ValueError('Family not recognized')

    nll = - np.sum(Ty * Theta - b + Tys) / type_f(n) # * size_factor to get back the likelihood

    return nll



@njit
def grad(Y, A, B, family, nuisance=np.ones((1,1)), thres_disp=10.
    #size_factor=np.ones((1,1)),
        ):
    """
    Compute the gradient of log likelihood with respect to B
    for generalized linear models with optional nuisance parameters.
    
    The natural parameter of Y is Theta = A @ B^T.
    
    Parameters:
    Y : array-like of shape (n_samples, n_features)
        The response variable.
    A : array-like of shape (n_samples, n_factors)
        The input data matrix.
    B : array-like of shape (n_features, n_factors)
        The input data matrix.
    family : str, optional (default='gaussian')
        The family of the generalized linear model. Options include 'poisson', and 'nb'.
    nuisance : float or array-like of shape (n_samples,), optional (default=1)
        The nuisance parameter for the family. For the Gaussian family, this is the variance; for the Poisson
        family, this is the scaling factor; and for the negative binomial family, this is the overdispersion
        parameter.
    
    Returns:
    grad : array-like of shape (n_features, n_factors)
        The gradient of log likelihood.
    """
    Theta = A @ B.T
    Ty = Y
    n = Y.shape[0]
    
    if family == 'nb':
        Xi = np.clip(Theta, -np.inf, type_f(1e2))
        b_p = np.exp(Xi)
        tmp = np.clip(1 / (type_f(1.) + b_p / nuisance), 1e-6, 1-1e-6)
        grad = - (Ty - b_p) * np.where(nuisance > thres_disp, 1., tmp) # * size_factor to get back the likelihood
    elif family == 'poisson':
        Theta = np.clip(Theta, -np.inf, type_f(1e2))
        b_p = np.exp(Theta)
        grad = - (Ty - b_p) # * size_factor to get back the likelihood
    else:
        raise ValueError('Family not recognized')
    grad = grad.T @ A / type_f(n)
    return grad



# ---------------------------------------------------------------------------
# Fused, parallel matrix kernels
# ---------------------------------------------------------------------------
# ``nll`` and ``grad`` above are written with whole-array NumPy expressions.
# Inside numba each expression allocates a full (n, p) temporary and runs on
# one core, so for the (n, p) calls made once per alternating-minimisation
# epoch (two gradients and one or two objective values) they cost about ten
# passes over the count matrix each and dominated the epoch: on a 3,000 x
# 3,000 problem ~2 s of a ~2.3 s epoch, against ~0.15 s for the prange line
# searches. The kernels below evaluate the same expressions in one fused pass
# per row, in parallel over rows, with no temporaries beyond the (n, p) linear
# predictor. Per-row partial sums are combined in a fixed order, so the result
# does not depend on the thread count.
#
# ``nuisance`` and ``Tys`` may be (1, p), (n, 1), (n, p) or (1, 1); they are
# broadcast by stride, as NumPy would.

_GENE_BLOCK = 128   # genes per work item in grad_genes; fixed so results do not depend on thread count


@njit(parallel=True)
def nll_mat(Y, A, B, family, nuisance, Tys, thres_disp):
    """Negative log-likelihood of ``Y`` (n, p) with natural parameter ``A @ B.T``.

    Same value as ``nll(Y, A, B, ...)`` up to floating-point summation order.
    """
    Theta = A @ np.ascontiguousarray(B.T)
    n, p = Theta.shape
    si_nu = 0 if nuisance.shape[0] == 1 else 1
    sj_nu = 0 if nuisance.shape[1] == 1 else 1
    si_t = 0 if Tys.shape[0] == 1 else 1
    sj_t = 0 if Tys.shape[1] == 1 else 1
    is_pois = family == 'poisson'
    hi = type_f(1e2)
    lo_p = type_f(1e-6)
    hi_p = type_f(1.) - type_f(1e-6)
    part = np.zeros(n, dtype=type_f)
    for i in prange(n):
        s = type_f(0.)
        for j in range(p):
            th = Theta[i, j]
            if th > hi:
                th = hi
            y = Y[i, j]
            ty = Tys[i * si_t, j * sj_t]
            e = np.exp(th)
            if is_pois:
                s += y * th - e + ty
            else:
                nu = nuisance[i * si_nu, j * sj_nu]
                if nu > thres_disp:
                    s += y * th - e + ty
                else:
                    tmp = type_f(1.) / (type_f(1.) + e / nu)
                    if tmp < lo_p:
                        tmp = lo_p
                    elif tmp > hi_p:
                        tmp = hi_p
                    s += y * np.log1p(-tmp) + nu * np.log(tmp) + ty
        part[i] = s
    total = type_f(0.)
    for i in range(n):          # serial: fixed summation order
        total += part[i]
    return -total / type_f(n)


@njit(inline='always')
def _resid_entry(y, th, nu, is_pois, thres_disp):
    """``-(y - mu) * w`` for one entry; ``w`` as in ``grad``."""
    if th > type_f(1e2):
        th = type_f(1e2)
    e = np.exp(th)
    if is_pois or nu > thres_disp:
        return -(y - e)
    tmp = type_f(1.) / (type_f(1.) + e / nu)
    if tmp < type_f(1e-6):
        tmp = type_f(1e-6)
    elif tmp > type_f(1.) - type_f(1e-6):
        tmp = type_f(1.) - type_f(1e-6)
    return -(y - e) * tmp


@njit(parallel=True)
def grad_genes(Y, A, B, family, nuisance, thres_disp):
    """Gradient w.r.t. ``B`` (p, d): equals ``grad(Y, A, B, ...)``.

    Work items are fixed blocks of ``_GENE_BLOCK`` genes; within a block the
    sum over cells runs in order, so the result is independent of the thread
    count.
    """
    Theta = A @ np.ascontiguousarray(B.T)
    n, p = Theta.shape
    d = A.shape[1]
    si_nu = 0 if nuisance.shape[0] == 1 else 1
    sj_nu = 0 if nuisance.shape[1] == 1 else 1
    is_pois = family == 'poisson'
    n_blocks = (p + _GENE_BLOCK - 1) // _GENE_BLOCK
    G = np.zeros((p, d), dtype=type_f)
    for blk in prange(n_blocks):
        j0 = blk * _GENE_BLOCK
        j1 = min(p, j0 + _GENE_BLOCK)
        for i in range(n):
            for j in range(j0, j1):
                r = _resid_entry(Y[i, j], Theta[i, j], nuisance[i * si_nu, j * sj_nu], is_pois, thres_disp)
                for k in range(d):
                    G[j, k] += r * A[i, k]
    return G / type_f(n)


@njit(parallel=True)
def grad_cells(Y, A, B, family, nuisance, thres_disp):
    """Gradient w.r.t. ``A`` (n, d): equals ``grad(Y.T, B, A, ..., nuisance.T)``.

    One work item per cell; the sum over genes runs in order.
    """
    Theta = A @ np.ascontiguousarray(B.T)
    n, p = Theta.shape
    d = B.shape[1]
    si_nu = 0 if nuisance.shape[0] == 1 else 1
    sj_nu = 0 if nuisance.shape[1] == 1 else 1
    is_pois = family == 'poisson'
    G = np.zeros((n, d), dtype=type_f)
    for i in prange(n):
        for j in range(p):
            r = _resid_entry(Y[i, j], Theta[i, j], nuisance[i * si_nu, j * sj_nu], is_pois, thres_disp)
            for k in range(d):
                G[i, k] += r * B[j, k]
    return G / type_f(p)
