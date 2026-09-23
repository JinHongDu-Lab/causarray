from causarray.utils import *
from causarray.utils import _filter_params
import contextlib
import numpy as np
import statsmodels as stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings('ignore')

from causarray.nb_glm_fast import fit_glm_fast, estimate_disp_fast, _resolve_offset

# ---------------------------------------------------------------------------
# Backend control flags
# ---------------------------------------------------------------------------

def _crispyx_available() -> bool:
    """Return True if crispyx is importable."""
    try:
        import crispyx  # noqa: F401
        return True
    except ImportError:
        return False

_CRISPYX_AVAILABLE: bool = _crispyx_available()  # evaluated once at import time

_USE_FAST_BACKEND: bool = True
"""Set to False to force statsmodels path everywhere (benchmarking / debugging).

Note: not thread-safe; use _backend_override() for scoped switching.
"""

_FAST_MIN_P: int = 10
"""Minimum number of genes for the batched crispyx path.

Benchmarked 2026-09-22 on crispyx 0.1.5 (`plan/glm_benchmark/bench_min_p.py`,
n = 1,000, 3 covariates, 3 treatments, NB): the batched path is faster at every
gene count measured -- 533x at p = 5, 59x at p = 50, 2.9x at p = 500 -- because
the statsmodels loop pays for a joblib pool before it fits anything, and it
agrees with statsmodels to max |dB| 1e-5 with the dispersion supplied and 1e-2
(median 3e-4) with it estimated.  So the old cap of 50, inherited from the
crispyx 0.1.4 era, was not paying for itself.  What does not survive small p is
the dispersion: crispyx's per-gene moments estimate sits within 5-12% of the
gene-by-gene estimate down to p = 10 and is twice it at p = 5, where a call is
too cheap for the routing to matter anyway.
"""

_FAST_MAX_COEF: float = 1e4
"""Maximum |coefficient| accepted from the batched path; beyond it, statsmodels.

The bound has not fired on any realistic design since the structured solver
arrived: measured 2026-09-22 (`plan/glm_benchmark/check_max_coef.py`) the
largest |coefficient| is 17.6 on a sparse tail of genes at 0.002 counts per
cell, 10.3 on latent-factor columns of standard deviation 0.009, 10.0 on an
empty treatment arm (the group clip) and 1.7 on a singular design with a
duplicated column, against a bound of 1e4.  That is expected -- crispyx clips
the linear predictor and the group coefficients, and ridges the rest -- so what
the guard really catches now is a non-finite fit.  It is kept because it costs
one `np.max` and the statsmodels fallback regularises.
"""


@contextlib.contextmanager
def _backend_override(backend: str):
    """Context manager to temporarily force 'fast' or 'original' GLM backend.

    Parameters
    ----------
    backend : str
        ``"fast"`` to force crispyx, ``"original"`` to force statsmodels,
        ``"auto"`` to leave the current setting unchanged.

    Notes
    -----
    Not thread-safe: mutates the module-level ``_USE_FAST_BACKEND`` flag.
    """
    global _USE_FAST_BACKEND
    old = _USE_FAST_BACKEND
    if backend == "fast":
        if not _CRISPYX_AVAILABLE:
            import warnings
            warnings.warn(
                "backend='fast' was requested but crispyx is not importable; "
                "falling back to the gene-by-gene statsmodels backend, which is "
                "much slower and can differ numerically. Install crispyx to use "
                "the fast path.",
                RuntimeWarning, stacklevel=3,
            )
        _USE_FAST_BACKEND = True
    elif backend == "original":
        _USE_FAST_BACKEND = False
    try:
        yield
    finally:
        _USE_FAST_BACKEND = old


def init_inv_link(Y, family, disp):
    if family=='gaussian':
        val = Y/disp
    elif family=='poisson':
        val = np.log1p(Y)
    elif family=='nb':
        val = np.log1p(Y)
    elif family=='binomial':
        eps = (np.mean(Y, axis=0) + np.mean(Y, axis=1)) / 2 
        val = np.log((Y + eps)/(disp - Y + eps))
    else:
        raise ValueError('Family not recognized')
    return val



def fit_glm(Y, X, A=None, family='gaussian', disp_family='poisson',
    disp_glm=None, impute=False, offset=None, offset_test=None, shrinkage=False,
    alpha=1e-4, maxiter=1000, thres_disp=100., n_jobs=-3, random_state=0, verbose=False,
    mem_limit_gb=None, **kwargs):
    '''
    Fit GLM to each column of Y, with covariate X and treatment A.

    Parameters
    ----------
    Y : array
        n x p matrix of outcomes
    X : array
        n x d matrix of covariates
    A : array
        n x 1 vector of treatments or None
    family : str
        Family of GLM to fit, can be one of: 'gaussian', 'poisson', 'nb'
    disp_glm : array or None
        Dispersion parameter for negative binomial GLM.
    impute : bool or None
        Whether to impute missing values in Y.        
    offset : bool
        Whether to use log of sum of Y as offset.
    shrinkage : bool
        Whether to use regularized GLM.
    alpha : float
        Regularization parameter for regularized GLM.
    maxiter : int
        Maximum number of iterations for GLM fitting.
    thres_disp : float
        Threshold for dispersion parameter for negative binomial GLM.
    n_jobs : int
        Number of jobs to run in parallel.
    random_state : int
        Random seed for reproducibility.
    verbose : bool
        Whether to print progress messages.
    kwargs : dict
        Additional arguments to pass to GLM fitting.

    Returns
    -------
    B : array
        d x p matrix of coefficients
    Yhat : array
        n x p x a matrix of predicted values
    disp_glm : array
        p x 1 vector of dispersion parameters
    offsets : array
        n x 1 vector of offsets
    resid_deviance : array
        n x p matrix of deviance residuals
    '''
    np.random.seed(random_state)
    
    if family not in ['gaussian', 'poisson', 'nb']:
        raise ValueError('Family not recognized')

    d = X.shape[1]

    if A is None:
        a = 1 # dummy treatment
        assert impute is False
    else:
        if A.ndim==1:
            A = A[:,None]
        if impute is not False and isinstance(impute, np.ndarray):
            X_test = impute
        else:
            X_test = X
        X_test = np.c_[X,np.zeros_like(A)]
        X = np.c_[X,A]
        a = A.shape[1]

    offsets = _resolve_offset(Y, offset, kwargs)

    # estimate dispersion parameter for negative binomial GLM if not provided
    if family=='nb' and disp_glm is None:
        disp_glm = estimate_disp(Y, X, offset=offsets, disp_family=disp_family, maxiter=1000, verbose=verbose, **kwargs)
    
    alpha = np.full(X.shape[1], alpha)
    pprint.pprint('Fitting {} GLM{}...'.format(family, '' if offsets is None else ' with offset'))
    is_constant = np.all(X == X[0, :], axis=0)
    alpha[is_constant] = 0


    families = {
        'gaussian': lambda disp: sm.families.Gaussian(),
        'poisson': lambda disp: sm.families.Poisson(),
        'nb': lambda disp: sm.families.NegativeBinomial(alpha=1/disp)
    }

    def fit_model(j, Y, X, offsets, family, disp, impute, alpha):
        if family=='nb' and disp[j]>thres_disp:
            family = 'poisson'
        try:            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                glm_family = families.get(family, lambda: ValueError('family must be one of: "gaussian", "poisson", "nb"'))(disp_glm[j] if family == 'nb' else None)

                try:
                    if shrinkage:
                        raise ValueError('fit regularized GLM')
                    mod = sm.GLM(Y[:,j], X, family=glm_family, offset=offsets).fit(maxiter=maxiter)
                    if not np.all(np.isfinite(mod.params)) or np.any(np.abs(mod.params[:d])>50) or np.any(np.abs(mod.params[d:])>10):
                        raise ValueError('GLM did not converge')
                    resid_deviance = mod.resid_deviance
                except:
                    mod = sm.GLM(Y[:,j], X, family=glm_family, offset=offsets).fit_regularized(alpha=alpha, cnvrg_tol=1e-5)
                    resid_deviance = np.full(Y.shape[0], 0.)

            B = mod.params

            Yhat_0 = np.zeros((Y.shape[0], a))
            Yhat_1 = np.zeros((Y.shape[0], a))
            if impute is not False:
                for k in range(a):
                    X_test_copy = X_test.copy()
                    Yhat_0[:,k] = mod.predict(X_test_copy, offset=offsets)                    
                    X_test_copy[:, d+k] = 1
                    Yhat_1[:,k] = mod.predict(X_test_copy, offset=offsets)
            else:
                # One fitted mean per cell, broadcast across the treatment axis;
                # reshape(-1, a) used to mis-shape it whenever a > 1.
                Yhat_0[:,:] = Yhat_1[:,:] = mod.predict(X, offset=offsets)[:, None]
            
        except:
            pprint.pprint('Fitting GLM for column {} does not converge.'.format(j))
            B = np.full(X.shape[1], 0.)
            
            Yhat_0 = np.full((Y.shape[0], a), 0.)
            Yhat_1 = np.full((Y.shape[0], a), 0.)
            if impute is not False:
                for k in range(a):
                    Yhat_0[:, k] = np.mean(Y[A[:, k] == 1, j])
                    Yhat_1[:, k] = np.mean(Y[A[:, k] == 0, j])
            resid_deviance = np.full(Y.shape[0], 0.)
        return B, Yhat_0, Yhat_1, resid_deviance


    results = Parallel(n_jobs=n_jobs)(delayed(fit_model)(
        j, Y, X, offsets, family, disp_glm, impute, alpha) for j in tqdm(range(Y.shape[1]), disable=not verbose))
    if verbose: pprint.pprint('Fitting GLM done.')

    B, Yhat_0, Yhat_1, resid_deviance = zip(*results)
    B = np.array(B)
    Yhat_0 = np.array(Yhat_0).transpose(1, 0, 2)
    Yhat_1 = np.array(Yhat_1).transpose(1, 0, 2)
    resid_deviance = np.array(resid_deviance).T

    # Match the fast path: when ``mem_limit_gb`` is set and the aggregated
    # imputation tensors would exceed the bound, downcast Yhat_0/Yhat_1 to
    # float32.  Peak memory still hits the float64 allocation produced by
    # the parallel ``fit_model`` calls (each worker materialises an
    # ``(n, a)`` slice), but the returned arrays are float32 so downstream
    # ``cross_fitting`` / ``AIPW_mean`` stay within budget.
    if mem_limit_gb is not None and impute is not False:
        _yhat_gb = (Yhat_0.size + Yhat_1.size) * 8 / 1e9
        if _yhat_gb > mem_limit_gb:
            warnings.warn(
                f"Imputation arrays ({_yhat_gb:.1f} GB as float64) exceed "
                f"mem_limit_gb={mem_limit_gb} GB; downcasting to float32 to "
                f"halve memory footprint of the returned Y_hat.",
                ResourceWarning, stacklevel=2,
            )
            Yhat_0 = Yhat_0.astype(np.float32)
            Yhat_1 = Yhat_1.astype(np.float32)

    if impute is not False:
        Yhat = (Yhat_0, Yhat_1)
    else:
        Yhat = np.array(Yhat_0)[:,:,0]

    return B, Yhat, disp_glm, offsets, resid_deviance


def estimate_disp(Y, X=None, A=None, Y_hat=None, disp_family='gaussian', offset=None, verbose=False, **kwargs):
    offsets = _resolve_offset(Y, offset, kwargs)
    sf = 1. if offsets is None else np.exp(offsets)[:,None]

    if Y_hat is None:        
        if verbose:
            pprint.pprint('Estimating dispersion parameter...')

        if A is not None:
            X = np.c_[X,A]

        if disp_family=='gaussian':
            Y_norm = Y/sf
            reg = LinearRegression(fit_intercept=False).fit(X, Y_norm)
            Y_hat = reg.predict(X)     
        elif disp_family=='poisson':
            Y_hat = fit_glm(Y, X, None, offset=offsets, family='poisson', impute=False, **kwargs)[1]      
            Y_hat /= sf

    # Clip Y_hat based on the range of Y per column
    Y_hat = np.clip(Y_hat, 0., np.max(Y/sf, axis=0))

    disp_glm = np.mean((Y/sf - Y_hat)**2 - Y_hat, axis=0) / np.mean(Y_hat**2, axis=0)
    disp_glm = 1./np.clip(disp_glm, 0.01, 100.)
    disp_glm[np.isnan(disp_glm)] = 1.

    return disp_glm




def loess_fit(Y, X, n_jobs=-3, **kwargs):
    
    def _loess_fit(y, x, **kwargs):
        try:
            from skmisc.loess import loess
            l = loess(x, y, **kwargs)
            l.fit()
            pred = l.predict(x, stderror=True)
            conf = pred.confidence()
            pred, lower, upper = pred.values, conf.lower, conf.upper
        except:
            pred, lower, upper = np.full(y.shape[0], np.nan), np.full(y.shape[0], np.nan), np.full(y.shape[0], np.nan)

        return pred, lower, upper

    results = Parallel(n_jobs=n_jobs)(delayed(_loess_fit)(Y[:,j], X, **kwargs) for j in range(Y.shape[1]))

    CATE, CATE_lower, CATE_upper = zip(*results)
    CATE = np.array(CATE).T
    CATE_lower = np.array(CATE_lower).T
    CATE_upper = np.array(CATE_upper).T
    return CATE, CATE_lower, CATE_upper



def ls_fit(Y, X, n_jobs=-3, **kwargs):
    
    def _ls_fit(y, x, **kwargs):
        # try:
        model = sm.OLS(y, x)
        result = model.fit(disp=False)

        # Get the predicted values
        pred = result.predict(x)

        # Get the confidence intervals
        conf = result.conf_int()    
        pred, lower, upper = pred, np.full(y.shape[0], conf[0][0]), np.full(y.shape[0], conf[0][1])
        # except:
        #     pred, lower, upper = np.full(y.shape[0], np.nan), np.full(y.shape[0], np.nan), np.full(y.shape[0], np.nan)

        return pred, lower, upper

    results = Parallel(n_jobs=n_jobs)(delayed(_ls_fit)(Y[:,j], X, **kwargs) for j in range(Y.shape[1]))

    CATE, CATE_lower, CATE_upper = zip(*results)
    CATE = np.array(CATE).T
    CATE_lower = np.array(CATE_lower).T
    CATE_upper = np.array(CATE_upper).T
    return CATE, CATE_lower, CATE_upper


def fit_glm_auto(Y, X, A=None, family='gaussian', disp_family='poisson',
    disp_glm=None, impute=False, offset=None, offset_test=None, shrinkage=False,
    alpha=1e-4, maxiter=1000, thres_disp=100., n_jobs=-3, random_state=0,
    verbose=False, mem_limit_gb=None, **kwargs):
    """Fit a GLM with crispyx's batched solvers, falling back to statsmodels.

    Routing:

    1. ``backend='original'`` (``_USE_FAST_BACKEND is False``), crispyx not
       installed, a Gaussian family, a ``shrinkage`` fit, or fewer than
       ``_FAST_MIN_P`` genes -> the gene-by-gene statsmodels path
       (:func:`fit_glm`).
    2. Otherwise :func:`causarray.nb_glm_fast.fit_glm_fast`, which fits every
       gene at once: crispyx's structured solver when the design carries a
       block of one-hot treatment indicators (exact, and its cost does not
       grow with the square of the number of treatments), its dense batch
       fitter otherwise.
    3. If those coefficients diverge, statsmodels as a last resort.

    Module-level knobs
    ------------------
    ``_USE_FAST_BACKEND`` : bool
        Master on/off switch.  Use ``_backend_override()`` for scoped changes.
    ``_FAST_MIN_P`` : int
        Minimum gene count for the batched path (default 10).
    ``_CRISPYX_AVAILABLE`` : bool
        Auto-detected at import time; set to False to simulate missing crispyx.

    Parameters and return values are identical to ``fit_glm``.
    """
    p = Y.shape[1]
    use_fast = (
        _USE_FAST_BACKEND
        and _CRISPYX_AVAILABLE
        and family in ('poisson', 'nb')
        and not shrinkage
        and p >= _FAST_MIN_P
    )
    if use_fast:
        try:
            result = fit_glm_fast(
                Y, X, A=A, family=family, disp_family=disp_family,
                disp_glm=disp_glm, impute=impute, offset=offset, offset_test=offset_test,
                shrinkage=shrinkage, alpha=alpha, maxiter=maxiter,
                thres_disp=thres_disp, n_jobs=n_jobs,
                random_state=random_state, verbose=verbose,
                mem_limit_gb=mem_limit_gb, **kwargs,
            )
        except ImportError:
            pass  # crispyx import failed at call time; fall through to statsmodels
        else:
            # Divergence trip-wire: crispyx clips the linear predictor, so a
            # non-finite or absurd coefficient means the design was singular
            # rather than that the gene is extreme.  Those calls go to
            # statsmodels, which regularises as a last resort.
            B = result[0]
            finite_ok = bool(np.all(np.isfinite(B)))
            max_abs = float(np.max(np.abs(B))) if finite_ok else float('inf')
            if finite_ok and max_abs <= _FAST_MAX_COEF:
                return result
            if verbose:
                pprint.pprint(
                    'Fast GLM diverged (NaN/inf), falling back to statsmodels...'
                    if not finite_ok else
                    f'Fast GLM coefficients exceed bound '
                    f'(max|B|={max_abs:.2e} > {_FAST_MAX_COEF:.0e}); '
                    f'falling back to statsmodels...'
                )
    return fit_glm(
        Y, X, A=A, family=family, disp_family=disp_family,
        disp_glm=disp_glm, impute=impute, offset=offset, offset_test=offset_test,
        shrinkage=shrinkage, alpha=alpha, maxiter=maxiter,
        thres_disp=thres_disp, n_jobs=n_jobs,
        random_state=random_state, verbose=verbose,
        mem_limit_gb=mem_limit_gb, **kwargs,
    )


def estimate_disp_auto(Y, X=None, A=None, Y_hat=None, disp_family='gaussian',
    offset=None, verbose=False, **kwargs):
    """Batch NB dispersion estimate, or None when there is no cheap one.

    Returns ``None`` when the crispyx path is unavailable (backend forced to
    ``'original'``, crispyx not installed, or fewer than ``_FAST_MIN_P``
    genes).  ``None`` means "no estimate supplied", which every caller
    already handles by letting the fitter estimate the dispersion itself --
    gene by gene in :func:`fit_glm`, or by method of moments inside crispyx.
    Returning a pooled estimate here instead cost 0.15 of correlation with the
    truth on the deconfounding benchmark (2026-09-22), because it replaced a
    per-gene estimate with a worse one.
    """
    if not (_USE_FAST_BACKEND and _CRISPYX_AVAILABLE and Y.shape[1] >= _FAST_MIN_P):
        return None
    X_disp = X if X is not None else np.ones((Y.shape[0], 1))
    offsets = _resolve_offset(Y, offset, kwargs)
    try:
        return estimate_disp_fast(Y, X_disp, A=A, offset=offsets)
    except ImportError:
        return None
