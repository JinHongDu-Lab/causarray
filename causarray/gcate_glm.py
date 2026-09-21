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

from causarray.nb_glm_fast import fit_glm_fast, estimate_disp_fast

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

_FAST_MAX_D: int = 50
"""Maximum effective design width (d_eff) for the crispyx fast path.

Increase to enable the fast path for wider designs; decrease to restrict it.
Raised from 30→50 so that GCATE's B-initialisation (d_eff = d+a+r, up to ~35
on typical datasets) uses the fast crispyx path instead of statsmodels.
"""

_FAST_MAX_COEF: float = 1e4
"""Maximum |coefficient| accepted from the crispyx fast path.

Column preconditioning eliminates the ~5× blow-up observed before v0.0.6,
so we keep only an extreme-magnitude trip-wire (default 1e4 — orders of
magnitude beyond any legitimate latent-factor coefficient).  Fits exceeding
this bound fall back to statsmodels.
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

    if offset is not None and offset is not False:
        if type(offset)==bool and offset is True:
            offsets = np.log(comp_size_factor(Y, **_filter_params(comp_size_factor, kwargs)))
        else:
            offsets = offset
    else:
        offsets = None

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
                Yhat_0[:,:] = Yhat_1[:,:] = mod.predict(X, offset=offsets).reshape(-1, a)
            
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
    if offset is not None:
        if type(offset)==bool and offset is True:
            offsets = np.log(comp_size_factor(Y, **_filter_params(comp_size_factor, kwargs)))
        else:
            offsets = offset
        sf = np.exp(offsets)[:,None]
    else:
        offsets = None
        sf = 1.

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
    """Fit GLM using crispyx's fast backend when available, falling back to statsmodels.

    Routing logic (evaluated in order):

    1. ``_USE_FAST_BACKEND is False``  → always use statsmodels.
    2. ``_CRISPYX_AVAILABLE is False`` → crispyx not installed; use statsmodels.
    3. ``family not in ('poisson', 'nb')`` → Gaussian; use statsmodels.
    4. ``p < 50`` or ``d_eff > _FAST_MAX_D`` or throughput heuristic fails → use statsmodels.
    5. crispyx path taken; if coefficients diverge → fall back to statsmodels.

    Module-level knobs
    ------------------
    ``_USE_FAST_BACKEND`` : bool
        Master on/off switch.  Use ``_backend_override()`` for scoped changes.
    ``_FAST_MAX_D`` : int
        Maximum effective design width for the fast path (default 30).
    ``_CRISPYX_AVAILABLE`` : bool
        Auto-detected at import time; set to False to simulate missing crispyx.

    Parameters and return values are identical to ``fit_glm``.
    """
    if not _USE_FAST_BACKEND:
        return fit_glm(
            Y, X, A=A, family=family, disp_family=disp_family,
            disp_glm=disp_glm, impute=impute, offset=offset, offset_test=offset_test,
            shrinkage=shrinkage, alpha=alpha, maxiter=maxiter,
            thres_disp=thres_disp, n_jobs=n_jobs,
            random_state=random_state, verbose=verbose,
            mem_limit_gb=mem_limit_gb, **kwargs,
        )

    n, p = Y.shape
    # When A is provided each perturbation is fit with a binary design of
    # width d_cov+1, so d_eff stays small regardless of a.  When A is None
    # the full X width drives crispyx's O(n*p*d²) einsum; for very wide X
    # the per-gene statsmodels path is faster.
    d_eff = X.shape[1] if A is None else X.shape[1] + 1  # effective per-model width
    use_fast = (
        _CRISPYX_AVAILABLE
        and family in ('poisson', 'nb')
        and p >= 50
        and d_eff <= _FAST_MAX_D
        and (n * p / d_eff ** 2) > 5_000  # throughput heuristic
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
            # Sanity check: crispyx IRLS should produce finite, well-bounded
            # coefficients.  Column preconditioning in _fit_glm_fast_single
            # eliminates the ill-conditioning that previously caused ~5×
            # coefficient blow-up vs statsmodels, so a tight 50/10 threshold
            # would fire spuriously on latent-factor designs where unscaled
            # coefs can legitimately be larger.  Keep only an extreme-
            # magnitude trip-wire (`max|B| > _FAST_MAX_COEF`) so that
            # pathological divergence (e.g. near-rank-deficient designs that
            # slip past preconditioning) still falls back to statsmodels.
            B = result[0]
            finite_ok = bool(np.all(np.isfinite(B)))
            max_abs = float(np.max(np.abs(B))) if finite_ok else float('inf')
            coef_ok = finite_ok and max_abs <= _FAST_MAX_COEF
            if coef_ok:
                return result
            if verbose:
                if not finite_ok:
                    pprint.pprint('Fast GLM diverged (NaN/inf), falling back to statsmodels...')
                else:
                    pprint.pprint(
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
    """Estimate NB dispersion using crispyx when available, falling back to statsmodels.

    Respects ``_USE_FAST_BACKEND`` and ``_CRISPYX_AVAILABLE`` flags.
    Parameters and return values are identical to ``estimate_disp``.
    """
    p = Y.shape[1]
    if _USE_FAST_BACKEND and _CRISPYX_AVAILABLE and p >= 50:
        # Use covariate-only design for dispersion — treatment indicators
        # don't affect gene-level overdispersion, and including many
        # treatment columns makes the batch fitter very slow.
        X_disp = X if X is not None else np.ones((Y.shape[0], 1))
        try:
            return estimate_disp_fast(Y, X_disp, offset=offset, method='moments')
        except ImportError:
            pass  # crispyx import failed at call time; fall through
    