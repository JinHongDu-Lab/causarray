import numpy as np
import contextlib
import inspect
import pandas as pd
import warnings
from typing import Literal
from causarray.DR_estimation import AIPW_mean, cross_fitting
from causarray.gcate_glm import loess_fit, ls_fit
import causarray.gcate_glm as _gcate_glm  # for _backend_override
from causarray.DR_inference import fdx_control, bh_correction
from causarray.utils import reset_random_seeds, pprint, tqdm, comp_size_factor, _filter_params


def _resolve_ps_clip(ps_clip, A, mask=None):
    """Resolve ``ps_clip`` to per-treatment ``(lower, upper)`` arrays or ``None``.

    ``'auto'`` gives ``lower_j = min(0.01, prevalence_j / 10)`` and
    ``upper_j = 1 - min(0.01, (1 - prevalence_j) / 10)``, with prevalence
    computed on the cells eligible for treatment ``j`` (its cases plus the
    shared controls, or ``mask[:, j]``).
    """
    if ps_clip is None:
        return None
    A = np.asarray(A, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
    a = A.shape[1]
    if isinstance(ps_clip, str):
        if ps_clip != 'auto':
            raise ValueError("ps_clip must be 'auto', None, or a pair 0 <= lower < upper <= 1")
        ctrl = np.sum(A, axis=1) == 0
        lower = np.empty(a); upper = np.empty(a)
        for j in range(a):
            eligible = np.asarray(mask)[:, j].astype(bool) if mask is not None else (ctrl | (A[:, j] == 1))
            prevalence = float(np.mean(A[eligible, j])) if eligible.any() else 0.5
            lower[j] = min(0.01, prevalence / 10.0)
            upper[j] = 1.0 - min(0.01, (1.0 - prevalence) / 10.0)
        return lower, upper
    if len(ps_clip) != 2 or not 0 <= ps_clip[0] < ps_clip[1] <= 1:
        raise ValueError("ps_clip must be 'auto', None, or a pair 0 <= lower < upper <= 1")
    return np.full(a, float(ps_clip[0])), np.full(a, float(ps_clip[1]))


def _add_log2fc_columns(df_res):
    """Add base-2 LFC aliases while retaining natural-log result columns."""
    log2 = np.log(2.0)
    log2fc = df_res['tau'].to_numpy() / log2
    log2fc_se = df_res['std'].to_numpy() / log2

    # Recompute existing aliases as well as missing ones.  This normalizes
    # mixed old/new frames loaded from resumable batch caches and guarantees
    # that the aliases remain exact transformations of tau and std.
    for name in ('log2fc', 'log2fc_se'):
        if name in df_res.columns:
            df_res.pop(name)
    insert_at = df_res.columns.get_loc('std') + 1
    df_res.insert(insert_at, 'log2fc', log2fc)
    df_res.insert(insert_at + 1, 'log2fc_se', log2fc_se)
    return df_res



def compute_causal_estimand(
    estimand,
    Y, W, A, W_A=None, family='nb', offset=False,
    Y_hat=None, pi_hat=None, mask=None,
    fdx=False, fdx_B=1000, fdx_alpha=0.05, fdx_c=0.1,
    verbose=False, random_state=0, backend: str = "auto", K=1,
    ps_clip='auto', ps_class_weight=None,
    **kwargs):
    """Estimate causal treatment effects using AIPW with a user-supplied estimand.

    Parameters
    ----------
    estimand : callable
        Function that maps influence function values ``(etas, A)`` to
        ``(eta_est, tau_est, var_est[, df_eff])``.  See :func:`LFC` for an
        example implementation.
    Y : array, shape (n, p)
        Count matrix of outcomes.
    W : array, shape (n, d)
        Covariate matrix (including latent factors from GCATE).
    A : array, shape (n, a)
        Binary treatment indicator matrix.
    W_A : array or None, shape (n, d_A)
        Covariate matrix for the propensity model.  If ``None``, ``W`` is used.
    family : str
        GLM family for the outcome model: ``'nb'`` (default) or ``'poisson'``.
    offset : bool or array-like
        Log-scale offset for the outcome model.  ``True`` computes size factors
        automatically; ``False`` or ``None`` disables the offset.
    Y_hat : array or None, shape (n, p, a, 2)
        Pre-computed counterfactual predictions.  When provided, cross-fitting
        is skipped.
    pi_hat : array or None, shape (n, a)
        Pre-computed propensity scores.  When provided, propensity fitting is
        skipped.
    K : int
        Number of folds used for nuisance estimation.  The default ``1``
        preserves in-sample fitting.
    ps_clip : {'auto'}, tuple(float, float) or None
        Bounds applied to propensity scores used by AIPW. ``'auto'`` (default)
        uses a prevalence-aware bound per treatment,
        ``lower_j = min(0.01, prevalence_j / 10)`` and
        ``upper_j = 1 - min(0.01, (1 - prevalence_j) / 10)``, so that calibrated
        scores of a rare treatment are not clipped wholesale. A tuple applies
        one fixed bound to every treatment; ``None`` disables clipping.

        .. versionchanged:: 0.0.10
            Default changed from ``(0.01, 0.99)`` to ``'auto'``.
    ps_class_weight : str, dict or None
        Class weighting for the propensity model. ``None`` (default) fits
        calibrated treatment probabilities, which is what the AIPW weights
        ``A / pi`` require. ``'balanced'`` is the pre-0.0.10 default and is kept
        as a legacy option; it centres scores near 0.5 regardless of prevalence
        and turns the estimator into an outcome-model plug-in.

        .. versionchanged:: 0.0.10
            Default changed from ``'balanced'`` to ``None``.
    mask : array or None, shape (n, a)
        Boolean mask indicating eligible cells for each treatment. It limits
        propensity-model fitting and final estimand computation.
    fdx : bool
        Whether to apply FDX control (``P(FDP > fdx_c) < fdx_alpha``).
    fdx_B : int
        Number of bootstrap samples for FDX control.
    fdx_alpha : float
        Significance level for FDX control.
    fdx_c : float
        FDP threshold for FDX control.
    backend : str
        GLM backend: ``"auto"`` (default), ``"fast"`` (force crispyx),
        or ``"original"`` (force statsmodels).
    verbose : bool
        Print progress information.
    **kwargs
        Additional arguments forwarded to the GLM fitting functions.

    Returns
    -------
    df_res : DataFrame
        Test results produced by ``estimand``. An estimand may optionally return
        a fifth dictionary whose arrays are added as diagnostic columns. The
        frame also carries per-pair support columns ``n_treated``,
        ``n_control``, ``count_treated`` and ``count_control`` (cells and raw
        summed counts in each arm), added in 0.0.10 so that arms with no
        observed counts can be audited without recomputation.
    """
    reset_random_seeds(random_state)

    if 'clip_pseudo_outcomes' in kwargs:
        raise TypeError(
            'clip_pseudo_outcomes has been removed; AIPW pseudo-outcomes are '
            'always left unclipped'
        )

    ctx = _gcate_glm._backend_override(backend) if backend != "auto" else contextlib.nullcontext()

    # check the input data
    if isinstance(Y, pd.DataFrame):
        gene_names = Y.columns
        Y = Y.values
    else:
        gene_names = range(Y.shape[1])
    # When Y_hat (n×p×a×2 float64) would exceed mem_limit_gb, downcast to
    # float32 to halve peak memory.  Opt-in only: pass ``mem_limit_gb=<GB>``.
    _a_shape = A.shape[1] if hasattr(A, 'shape') and len(A.shape) > 1 else 1
    _yhat_gb = Y.shape[0] * Y.shape[1] * _a_shape * 2 * 8 / 1e9
    _mem_limit = kwargs.get('mem_limit_gb', None)
    _use_f32 = _mem_limit is not None and _yhat_gb > _mem_limit
    if _use_f32:
        warnings.warn(
            f"AIPW intermediate Y ({_yhat_gb:.1f} GB as float64) exceeds "
            f"mem_limit_gb={_mem_limit} GB; downcasting Y, A, and pi_hat to "
            f"float32 to halve peak memory.",
            ResourceWarning, stacklevel=2,
        )
    Y = Y.astype(np.float32 if _use_f32 else float)
    n, p = Y.shape
    if not np.all(np.isfinite(Y)):
        raise ValueError('Y must contain only finite values')

    if len(A.shape) == 1:
        A = A.reshape(-1,1)
    if isinstance(A, pd.DataFrame):
        trt_names = A.columns
        A = A.values
    else:
        trt_names = range(A.shape[1])

    if isinstance(W, pd.DataFrame):
        cov_names = W.columns
        W = W.values
    if W_A is None:
        W_A = W
    elif isinstance(W_A, pd.DataFrame):
        W_A = W_A.values

    if mask is not None:
        mask = np.array(mask).astype(bool)
        if len(mask.shape) == 1: mask = mask.reshape(-1,1)
        if mask.shape != A.shape:
            raise ValueError('Mask must have the same shape as the treatment matrix')

    kwargs = {k:v for k,v in kwargs.items() if k not in 
        ['kwargs_ls_1', 'kwargs_ls_2', 'kwargs_es_1', 'kwargs_es_2', 'c1', 'num_d']
    }

    if verbose:
        d_A = W_A.shape[1]
        pprint.pprint('Estimating LFC...')
        pprint.pprint({'estimands':'LFC','n':n,'p':p,'d':W.shape[1], 'd_A':d_A, 'a':A.shape[1]}, compact=True)

    if offset is not None and offset is not False:
        if type(offset)==bool and offset is True:
            size_factors = comp_size_factor(Y, **_filter_params(comp_size_factor, kwargs))
            offset = np.log(size_factors)
        else:
            size_factors = np.exp(offset)
    else:
        offset = None
        size_factors = np.ones(n)
    size_factors = np.asarray(size_factors, dtype=float)
    if size_factors.shape != (n,) or not np.all(np.isfinite(size_factors)) or np.any(size_factors <= 0):
        raise ValueError('Size factors must be finite, positive, and have length n')
    
    ps_clip_bounds = _resolve_ps_clip(ps_clip, A, mask)
    with ctx:
        Y_hat, pi_hat, pi_hat_raw = cross_fitting(
            Y, A, W, W_A, family=family, K=K, offset=offset,
            Y_hat=Y_hat, pi_hat=pi_hat, mask=mask, ps_clip=ps_clip_bounds,
            ps_class_weight=ps_class_weight,
            return_raw_pi=True, random_state=random_state, verbose=verbose,
            **kwargs,
        )
    pi_hat = pi_hat.reshape(*A.shape)
    if not np.all(np.isfinite(Y_hat)):
        raise ValueError('Outcome-model predictions must contain only finite values')
    if not np.all(np.isfinite(pi_hat)) or np.any((pi_hat <= 0) | (pi_hat >= 1)):
        raise ValueError(
            'Propensity scores used by AIPW must be finite and strictly between 0 and 1; '
            'pass ps_clip bounds or provide valid pi_hat values')

    if verbose: pprint.pprint('Estimating AIPW mean...')
    _aipw_dtype = Y_hat.dtype if _use_f32 else None
    A_aipw     = A.astype(_aipw_dtype) if _aipw_dtype is not None else A
    pi_hat_aipw = pi_hat.astype(_aipw_dtype) if _aipw_dtype is not None else pi_hat

    # point estimation of the treatment effect
    _, etas = AIPW_mean(
        Y,
        np.stack([1-A_aipw, A_aipw], axis=-1),
        Y_hat,
        np.stack([1-pi_hat_aipw, pi_hat_aipw], axis=-1),
    )

    # normalize the influence function values
    etas /= size_factors[:,None,None,None]

    # Preserve the public two-argument callback contract. Private inference
    # metadata is only supplied when explicitly accepted or via **kwargs.
    callback_params = inspect.signature(estimand).parameters
    accepts_metadata = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in callback_params.values())
    res = []
    _count_control_shared = None
    _small_arm_floored = {}
    iters = range(A.shape[1]) if A.shape[1]==1 else tqdm(range(A.shape[1]))
    for j in iters:
        if mask is not None:
            i_cells = mask[:, j]
        else:
            i_ctrl = (np.sum(A, axis=1) == 0.)
            i_case = (A[:,j] == 1.)
            i_cells = i_ctrl | i_case
        # Raw support per arm: cells and summed counts.  Treated rows are few;
        # the shared control block is summed once when no mask is given.
        idx_treated = np.flatnonzero(i_cells & (A[:, j] == 1.))
        idx_control = np.flatnonzero(i_cells & (A[:, j] == 0.))
        count_treated = Y[idx_treated].sum(axis=0, dtype=np.float64)
        if mask is None:
            if _count_control_shared is None:
                _count_control_shared = Y[idx_control].sum(axis=0, dtype=np.float64)
            count_control = _count_control_shared
        else:
            count_control = Y[idx_control].sum(axis=0, dtype=np.float64)
        metadata = dict(
            _n_params=W.shape[1] + 1, _in_sample=(K == 1),
            _obs_mean_treated=count_treated / max(idx_treated.size, 1),
            _obs_mean_control=count_control / max(idx_control.size, 1),
            _size_factors=size_factors[i_cells],
        )
        if not accepts_metadata:
            metadata = {key: value for key, value in metadata.items()
                        if key in callback_params and callback_params[key].kind in
                        (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                         inspect.Parameter.KEYWORD_ONLY)}
        metadata.update(kwargs)
        _ret = estimand(etas[i_cells,:,j], A[i_cells,j], **metadata)
        eta_est, tau_est, var_est = _ret[:3]
        df_est = _ret[3] if len(_ret) > 3 else None
        estimand_info = _ret[4] if len(_ret) > 4 else None

        std_est = np.sqrt(var_est)
        tvalues_init = tau_est / std_est

        # Multiple testing procedure
        V = fdx_control(tau_est, tvalues_init, eta_est, std_est, fdx, fdx_B, fdx_alpha, fdx_c)

        # BH correction
        tvalues_init[np.isinf(std_est)] = np.nan
        pvals, qvals, pvals_adj, qvals_adj = bh_correction(tvalues_init, df=df_est)
        
        df_res = pd.DataFrame({
            'gene_names': gene_names,            
            'tau': tau_est,
            'std': std_est,
            'stat': tvalues_init,
            'rej': V,
            'pvalue': pvals,
            'padj': qvals,
            'pvalue_emp_null_adj': pvals_adj,
            'padj_emp_null_adj': qvals_adj,            
            })
        df_res['n_treated'] = int(idx_treated.size)
        df_res['n_control'] = int(idx_control.size)
        df_res['count_treated'] = count_treated
        df_res['count_control'] = count_control
        if estimand_info is not None:
            for name, values in estimand_info.items():
                df_res[name] = values
            if 'var_floored' in estimand_info and min(idx_treated.size, idx_control.size) < 200:
                n_floored = int(np.sum(np.asarray(estimand_info['var_floored'], dtype=bool)))
                if n_floored:
                    _small_arm_floored[trt_names[j] if A.shape[1] > 1 else 0] = n_floored
            n_nonestimable = int(np.sum(~np.asarray(estimand_info['estimable'], dtype=bool)))
            if n_nonestimable:
                treatment = trt_names[j] if A.shape[1] > 1 else 0
                warnings.warn(
                    f'{n_nonestimable} gene(s) for treatment {treatment!r} have '
                    'nonfinite or nonpositive AIPW counterfactual means and are '
                    'reported as non-estimable.',
                    RuntimeWarning, stacklevel=2,
                )
        if A.shape[1]>1:
            df_res['trt'] = trt_names[j]
        res.append(df_res)
    df_res = pd.concat(res, axis=0).reset_index(drop=True)
    if _small_arm_floored:
        n_pairs = int(sum(_small_arm_floored.values()))
        warnings.warn(
            f'{len(_small_arm_floored)} treatment(s) have fewer than 200 cells in one '
            f'arm and the model-based variance floor bound for {n_pairs} gene-treatment '
            'pair(s) (column var_floored). Such pairs typically have zero or near-zero '
            'counts in the small arm; inspect count_treated / count_control before '
            'interpreting their effects.',
            RuntimeWarning, stacklevel=2,
        )
    estimation = {**{
        'pi_hat': pi_hat,
        'pi_hat_raw': pi_hat_raw,
        'ps_clip_bounds': ps_clip_bounds,
        'Y_hat': Y_hat,
        'offset': offset,
        'size_factors': size_factors,
        'ps_class_weight': ps_class_weight,
    }, **kwargs}
    return df_res, estimation


def LFC(
    Y, W, A, W_A=None, family='nb', offset=False,
    Y_hat=None, pi_hat=None, cross_est=False, K=None, mask=None,
    usevar: Literal['pooled'] = 'pooled',
    thres_min='auto', thres_diff=1e-2, eps_var=None, min_counts=5.0,
    fdx=False, fdx_alpha=0.05, fdx_c=0.1,
    verbose=False, backend: str = "auto", ps_clip='auto',
    ps_class_weight=None, **kwargs):
    """Estimate log-fold changes of treatment effects (LFCs) using AIPW.

    Fits a doubly-robust AIPW estimator for the log-ratio of counterfactual
    means E[Y(1)] / E[Y(0)].  Call this after :func:`fit_gcate` to incorporate
    estimated latent factors into the covariate matrix ``W``.

    Parameters
    ----------
    Y : array, shape (n, p)
        Count matrix of outcomes.
    W : array, shape (n, d)
        Covariate matrix, typically ``[X | U]`` where ``U`` are the latent
        factors from GCATE.
    A : array, shape (n, a)
        Binary treatment indicator matrix.
    W_A : array or None, shape (n, d_A)
        Covariate matrix for the propensity model.  If ``None``, ``W`` is used.
    family : str
        GLM family for the outcome model: ``'nb'`` (default) or ``'poisson'``.
    offset : bool or array-like
        Log-scale offset for the outcome model.  ``True`` computes size factors
        automatically; ``False`` or ``None`` disables the offset.
    Y_hat : array or None, shape (n, p, a, 2)
        Pre-computed counterfactual predictions.  When provided, cross-fitting
        is skipped.
    pi_hat : array or None, shape (n, a)
        Pre-computed propensity scores.  When provided, propensity fitting is
        skipped.
    cross_est : bool
        Whether to use two-fold cross-estimation for nuisance parameters.
        An explicit ``K`` takes precedence.
    K : int or None
        Number of nuisance-estimation folds.  ``None`` uses 2 when
        ``cross_est=True`` and 1 otherwise.
    mask : array or None, shape (n, a)
        Boolean mask indicating eligible cells for each treatment. It limits
        propensity-model fitting and final estimand computation.
    usevar : str
        Variance estimator for the AIPW pseudo-outcomes. Only ``'pooled'``
        remains: the influence-function (sandwich) variance ``var(eta) / n``
        of the estimator, where ``eta`` are the per-cell influence values of
        the log-ratio and ``n`` counts every cell entering the estimand. With
        calibrated propensity scores this equals the efficient two-sample form
        ``Var(Y|A=1)/n₁ + Var(Y|A=0)/n₀`` up to the outcome-model correction,
        and it matches the estimator's actual sampling variability in oracle
        simulations. For in-sample nuisance fits the variance is rescaled by
        ``n/(n-d)`` and p-values use a t reference with ``n-d`` degrees of
        freedom (see Notes).

        ``'unequal'`` (the 0.0.6-0.0.9 default) applied a two-sample Welch
        formula ``s₀²/n₀ + s₁²/n₁`` by arm. That is not the variance of an
        estimator that averages pseudo-outcomes over all cells: for equal arm
        sizes it is exactly twice the correct standard error, and for a rare
        treatment fitted with class-balanced propensity scores it is an order
        of magnitude too large. It was removed in 0.0.10 after validation on
        the Perturb-seq, SEA-AD and Adamson tutorials; the argument is
        accepted as an alias of ``'pooled'`` with a ``FutureWarning`` for one
        release.

        Neither estimator models within-donor or within-subject correlation.
        Repeated cells from the same biological unit should still be
        pseudo-bulked or analysed with a cluster-aware method.

        .. versionchanged:: 0.0.10
            ``'unequal'`` removed; ``'pooled'`` is the only estimator.
        .. versionchanged:: 0.0.6
            Default changed from ``'pooled'`` to ``'unequal'``.
    thres_min : {'auto'} or float
        Genes whose larger counterfactual arm mean (size-factor-normalised
        counts per cell) is below this threshold are excluded (reported as
        ``tau=0``, ``padj=NaN``). ``'auto'`` (default) sets the threshold per
        treatment to ``min_counts / min(n_0, n_1)``, i.e. it requires about
        ``min_counts`` expected counts in the smaller arm: 0.05 counts per cell
        for a 100-cell arm, 0.007 for a 700-cell arm. Below that the
        log-scale statistic of a sparse gene is positively skewed under the
        null. The test is applied to both the counterfactual arm means and
        the observed per-cell arm means (raw counts), so an inflated model
        prediction for an all-zero arm cannot pass it. A float applies one
        fixed threshold.

        .. versionchanged:: 0.0.10
            Default changed from the fixed ``0.01`` to ``'auto'``.
    min_counts : float
        Expected-count requirement used by ``thres_min='auto'``.

        .. versionadded:: 0.0.10
    thres_diff : float
        Floor applied to each arm mean before the logarithm, and the minimum
        absolute difference between arm means for a gene to be tested.
    eps_var : None
        Deprecated and ignored. A model-based variance floor (see Notes)
        replaces the additive constant.

        .. deprecated:: 0.0.10
    fdx : bool
        Whether to apply FDX control (``P(FDP > fdx_c) < fdx_alpha``).
    fdx_alpha : float
        Significance level for FDX control.
    fdx_c : float
        FDP threshold for FDX control.
    verbose : bool
        Print progress information.
    backend : str
        GLM backend: ``"auto"`` (default), ``"fast"`` (force crispyx),
        or ``"original"`` (force statsmodels).
    ps_clip : {'auto'}, tuple(float, float) or None
        Bounds applied to propensity scores used by AIPW. ``'auto'`` (default)
        is prevalence-aware per treatment: ``lower_j = min(0.01,
        prevalence_j / 10)`` and symmetrically for the upper bound. Raw,
        unclipped scores remain available as ``estimation['pi_hat_raw']`` and
        the resolved bounds as ``estimation['ps_clip_bounds']``.

        .. versionchanged:: 0.0.10
            Default changed from ``(0.01, 0.99)``, which clipped every
            calibrated score of a treatment with prevalence below 1%.
    ps_class_weight : str, dict or None
        Class weighting for the propensity model. ``None`` (default) gives
        calibrated probabilities, which the AIPW weights ``A / pi`` require.
        ``'balanced'`` (pre-0.0.10 default) centres scores near 0.5 whatever
        the prevalence, shrinking the AIPW correction term by roughly twice
        the prevalence and turning the estimator into an outcome-model
        plug-in whose uncertainty the influence function no longer reflects.

        .. versionchanged:: 0.0.10
            Default changed from ``'balanced'`` to ``None``.
    **kwargs
        Additional arguments forwarded to the GLM fitting functions.

    Returns
    -------
    df_res : DataFrame
        Test results with natural-log effect estimate ``tau`` and standard
        error ``std``, together with their base-2 equivalents ``log2fc`` and
        ``log2fc_se``. The fold change is treatment relative to control, and
        ``log2fc_se`` is a standard error (not a sample standard deviation).
        The result also contains inference columns, raw ``mean_control`` and
        ``mean_treated`` counterfactual means, an ``estimable`` flag, a
        ``var_floored`` flag (True where the model-based variance floor
        bound) with the pre-floor standard error ``std_raw``, per-arm support ``n_treated``, ``n_control``,
        ``count_treated``, ``count_control`` (plus ``trt`` for multiple
        treatments).

        .. versionadded:: 0.0.10
            ``var_floored`` and the four support columns.
        .. versionadded:: 0.0.8
            Added the ``log2fc`` and ``log2fc_se`` convenience columns. The
            original natural-log ``tau`` and ``std`` columns remain unchanged.

    Notes
    -----
    **Model-based variance floor (0.0.10).** The empirical influence-function
    variance of an arm whose cells all have zero counts is zero, and the
    logarithm of its floored mean ``max(mean, thres_diff)`` is then reported
    with a spuriously tiny standard error. A mean estimated from ``n_k`` cells
    cannot be more precise than Poisson sampling allows, so the variance of
    the log-ratio uses the working Poisson floor
    ``mean(1/s₁)/(n₁ τ₁) + mean(1/s₀)/(n₀ τ₀)``, where ``s_k`` are
    the size factors in arm ``k`` (one without offsets), and
    ``var_est = max(var_est, floor)`` is used. With 100 perturbed cells and a
    floored mean of 0.01 this gives a standard error of at least 1, so a
    chance all-zero arm of a sparse gene is not called, while a genuine
    complete knockout of a gene with control mean 2 (``tau ≈ -5.3``) remains
    significant. An arm with no observed counts is kept estimable at the
    floor ``thres_diff`` (its AIPW mean is exactly zero), so complete
    knockouts of expressed genes are reported rather than dropped as
    non-estimable. The SCARF tutorial documents the failure this prevents.

    **Expression threshold (0.0.10).** ``thres_min='auto'`` requires about
    ``min_counts`` (5) expected counts in the smaller arm, i.e. a larger-arm
    mean of at least ``5 / min(n_0, n_1)`` counts per cell. Below that level a
    handful of counts in the small arm yields large positively skewed
    statistics under the null (SCARF and Adamson negative controls); above it
    the variance floor suffices.

    **Small-sample correction (0.0.10).** With in-sample nuisance fits
    (``K=1``) the pooled variance is multiplied by ``n / (n - d)``, where ``d``
    is the number of outcome-model parameters (``W.shape[1] + 1``), and
    p-values use a t reference with ``n - d`` degrees of freedom. Both are
    no-ops for large ``n``; on 85-donor pseudo-bulk data and 100-cell
    perturbation arms they bring the null t-statistics from SD ≈ 1.08-1.10 to
    ≈ 1.0 (label-permutation and fake-perturbation nulls on the SEA-AD and
    Perturb-seq tutorials).
    """
    if eps_var is not None:
        warnings.warn(
            'eps_var is deprecated and ignored since 0.0.10; a model-based '
            'variance floor replaces the additive constant.',
            FutureWarning, stacklevel=2,
        )
    if usevar == 'unequal':
        warnings.warn(
            "usevar='unequal' (the 0.0.6-0.0.9 Welch-by-arm formula) was removed in "
            "0.0.10 after validation on the Perturb-seq, SEA-AD and Adamson tutorials: "
            "it is not the variance of the AIPW estimator (2x the correct SE for equal "
            "arms, far larger for rare treatments). The argument is accepted as an alias "
            "of 'pooled' for one release and will then raise.",
            FutureWarning, stacklevel=2,
        )
        usevar = 'pooled'
    elif usevar != 'pooled':
        raise ValueError("usevar must be 'pooled'")

    def estimand(etas, A, **kwargs):
        eta_0, eta_1 = etas[..., 0], etas[..., 1]
        mean_0 = np.mean(eta_0, axis=0, dtype=np.float64)
        mean_1 = np.mean(eta_1, axis=0, dtype=np.float64)
        finite_means = np.isfinite(mean_0) & np.isfinite(mean_1)
        # An arm whose observed counts are all zero has an AIPW mean of
        # exactly zero (or numerically negative) although the gene may be a
        # genuine complete knockout.  Since 0.0.10 such arms stay estimable at
        # the floor ``thres_diff`` and inherit the model-based variance floor;
        # the observed-support threshold below removes the cases where the
        # other arm is too sparse to support the comparison.  Nonpositive
        # means with nonzero observed counts remain non-estimable.
        _obs1 = kwargs.get('_obs_mean_treated'); _obs0 = kwargs.get('_obs_mean_control')
        zero_arm_1 = np.asarray(_obs1) == 0 if _obs1 is not None else np.zeros_like(mean_1, dtype=bool)
        zero_arm_0 = np.asarray(_obs0) == 0 if _obs0 is not None else np.zeros_like(mean_0, dtype=bool)
        estimable = finite_means & ((mean_0 > 0) | zero_arm_0) & ((mean_1 > 0) | zero_arm_1)

        # Apply the count-mean parameter-space constraint only after averaging
        # the unmodified AIPW pseudo-outcomes.  The floor makes the logarithm
        # and its delta-method denominator numerically safe.  Nonpositive raw
        # means remain non-estimable and are filtered below, so this safeguard
        # cannot turn an invalid aggregate into a discovery.
        tau_0 = np.where(estimable, np.maximum(mean_0, thres_diff), 1.0)
        tau_1 = np.where(estimable, np.maximum(mean_1, thres_diff), 1.0)
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            tau_est = np.log(tau_1/tau_0)
            eta_est = eta_1 / tau_1[None,:] - eta_0 / tau_0[None,:]

        df_eff = None
        n_0 = int(np.sum(A==0))
        n_1 = int(np.sum(A==1))
        n_cells = eta_est.shape[0]
        n_params = int(kwargs.get('_n_params', 1))
        in_sample = bool(kwargs.get('_in_sample', True))
        df_resid = max(n_cells - n_params, 2)
        var_est = np.var(eta_est, axis=0, ddof=1) / n_cells
        if in_sample:
            # Residuals of an outcome model fitted on the same cells are
            # deflated by ~(n - d)/n; rescale (HC1-style) so that small
            # designs (donor-level pseudo-bulk, ~100-cell arms) are not
            # anti-conservative.  Cross-fitted nuisances (K > 1) need no
            # rescaling.
            var_est = var_est * (n_cells / df_resid)
        # t reference with residual degrees of freedom; equals the normal
        # reference for large n.
        df_eff = np.full(var_est.shape, float(df_resid))

        # Under the working Poisson model Y_i ~ Poisson(s_i * tau_k),
        # Var(Y_i / s_i) = tau_k / s_i. The unweighted arm mean thus has
        # log-scale variance sum(1 / s_i) / (n_k**2 * tau_k).
        sf = kwargs.get('_size_factors', np.ones(n_cells))
        inv_sf_1 = np.sum(1.0 / sf[A == 1]) / max(n_1, 1)**2
        inv_sf_0 = np.sum(1.0 / sf[A == 0]) / max(n_0, 1)**2
        with np.errstate(invalid='ignore', divide='ignore'):
            std_raw = np.sqrt(var_est)
            var_floor = inv_sf_1 / tau_1 + inv_sf_0 / tau_0
            var_floored = np.asarray(var_est < var_floor) & estimable
            var_est = np.where(var_floored, var_floor, var_est)

        # Filter on the raw aggregate estimates, not on cell-level projections
        # or the numerical floor used for the log transform.
        if isinstance(thres_min, str):
            if thres_min != 'auto':
                raise ValueError("thres_min must be 'auto' or a non-negative float")
            thres_min_j = float(min_counts) / max(min(n_0, n_1), 1)
        else:
            thres_min_j = float(thres_min)
        # The support test uses *observed* arm means (raw counts per cell) as
        # well as the counterfactual means: for a gene with zero counts in a
        # small arm the outcome model's prediction for that arm is unreliable
        # and can be inflated, which would otherwise let the pair through.
        obs_mean_1 = kwargs.get('_obs_mean_treated')
        obs_mean_0 = kwargs.get('_obs_mean_control')
        low_support = np.maximum(mean_0, mean_1) < thres_min_j
        if obs_mean_1 is not None and obs_mean_0 is not None:
            low_support |= np.maximum(np.asarray(obs_mean_0), np.asarray(obs_mean_1)) < thres_min_j
        idx = (
            ~estimable |
            low_support |
            (np.abs(mean_1 - mean_0) < thres_diff)
        )
        tau_est[idx] = 0.; eta_est[:,idx] = 0.; var_est[idx] = np.inf
        if df_eff is not None:
            df_eff[idx] = np.nan

        # Count genes whose Welch df is NaN for reasons OTHER than the
        # low-expression filter (which is intentional) — typically caused by
        # ``var = NaN`` from very small per-arm counts.  These rows are
        # silently dropped by ``bh_correction``'s ``~np.isnan(t)`` mask, so
        # surface a single warning if any survive the filter.
        if df_eff is not None:
            silent_nan = int(np.sum(np.isnan(df_eff) & ~idx))
            if silent_nan > 0:
                import warnings
                warnings.warn(
                    f"{silent_nan} gene(s) produced NaN Welch df despite "
                    f"passing the low-expression filter; these will be "
                    f"reported as NaN p-values in the result DataFrame.",
                    RuntimeWarning, stacklevel=3,
                )

        var_floored = var_floored & ~idx
        std_raw = np.where(idx, np.inf, std_raw)
        info = {
            'mean_control': mean_0,
            'mean_treated': mean_1,
            'estimable': estimable,
            'var_floored': var_floored,
            'std_raw': std_raw,
        }
        return eta_est, tau_est, var_est, df_eff, info

    if K is None:
        K = 2 if cross_est else 1

    df_res, estimation = compute_causal_estimand(
        estimand, Y, W, A, W_A, family, offset,    
        Y_hat=Y_hat, pi_hat=pi_hat, mask=mask,
        fdx=fdx, fdx_alpha=fdx_alpha, fdx_c=fdx_c,
        verbose=verbose, backend=backend, K=K, ps_clip=ps_clip,
        ps_class_weight=ps_class_weight, **kwargs)
    return _add_log2fc_columns(df_res), estimation





def VIM(eta_est, X, id_covs, **kwargs):
    """Estimate variable importance measures (VIM) for heterogeneous treatment effects.

    Decomposes treatment effect variance into components explained by each
    covariate using conditional average treatment effect (CATE) regression.

    Parameters
    ----------
    eta_est : array, shape (n, p)
        Influence function values from :func:`LFC` or
        :func:`compute_causal_estimand`.
    X : array, shape (n, d)
        Covariate matrix.
    id_covs : int or array-like of int
        Column indices of ``X`` to compute VIM for.  An integer ``k`` is
        treated as ``range(k)``.

    Returns
    -------
    estimation : dict
        Dictionary with keys:

        ``'CATE'``, ``'CATE_lower'``, ``'CATE_upper'`` : array, shape (n_covs, n, p)
            Conditional average treatment effect and pointwise confidence band.
        ``'VTE'`` : array, shape (p,)
            Total variance of the treatment effect (marginal).
        ``'CVTE'`` : array, shape (n_covs, p)
            Conditional variance of the treatment effect given each covariate.
        ``'VIM_mean'`` : array, shape (n_covs, p)
            VIM point estimate ``CVTE / VTE - 1`` for each covariate and gene.
        ``'VIM_sd'`` : array, shape (n_covs, p)
            Standard deviation of the VIM estimate.
    """
    if len(X.shape)==1:
        X = X[:,None]

    n, p = eta_est.shape
    d = X.shape[1]
    if id_covs is None:
        id_covs = range(d)
    if np.isscalar(id_covs):
        id_covs = range(id_covs)

    n_covs = len(id_covs)

    emp_VTE = (eta_est - np.mean(eta_est, axis=0, keepdims=True))**2
    VTE = np.mean(emp_VTE, axis=0)
    VIM_mean = np.zeros((n_covs, p))
    VIM_sd = np.zeros((n_covs, p))
    emp_CVTE = np.zeros((n_covs, n, p))
    CVTE = np.zeros((n_covs, p))
    CATE = np.zeros((n_covs, n, p))
    CATE_lower = np.zeros((n_covs, n, p))
    CATE_upper = np.zeros((n_covs, n, p))

    for j,i in enumerate(id_covs):
        print(j,i)
        # regression eta_est on X to get predicted values
        if np.all(np.modf(X[:,i:i+1])[0] == 0):
            CATE[j], CATE_lower[j], CATE_upper[i] = ls_fit(eta_est, X[:,i], **kwargs)
        else:
            CATE[j], CATE_lower[j], CATE_upper[j] = loess_fit(eta_est, X[:,i], **kwargs)
        # compute the variance of treatment effect        
        _emp_CVTE = (eta_est - CATE[j])**2
        _CVTE = np.nanmean(_emp_CVTE, axis=0)
        emp_CVTE[j] = _emp_CVTE
        CVTE[j] = _CVTE

        VIM_mean[j] = _CVTE / VTE - 1
        VIM_sd[j] = np.nanstd((emp_VTE - _emp_CVTE), axis=0, ddof=1)/VTE

    estimation = {
        'CATE': CATE,
        'CATE_lower': CATE_lower,
        'CATE_upper': CATE_upper,
        'emp_VTE': emp_VTE,
        'VTE': VTE,
        'emp_CVTE' : emp_CVTE,
        'CVTE' :CVTE,
        'VIM_mean' : VIM_mean,
        'VIM_sd' : VIM_sd
    }
    return estimation


def _nuisance_path_for(cache_path):
    """Sibling file holding the nuisance store for ``cache_path``."""
    if cache_path is None:
        raise ValueError('save_nuisances=True requires cache_path')
    base = str(cache_path)
    stem = base[:-3] if base.endswith('.h5') else base
    return f'{stem}.nuisances.h5'


def _save_batch_nuisances(path, batch_i, estimation, cell_idx, pert_names,
                          offset, U, gene_names):
    """Persist one batch's outcome-model predictions for later re-estimation.

    The outcome model is independent of the propensity design, so storing
    ``Y_hat`` (with the batch's cells, latent factors and offset) is enough to
    re-run :func:`LFC` under a different propensity specification without
    refitting it.  ``Y_hat`` dominates the file: it is one float32 per cell and
    gene in the batch, so budget roughly ``4 * n_cells * n_genes`` bytes per
    batch before compression.

    Parameters
    ----------
    path : str
        HDF5 file to append to.
    batch_i : int
        Index of the batch, used as the group name.
    estimation : dict
        Second return value of :func:`LFC`, providing ``Y_hat`` and ``pi_hat``.
    cell_idx : array
        Row indices of this batch's cells in the full matrix.
    pert_names : sequence
        Perturbation columns fitted in this batch.
    offset : array
        Log size factors used by the batch fit.
    U : array
        Latent factors estimated for the batch.
    gene_names : sequence or None
        Column labels for ``Y_hat``.
    """
    import h5py

    with h5py.File(path, 'a') as handle:
        group_name = f'batch_{batch_i:04d}'
        if group_name in handle:
            del handle[group_name]
        group = handle.create_group(group_name)
        group.create_dataset(
            'Y_hat', data=np.asarray(estimation['Y_hat'], dtype=np.float32),
            compression='gzip', compression_opts=1,
        )
        group.create_dataset(
            'pi_hat', data=np.asarray(estimation['pi_hat'], dtype=np.float32),
            compression='gzip', compression_opts=1,
        )
        group.create_dataset('cell_idx', data=np.asarray(cell_idx, dtype=np.int64))
        group.create_dataset('offset', data=np.asarray(offset, dtype=np.float64))
        group.create_dataset('U', data=np.asarray(U, dtype=np.float64))
        dt = h5py.special_dtype(vlen=str)
        group.create_dataset('pert_names',
                             data=np.asarray([str(n) for n in pert_names], dtype=object),
                             dtype=dt)
        if gene_names is not None:
            group.create_dataset('gene_names',
                                 data=np.asarray([str(g) for g in gene_names], dtype=object),
                                 dtype=dt)


def gcate_lfc_batch(
    Y, X, A, r,
    W_A=None,
    batch_size=10,
    n_batches=None,
    max_cells=2000,
    n_ctrl=2000,
    family='nb',
    offset=True,
    warm_start_U=False,
    cache_path=None,
    save_nuisances=False,
    random_state=0,
    verbose=False,
    gcate_kwargs=None,
    lfc_kwargs=None,
    **kwargs,
):
    """Batch-wise GCATE + doubly-robust LFC estimation.

    Partitions perturbations into chunks of ``batch_size``, runs
    :func:`fit_gcate_batch` to estimate per-batch latent confounders, then
    calls :func:`LFC` on each batch independently.  All large intermediate
    arrays (``res_1``, ``res_2``, ``Y_hat``, ``pi_hat``) are freed immediately
    after each batch so that peak memory is bounded by one batch's worth of
    data regardless of the total number of perturbations.

    Pass ``save_nuisances=True`` (which requires ``cache_path``) to keep each
    batch's outcome-model predictions before they are freed.  They are written
    beside the result cache as ``<cache_path stem>.nuisances.h5``, kept in a
    separate file because they are orders of magnitude larger and must not
    disturb the ``/batch_*`` keys that drive resumption.  Because the outcome
    model does not depend on the propensity design, those predictions can be
    fed back to :func:`LFC` as ``Y_hat`` to re-estimate under different
    propensity scores without refitting the outcome model, the expensive stage.
    ``Y_hat`` holds counterfactual predictions with shape
    ``(n_cells, n_genes, n_treatments, 2)``, so budget roughly
    ``8 * n_cells * n_genes * n_treatments`` bytes per batch before
    compression -- a few GB for a typical screen batch.

    Results can optionally be cached to an HDF5 file (``cache_path``) so that
    interrupted runs can be resumed without re-processing completed batches.

    Parameters
    ----------
    Y : array-like or DataFrame, shape (n, p)
        Count matrix.
    X : array, shape (n, d)
        Covariate matrix (intercept column should be included).
    A : array-like or DataFrame, shape (n, a)
        Binary treatment indicator matrix; control cells have all-zero rows.
    r : int
        Number of latent factors.
    W_A : array or None, shape (n, d_A)
        Propensity-score covariate matrix.  If ``None``, ``X`` is used.
    batch_size : int
        Perturbations per batch (default 10).  Ignored when ``n_batches`` is
        set.  Batches are sized evenly with :func:`numpy.array_split` so the
        last batch is never drastically smaller than the others.
    n_batches : int or None
        Total number of batches.  When set, overrides ``batch_size`` and
        perturbations are split as evenly as possible across exactly
        ``n_batches`` batches (e.g. ``n_batches=2`` on a 29-pert dataset
        gives two batches of 15 and 14).
    max_cells : int or None
        Maximum **pert** cells per batch (default 2 000).  ``None`` means no
        cap.  Ctrl cells are added on top so the actual batch size is at most
        ``n_ctrl + max_cells``.  The cap is rarely active because typical
        Perturb-seq datasets have only a few hundred cells per perturbation.
    n_ctrl : int
        Number of ctrl cells in the fixed subsample (default 2 000).
    family : str
        GLM family (default ``'nb'``).
    offset : bool or array-like
        Offset specification passed to :func:`fit_gcate_batch`.
    warm_start_U : bool
        Passed to :func:`fit_gcate_batch`.
    cache_path : str or None
        Path to an HDF5 file used for incremental caching.  When set:

        - On entry, any already-computed batches are loaded from the store and
          their indices are skipped by :func:`fit_gcate_batch`.
        - After each new batch, the result DataFrame is appended to the store
          under key ``/batch_{i:04d}``.
        - On exit, all batches (cached + newly computed) are concatenated and
          returned.

        This lets you resume an interrupted run by re-calling the function
        with the same ``cache_path`` — completed batches are not re-run.
    random_state : int
        RNG seed.
    verbose : bool
        Print per-batch timing.
    gcate_kwargs : dict or None
        Extra keyword arguments forwarded to :func:`fit_gcate_batch`
        (and ultimately :func:`fit_gcate`).  E.g.::

            gcate_kwargs=dict(backend='fast',
                              kwargs_es_1=dict(max_iters=10, rel_tol=2e-4),
                              kwargs_es_2=dict(max_iters=10, rel_tol=2e-4))

    lfc_kwargs : dict or None
        Extra keyword arguments forwarded to :func:`LFC`
        (e.g. ``fdx``, ``thres_min``). Since 0.0.10 the influence-function
        variance (``usevar='pooled'``) is the only estimator and applies to
        balanced and unbalanced arms alike; ``usevar='unequal'`` is accepted
        as a deprecated alias (see :func:`LFC`).
    **kwargs
        Additional arguments forwarded to both :func:`fit_gcate_batch` and
        :func:`LFC`.  When a key collides with ``gcate_kwargs`` /
        ``lfc_kwargs``, the **stage-specific dict wins** — this lets you
        scope a kwarg to one stage (e.g. ``gcate_kwargs=dict(
        backend='fast')`` paired with a top-level ``backend='original'``
        targeting LFC).

    Returns
    -------
    df_res : DataFrame
        Concatenated result from all batches. Includes natural-log ``tau`` and
        ``std`` columns, base-2 ``log2fc`` and ``log2fc_se`` columns, and a
        ``'batch'`` column with the 0-based batch index. Older compatible
        caches containing only ``tau`` and ``std`` are upgraded in memory.
    """
    import gc
    from causarray.gcate import fit_gcate_batch

    if gcate_kwargs is None:
        gcate_kwargs = {}
    if lfc_kwargs is None:
        lfc_kwargs = {}

    import scipy.sparse as _sp
    from causarray.nb_glm_fast import _maybe_densify
    if _sp.issparse(Y):
        Y_np = _maybe_densify(Y)
        gene_names = None
    elif isinstance(Y, pd.DataFrame):
        Y_np = Y.values
        gene_names = list(Y.columns)
    else:
        Y_np = np.asarray(Y)
        gene_names = None  # will fall back to range(p) inside LFC
    X_np = np.asarray(X)
    W_A_np = np.asarray(W_A) if W_A is not None else None

    if isinstance(A, pd.DataFrame):
        A_np = A.values.astype(float)
        pert_names_all = list(A.columns)
    else:
        A_np = np.asarray(A, dtype=float)
        pert_names_all = list(range(A_np.shape[1]))

    # ── Disk-cache: identify cached batches and validate schema ─────────
    from causarray.__about__ import __version__ as _causarray_version
    cached_keys = {}  # batch_i → "/batch_NNNN" hdf5 key
    cache_meta_existing = None
    skip_batches = set()
    if cache_path is not None:
        try:
            with pd.HDFStore(cache_path, mode='r') as store:
                keys = store.keys()
                if '/meta' in keys:
                    cache_meta_existing = store['/meta']
                for key in keys:
                    # keys look like '/batch_0000'
                    if key.startswith('/batch_'):
                        try:
                            idx = int(key.split('_')[1])
                            cached_keys[idx] = key
                            skip_batches.add(idx)
                        except (ValueError, IndexError):
                            pass
        except (FileNotFoundError, OSError):
            pass  # cache file doesn't exist yet — start fresh
        if cache_meta_existing is not None:
            _expected = {
                'family': family,
                'a_total': int(A_np.shape[1]),
            }
            for key_, val_ in _expected.items():
                if key_ in cache_meta_existing.columns:
                    got = cache_meta_existing.iloc[0][key_]
                    if got != val_:
                        raise ValueError(
                            f"gcate_lfc_batch cache at {cache_path!r} was written "
                            f"with {key_}={got!r}, but the current call uses "
                            f"{key_}={val_!r}.  Refusing to mix incompatible "
                            f"schemas — delete the cache file or pass a fresh "
                            f"cache_path to start over."
                        )
        if verbose and skip_batches:
            print(f'[gcate_lfc_batch] Resuming: {len(skip_batches)} batches '
                  f'already cached in {cache_path!r}')

    # ── Run GCATE in batches ─────────────────────────────────────────────
    # Explicit ``gcate_kwargs`` wins over the generic ``**kwargs`` so a kwarg
    # can be scoped to one stage (e.g. ``gcate_kwargs=dict(backend='fast')``
    # with a top-level ``backend='original'`` targeting LFC only).
    batch_results = fit_gcate_batch(
        Y_np, X_np, A, r,
        batch_size=batch_size,        n_batches=n_batches,        max_cells=max_cells,
        n_ctrl=n_ctrl,
        family=family,
        offset=offset,
        warm_start_U=warm_start_U,
        skip_batches=skip_batches,
        random_state=random_state,
        verbose=verbose,
        **{**kwargs, **gcate_kwargs},
    )

    # Build a column-name lookup once
    if isinstance(A, pd.DataFrame):
        pert_col_map = {name: i for i, name in enumerate(pert_names_all)}
    else:
        pert_col_map = {i: i for i in range(A_np.shape[1])}

    new_dfs = {}
    for batch_i, br in enumerate(batch_results):
        # Skipped batches already have their DataFrame in cached_dfs
        if br.get('skipped'):
            continue

        cell_idx = br['cell_idx']
        chunk_pert_names = br['pert_names']
        res_2 = br['res_2']

        # Extract latent factors (copy needed — 'U' is a view of 'X_U')
        U_b = res_2['U'].copy()
        # Use offset already computed by fit_gcate (consistent with GCATE fitting)
        offset_b = np.log(res_2['kwargs_glm']['size_factor'])

        Y_b_np = Y_np[cell_idx]
        # Preserve gene names so df_b['gene_names'] has the same type as df_full
        Y_b = pd.DataFrame(Y_b_np, columns=gene_names) if gene_names is not None else Y_b_np
        X_b = X_np[cell_idx]

        # Recover pert columns for this batch
        chunk_cols = [pert_col_map[name] for name in chunk_pert_names]
        A_b = A_np[np.ix_(cell_idx, chunk_cols)]

        W_b = np.c_[X_b, U_b]
        if W_A_np is not None:
            W_A_b = np.c_[W_A_np[cell_idx], U_b]
        else:
            W_A_b = W_b

        A_df_b = pd.DataFrame(A_b, columns=chunk_pert_names)

        df_b, estimation_b = LFC(
            Y_b, W_b, A_df_b, W_A_b,
            family=family,
            offset=offset_b,
            verbose=verbose,
            **{**kwargs, **lfc_kwargs},
        )
        df_b['batch'] = batch_i
        new_dfs[batch_i] = df_b

        if cache_path is not None:
            with pd.HDFStore(cache_path, mode='a') as store:
                if '/meta' not in store.keys():
                    _meta = pd.DataFrame({
                        'causarray_version': [_causarray_version],
                        'family': [family],
                        'a_total': [int(A_np.shape[1])],
                        'columns': [','.join(df_b.columns.astype(str))],
                    })
                    store.put('/meta', _meta, format='fixed')
                store.put(f'batch_{batch_i:04d}', df_b, format='fixed')

        if save_nuisances:
            _save_batch_nuisances(
                _nuisance_path_for(cache_path), batch_i, estimation_b, cell_idx,
                chunk_pert_names, offset_b, U_b, gene_names,
            )

        del estimation_b   # releases Y_hat and pi_hat
        del U_b, Y_b, Y_b_np, X_b, A_b, W_b, W_A_b
        br['res_1'] = None
        br['res_2'] = None
        gc.collect()

    new_indices = set(new_dfs)
    cached_indices = set(cached_keys)
    all_indices = sorted(new_indices | cached_indices)

    if cached_indices and cache_path is not None:
        with pd.HDFStore(cache_path, mode='r') as store:
            frames = [
                new_dfs[i] if i in new_indices else store[cached_keys[i]]
                for i in all_indices
            ]
            result = pd.concat(frames, axis=0).reset_index(drop=True)
    else:
        result = pd.concat(
            [new_dfs[i] for i in all_indices], axis=0
        ).reset_index(drop=True)
    return _add_log2fc_columns(result)


def LFC_batch(*args, **kwargs):
    """Deprecated alias for :func:`gcate_lfc_batch`.

    .. deprecated::
        Use ``gcate_lfc_batch`` instead.  ``LFC_batch`` will be removed in a
        future release.
    """
    import warnings
    warnings.warn(
        "LFC_batch is deprecated and will be removed in a future release. "
        "Use gcate_lfc_batch instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return gcate_lfc_batch(*args, **kwargs)
