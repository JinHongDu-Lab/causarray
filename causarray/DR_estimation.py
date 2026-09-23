import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn_ensemble_cv import reset_random_seeds, Ensemble, ECV
from causarray.gcate_glm import fit_glm
import causarray.gcate_glm as _gcate_glm  # module-qualified so _USE_FAST_BACKEND changes take effect at call time
from causarray.utils import *
from causarray.utils import _filter_params
from joblib import Parallel, delayed
from tqdm import tqdm
import pprint
import warnings
from collections.abc import Mapping

from sklearn.model_selection import KFold, ShuffleSplit

def _get_func_ps(ps_model, **kwargs):
    if ps_model=='random_forest_cv':
        params_ps = _filter_params(fit_rf, kwargs)
        func_ps = lambda X, Y, X_test:fit_rf_ind_ps(X, Y[:,None], X_test=X_test, **params_ps)[:,0]
    elif ps_model=='logistic':
        clf_ps = LogisticRegression
        kwargs = {**{'fit_intercept':False, 'C':1e0, 'class_weight':None, 'random_state':0}, **kwargs}
        params_ps = _filter_params(clf_ps().get_params(), kwargs)
        func_ps = lambda X, Y, X_test: clf_ps(**params_ps).fit(X, Y).predict_proba(X_test)[:,1]
    elif ps_model=='ensemble':
        kwargs = dict(kwargs)
        logistic_class_weight = kwargs.pop('class_weight', None)
        params_ps_rf = _filter_params(fit_rf, kwargs)
        clf_ps = LogisticRegression
        kwargs = {**{
            'fit_intercept': False, 'C': 1e0,
            'class_weight': logistic_class_weight, 'random_state': 0,
        }, **kwargs}
        params_ps_lr = _filter_params(clf_ps().get_params(), kwargs)
        params_ps = {'params_ps_rf':params_ps_rf, 'params_ps_lr':params_ps_lr}
        func_ps = lambda X, Y, X_test:(fit_rf_ind(X, Y[:,None], X_test=X_test, **params_ps_rf)[:,0] + clf_ps(**params_ps_lr).fit(X, Y).predict_proba(X_test)[:,1])/2
    else:
        raise ValueError('Invalid propensity score model.')

    return func_ps, params_ps


def _validate_clip(clip):
    """Validate a propensity clipping bound and return it as ``(lower, upper)``.

    Raises
    ------
    ValueError
        If ``clip`` is not a pair of numbers satisfying
        ``0 <= lower < upper <= 1``.
    """
    message = 'clip must be None or a pair 0 <= lower < upper <= 1'
    if isinstance(clip, (str, bytes)) or np.ndim(clip) != 1:
        raise ValueError(message)
    try:
        lower, upper = (float(bound) for bound in clip)
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if not 0 <= lower < upper <= 1:
        raise ValueError(message)
    return lower, upper


def estimate_propensity_scores(
    A, X_A, K=1, ps_model='logistic', mask=None, clip=None,
    random_state=0, verbose=False, class_weight=None, **kwargs,
):
    """Estimate per-treatment propensity scores.

    Each treatment is compared with the shared all-zero control group.  With
    ``K > 1``, every returned score is predicted by a model that did not train
    on that cell. Logistic models return calibrated treatment probabilities
    (``class_weight=None``) by default, matching :func:`LFC`. Pass
    ``class_weight='balanced'`` to reproduce pre-0.0.10 fits, whose scores are
    centred near 0.5 regardless of prevalence.

    .. versionchanged:: 0.0.10
        Default ``class_weight`` changed from ``'balanced'`` to ``None``.

    Parameters
    ----------
    A : array-like, shape (n,) or (n, a)
        Binary treatment indicators.  Rows containing only zeros are controls.
    X_A : array-like, shape (n, d_A)
        Covariates used by the propensity model, including an intercept column
        when ``fit_intercept=False``.
    K : int, optional
        Number of folds.  ``1`` fits and predicts on all eligible cells;
        values greater than one produce out-of-fold predictions.
    ps_model : {'logistic', 'random_forest_cv', 'ensemble'}, optional
        Propensity model.
    mask : array-like or None, shape (n,) or (n, a)
        Optional per-treatment eligibility mask for model fitting.
    clip : tuple(float, float) or None, optional
        Bounds applied after prediction.  ``None`` returns raw probabilities.
    random_state : int, optional
        Random seed used for fold construction and supported estimators.
    class_weight : str, dict or None, optional
        Class weighting for logistic propensity estimation. ``None`` (default)
        gives calibrated probabilities and matches :func:`LFC`;
        ``'balanced'`` reproduces the pre-0.0.10 behaviour.

    Returns
    -------
    pi_hat : ndarray, shape (n, a)
        Estimated probabilities ``P(A_j=1 | X_A)``.
    """
    A = np.asarray(A)
    if A.ndim == 1:
        A = A[:, None]
    X_A = np.asarray(X_A)
    if A.ndim != 2 or X_A.ndim != 2 or A.shape[0] != X_A.shape[0]:
        raise ValueError('A and X_A must be two-dimensional with matching rows')
    if not np.all(np.isin(A, (0, 1))):
        raise ValueError('A must contain only binary treatment indicators')

    try:
        K_int = int(K)
    except (TypeError, ValueError) as exc:
        raise ValueError('K must be a positive integer') from exc
    if K_int != K:
        raise ValueError('K must be a positive integer')
    K = K_int
    if K < 1:
        raise ValueError('K must be a positive integer')
    if K > A.shape[0]:
        raise ValueError('K cannot exceed the number of samples')

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim == 1:
            mask = mask[:, None]
        if mask.shape != A.shape:
            raise ValueError('Mask must have the same shape as the treatment matrix')

    func_ps, params_ps = _get_func_ps(
        ps_model, verbose=False, random_state=random_state,
        class_weight=class_weight, **kwargs)
    if verbose:
        pprint.pprint(params_ps)

    if ps_model == 'random_forest_cv':
        info_ecv = run_ecv(X_A, A, **params_ps)
        func_ps, params_ps = _get_func_ps(
            ps_model, verbose=False, ecv=False,
            kwargs_ensemble=info_ecv['best_params_ensemble'],
            kwargs_regr=info_ecv['best_params_regr'],
        )
        if verbose:
            pprint.pprint('Best parameters for the regression model:')
            pprint.pprint(info_ecv['best_params_regr'])
            pprint.pprint('Best parameters for the ensemble model:')
            pprint.pprint(info_ecv['best_params_ensemble'])

    n = A.shape[0]
    if K == 1:
        folds = [(np.arange(n), np.arange(n))]
    elif K == n:
        folds = [
            (np.delete(np.arange(n), j), np.array([j])) for j in range(n)
        ]
    else:
        folds = KFold(
            n_splits=K, random_state=random_state, shuffle=True,
        ).split(X_A)

    pi_hat = np.zeros(A.shape, dtype=float)
    for train_index, test_index in folds:
        A_train = A[train_index]
        XA_train, XA_test = X_A[train_index], X_A[test_index]
        i_ctrl = np.sum(A_train, axis=1) == 0

        for j in range(A.shape[1]):
            i_case = A_train[:, j] == 1
            eligible = mask[train_index, j] if mask is not None else (i_ctrl | i_case)
            y_train = A_train[eligible, j]
            if y_train.size == 0 or np.unique(y_train).size != 2:
                raise ValueError(
                    f'Treatment {j} needs at least one eligible control and case '
                    'in every training fold'
                )

            x_train = XA_train[eligible]
            constant_design = np.all(np.ptp(x_train, axis=0) == 0)
            if ps_model == 'logistic' and constant_design:
                if class_weight == 'balanced':
                    pi_hat[test_index, j] = 0.5
                elif isinstance(class_weight, dict):
                    sample_weight = np.asarray([
                        class_weight.get(int(value), 1.0) for value in y_train
                    ])
                    pi_hat[test_index, j] = np.average(
                        y_train, weights=sample_weight)
                else:
                    pi_hat[test_index, j] = np.mean(y_train)
            else:
                pi_hat[test_index, j] = func_ps(x_train, y_train, XA_test)

    if clip is not None:
        lower, upper = _validate_clip(clip)
        pi_hat = np.clip(pi_hat, lower, upper)
    return pi_hat


def _arm_support_metrics(A, pi_hat, j, bins=40, mask=None):
    """Support metrics for one treatment column, matching
    :func:`~causarray.diagnostics.summarize_propensity_scores`.

    Scoring a single arm avoids summarizing every treatment on each candidate
    penalty, which dominates the cost of a search.  With ``mask``, only the
    cells the propensity model was fitted on are scored.
    """
    from sklearn.metrics import roc_auc_score

    from causarray.diagnostics import _effective_sample_size

    A = np.asarray(A, dtype=float)
    ctrl = A.sum(axis=1) == 0
    case = A[:, j] == 1
    eligible = ctrl | case
    if mask is not None:
        eligible &= mask[:, j]
    y = case[eligible].astype(int)
    p = np.asarray(pi_hat)[eligible, j]
    p_ctrl, p_case = p[y == 0], p[y == 1]

    h_ctrl, edges = np.histogram(p_ctrl, bins=bins, range=(0, 1))
    h_case, _ = np.histogram(p_case, bins=edges)
    h_ctrl = h_ctrl / h_ctrl.sum() if h_ctrl.sum() else h_ctrl
    h_case = h_case / h_case.sum() if h_case.sum() else h_case
    eps = np.finfo(float).eps
    return {
        'auc': float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan,
        'overlap_ratio': float(np.minimum(h_ctrl, h_case).sum()),
        'ess_treated_fraction': (
            _effective_sample_size(1 / np.clip(p_case, eps, None)) / max(len(p_case), 1)),
        'ess_control_fraction': (
            _effective_sample_size(1 / np.clip(1 - p_ctrl, eps, None)) / max(len(p_ctrl), 1)),
    }


def _meets(metrics, target):
    """True when every ``<metric>_lt`` / ``<metric>_gt`` condition holds.

    The prefix must name a metric exactly, for example
    ``ess_treated_fraction_lt``, not ``ess_treated_lt``.
    """
    for key, bound in target.items():
        if key.endswith('_lt'):
            name, op = key[:-3], 'lt'
        elif key.endswith('_gt'):
            name, op = key[:-3], 'gt'
        else:
            raise ValueError(f"condition {key!r} must end with '_lt' or '_gt'")
        if name not in metrics:
            raise ValueError(
                f"condition {key!r} names no metric; available metrics are "
                f"{sorted(metrics)}")
        value = metrics[name]
        if not np.isfinite(value):
            return False
        if op == 'lt' and not value < bound:
            return False
        if op == 'gt' and not value > bound:
            return False
    return True


def tune_penalty_factor(
    A, X_A, covariate, treatment_names=None, covariate_names=None,
    trigger=None, target=None, bracket=(1.0, 1e4), tol=0.15,
    on_infeasible='best',
    K=1, ps_model='logistic', mask=None, random_state=0, verbose=False,
    class_weight=None, **kwargs,
):
    """Choose a per-treatment L2 penalty for one propensity covariate.

    Some perturbations shift a covariate so strongly that the propensity model
    separates them from the controls, and their inverse-probability weights
    collapse onto a few cells.  :func:`refit_propensity_scores` can penalize
    that coefficient for selected treatments; this picks the factor.

    For each triggered treatment the search is bracketed by two endpoints: the
    unpenalized fit and the fit with the covariate dropped, which is the
    infinite-penalty limit.  **If the dropped fit misses ``target``, no finite
    penalty can reach it**, so the treatment is reported as infeasible after a
    single fit instead of an exhausted search.  Otherwise the factor is found by
    bisection on a log scale and the *smallest* qualifying value is returned, so
    the covariate keeps as much of its adjustment role as the data support.

    Target a monotone metric.  ``auc`` and ``overlap_ratio`` move monotonically
    with the penalty; ``ess_treated_fraction`` does not, because an arm that is
    completely separated has near-uniform weights and a deceptively high ESS
    that *falls* as the penalty restores genuine overlap.

    Penalizing a covariate is a soft version of dropping it, so the two are
    endpoints of one continuum.  When the covariate is affected by treatment,
    no factor makes that contrast identified; the choice trades a known bias
    against precision and belongs in the analysis plan, not in this search.

    Parameters
    ----------
    A : array-like, shape (n,) or (n, a)
        Binary treatment indicators; all-zero rows are the shared controls.
    X_A : array-like, shape (n, d_A)
        Propensity covariates, including the intercept column.
    covariate : str or int
        The single covariate whose penalty is tuned.
    treatment_names, covariate_names : sequence, optional
        Labels for ``A`` columns and ``X_A`` columns.
    trigger : mapping or None
        Conditions selecting which treatments to tune, as ``{'auc_gt': 0.9,
        'ess_treated_fraction_lt': 0.5}``.  A treatment is triggered when **any**
        condition holds.  Defaults to ``{'auc_gt': 0.9}``.
    target : mapping or None
        Conditions a factor must satisfy, in the same form, combined with
        **and**.  Defaults to ``{'auc_lt': 0.9}``.
    bracket : tuple(float, float)
        Smallest and largest factors considered.  The lower end is evaluated as
        the unpenalized fit and the upper end as the dropped-covariate fit.
    on_infeasible : {'best', 'none'}
        What to do when the dropped-covariate endpoint already misses
        ``target``, so no finite factor can reach it.  ``'best'`` (default)
        applies the largest factor in ``bracket``, giving the arm the closest
        support the covariate allows.  ``'none'`` leaves it unpenalized, which
        keeps the arm at its worst-case weights -- the trigger has already said
        its support is inadequate, so doing nothing is not a neutral choice.
    tol : float
        Bisection stops when the bracket spans less than ``tol`` in natural log
        units.
    K, ps_model, mask, random_state, verbose, class_weight, **kwargs
        Passed to :func:`estimate_propensity_scores` for every candidate fit.

    Returns
    -------
    penalty_factors_by_treatment : dict
        ``{treatment: {covariate: factor}}`` for feasible triggered treatments,
        ready to hand to :func:`refit_propensity_scores`.  Treatments that were
        not triggered, already met ``target`` unpenalized, or are infeasible
        with ``on_infeasible='none'`` are absent.
    report : DataFrame
        One row per triggered treatment with the chosen factor, whether the
        target was feasible (``feasible``) and met at that factor
        (``target_met``), the number of fits used, and the metrics
        unpenalized, at the chosen factor, and with the covariate dropped.

    Examples
    --------
    >>> factors, report = tune_penalty_factor(
    ...     A, X_A, 'log_library_size', treatment_names=names,
    ...     covariate_names=cov, target={'auc_lt': 0.9})  # doctest: +SKIP
    >>> pi, _ = refit_propensity_scores(
    ...     A, X_A, pi_hat=pi, treatment_names=names, covariate_names=cov,
    ...     penalty_factors_by_treatment=factors)  # doctest: +SKIP
    """
    trigger = {'auc_gt': 0.9} if trigger is None else dict(trigger)
    target = {'auc_lt': 0.9} if target is None else dict(target)
    low, high = (float(b) for b in bracket)
    if not 1.0 <= low < high:
        raise ValueError('bracket must satisfy 1 <= low < high')
    if tol <= 0:
        raise ValueError('tol must be positive')

    A_arr = np.asarray(A, dtype=float)
    if A_arr.ndim == 1:
        A_arr = A_arr[:, None]
    if treatment_names is None:
        treatment_names = (list(A.columns) if hasattr(A, 'columns')
                           else list(range(A_arr.shape[1])))
    treatment_names = list(treatment_names)
    if covariate_names is None:
        covariate_names = (list(X_A.columns) if hasattr(X_A, 'columns')
                           else [f'covariate_{j + 1}' for j in range(np.shape(X_A)[1])])
    covariate_names = list(covariate_names)
    if covariate not in covariate_names:
        if isinstance(covariate, (int, np.integer)) and 0 <= covariate < len(covariate_names):
            covariate = covariate_names[covariate]
        else:
            raise ValueError(f'covariate {covariate!r} is not in covariate_names')

    mask_arr = None
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.ndim == 1:
            mask_arr = mask_arr[:, None]
    fit_kwargs = dict(K=K, ps_model=ps_model, mask=mask, clip=None,
                      random_state=random_state, verbose=False,
                      class_weight=class_weight, **kwargs)
    pi_base = estimate_propensity_scores(A_arr, X_A, **fit_kwargs)

    def scored(pi, j):
        return _arm_support_metrics(A_arr, pi, j, mask=mask_arr)

    rows, factors = [], {}
    for j, name in enumerate(treatment_names):
        base = scored(pi_base, j)
        if not any(_meets(base, {key: bound}) for key, bound in trigger.items()):
            continue
        n_fits = 1

        def evaluate(factor, name=name, j=j):
            pi_try, _ = refit_propensity_scores(
                A_arr, X_A, pi_hat=pi_base.copy(), treatment_names=treatment_names,
                covariate_names=covariate_names,
                penalty_factors_by_treatment={name: {covariate: float(factor)}},
                **fit_kwargs)
            return scored(pi_try, j)

        # factor -> infinity is the covariate dropped; it bounds what any
        # finite penalty can achieve.
        pi_drop, _ = refit_propensity_scores(
            A_arr, X_A, pi_hat=pi_base.copy(), treatment_names=treatment_names,
            covariate_names=covariate_names, drop_by_treatment={name: [covariate]},
            **fit_kwargs)
        dropped = scored(pi_drop, j)
        n_fits += 1
        feasible = _meets(dropped, target)

        if not feasible:
            if on_infeasible == 'best':
                chosen_factor, chosen = high, evaluate(high)
                n_fits += 1
                factors[name] = {covariate: high}
            elif on_infeasible == 'none':
                chosen_factor, chosen = 1.0, base
            else:
                raise ValueError("on_infeasible must be 'best' or 'none'")
        elif _meets(base, target):
            chosen_factor, chosen = low, base      # nothing to do beyond the trigger
        else:
            lo_log, hi_log = np.log(low), np.log(high)
            chosen_factor, chosen = None, None
            while hi_log - lo_log > tol:
                mid_log = 0.5 * (lo_log + hi_log)
                metrics = evaluate(np.exp(mid_log))
                n_fits += 1
                if _meets(metrics, target):
                    hi_log, chosen_factor, chosen = mid_log, float(np.exp(mid_log)), metrics
                else:
                    lo_log = mid_log
            if chosen is None:
                # No bisection point qualified; score the largest factor
                # itself rather than borrowing the dropped fit's metrics.
                chosen_factor, chosen = high, evaluate(high)
                n_fits += 1
            factors[name] = {covariate: chosen_factor}

        rows.append({'treatment': name, 'penalty_factor': chosen_factor,
                     'feasible': feasible, 'target_met': _meets(chosen, target),
                     'n_fits': n_fits,
                     **{f'{k}_unpenalized': v for k, v in base.items()},
                     **{f'{k}_chosen': v for k, v in chosen.items()},
                     **{f'{k}_dropped': v for k, v in dropped.items()}})

    report = pd.DataFrame(rows)
    if verbose and len(report):
        print(f'[tune_penalty_factor] {len(report)} treatments triggered, '
              f'{int(report.feasible.sum())} feasible, '
              f'{report.n_fits.sum()} fits', flush=True)
    return factors, report


def refit_propensity_scores(
    A, X_A, drop_by_treatment=None, pi_hat=None, treatment_names=None,
    covariate_names=None, penalty_factors_by_treatment=None, K=1,
    ps_model='logistic', mask=None, clip=None, random_state=0, verbose=False,
    class_weight=None, **kwargs,
):
    """Refit propensity scores with treatment-specific covariate filtering.

    This helper supports sensitivity analyses in which different treatments
    omit different observed covariates or latent factors, or apply stronger L2
    regularization to selected covariates. When existing scores are supplied,
    only treatments named in either treatment-specific mapping are refitted;
    the remaining columns are carried over from ``pi_hat`` unchanged, up to the
    shared ``clip`` described below. Outcome models are not fit by this function
    and cached ``Y_hat`` values can be reused in :func:`LFC`.

    Parameters
    ----------
    A : array-like, shape (n, a)
        Binary treatment indicator matrix.
    X_A : array-like, shape (n, d_A)
        Full propensity-model design before treatment-specific filtering.
    drop_by_treatment : mapping or None
        Treatment names or indices mapped to covariate names or indices to
        remove for that treatment. Defaults to no removals.
    pi_hat : array-like or None, shape (n, a)
        Existing raw propensity scores. If supplied, treatments absent from
        ``drop_by_treatment`` are not refitted.
    treatment_names, covariate_names : sequence, optional
        Column labels, inferred from DataFrames when possible.
    penalty_factors_by_treatment : mapping or None
        Treatment names or indices mapped to ``{covariate: factor}`` mappings.
        A factor greater than one applies that multiple of the ordinary L2
        penalty to the named coefficient. This is implemented by dividing the
        covariate by ``sqrt(factor)`` during both fitting and prediction and is
        available only for ``ps_model='logistic'`` with an L2 penalty. A factor
        of one leaves the covariate unchanged. Interpret relative penalties on
        a common scale; ``prep_causarray_data`` standardizes log-library size.
    K, ps_model, mask, random_state, verbose, class_weight, **kwargs
        Passed to :func:`estimate_propensity_scores` for each refitted model.
    clip : tuple(float, float) or None, optional
        Bounds applied to the **whole** returned matrix, refitted columns and
        carried-over columns alike, so that a single consistent bound reaches
        :func:`LFC`. Pass ``None`` to leave carried-over scores exactly as
        supplied.

    Returns
    -------
    pi_updated : ndarray, shape (n, a)
        Updated propensity scores.
    report : DataFrame
        Audit table of refitted treatments, retained/dropped covariates, and
        the resulting score spread. ``degenerate_design`` flags a treatment
        whose retained design is constant, and ``score_std`` reports the
        standard deviation of its refitted scores on eligible rows; both make a
        collapsed propensity model visible without reading the warning stream.

    Warns
    -----
    RuntimeWarning
        If filtering leaves a constant design for some treatment, in which case
        its scores carry no covariate information.

    Notes
    -----
    A very large penalty factor shrinks a coefficient towards zero without
    making the design constant. That case leaves ``degenerate_design`` False
    but drives ``score_std`` towards zero.

    .. versionadded:: 0.0.9
    .. versionchanged:: 0.0.10
        Default ``class_weight`` changed from ``'balanced'`` to ``None`` to
        match :func:`estimate_propensity_scores` and :func:`LFC`.
    """
    if drop_by_treatment is None:
        drop_by_treatment = {}
    if penalty_factors_by_treatment is None:
        penalty_factors_by_treatment = {}
    if not isinstance(drop_by_treatment, Mapping):
        raise ValueError('drop_by_treatment must be a mapping')
    if not isinstance(penalty_factors_by_treatment, Mapping):
        raise ValueError('penalty_factors_by_treatment must be a mapping')
    if penalty_factors_by_treatment:
        if ps_model != 'logistic':
            raise ValueError(
                'penalty_factors_by_treatment is supported only for logistic models'
            )
        if kwargs.get('penalty', 'l2') != 'l2':
            raise ValueError(
                'penalty_factors_by_treatment requires logistic penalty="l2"'
            )

    if isinstance(A, pd.DataFrame):
        if treatment_names is None:
            treatment_names = list(A.columns)
        A_array = A.to_numpy()
    else:
        A_array = np.asarray(A)
    if A_array.ndim == 1:
        A_array = A_array[:, None]
    if A_array.ndim != 2 or not np.all(np.isin(A_array, (0, 1))):
        raise ValueError('A must be a one- or two-dimensional binary matrix')
    if treatment_names is None:
        treatment_names = list(range(A_array.shape[1]))
    treatment_names = list(treatment_names)
    if len(treatment_names) != A_array.shape[1]:
        raise ValueError('treatment_names must match the number of treatments')
    if len(set(treatment_names)) != len(treatment_names):
        raise ValueError('treatment_names must be unique')

    if isinstance(X_A, pd.DataFrame):
        if covariate_names is None:
            covariate_names = list(X_A.columns)
        X_array = X_A.to_numpy()
    else:
        X_array = np.asarray(X_A)
    if X_array.ndim != 2 or X_array.shape[0] != A_array.shape[0]:
        raise ValueError('X_A must be two-dimensional with the same rows as A')
    try:
        X_array = np.asarray(X_array, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('X_A must contain numeric covariates') from exc
    if not np.all(np.isfinite(X_array)):
        raise ValueError('X_A must contain only finite values')
    if covariate_names is None:
        covariate_names = [f'covariate_{j + 1}' for j in range(X_array.shape[1])]
    covariate_names = list(covariate_names)
    if len(covariate_names) != X_array.shape[1]:
        raise ValueError('covariate_names must match the number of covariates')
    if len(set(covariate_names)) != len(covariate_names):
        raise ValueError('covariate_names must be unique')

    treatment_lookup = {name: j for j, name in enumerate(treatment_names)}
    covariate_lookup = {name: j for j, name in enumerate(covariate_names)}

    def resolve_treatment(value):
        if value in treatment_lookup:
            return treatment_lookup[value]
        if isinstance(value, (int, np.integer)) and 0 <= int(value) < A_array.shape[1]:
            return int(value)
        raise ValueError(f'Unknown treatment: {value}')

    def resolve_covariate(value):
        if value in covariate_lookup:
            return covariate_lookup[value]
        if isinstance(value, (int, np.integer)) and 0 <= int(value) < X_array.shape[1]:
            return int(value)
        raise ValueError(f'Unknown covariate: {value}')

    drops = {}
    for treatment, covariates in drop_by_treatment.items():
        j = resolve_treatment(treatment)
        if j in drops:
            raise ValueError(f'Treatment {treatment_names[j]} is specified more than once')
        if isinstance(covariates, (str, bytes)):
            covariates = [covariates]
        indices = [resolve_covariate(value) for value in covariates]
        if len(set(indices)) != len(indices):
            raise ValueError(
                f'Duplicate dropped covariates for treatment {treatment_names[j]}'
            )
        drops[j] = set(indices)

    penalty_factors = {}
    for treatment, factors in penalty_factors_by_treatment.items():
        j = resolve_treatment(treatment)
        if j in penalty_factors:
            raise ValueError(f'Treatment {treatment_names[j]} is specified more than once')
        if not isinstance(factors, Mapping):
            raise ValueError(
                f'Penalty factors for treatment {treatment_names[j]} must be a mapping'
            )
        resolved = {}
        for covariate, factor in factors.items():
            k = resolve_covariate(covariate)
            try:
                factor = float(factor)
            except (TypeError, ValueError) as exc:
                raise ValueError('Penalty factors must be finite numbers at least 1') from exc
            if not np.isfinite(factor) or factor < 1:
                raise ValueError('Penalty factors must be finite numbers at least 1')
            if k in resolved:
                raise ValueError(
                    f'Duplicate penalty factors for covariate {covariate_names[k]}'
                )
            resolved[k] = factor
        penalty_factors[j] = resolved

    for j in set(drops).intersection(penalty_factors):
        conflict = drops[j].intersection(penalty_factors[j])
        if conflict:
            names = [covariate_names[k] for k in sorted(conflict)]
            raise ValueError(
                f'Covariates cannot be both dropped and penalized for '
                f'{treatment_names[j]}: {names}'
            )

    if pi_hat is None:
        pi_updated = np.empty(A_array.shape, dtype=float)
        refit_indices = range(A_array.shape[1])
    else:
        pi_updated = np.asarray(pi_hat, dtype=float).copy()
        if pi_updated.shape != A_array.shape:
            raise ValueError('pi_hat must have the same shape as A')
        if (not np.all(np.isfinite(pi_updated))
                or np.any((pi_updated < 0) | (pi_updated > 1))):
            raise ValueError('pi_hat must contain finite probabilities in [0, 1]')
        refit_indices = sorted(set(drops).union(penalty_factors))

    mask_array = None
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.ndim == 1:
            mask_array = mask_array[:, None]
        if mask_array.shape != A_array.shape:
            raise ValueError('Mask must have the same shape as the treatment matrix')

    ctrl = np.sum(A_array, axis=1) == 0
    rows = []
    for j in refit_indices:
        dropped = drops.get(j, set())
        retained = [k for k in range(X_array.shape[1]) if k not in dropped]
        if not retained:
            raise ValueError(
                f'Filtering removes every covariate for treatment {treatment_names[j]}'
            )
        eligible = ctrl | (A_array[:, j] == 1)
        if mask_array is not None:
            eligible &= mask_array[:, j]
        X_treatment = X_array[:, retained].copy()
        treatment_penalties = penalty_factors.get(j, {})
        for position, k in enumerate(retained):
            factor = treatment_penalties.get(k, 1.0)
            X_treatment[:, position] /= np.sqrt(factor)
        degenerate = bool(np.all(np.ptp(X_treatment[eligible], axis=0) == 0))
        if degenerate:
            warnings.warn(
                f'The propensity design for treatment {treatment_names[j]} is '
                'constant after filtering; its scores fall back to a '
                'class-weighted prevalence and carry no covariate information.',
                RuntimeWarning, stacklevel=2,
            )
        scores = estimate_propensity_scores(
            A_array[:, [j]], X_treatment, K=K,
            ps_model=ps_model, mask=eligible[:, None], clip=None,
            random_state=random_state, verbose=verbose,
            class_weight=class_weight, **kwargs,
        )
        pi_updated[:, j] = scores[:, 0]
        rows.append({
            'treatment': treatment_names[j],
            'dropped_covariates': [covariate_names[k] for k in sorted(dropped)],
            'retained_covariates': [covariate_names[k] for k in retained],
            'penalty_factors': {
                covariate_names[k]: treatment_penalties[k]
                for k in retained if treatment_penalties.get(k, 1.0) != 1.0
            },
            'n_retained': len(retained),
            'degenerate_design': degenerate,
            'score_std': float(np.std(scores[eligible, 0])),
        })

    if clip is not None:
        lower, upper = _validate_clip(clip)
        pi_updated = np.clip(pi_updated, lower, upper)
    report = pd.DataFrame(rows, columns=[
        'treatment', 'dropped_covariates', 'retained_covariates',
        'penalty_factors', 'n_retained', 'degenerate_design', 'score_std',
    ])
    return pi_updated, report


def cross_fitting(
    Y, A, X, X_A, family='poisson', K=1, glm_alpha=1e-4,
    ps_model='logistic', ps_class_weight=None,
    Y_hat=None, pi_hat=None, mask=None, ps_clip='auto',
    return_raw_pi=False, verbose=False, **kwargs):
    '''
    Cross-fitting for causal estimands.

    Parameters
    ----------
    Y : array
        Outcomes.    
    A : array
        Binary treatment indicator.
    X : array
        Covariates.
    X_A : array
        Covariates for the propensity score model.
    family : str, optional
        The family of the generalized linear model. The default is 'poisson'.
    K : int, optional
        The number of folds for cross-validation. The default is 1.
    glm_alpha : float, optional
        The regularization parameter for the generalized linear model. The default is 1e-4.
    ps_model : str, optional
        The propensity score model. The default is 'logistic'.
    ps_class_weight : str, dict or None, optional
        Class weighting used by the propensity model. ``None`` (default since
        0.0.10) gives calibrated treatment probabilities; ``'balanced'``
        reproduces the pre-0.0.10 nuisance fit.
    
    Y_hat : array, optional
        Estimated potential outcome of shape (n, p, a, 2). The default is None.
    pi_hat : array, optional
        Propensity score of shape (n, a). The default is None.
    mask : array, optional
        Boolean mask of shape (n, a) for the treatment, indicating which samples are used for 
        propensity-model fitting and the downstream estimand.
    ps_clip : {'auto'}, tuple(float, float), (lower_array, upper_array) or None, optional
        Bounds applied to scores used by AIPW. ``'auto'`` (default) resolves
        to a prevalence-aware bound per treatment (see
        :func:`causarray.DR_learner._resolve_ps_clip`); a pair of scalars
        applies one bound to all treatments; a pair of length-``a`` arrays
        gives per-treatment bounds; ``None`` disables clipping.
    return_raw_pi : bool, optional
        Return raw scores as a third result when true.

    **kwargs : dict
        Additional arguments to pass to the model.

    Returns
    -------    
    Y_hat : array
        Estimated potential outcome under control.
    pi_hat : array
        Estimated propensity score.
    pi_hat_raw : array
        Unclipped propensity score, returned only when ``return_raw_pi=True``.
    '''
    kwargs = dict(kwargs)
    if 'class_weight' in kwargs:
        legacy_class_weight = kwargs.pop('class_weight')
        warnings.warn(
            'Passing class_weight through LFC/cross_fitting is deprecated; '
            'use ps_class_weight instead.',
            FutureWarning, stacklevel=2,
        )
        ps_class_weight = legacy_class_weight

    params_glm = _filter_params(fit_glm, {**kwargs, 'verbose': verbose})

    if verbose:
        pprint.pprint(params_glm)
    
    if K > 1:
        n_samples = X.shape[0]
        if K >= n_samples:
            # Use Leave-One-Out Cross-Validation
            folds = [([i for i in range(n_samples) if i != j], [j]) for j in range(n_samples)]
        else:
            # Initialize KFold cross-validator
            kf = KFold(n_splits=int(K), random_state=0, shuffle=True)
            folds = kf.split(X)
    else:
        folds = [(np.arange(X.shape[0]), np.arange(X.shape[0]))]

    # Initialize lists to store results
    if pi_hat is None:
        if verbose:
            pprint.pprint('Fit propensity score models...')
        ps_kwargs = {k: v for k, v in kwargs.items() if k != 'random_state'}
        if ps_model in ('logistic', 'ensemble'):
            ps_kwargs['class_weight'] = ps_class_weight
        pi_hat_raw = estimate_propensity_scores(
            A, X_A, K=K, ps_model=ps_model, mask=mask,
            random_state=kwargs.get('random_state', 0), verbose=verbose,
            **ps_kwargs,
        )
    else:
        pi_hat_raw = np.asarray(pi_hat, dtype=float).reshape(A.shape)
    if ps_clip is None:
        pi_hat = pi_hat_raw.copy()
    else:
        if isinstance(ps_clip, str):
            from causarray.DR_learner import _resolve_ps_clip
            ps_clip = _resolve_ps_clip(ps_clip, A, mask)
        if len(ps_clip) != 2:
            raise ValueError(
                "ps_clip must be 'auto', None, or a pair 0 <= lower < upper <= 1")
        lower = np.broadcast_to(np.asarray(ps_clip[0], dtype=float), (A.shape[1],))
        upper = np.broadcast_to(np.asarray(ps_clip[1], dtype=float), (A.shape[1],))
        if not (np.all(0 <= lower) and np.all(lower < upper) and np.all(upper <= 1)):
            raise ValueError(
                "ps_clip must be 'auto', None, or a pair 0 <= lower < upper <= 1")
        pi_hat = np.clip(pi_hat_raw, lower[None, :], upper[None, :])
    fit_Y = True if Y_hat is None else False
    if fit_Y:
        _yhat_gb = Y.shape[0] * Y.shape[1] * A.shape[1] * 2 * 8 / 1e9
        _mem_limit_gb = kwargs.get('mem_limit_gb', None)
        if _mem_limit_gb is not None and _yhat_gb > _mem_limit_gb:
            warnings.warn(
                f"Y_hat allocation ({_yhat_gb:.1f} GB as float64) exceeds "
                f"mem_limit_gb={_mem_limit_gb} GB; using float32 to halve peak memory.",
                ResourceWarning, stacklevel=3,
            )
            Y_hat = np.zeros((Y.shape[0], Y.shape[1], A.shape[1], 2), dtype=np.float32)
        else:
            Y_hat = np.zeros((Y.shape[0], Y.shape[1], A.shape[1], 2), dtype=float)

    # Perform cross-fitting
    for train_index, test_index in folds:
        # Split data
        X_train, X_test = X[train_index], X[test_index]
        XA_train, XA_test = X_A[train_index], X_A[test_index]
        A_train, A_test = A[train_index], A[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        if fit_Y:
            if verbose: pprint.pprint('Fit outcome models...')
            # Subset offset to training fold (for fitting) and test fold (for
            # imputation) when it is a pre-computed array, so that
            # ``fit_glm_auto`` receives arrays with matching leading
            # dimensions in both stages.
            params_glm_fold = params_glm
            offset_test_arr = None
            if 'offset' in params_glm and isinstance(params_glm['offset'], np.ndarray):
                params_glm_fold = dict(params_glm)
                params_glm_fold['offset'] = params_glm['offset'][train_index]
                offset_test_arr = params_glm['offset'][test_index]
            # Fit GLM on training data and predict on test data
            res = _gcate_glm.fit_glm_auto(Y_train, X_train, A_train, family=family, alpha=glm_alpha,
                impute=X_test, offset_test=offset_test_arr, **params_glm_fold)
            Y_hat[test_index,:,:,0] = res[1][0]
            Y_hat[test_index,:,:,1] = res[1][1]

    Y_hat = np.clip(Y_hat, None, 1e5)
    if return_raw_pi:
        return Y_hat, pi_hat, pi_hat_raw
    return Y_hat, pi_hat





def AIPW_mean(Y, A, mu, pi):
    '''
    Augmented inverse probability weighted estimator (AIPW)

    Parameters
    ----------
    Y : array
        Outcomes of shape (n, p).
    A : array
        Binary treatment indicator of shape (n, a, 2).
    mu : array
        Conditional outcome distribution estimate of shape (n, p, a, 2).
    pi : array
        Propensity score of shape (n, a, 2).
    Returns
    -------
    tau : array
        A point estimate of the expected potential outcome of shape (p, a, 2).
    pseudo_y : array
        Pseudo-outcome of shape (n, p, a, 2).
    '''
    
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        weight = A / pi
    weight = weight[:, None, ...]
    Y = Y[:, :, None, None]

    # Influence-function values are intentionally left unconstrained.  Even
    # for a nonnegative outcome, individual AIPW pseudo-outcomes may be
    # negative; projecting them cell by cell changes their mean and biases the
    # estimator.  Parameter-space constraints belong after aggregation.
    pseudo_y = weight * (Y - mu) + mu

    tau = np.mean(pseudo_y, axis=0, dtype=np.float64)

    return tau, pseudo_y








def run_ecv(
    X, y, M=200, M_max=1000,
    # fixed parameters for bagging regressor
    kwargs_ensemble={},
    # fixed parameters for decision tree
    kwargs_regr={},
    # grid search parameters
    grid_regr={},
    grid_ensemble={}
):
    """
    Runs Ensemble Cross-Validation (ECV) to find the best hyperparameters.
    """
    kwargs_ensemble = {**{'verbose': 1, 'bootstrap': True}, **kwargs_ensemble}
    kwargs_regr = {**{'min_samples_split': 20, 'min_samples_leaf': 10, 'max_features': 'sqrt', 'ccp_alpha': 0.02, 'class_weight': 'balanced'}, **kwargs_regr}
    grid_regr = {**{'max_depth': [3, 5, 7]}, **grid_regr}
    grid_ensemble = {**{'random_state': 0, 'max_samples': [0.4, 0.6, 0.8, 1.]}, **grid_ensemble}

    # Validate integer parameters
    M = int(M)
    M_max = int(M_max)

    # Make sure y is 2D
    y = y.reshape(-1, 1) if y.ndim == 1 else y

    # Run ECV
    _, info_ecv = ECV(
        X, y, DecisionTreeClassifier, grid_regr, grid_ensemble,
        kwargs_regr, kwargs_ensemble,
        M=M, M0=M, M_max=M_max, return_df=True
    )

    # Replace the in-sample best parameter for 'n_estimators' with extrapolated best parameter
    info_ecv['best_params_ensemble']['n_estimators'] = info_ecv['best_n_estimators_extrapolate']

    return info_ecv


def fit_rf(
    X, y, X_test=None, M=100, M_max=1000, ecv=True,
    # fixed parameters for bagging regressor
    kwargs_ensemble={},
    # fixed parameters for decision tree
    kwargs_regr={},
    # grid search parameters
    grid_regr={},
    grid_ensemble={}
):
    """
    Fits a Random Forest model using parameters found by ECV.
    """

    kwargs_ensemble = {**{'verbose': 1, 'bootstrap': True}, **kwargs_ensemble}
    kwargs_regr = {**{'min_samples_split': 20, 'min_samples_leaf': 10, 'max_features': 'sqrt', 'ccp_alpha': 0.02, 'class_weight': 'balanced'}, **kwargs_regr}
    grid_regr = {**{'max_depth': [3, 5, 7]}, **grid_regr}
    grid_ensemble = {**{'random_state': 0, 'max_samples': [0.4, 0.6, 0.8, 1.]}, **grid_ensemble}

    # Make sure y is 2D
    y_2d = y.reshape(-1, 1) if y.ndim == 1 else y

    if ecv:
        # Get best parameters from ECV
        info_ecv = run_ecv(
            X, y_2d, M=M, M_max=M_max,
            kwargs_ensemble=kwargs_ensemble,
            kwargs_regr=kwargs_regr,
            grid_regr=grid_regr,
            grid_ensemble=grid_ensemble
        )
        params_regr = info_ecv['best_params_regr']
        params_ensemble = info_ecv['best_params_ensemble']
    else:
        params_regr = kwargs_regr
        params_ensemble = kwargs_ensemble
        
    # Fit the ensemble with the best CV parameters
    regr = Ensemble(
        estimator=DecisionTreeClassifier(**params_regr), **params_ensemble).fit(X, y_2d)

    # Predict
    if X_test is None:
        X_test = X
    return regr.predict(X_test).reshape(-1, y_2d.shape[1])



def fit_rf_ind(X, Y, *args, **kwargs):
    Y_hat = Parallel(n_jobs=-1)(delayed(fit_rf)(X, Y[:,j], *args, **kwargs)
        for j in tqdm(range(Y.shape[1])))
    Y_pred = np.concatenate(Y_hat, axis=-1)
    return Y_pred


def fit_rf_ind_ps(X, Y, *args, **kwargs):
    i_ctrl = (np.sum(Y, axis=1) == 0.)

    if 'X_test' not in kwargs:
        kwargs['X_test'] = X

    def _fit(X, y, i_ctrl, *args, **kwargs):        
        i_case = (y == 1.)
        i_cells = i_ctrl | i_case
        return fit_rf(X[i_cells], y[i_cells], *args, **kwargs)

    Y_hat = Parallel(n_jobs=-1)(delayed(_fit)(X, Y[:,j], i_ctrl, *args, **kwargs)
        for j in tqdm(range(Y.shape[1])))
    Y_pred = np.concatenate(Y_hat, axis=-1)

    return Y_pred


def fit_rf_ind_outcome(W, Y, A, *args, **kwargs):
    d = W.shape[1]
    a = A.shape[1]
    X = np.c_[W, A]
    X_test = np.tile(np.c_[W, np.zeros_like(A)][:,None,:], (1,1+a,1))
    for j in range(a):
        X_test[:,1+j,d+j] = 1
    X_test = X_test.reshape(-1, X_test.shape[-1])
    Y_pred = fit_rf_ind(X, Y, X_test=X_test)
    Y_pred = Y_pred.reshape(X.shape[0],1+a,Y.shape[1])
    Yhat_1 = Y_pred[:,1:,:].transpose(0,2,1)
    Yhat_0 = np.tile(Y_pred[:,0,:][:,:,None], (1,1,a))
    return Yhat_0, Yhat_1
