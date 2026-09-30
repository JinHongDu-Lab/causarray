import os
import random
import numpy as np
import pandas as pd

import inspect

import pprint
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

np.set_printoptions(threshold=10)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot


def prep_causarray_data(Y, A, X=None, X_A=None, intercept=True):
    """
    Prepares the input data for the causarray model.

    Parameters
    ----------
    Y : array-like
        The response matrix.
    A : array-like
        The treatment matrix.
    X : array-like, optional
        The covariate matrix. Defaults to None.
    X_A : array-like, optional
        The covariate matrix for the treatment. Defaults to None.
    intercept : bool, optional
        Whether to include an intercept in the covariate matrix. Defaults to True.

    Returns
    -------
    Y : array
        The processed response matrix.
    A : array
        The processed treatment matrix.
    X : array
        The processed covariate matrix.
    X_A : array
        The processed covariate matrix with the log library size.
    """
    if not isinstance(Y, pd.DataFrame):
        Y = np.asarray(Y)
    Y = np.minimum(Y, np.round(np.quantile(np.max(Y, 0), 0.999)))
    if not isinstance(A, pd.DataFrame):
        A = np.asarray(A)
    if A.ndim == 1:
        A = A[:, None]

    X = np.zeros((Y.shape[0], 0)) if X is None else np.asarray(X)        
    X_A = X if X_A is None else np.asarray(X_A)
    loglibsize = np.log2(np.sum(np.asarray(Y), axis=1))
    loglibsize = (loglibsize - np.mean(loglibsize)) / np.std(loglibsize, ddof=1)
    X_A = np.hstack((X_A, loglibsize[:, None]))

    intercept_col = np.ones((X.shape[0], 1)) if intercept else np.empty((X.shape[0], 0))
    X = np.hstack((intercept_col, X))
    X_A = np.hstack((intercept_col, X_A))

    return Y, A, X, X_A


def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    random.seed(seed)


def _filter_params(func, kwargs):
    '''
    Filter the parameters of a function.

    Parameters
    ----------
    func : function
        The function to filter the parameters.
    kwargs : dict
        The input parameters.

    Returns
    -------
    filtered_kwargs : dict
        The filtered parameters.
    '''
    if isinstance(func, dict):
        valid_params = func.keys()
    elif callable(func):
        valid_params = inspect.signature(func).parameters.keys()
    else:
        raise ValueError("The provided func is not a callable function, or a dict.")
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    
    return filtered_kwargs


class Early_Stopping():
    '''
    The early-stopping monitor.

    Parameters
    ----------
    tolerance : float
        Minimum absolute improvement in the metric to count as progress.
        Default 0 (disabled).  Can be combined with ``rel_tol``.
    rel_tol : float
        Minimum *relative* improvement in the metric to count as progress,
        i.e. the threshold is ``rel_tol * |best_metric|``.  Default 1e-4.
        This is scale-invariant and avoids the false-convergence problem that
        occurs with a fixed absolute threshold when the per-gene NLL is small
        (e.g. many lowly-expressed genes dilute the average NLL, making
        per-epoch improvements far smaller than any fixed absolute threshold).
    '''
    def __init__(self, warmup=25, patience=25, tolerance=0., rel_tol=1e-4,
                 is_minimize=True, **kwargs):
        self.warmup = warmup
        self.patience = patience
        self.tolerance = tolerance
        self.rel_tol = rel_tol
        self.is_minimize = is_minimize

        self.step = -1
        self.best_step = -1
        self.best_metric = np.inf

        if not self.is_minimize:
            self.factor = -1.0
        else:
            self.factor = 1.0
        self.info = None

    def __call__(self, metric):
        self.step += 1

        if self.step < self.warmup:
            return False

        # Threshold: max of absolute and relative tolerance.
        # rel_tol is scale-invariant (handles small per-gene NLL when p is large).
        if np.isfinite(self.best_metric):
            threshold = self.tolerance + self.rel_tol * abs(self.best_metric)
        else:
            threshold = 0.

        if self.factor * metric < self.factor * self.best_metric - threshold:
            self.best_metric = metric
            self.best_step = self.step
            return False
        elif self.step - self.best_step > self.patience:
            self.info = 'Best Epoch: %d. Best Metric: %f.' % (self.best_step, self.best_metric)
            return True
        else:
            return False

    def reset_state(self):
        self.best_step = self.step
        self.best_metric = np.inf



def comp_size_factor(counts, method='geomeans', lib_size=1e4, min_mean=2.0,
                     min_genes=100, **kwargs):
    '''
    Compute the size factors of the rows of the count matrix.

    Parameters
    ----------
    counts : array-like
        The input raw count matrix.
    method : str
        ``'geomeans'`` (default): median of each row's log-ratios to the
        per-gene geometric means, over the row's nonzero counts in genes
        with mean count at least ``min_mean``. ``'libsize'``: each row's
        total count. ``'scale'``: each row's total count divided by
        ``lib_size``, so that ``counts / size_factor`` has ``lib_size`` counts
        per row.
    lib_size : float
        The library size after normalization for ``'scale'``.
    min_mean : float
        For ``'geomeans'``, genes with a mean count below this are left out
        of the ratios; if fewer than ``min_genes`` genes pass, the
        ``min_genes`` genes with the highest mean are used. ``0`` uses every
        gene. A row with no counts in these genes gets its total count,
        rescaled by the median ratio of size factor to total count over the
        other rows.
    min_genes : int
        Minimum number of genes used by ``'geomeans'``.

    Returns
    -------
    size_factor : array-like
        The size factors of the rows, with geometric mean one for
        ``'geomeans'`` and ``'libsize'``. A row without counts gets zero.

    Notes
    -----
    The median of ratios resists composition changes: a perturbation or a
    latent factor that moves a minority of genes does not move the size
    factor. In sparse single-cell data, however, most nonzero counts are 1 or
    2, and their ratios capture only part of each cell's sequencing depth.
    The remainder stays in highly expressed genes, which scale fully with
    depth, and shifts all of them together whenever two groups of cells
    differ in average depth by chance; their test statistics are then too
    spread under the null. Restricting the ratios to genes with enough counts
    keeps the median's robustness while tracking depth.

    .. versionchanged:: 0.1.1
        ``'geomeans'`` uses genes with mean count at least ``min_mean``
        (previously every gene); ``'libsize'`` added; ``'scale'`` returns
        one value per row (previously one per column).
    '''
    counts = np.asarray(counts, dtype=np.float64)
    if method == 'libsize':
        totals = counts.sum(axis=1)
        positive = totals > 0
        size_factor = np.zeros_like(totals)
        if positive.any():
            log_totals = np.log(totals[positive])
            size_factor[positive] = np.exp(log_totals - np.mean(log_totals))
        return size_factor
    if method == 'geomeans':
        gene_mean = counts.mean(axis=0)
        keep = gene_mean >= min_mean
        if keep.sum() < min(min_genes, counts.shape[1]):
            keep = np.zeros(counts.shape[1], dtype=bool)
            keep[np.argsort(gene_mean)[::-1][:min_genes]] = True
        keep &= gene_mean > 0

        def _log_median_ratio(sub):
            log_sub = np.where(sub > 0, np.log(np.where(sub > 0, sub, 1.0)), np.nan)
            log_geo_means = np.nanmean(log_sub, axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN rows
                return np.nanmedian(log_sub - log_geo_means[None, :], axis=1)

        log_size_factor = _log_median_ratio(counts[:, keep])
        # A row with no counts in the selected genes falls back to its total
        # count, put on the scale of the other rows by their median offset
        # between log size factor and log total.
        log_totals = np.log(np.where(counts.sum(axis=1) > 0, counts.sum(axis=1), np.nan))
        missing = ~np.isfinite(log_size_factor)
        found = ~missing
        if missing.any() and found.any():
            shift = np.median(log_size_factor[found] - log_totals[found])
            log_size_factor[missing] = log_totals[missing] + shift
        finite = np.isfinite(log_size_factor)
        size_factor = np.zeros(counts.shape[0])
        size_factor[finite] = np.exp(log_size_factor[finite] - np.mean(log_size_factor[finite]))
    elif method == 'scale':
        size_factor = counts.sum(axis=1) / lib_size
    else:
        raise ValueError("Method must be in {'geomeans', 'libsize', 'scale'}.")

    return size_factor


def subsample_ctrl_cells(ctrl_idx, n_ctrl=2000, random_state=0):
    """Draw a fixed subsample of control cell indices.

    Called once before the batch loop in ``fit_gcate_batch`` / ``LFC_batch``.
    The same returned indices are reused for every batch so that all batches
    see the same reference distribution and warm-started U rows map 1-to-1.

    Parameters
    ----------
    ctrl_idx : array-like of int
        Row indices of all control cells in the full dataset.
    n_ctrl : int
        Number of ctrl cells to select (default 2 000).  If the pool is
        smaller, all are returned unchanged.
    random_state : int
        RNG seed for reproducibility.

    Returns
    -------
    ctrl_sel : ndarray of int, shape (min(len(ctrl_idx), n_ctrl),)
        Sorted ctrl cell indices.
    """
    ctrl_idx = np.asarray(ctrl_idx)
    if len(ctrl_idx) <= n_ctrl:
        return ctrl_idx
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(ctrl_idx, size=n_ctrl, replace=False))


def subsample_pert_cells(pert_idx, max_cells=2000, random_state=0):
    """Subsample perturbation cells to at most *max_cells*.

    The cap applies to pert cells only; ctrl cells are added on top by the
    caller.  Typical Perturb-seq datasets have a few hundred cells per
    perturbation, so the default cap of 2 000 is rarely active.

    Parameters
    ----------
    pert_idx : array-like of int
        Row indices of pert cells for this batch.
    max_cells : int or None
        Maximum number of pert cells to keep (default 2 000).  ``None`` keeps
        all.
    random_state : int
        RNG seed for reproducibility.

    Returns
    -------
    pert_sel : ndarray of int
        Sorted pert cell indices (len ≤ max_cells).
    """
    pert_idx = np.asarray(pert_idx)
    if max_cells is None or len(pert_idx) <= max_cells:
        return pert_idx
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(pert_idx, size=max_cells, replace=False))


def plot_r(df_r, c=1):
    '''
    Plot the results of the estimation of the number of latent factors.

    Parameters
    ----------
    df_r : DataFrame
        Results of the number of latent factors.
    c : float
        The constant factor for the complexity term.

    Returns
    -------
    fig : Figure
        The figure of the plot.
    '''
    
    
    fig = plt.figure(figsize=[18,6])
    host = host_subplot(121)
    par = host.twinx()

    host.set_xlabel("Number of factors $r$")
    host.set_ylabel("Deviance")
    # par.set_ylabel("$\nu$")


    p1, = host.plot(df_r['r'], df_r['deviance'], '-o', label="Deviance")
    p2, = par.plot(df_r['r'], df_r['nu']*c, '-o', label=r"$\nu$")


    host.set_xticks(df_r['r'])
    host.yaxis.get_label().set_color(p1.get_color())
    par.tick_params(axis='y', colors=p2.get_color(), labelsize=14)
    host.tick_params(axis='y', colors=p1.get_color(), labelsize=14)

    p1, = host.plot(df_r['r'], df_r['deviance']+df_r['nu']*c, '-o', label="JIC")
    host.legend(labelcolor="linecolor")


    host = host_subplot(122)
    par = host.twinx()
    host.set_xlabel("Number of factors $r$")
    par.set_ylabel(r"$\nu$")

    p1, = host.plot(df_r['r'].iloc[1:], -np.diff(df_r['deviance']), '-o', label='diff dev')
    p2, = par.plot(df_r['r'].iloc[1:], np.diff(df_r['nu'])*c,  '-o', label=r'diff $\nu$')

    host.legend(labelcolor="linecolor")
    host.set_xticks(df_r['r'].iloc[1:])
    par.set_ylim(*host.get_ylim())
    
    par.yaxis.get_label().set_color(p2.get_color())
    par.tick_params(axis='y', colors=p2.get_color(), labelsize=14)
    host.tick_params(axis='y', colors=p1.get_color(), labelsize=14)

    return fig


