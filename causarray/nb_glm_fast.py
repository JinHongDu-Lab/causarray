"""Batch GLM fitting for causarray, backed by crispyx.

causarray's designs are always ``[covariates | one-hot treatments]``: a few
dense columns (intercept, covariates, latent factors) beside one indicator
column per perturbation, at most one of which is set for any cell.  crispyx
solves exactly that shape with :class:`crispyx.glm.StructuredGLMBatchFitter`
(the arrowhead Hessian is solved through the Schur complement of its diagonal
block, so the cost does not grow with the square of the number of treatments),
and covariate-only designs with :func:`crispyx.glm.fit_nb_glm_batch_auto`.
This module is the adaptor between the two conventions -- causarray's NB size
``r`` against crispyx's ``alpha = 1/r``, and causarray's counterfactual
``(Y_hat_0, Y_hat_1)`` tensors -- and nothing more.

Key functions:

- ``estimate_disp_fast``: batch dispersion estimation.
- ``fit_glm_fast``: batch Poisson/NB fitting, with counterfactual imputation.
- ``fit_glm_ondisk``: the same for an h5ad file too large to hold in memory.
"""

from __future__ import annotations

import logging
import warnings
from typing import Literal

import numpy as np
import scipy.sparse as sp

from causarray.utils import comp_size_factor, _filter_params, pprint

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

_SPARSE_WARN_GB: float = 4.0
"""Emit ResourceWarning when materialising sparse Y into dense float64 would
exceed this many gigabytes."""

_MIN_MU: float = 0.0
"""Floor on fitted means, passed to every crispyx fitter.

DESeq2 floors means at 0.5 to stabilise low-count genes; that floor biases
every gene below about one count per cell (median |dB| against statsmodels
0.1-0.2 on the sparse tail) and, in the imputation path, lifts ``Y_hat`` above
``thres_min = 0.01`` in :mod:`causarray.DR_learner` for genes whose true mean
is ~1e-3, which used to manufacture spurious effects.  causarray therefore
fits unfloored; crispyx clips the linear predictor to ``[-30, 20]`` regardless,
which is what keeps a diverging gene finite.
"""

_MAX_IRLS_ITER: int = 50
"""Cap on IRLS iterations handed to crispyx (causarray's ``maxiter`` default of
1000 is a statsmodels-era number; the batched solvers converge in well under
50 or not at all)."""


def _maybe_densify(Y) -> np.ndarray:
    """Return ``Y`` as a dense float64 array, warning if that is large."""
    if sp.issparse(Y):
        gb = Y.shape[0] * Y.shape[1] * 8 / 1e9
        if gb > _SPARSE_WARN_GB:
            warnings.warn(
                f"Materialising sparse Y as dense float64 costs {gb:.1f} GB.",
                ResourceWarning, stacklevel=3,
            )
        return np.asarray(Y.toarray(), dtype=np.float64)
    return np.asarray(Y, dtype=np.float64)


def _resolve_offset(Y, offset, kwargs) -> np.ndarray | None:
    """Normalise causarray's ``offset`` spellings to an ``(n,)`` array or None.

    ``True`` means "log size factors computed from ``Y``"; ``None`` and
    ``False`` both mean "no offset"; anything else is taken as the offset
    itself.
    """
    if offset is None or offset is False:
        return None
    if offset is True:
        sf = comp_size_factor(Y, **_filter_params(comp_size_factor, kwargs))
        empty = np.asarray(sf).ravel() <= 0
        if empty.any():
            # log(0) is -inf, which turns the whole row's fit into NaN without
            # saying why.  A cell with no counts carries no information here.
            raise ValueError(
                f"{int(empty.sum())} of {len(empty)} cells have no counts, so "
                "their size factor is zero and offset=True is undefined for "
                "them; drop them (or pass an explicit offset) before fitting."
            )
        return np.log(sf)
    return np.asarray(offset, dtype=np.float64).ravel()


def _is_onehot(G) -> bool:
    """True if ``G`` is a block of binary columns with disjoint supports."""
    G = np.asarray(G, dtype=np.float64)
    if G.ndim != 2 or G.shape[1] == 0:
        return False
    return bool(np.all((G == 0) | (G == 1)) and np.all(G.sum(axis=1) <= 1))


def _to_alpha(disp_glm) -> np.ndarray | None:
    """causarray's NB size ``r`` -> crispyx's dispersion ``alpha = 1/r``."""
    if disp_glm is None:
        return None
    return 1.0 / np.clip(np.asarray(disp_glm, dtype=np.float64).ravel(), 1e-8, 1e8)


def _to_size(alpha) -> np.ndarray:
    """crispyx's ``alpha`` -> causarray's NB size ``r = 1/alpha``."""
    r = 1.0 / np.clip(np.asarray(alpha, dtype=np.float64).ravel(), 1e-8, 1e8)
    r[~np.isfinite(r)] = 1.0
    return r


def _constant_columns(X: np.ndarray) -> np.ndarray:
    """Indices of columns that do not vary (the intercept, typically)."""
    return np.flatnonzero(np.all(X == X[0, :], axis=0))


# ---------------------------------------------------------------------------
# Dispersion
# ---------------------------------------------------------------------------


def estimate_disp_fast(
    Y,
    X: np.ndarray,
    A: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    method: Literal["moments"] = "moments",
    **kwargs,
) -> np.ndarray:
    """Batch method-of-moments NB dispersion.

    Parameters
    ----------
    Y : (n, p) array or sparse matrix
        Counts.
    X : (n, d) array
        Covariates, including the intercept.
    A : (n, a) array, optional
        Treatment indicators.  They stay in the dispersion model: dropping
        them lowers the estimated size parameter by about 13% on the Replogle
        screen and moves the downstream results (measured 2026-09-21).
    offset : (n,) array, optional
        Log-scale offset.
    method : str
        Only ``'moments'`` is supported; the argument is kept because callers
        pass it.

    Returns
    -------
    disp : (p,) array
        NB size parameter ``r`` (causarray's convention, ``r = 1/alpha``).
    """
    from crispyx.glm import fit_nb_glm_batch_auto

    if method != "moments":
        raise ValueError(f"method must be 'moments', got {method!r}")

    design = np.asarray(X, dtype=np.float64)
    if A is not None:
        A = np.asarray(A, dtype=np.float64)
        design = np.c_[design, A[:, None] if A.ndim == 1 else A]

    result = fit_nb_glm_batch_auto(
        design,
        _maybe_densify(Y),
        offset=None if offset is None else np.asarray(offset, dtype=np.float64).ravel(),
        protected_columns=_constant_columns(design),
        max_iter=10,
        poisson_init_iter=5,
        dispersion_method="moments",
        min_mu=_MIN_MU,
    )
    return _to_size(np.clip(result.dispersion, 1e-8, 100.0))


# ---------------------------------------------------------------------------
# Batch fitting
# ---------------------------------------------------------------------------


def _split_design(X: np.ndarray, A: np.ndarray | None):
    """Split a design into a covariate block and a one-hot group block.

    Returns ``(X_cov, G, cov_idx, grp_idx)`` where ``cov_idx`` and ``grp_idx``
    index the columns of the caller's design ``[X | A]``, so that coefficients
    can be returned in the caller's column order.  ``G`` is None when no
    group block is usable and the whole design must go to the dense solver.
    """
    from crispyx.glm import detect_onehot_block

    if A is not None:
        width = X.shape[1] + A.shape[1]
        if _is_onehot(A):
            return X, A, np.arange(X.shape[1]), np.arange(X.shape[1], width)
        return np.c_[X, A], None, np.arange(width), np.empty(0, dtype=int)

    # GCATE hands the treatments inside X (the design is [X | A | U]); find
    # them so that a wide screen does not fall onto the dense path.
    grp_idx = detect_onehot_block(X, min_block=2)
    if grp_idx.size < 2:
        return X, None, np.arange(X.shape[1]), np.empty(0, dtype=int)
    cov_idx = np.setdiff1d(np.arange(X.shape[1]), grp_idx)
    return X[:, cov_idx], X[:, grp_idx], cov_idx, grp_idx


def _counterfactual_dense(B, X_test, d, a, offsets_test, dtype):
    """``(Y_hat_0, Y_hat_1)`` from a dense ``[X | A]`` coefficient matrix."""
    from crispyx._irls import ETA_MAX, ETA_MIN

    eta_0 = X_test @ B[:, :d].T
    if offsets_test is not None:
        eta_0 = eta_0 + offsets_test[:, None]
    baseline = np.exp(np.clip(eta_0, ETA_MIN, ETA_MAX))
    Yhat_0 = np.repeat(baseline[:, :, None], a, axis=2).astype(dtype, copy=False)
    Yhat_1 = np.exp(
        np.clip(eta_0[:, :, None] + B[None, :, d:], ETA_MIN, ETA_MAX)
    ).astype(dtype, copy=False)
    return Yhat_0, Yhat_1


def _impute_dtype(n_test, p, a, mem_limit_gb):
    """float32 when the two imputation tensors would exceed ``mem_limit_gb``."""
    gb = n_test * p * a * 2 * 8 / 1e9
    if mem_limit_gb is not None and gb > mem_limit_gb:
        warnings.warn(
            f"Imputation arrays ({gb:.1f} GB as float64) exceed "
            f"mem_limit_gb={mem_limit_gb} GB; using float32 to halve peak memory.",
            ResourceWarning, stacklevel=3,
        )
        return np.float32
    return np.float64


def fit_glm_fast(
    Y,
    X: np.ndarray,
    A: np.ndarray | None = None,
    family: str = "gaussian",
    disp_family: str = "poisson",
    disp_glm: np.ndarray | None = None,
    impute: bool | np.ndarray = False,
    offset: np.ndarray | bool | None = None,
    offset_test: np.ndarray | None = None,
    shrinkage: bool = False,
    alpha: float = 1e-4,
    maxiter: int = 1000,
    thres_disp: float = 100.0,
    n_jobs: int = -3,
    random_state: int = 0,
    verbose: bool = False,
    mem_limit_gb: float | None = None,
    **kwargs,
):
    """Batch Poisson/NB GLM fitting; a drop-in replacement for ``fit_glm``.

    The design ``[X | A]`` is fitted jointly for every gene, by crispyx's
    structured solver when it carries a block of one-hot treatment indicators
    and by its dense batch fitter otherwise.  Parameters and returns are
    identical to :func:`causarray.gcate_glm.fit_glm`; ``family='gaussian'``
    is delegated to it, as are ``shrinkage`` fits.

    Measured on the Perturb-seq tutorial (29 perturbations, 2,926 cells,
    3,221 genes, 2026-09-21): 10.7 s against 46.8 s for the statsmodels pool,
    tau correlation 0.9987 against it, 7,460 of 7,468 / 7,482 discoveries
    shared.
    """
    np.random.seed(random_state)

    if family not in ("gaussian", "poisson", "nb"):
        raise ValueError("Family not recognized")

    if family == "gaussian" or shrinkage:
        from causarray.gcate_glm import fit_glm as _fit_glm_orig
        return _fit_glm_orig(
            Y, X, A=A, family=family, disp_family=disp_family,
            disp_glm=disp_glm, impute=impute, offset=offset,
            offset_test=offset_test, shrinkage=shrinkage, alpha=alpha,
            maxiter=maxiter, thres_disp=thres_disp, n_jobs=n_jobs,
            random_state=random_state, verbose=verbose,
            mem_limit_gb=mem_limit_gb, **kwargs,
        )

    from crispyx._irls import ETA_MAX, ETA_MIN, Deviance
    from crispyx.glm import NBGLMBatchFitter, StructuredGLMBatchFitter

    X = np.asarray(X, dtype=np.float64)
    n, p = Y.shape
    d = X.shape[1]
    if A is not None:
        A = np.asarray(A, dtype=np.float64)
        if A.ndim == 1:
            A = A[:, None]
    a = 1 if A is None else A.shape[1]

    Y_float = _maybe_densify(Y)
    offsets = _resolve_offset(Y, offset, kwargs)
    offset_arr = np.zeros(n) if offsets is None else offsets

    do_impute = impute is not False and A is not None
    X_test = impute if isinstance(impute, np.ndarray) else X
    if do_impute:
        n_test = X_test.shape[0]
        if offset_test is not None and np.asarray(offset_test).shape[0] == n_test:
            offsets_test = np.asarray(offset_test, dtype=np.float64).ravel()
        elif offsets is not None and offsets.shape[0] == n_test:
            offsets_test = offsets
        else:
            offsets_test = None
        dtype = _impute_dtype(n_test, p, a, mem_limit_gb)

    X_cov, G, cov_idx, grp_idx = _split_design(X, A)

    if verbose:
        pprint.pprint(
            "Fitting {} GLM ({})...".format(
                family,
                "structured, {} covariates + {} treatments".format(
                    X_cov.shape[1], 0 if G is None else G.shape[1])
                if G is not None else "dense, {} columns".format(X_cov.shape[1]),
            )
        )

    if G is not None:
        fitter = StructuredGLMBatchFitter(
            X_cov, G, offset=offset_arr, family=family,
            max_iter=min(maxiter, _MAX_IRLS_ITER), min_mu=_MIN_MU,
            dispersion_method="moments",
        )
        result = fitter.fit_batch(
            Y_float, dispersion=_to_alpha(disp_glm), return_mu=not do_impute,
        )
        B = np.empty((p, cov_idx.size + grp_idx.size))
        B[:, cov_idx] = result.coef[:, : cov_idx.size]
        B[:, grp_idx] = result.coef[:, cov_idx.size :]
        if do_impute:
            baseline, per_group = fitter.counterfactual_means(
                result, design=X_test,
                offset=np.zeros(n_test) if offsets_test is None else offsets_test,
            )
            Yhat = (
                np.repeat(baseline[:, :, None], a, axis=2).astype(dtype, copy=False),
                per_group.astype(dtype, copy=False),
            )
        else:
            Yhat = result.mu
        dispersion, dev_resid = result.dispersion, result.dev_resid
    else:
        fitter = NBGLMBatchFitter(
            X_cov, offset=offset_arr, family=family,
            max_iter=min(maxiter, _MAX_IRLS_ITER), poisson_init_iter=5,
            dispersion_method="moments", min_mu=_MIN_MU,
        )
        result = fitter.fit_batch(Y_float)
        B = result.coef
        eta = offset_arr[:, None] + X_cov @ B.T
        np.clip(eta, ETA_MIN, ETA_MAX, out=eta)
        mu = np.exp(eta)
        dispersion = result.dispersion
        dev_resid = Deviance(
            Y_float, family, dispersion if family == "nb" else None
        ).residuals(eta, mu)
        Yhat = (
            _counterfactual_dense(B, X_test, d, a, offsets_test, dtype)
            if do_impute else mu
        )

    disp_out = _to_size(dispersion) if family == "nb" else disp_glm
    return B, Yhat, disp_out, offsets, dev_resid


# ---------------------------------------------------------------------------
# On-disk NB-GLM fitting
# ---------------------------------------------------------------------------


def fit_glm_ondisk(
    path: str,
    perturbation_col: str = "perturbation",
    control_label: str = "control",
    target_label: str | None = None,
    gene_indices: np.ndarray | None = None,
    covariate_columns: list[str] | None = None,
    chunk_size: int = 2048,
    max_iter: int = 25,
    verbose: bool = False,
):
    """On-disk NB-GLM fitting: read an h5ad in chunks, then ``fit_glm_fast``.

    Parameters
    ----------
    path : str
        Path to the h5ad file.
    perturbation_col : str
        Column in obs containing perturbation labels.
    control_label : str
        Label for control cells.
    target_label : str or None
        Label for the target perturbation.  If None, all non-control
        perturbations are compared to control.
    gene_indices : array or None
        Indices of genes to fit.  If None, all genes are used.
    covariate_columns : list or None
        Additional covariate columns from obs.
    chunk_size : int
        Number of cells per chunk for streaming.
    max_iter : int
        Maximum IRLS iterations.
    verbose : bool
        Print progress.

    Returns
    -------
    B : (p, d) array
        Coefficient matrix.
    Yhat : (n, p) array
        Fitted values.  ``n`` counts only the cells that carry at least one
        count among the genes read; the rest are dropped with a warning,
        because their size factor is undefined.
    disp : (p,) array
        Dispersion parameters.
    offsets : (n,) array
        Log size factors.
    resid_deviance : (n, p) array
        Deviance residuals.
    """
    import anndata as ad
    from crispyx.data import read_backed

    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs
    n_cells, n_genes_total = adata.shape

    perturbations = obs[perturbation_col].values
    if target_label is not None:
        mask = np.isin(perturbations, [control_label, target_label])
    else:
        mask = np.ones(n_cells, dtype=bool)

    cell_indices = np.where(mask)[0]
    n = len(cell_indices)
    pert_labels = perturbations[cell_indices]
    A_vec = (pert_labels != control_label).astype(np.float64)

    if gene_indices is None:
        gene_indices = np.arange(n_genes_total)
    p = len(gene_indices)

    if verbose:
        logger.info(f"On-disk NB-GLM: {n} cells x {p} genes")

    Y = np.zeros((n, p), dtype=np.float64)
    backed = read_backed(path)
    try:
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            idx = cell_indices[start:end]
            chunk = backed.X[idx][:, gene_indices]
            if sp.issparse(chunk):
                chunk = np.asarray(chunk.toarray(), dtype=np.float64)
            else:
                chunk = np.asarray(chunk, dtype=np.float64)
            Y[start:end] = chunk
    finally:
        backed.file.close()

    # Size factors come from the genes actually read, so a cell with no counts
    # among them has no offset and would poison its whole row with -inf.  Those
    # cells carry no information about these genes; drop them.
    keep = Y.sum(axis=1) > 0
    if not keep.all():
        warnings.warn(
            f"{int((~keep).sum())} of {n} cells have no counts among the "
            f"{p} genes read and were dropped before fitting.",
            RuntimeWarning, stacklevel=2,
        )
        Y = Y[keep]
        A_vec = A_vec[keep]
        n = int(keep.sum())

    offsets = np.log(comp_size_factor(Y))

    # Covariates are appended after the intercept so that fit_glm_fast
    # receives a covariate-only X and the treatment vector as A.
    if covariate_columns:
        cov_data = np.column_stack(
            [obs[col].values[cell_indices].astype(np.float64) for col in covariate_columns]
        )
        X_cov = np.column_stack([np.ones(n), cov_data])
    else:
        X_cov = np.ones((n, 1))

    B, Yhat, disp, _, resid_dev = fit_glm_fast(
        Y, X_cov, A=A_vec[:, None], family="nb", offset=offsets,
        maxiter=max_iter, verbose=verbose,
    )

    adata.file.close()
    return B, Yhat, disp, offsets, resid_dev
