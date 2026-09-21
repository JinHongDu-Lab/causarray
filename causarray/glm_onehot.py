"""Batched IRLS for GLM designs of the form ``[X | G]`` with one-hot ``G``.

causarray's GCATE initialisation and its propensity/outcome models fit, for
every gene, a Poisson or negative-binomial GLM on a design whose columns are a
few dense covariates ``X`` (intercept, library size, latent factors) followed
by many one-hot treatment indicators ``G`` (one column per perturbation, all
zero for control cells).  For ``a`` perturbations the generic batched solver
forms a dense ``(p, d, d)`` Hessian per IRLS step, which costs ``O(n p d^2)``
with ``d = d_X + a``; with ``a = 200`` this is hours even for a few thousand
cells.

Because the columns of ``G`` have disjoint supports, the per-gene Hessian is
block structured::

    H = [[ X'WX   X'WG ],        X'WG  : (d_X, a)   group-wise weighted sums of X
         [ G'WX   D    ]]        D     : diag(a)    group-wise weight sums

so each Newton step can be solved through the Schur complement of ``D`` at a
cost of ``O(n p (d_X^2 + a) + p a d_X^2)`` using BLAS matmuls and sparse
group aggregation, with no ``d^2`` term in the cell count.  The result is the
exact IRLS solution of the same GLM (up to a small ridge on the treatment
block that keeps groups with no counts finite), so it can replace both the
statsmodels gene-by-gene path and the dense batch path for such designs.

Public entry point: :func:`fit_glm_onehot`.  :func:`detect_onehot_block`
finds a maximal block of disjoint binary columns in a design matrix so that
:func:`causarray.gcate_glm.fit_glm_auto` can route to this solver.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sps

__all__ = ['fit_glm_onehot', 'detect_onehot_block']

_ETA_CLIP = 30.0


def detect_onehot_block(X, min_block=2):
    """Return indices of a maximal block of disjoint one-hot columns in ``X``.

    A column qualifies if it is binary (only 0 and 1) and its support does not
    overlap the support of any other selected column.  Columns are scanned from
    the smallest support upwards and greedily added, and the returned indices
    are sorted in their original column order.  Returns an empty array when fewer than
    ``min_block`` qualifying columns exist.
    """
    X = np.asarray(X)
    n, d = X.shape
    binary = [j for j in range(d) if np.all((X[:, j] == 0) | (X[:, j] == 1)) and X[:, j].any()]
    # Smallest supports first: one-hot treatment columns each cover a small
    # fraction of cells, whereas a binary covariate (e.g. sex) covers about
    # half of them and would otherwise block every group it overlaps.
    binary.sort(key=lambda j: int(X[:, j].sum()))
    taken = np.zeros(n, dtype=bool)
    block = []
    for j in binary:
        col = X[:, j] == 1
        if not np.any(taken & col):
            block.append(j)
            taken |= col
    if len(block) < min_block:
        return np.array([], dtype=int)
    return np.sort(np.asarray(block, dtype=int))


class _Deviance:
    """Per-gene deviance with the data-only terms precomputed.

    Poisson: ``2 * sum(y log y - y eta - y + mu)``; NB with size ``r``:
    ``2 * sum(y log y - y eta - (y + r) log(y + r) + (y + r) log(mu + r))``,
    where ``eta = log mu``.  Only ``log(mu + r)`` (NB) is evaluated per call, so a
    deviance evaluation costs one log over the matrix instead of three.
    """

    def __init__(self, Y, family, disp, const=None):
        self.family = family
        self.Y = Y
        if family != 'poisson':
            self.r = disp[None, :]
        if const is not None:
            self.const = const
            return
        with np.errstate(divide='ignore', invalid='ignore'):
            ylogy = np.where(Y > 0, Y * np.log(np.where(Y > 0, Y, 1.0)), 0.0)
        if family == 'poisson':
            self.const = (ylogy - Y).sum(axis=0)
        else:
            yr = Y + self.r
            self.const = (ylogy - yr * np.log(yr)).sum(axis=0)

    def subset(self, idx):
        """The same object restricted to genes ``idx`` (no recomputation)."""
        disp = None if self.family == 'poisson' else self.r[0, idx]
        return _Deviance(self.Y[:, idx], self.family, disp, const=self.const[idx])

    def total(self, eta, mu):
        """Deviance per gene, shape (p,)."""
        if self.family == 'poisson':
            return 2.0 * (self.const - (self.Y * eta).sum(axis=0) + mu.sum(axis=0))
        return 2.0 * (self.const - (self.Y * eta).sum(axis=0) + ((self.Y + self.r) * np.log(mu + self.r)).sum(axis=0))

    def residuals(self, eta, mu):
        """Deviance residuals, shape (n, p)."""
        Y = self.Y
        with np.errstate(divide='ignore', invalid='ignore'):
            ylogy = np.where(Y > 0, Y * np.log(np.where(Y > 0, Y, 1.0)), 0.0)
            if self.family == 'poisson':
                unit = 2.0 * (ylogy - Y * eta - (Y - mu))
            else:
                yr = Y + self.r
                unit = 2.0 * (ylogy - Y * eta - yr * np.log(yr) + yr * np.log(mu + self.r))
        return np.sign(Y - mu) * np.sqrt(np.maximum(unit, 0.0))


def fit_glm_onehot(Y, X, G, family='poisson', disp=None, offset=None, *,
                   max_iter=50, tol=1e-8, ridge=1e-6, ridge_group=1e-4,
                   clip_group=10.0, return_mu=True, verbose=False):
    """Fit ``p`` GLMs with design ``[X | G]`` by block-structured batched IRLS.

    Parameters
    ----------
    Y : (n, p) array
        Counts.
    X : (n, d_X) array
        Dense covariates (include the intercept here).
    G : (n, a) array or sparse matrix
        One-hot group indicators with disjoint supports; rows of zeros are
        cells that belong to no group (controls).
    family : {'poisson', 'nb'}
        With ``'nb'`` the size parameter ``disp`` (``r`` in ``NB(r, p)``, as
        returned by causarray's dispersion estimators) is held fixed.
    disp : (p,) array, optional
        NB size per gene.  Required for ``family='nb'``.
    offset : (n,) array, optional
        Log-scale offset (log size factors).
    max_iter, tol
        IRLS iterations and relative deviance tolerance.
    ridge, ridge_group
        L2 penalties on the covariate and group coefficients.  ``ridge_group``
        keeps a group whose cells have no counts for a gene finite; its
        coefficient then tends to ``-clip_group``.
    clip_group
        Bound on group coefficients (log scale), matching the sanity bound the
        statsmodels path applies before falling back to a regularised fit.
    return_mu
        Also return the fitted means ``mu`` (n, p).

    Returns
    -------
    B : (p, d_X + a) array
        Coefficients, covariates first, then groups in the column order of ``G``.
    mu : (n, p) array or None
        Fitted means including the offset.
    dev_resid : (n, p) array
        Deviance residuals.
    info : dict
        ``n_iter``, ``converged`` (p,) and ``deviance`` (p,).
    """
    Y = np.asarray(Y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    n, p = Y.shape
    dx = X.shape[1]
    G = sps.csr_matrix(G) if not sps.issparse(G) else G.tocsr()
    a = G.shape[1]
    if family not in ('poisson', 'nb'):
        raise ValueError("family must be 'poisson' or 'nb'")
    if family == 'nb':
        if disp is None:
            raise ValueError("family='nb' requires disp")
        disp = np.asarray(disp, dtype=np.float64).ravel()
        alpha = 1.0 / np.clip(disp, 1e-8, 1e8)          # NB variance mu + alpha mu^2
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=np.float64).ravel()
    GT = G.T.tocsr()                                     # (a, n)
    # X x X outer products per cell, flattened: (n, dx*dx); used for X'WX via one matmul.
    XX = (X[:, :, None] * X[:, None, :]).reshape(n, dx * dx)
    eye_x = ridge * np.eye(dx)

    # --- initialisation: constant eta per gene at the offset-adjusted mean ---
    mean_y = Y.mean(axis=0)
    mean_e = np.exp(off).mean()
    Bx = np.zeros((p, dx))
    has_intercept = np.flatnonzero(np.all(X == 1.0, axis=0))
    eta0 = np.log(np.maximum(mean_y, 1e-3) / mean_e)
    if has_intercept.size:
        Bx[:, has_intercept[0]] = eta0
        eta = off[:, None] + eta0[None, :]
    else:
        eta = off[:, None] + eta0[None, :]
    Ba = np.zeros((p, a))
    eta = np.clip(eta, -_ETA_CLIP, _ETA_CLIP)
    mu = np.exp(eta)
    devf = _Deviance(Y, family, disp if family == 'nb' else None)
    dev = devf.total(eta, mu)
    converged = np.zeros(p, dtype=bool)
    n_iter = 0
    active = np.arange(p)          # genes still iterating; converged genes are frozen

    for it in range(max_iter):
        n_iter = it + 1
        # Work only on the active genes.  Once most genes have converged the
        # per-iteration cost is proportional to the number still iterating,
        # instead of the full matrix for every iteration a single slow gene needs.
        if active.size < p:
            Ya, eta_a, mu_a = Y[:, active], eta[:, active], mu[:, active]
            Bx_a, Ba_a, dev_a = Bx[active], Ba[active], dev[active]
            devf_a = devf.subset(active)
            alpha_a = alpha[active] if family == 'nb' else None
        else:
            Ya, eta_a, mu_a, Bx_a, Ba_a, dev_a, devf_a = Y, eta, mu, Bx, Ba, dev, devf
            alpha_a = alpha if family == 'nb' else None
        pa = active.size
        # IRLS weights and working response (log link)
        if family == 'poisson':
            W = mu_a
        else:
            W = mu_a / (1.0 + alpha_a[None, :] * mu_a)
        Z = (eta_a - off[:, None]) + (Ya - mu_a) / mu_a       # working response minus offset
        WZ = W * Z
        # --- Hessian blocks ---
        C = (W.T @ XX).reshape(pa, dx, dx) + eye_x[None]   # X'WX  (pa, dx, dx)
        bX = (X.T @ WZ).T                                  # X'Wz  (pa, dx)
        D = np.asarray(GT @ W) + ridge_group               # G'WG  (a, pa) diagonal
        bA = np.asarray(GT @ WZ)                           # G'Wz  (a, pa)
        Bblk = np.empty((a, dx, pa))                       # X'WG  per group: (a, dx, pa)
        for j in range(dx):
            Bblk[:, j, :] = np.asarray(GT @ (W * X[:, j][:, None]))
        # --- Schur complement solve ---
        invD = 1.0 / D                                     # (a, pa)
        S = C - np.einsum('kip,kjp,kp->pij', Bblk, Bblk, invD, optimize=True)
        rhs = bX - np.einsum('kip,kp,kp->pi', Bblk, bA, invD, optimize=True)
        Bx_new = np.linalg.solve(S, rhs[:, :, None])[:, :, 0]
        Ba_new = ((bA - np.einsum('kip,pi->kp', Bblk, Bx_new)) * invD).T   # (pa, a)
        Ba_new = np.clip(Ba_new, -clip_group, clip_group)
        # --- step, with halving where the deviance increases ---
        eta_new = np.clip(off[:, None] + X @ Bx_new.T + np.asarray(G @ Ba_new.T), -_ETA_CLIP, _ETA_CLIP)
        mu_new = np.exp(eta_new)
        dev_new = devf_a.total(eta_new, mu_new)
        worse = dev_new > dev_a * (1 + 1e-9)
        step = 1.0
        for _ in range(8):
            if not np.any(worse):
                break
            step *= 0.5
            idx = np.flatnonzero(worse)                        # only recompute the genes that got worse
            Bx_h = Bx_a[idx] + step * (Bx_new[idx] - Bx_a[idx])
            Ba_h = Ba_a[idx] + step * (Ba_new[idx] - Ba_a[idx])
            eta_h = np.clip(off[:, None] + X @ Bx_h.T + np.asarray(G @ Ba_h.T), -_ETA_CLIP, _ETA_CLIP)
            mu_h = np.exp(eta_h)
            dev_h = devf_a.subset(idx).total(eta_h, mu_h)
            take = dev_h <= dev_new[idx]
            ti = idx[take]
            Bx_new[ti] = Bx_h[take]; Ba_new[ti] = Ba_h[take]
            eta_new[:, ti] = eta_h[:, take]; mu_new[:, ti] = mu_h[:, take]; dev_new[ti] = dev_h[take]
            worse = dev_new > dev_a * (1 + 1e-9)
        rel = np.abs(dev_new - dev_a) / np.maximum(np.abs(dev_a), 1e-8)
        # --- write back ---
        if pa < p:
            Bx[active] = Bx_new; Ba[active] = Ba_new
            eta[:, active] = eta_new; mu[:, active] = mu_new; dev[active] = dev_new
        else:
            Bx, Ba, eta, mu, dev = Bx_new, Ba_new, eta_new, mu_new, dev_new
        converged[active] = rel < tol
        active = np.flatnonzero(~converged)
        if verbose:
            print(f'  IRLS iter {n_iter}: converged {converged.mean():.3f}, median dev {np.median(dev):.3f}')
        if active.size == 0:
            break

    B = np.c_[Bx, Ba]
    dev_resid = devf.residuals(eta, mu)
    info = {'n_iter': n_iter, 'converged': converged, 'deviance': dev}
    return B, (mu if return_mu else None), dev_resid, info
