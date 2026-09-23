from causarray.gcate_likelihood import *
from causarray.utils import *
import causarray.gcate_glm as _gcate_glm  # module-qualified so _USE_FAST_BACKEND changes take effect at call time
import scipy as sp
from scipy.sparse.linalg import svds
from joblib import Parallel, delayed
from collections import defaultdict


@njit
def project_norm_ball(X, radius):
    """
    Projects a vector X to the norm ball of a given radius.

    Parameters:
        X (ndarray): The vector to be projected.
        radius (float): The radius of the norm ball.

    Returns:
        ndarray: The projected matrix.
    """
    norms = np.linalg.norm(X, ord=np.inf)
    if norms > radius:
        X *= radius / norms

    return X



@njit
def line_search(Y, A, x0, g, d, family, nuisance, Ys, thres_disp, start,
                lam, alpha=1., beta=0.5, max_iters=100, tol=1e-3, 
                ):
    """
    Performs line search to find the step size that minimizes a given function.
    
    Parameters:
        f (callable): A scalar-valued function of a vector argument.
        x0 (ndarray): The starting point for the line search.
        g (ndarray): The search direction vector.
        alpha (float, optional): The initial step size. Default is 10.
        beta (float, optional): The shrinkage factor used to reduce the step size. Default is 0.5.
        max_iters (int, optional): The maximum number of iterations. Default is 100.
        tol (float, optional): The tolerance for the step size. Default is 1e-3.
    
    Returns:
        float: The step size that minimizes the function along the search direction.
    """

    # Evaluate the function at the starting point.
    f0 = nll(Y, A, x0, family, nuisance, Ys, thres_disp)  + np.mean(lam * np.abs(x0[start:d]))
    
    # Initialize the step size.    
    alpha = type_f(alpha)
    beta = type_f(beta)
    t = alpha
    norm_g = np.linalg.norm(g)

    # Iterate until the maximum number of iterations is reached or the step size is small enough.
    for i in range(max_iters):
        
        # Compute the new point.
        x1 = x0 - t*g
        x1[start:d] = np.sign(x1[start:d]) * np.maximum(np.abs(x1[start:d]) - lam, type_f(0.))
        
        # Evaluate the function at the new point.
        f1 = nll(Y, A, x1, family, nuisance, Ys, thres_disp) + np.mean(lam * np.abs(x1[start:d]))
        if i==0:
            f01 = f1

        # Check if the function has decreased sufficiently.
        if f1 < f0 - tol*t*norm_g:
            return x1
        if t<1e-4:
            break
        t *= beta

    # Return the maximum step size.
    if f01<1.1*f1:
        x1 = x0 - alpha*g
        x1[start:d] = np.sign(x1[start:d]) * np.maximum(np.abs(x1[start:d]) - lam, type_f(0.))
    return x1






@njit(parallel=True)
def update(Y, A, B, d, lam, P1, P2,
          family, nuisance, Ys, thres_disp, a, C,
          alpha, beta, max_iters, tol):
    """Single alternating-minimisation step (no per-entity convergence mask).

    ``P1`` and ``P2`` follow the thin-Q convention used everywhere else in
    ``alter_min`` — they are the QR factor of the relevant design block,
    not the full ``(I − QQᵀ)`` projection matrix.  Projections are applied
    implicitly as ``v -= Q @ (Q.T @ v)`` so that the caller can hand in
    ``(n, d_X)`` / ``(p, r)`` arrays instead of dense ``(n, n)`` / ``(p, p)``
    matrices.

    Functionally equivalent to ``update_with_mask`` with all-True masks and
    a uniform ``alpha`` step size; kept around for the T4 parity test.
    """
    n, p = Y.shape

    g = grad_cells(Y, A, B, family, nuisance, thres_disp)
    g[:, :d] = 0.
    for i in prange(n):
        g[i, d:] = project_norm_ball(g[i, d:], 2*C)

        A[i, :] = line_search(Y[i, :], B, A[i, :], g[i, :], d,
                              family, nuisance[0], Ys[i, :], thres_disp, 0,
                              type_f(0.), alpha, beta, max_iters, tol)
    if P1 is not None:
        # Implicit projection: (I - Q @ Q.T) @ v = v - Q @ (Q.T @ v)
        # P1 is the thin Q factor (n, d_X), not the full (n, n) matrix.
        A[:, d:] -= P1 @ (P1.T @ A[:, d:])

    g = grad_genes(Y, A, B, family, nuisance, thres_disp)
    if P1 is not None:
        g[:, :d] = 0.
    elif P2 is not None:
        # Implicit projection: (I - Q @ Q.T) @ v = v - Q @ (Q.T @ v)
        # P2 is the thin Q factor (p, r), not the full (p, p) matrix.
        g[:, d-a:d] -= P2 @ (P2.T @ g[:, d-a:d])
        g[:, d:] = 0.
        g[:, :d-a] = 0.

    for j in prange(p):
        if P2 is None:
            g[j, d:] = project_norm_ball(g[j, d:], 2*C)
        else:
            g[j, :d] = project_norm_ball(g[j, :d], 2*C)

        B[j, :] = line_search(Y[:, j], A, B[j, :], g[j, :], d,
                              family, nuisance[:,j], Ys[:, j], thres_disp, d-a,
                              lam[j,:], alpha, beta, max_iters, tol
                             )

    if P2 is None:
        B[:, d:] = np.clip(B[:, d:], -10., 10.)
    func_val = nll_mat(Y, A, B, family, nuisance, Ys, thres_disp)

    return func_val, A, B


@njit(parallel=True)
def update_with_mask(Y, A, B, d, lam, P1, P2,
                     family, nuisance, Ys, thres_disp, a, C,
                     alpha, beta, max_iters, tol,
                     gene_active, cell_active, alpha_gene):
    """update() extended with per-gene and per-cell convergence masks.

    Parameters
    ----------
    gene_active : (p,) bool array
        Only genes where gene_active[j] == True get their B row updated.
    cell_active : (n,) bool array
        Only cells where cell_active[i] == True get their A row updated.
    alpha_gene : (p,) float64 array
        Per-gene initial step size for the backtracking line search.
        Use alpha_gene[j] = alpha for uniform step sizes.
    """
    n, p = Y.shape

    # ---- A-step (update cell latent factors) ----
    g = grad_cells(Y, A, B, family, nuisance, thres_disp)
    g[:, :d] = 0.
    for i in prange(n):
        if cell_active[i]:
            g[i, d:] = project_norm_ball(g[i, d:], 2*C)
            A[i, :] = line_search(Y[i, :], B, A[i, :], g[i, :], d,
                                  family, nuisance[0], Ys[i, :], thres_disp, 0,
                                  type_f(0.), alpha, beta, max_iters, tol)
    if P1 is not None:
        # Implicit projection: (I - Q @ Q.T) @ v = v - Q @ (Q.T @ v)
        # P1 is the thin Q factor (n, d_X), not the full (n, n) matrix.
        A[:, d:] -= P1 @ (P1.T @ A[:, d:])

    # ---- B-step (update gene coefficients) ----
    g = grad_genes(Y, A, B, family, nuisance, thres_disp)
    if P1 is not None:
        g[:, :d] = 0.
    elif P2 is not None:
        # Implicit projection: (I - Q @ Q.T) @ v = v - Q @ (Q.T @ v)
        # P2 is the thin Q factor (p, r), not the full (p, p) matrix.
        g[:, d-a:d] -= P2 @ (P2.T @ g[:, d-a:d])
        g[:, d:] = 0.
        g[:, :d-a] = 0.

    for j in prange(p):
        if gene_active[j]:
            if P2 is None:
                g[j, d:] = project_norm_ball(g[j, d:], 2*C)
            else:
                g[j, :d] = project_norm_ball(g[j, :d], 2*C)
            B[j, :] = line_search(Y[:, j], A, B[j, :], g[j, :], d,
                                  family, nuisance[:, j], Ys[:, j], thres_disp, d-a,
                                  lam[j, :], alpha_gene[j], beta, max_iters, tol)

    if P2 is None:
        B[:, d:] = np.clip(B[:, d:], -10., 10.)
    func_val = nll_mat(Y, A, B, family, nuisance, Ys, thres_disp)

    return func_val, A, B


def alter_min(
    Y, r, X=None, P1=None, P2=None, A=None, B=None,
    kwargs_glm={}, kwargs_ls={}, kwargs_es={}, 
    lam=0., a=None, verbose=False, thres_disp=100.):
    '''
    Alternative minimization of latent factorization for generalized linear models.

    Parameters
    ----------
    Y : array-like, shape (n, p)
        Response matrix.
    r : int
        The number of latent factors.
    X : array-like, shape (n, d)
        Observed covariate matrix.
    P1 : array-like, shape (n, n)
        The projection matrix for the rows.
    P2 : array-like, shape (p, p)
        The projection matrix onto the orthogonal column space of Gamma.
    A : array-like, shape (n, d+r)
        The initial matrix for the covariate and latent factors. If None, initialize internally.
    B : array-like, shape (p, d+r)
        The initial matrix for the covariate and latent coefficients. If None, initialize internally.
    kwargs_glm : dict
        Keyword arguments for the generalized linear model.
    kwargs_ls : dict
        Keyword arguments for the line search.
    kwargs_es : dict
        Keyword arguments for the early stopping.
    lam : float
        The regularization parameter for the l1 norm of the coefficients.
    a : int
        The number of columns to be regularized. Assume the last 'a' columns of the covariates are the regularized coefficients. If 'a' is None, it is set to be 'd-offset' by default.
    verbose : bool
        The indicator of whether to print the progress.
    thres_disp : float
        The threshold for the dispersion parameter.

    Returns
    -------
    res : dict
        A dictionary containing the information of the optimization, including 
            'A': the matrix (n, d+r) for the covariate and updated latent factors,
            'B': the matrix (p, d+r) for the updated covariate and latent coefficients,
            'U': the matrix (n, r) for the estimated confouders.
    '''

    n, p = Y.shape
    d = X.shape[1]
    
    assert d>0
    if verbose:
        pprint.pprint({'n':n,'p':p,'d':d,'r':r})

    kwargs_glm = {**{'family':'poisson', 'disp_glm':np.ones(p), 'size_factor':np.ones(n)
        }, **kwargs_glm}
    kwargs_ls = {**{
        'alpha': 0.1, 'beta': 0.5, 'max_iters': 20, 'tol': 1e-4, 'C': None,
        # Convergence mask parameters (G1-G3)
        'tol_gene': 1e-4,          # per-gene B-row step-norm threshold; 0 = disabled
        'tol_cell': 1e-4,          # per-cell A-row step-norm threshold; 0 = disabled
        'recheck_interval': 10,    # re-activate all entities every N iters
        'sparsity_threshold': 0.5, # zero-fraction above which gene is considered sparse
        'sparsity_boost': 2.0,     # alpha multiplier for sparse genes
        'warmup_iters': 0,         # B-only warm-up iterations for sparse genes after init
    }, **kwargs_ls}
    # Default ``max_iters=50`` (down from 500 in v0.0.5).  ``rel_tol=2e-4``
    # combined with ``patience=5`` reliably short-circuits the loop once the
    # NLL has plateaued in practice; if a user needs a tighter convergence
    # bound they can override via ``kwargs_es_1`` / ``kwargs_es_2``.  See
    # CHANGELOG → Changed for the migration note.
    kwargs_es = {**{'max_iters':50, 'warmup':0, 'patience':5, 'tolerance':0., 'rel_tol':2e-4}, **kwargs_es}

    if kwargs_es['max_iters'] == 0:
        return A, B, {'n_iter':0, 'func_val':0., 'resid':0., 'hist':[0.], 'kwargs_glm':kwargs_glm, 'kwargs_ls':kwargs_ls, 'kwargs_es':kwargs_es}
    
    family, nuisance, size_factor = kwargs_glm['family'], kwargs_glm['disp_glm'].astype(type_f), kwargs_glm['size_factor'].astype(type_f)
    nuisance = nuisance.reshape(1,-1)
    size_factor = size_factor.reshape(-1,1)

    if kwargs_ls['C'] is None:
        kwargs_ls['C'] = 1e3 if family=='nb' else 1e5
    C = kwargs_ls['C']

    if P1 is True:
        Q, _ = sp.linalg.qr(X, mode='economic')
        # Store only the thin Q factor (n, d_X) instead of the full (n, n)
        # projection matrix.  Memory: O(n*d_X) vs O(n^2).  For Adamson
        # (n=29963, d_X=6) this saves ~7.2 GB.  The projection
        # (I - Q @ Q.T) @ v is applied implicitly as v -= Q @ (Q.T @ v).
        P1 = Q.astype(type_f)
            
    if a is None:
        a = d

    # Compute gene sparsity before normalisation (zeros are preserved by /size_factor)
    gene_sparsity = (Y == 0).mean(axis=0).astype(type_f)  # (p,)

    # initialization for Theta = A @ B^T    
    if A is None:
        if verbose:
            pprint.pprint('Estimating initial latent variables with GLMs...')
        res_glm = _gcate_glm.fit_glm_auto(Y, X, offset=np.log(size_factor[:,0]), family=family, disp_glm=nuisance[0], maxiter=100, verbose=verbose)
        u, s, vt = svds(res_glm[-1], k=r)

        if u.shape[1]<r:
            raise ValueError(f'The number of latent factors is larger than the rank of deviance residuals ({u.shape[1]}). Try to decrease the value of r.')
        
        # Implicit projection: (I - Q @ Q.T) @ u = u - Q @ (Q.T @ u)
        # P1 is now the thin Q factor (n, d_X), not the full (n, n) matrix.
        u_proj = u - P1 @ (P1.T @ u) if P1 is not None else u
        A = np.c_[X, u_proj]
    else:
        assert A.shape[1] == d+r

    if B is None:
        if verbose:
            pprint.pprint('Estimating initial coefficients with GLMs...')
        
        B = _gcate_glm.fit_glm_auto(Y, A, offset=np.log(size_factor[:,0]), family=family, disp_glm=nuisance[0], maxiter=100, verbose=verbose)[0]
        
        E = A[:, -r:] @ B[:, -r:].T
        u, s, vh = sp.sparse.linalg.svds(E, k=r)        
        A[:, d:] = u * s[None,:]**(1/2)
        B[:, d:] = vh.T * s[None,:]**(1/2)
        del E, u, s, vh


    if P2 is not None:
        P2 = P2.astype(type_f)
        # Implicit projection: v -= Q @ (Q.T @ v) where P2 is the thin Q factor.
        B[:, d-a:d] -= P2 @ (P2.T @ B[:, d-a:d])
    

    Y = Y.astype(type_f)
    Y /= size_factor
    Ys = - gammaln_nb(Y+1)
    if family=='nb':
        Ys += np.where(nuisance > thres_disp, 0., gammaln_nb(nuisance+Y) - gammaln_nb(nuisance))    
    
    A = A.astype(type_f)
    B = B.astype(type_f)
    
    lam = type_f(lam)
    weights = lam / (np.abs(B[:, d-a:d]) + 1e-6)
    
    assert ~np.any(np.isnan(A))
    assert ~np.any(np.isnan(B))

    # ---- Convergence-mask parameters ----
    tol_gene        = type_f(kwargs_ls['tol_gene'])
    tol_cell        = type_f(kwargs_ls['tol_cell'])
    recheck_interval= int(kwargs_ls['recheck_interval'])
    sparsity_threshold = float(kwargs_ls['sparsity_threshold'])
    sparsity_boost  = float(kwargs_ls['sparsity_boost'])
    warmup_iters    = int(kwargs_ls['warmup_iters'])

    # Per-gene initial step sizes — boosted for sparse genes (G3)
    alpha_gene = np.full(p, kwargs_ls['alpha'], dtype=type_f)
    sparse_mask = gene_sparsity > sparsity_threshold
    if np.any(sparse_mask):
        alpha_gene[sparse_mask] *= (
            type_f(1.0) + type_f(sparsity_boost) * gene_sparsity[sparse_mask]
        )

    # Convergence masks (G1, G2)
    gene_active = np.ones(p, dtype=np.bool_)
    cell_active = np.ones(n, dtype=np.bool_)

    t = 0
    func_val_pre = (nll_mat(Y, A, B, family, nuisance, Ys, thres_disp) + np.sum(np.abs(B[:,d-a:d]) * weights)) / p
    func_val = func_val_pre

    # ---- Sparse-gene B-only warm-up (G6) ----
    if warmup_iters > 0 and np.any(sparse_mask):
        wu_gene_active = sparse_mask.astype(np.bool_)
        wu_cell_active = np.zeros(n, dtype=np.bool_)  # skip A-step during warmup
        for _ in range(warmup_iters):
            _, A, B = update_with_mask(
                Y, A, B, d, weights, P1, P2,
                family, nuisance, Ys, thres_disp, a, C,
                kwargs_ls['alpha'], kwargs_ls['beta'], kwargs_ls['max_iters'], kwargs_ls['tol'],
                wu_gene_active, wu_cell_active, alpha_gene,
            )
        # Refresh ``func_val_pre`` so ``hist[0]`` reflects the post-warmup
        # objective.  Without this, the first main-loop iteration's improvement
        # appears to include all of the warm-up's NLL drop (creating a one-step
        # cliff in the hist plot and misattributing the warm-up gain).
        func_val_pre = (nll_mat(Y, A, B, family, nuisance, Ys, thres_disp) + np.sum(np.abs(B[:,d-a:d]) * weights)) / p
        func_val = func_val_pre

    kwargs_ls['alpha'] = kwargs_ls['alpha']
    if verbose:
        pprint.pprint({'kwargs_glm':kwargs_glm,'kwargs_ls':kwargs_ls,'kwargs_es':kwargs_es}, compact=True)
    pprint.pprint(f'Fitting GCATE (step {2 if P1 is None else 1})...')
    hist = [func_val_pre]
    es = Early_Stopping(**kwargs_es)
    total_gene_updates = 0
    total_cell_updates = 0
    with tqdm(np.arange(kwargs_es['max_iters']), disable=not verbose) as pbar:
        for t in pbar:
            # Save previous state for mask update (only when masking is active)
            use_mask = tol_gene > 0 or tol_cell > 0
            if use_mask:
                B_prev = B.copy()
                A_prev_latent = A[:, d:].copy()

            func_val, A, B = update_with_mask(
                Y, A, B, d, weights, P1, P2,
                family, nuisance, Ys, thres_disp, a, C,
                kwargs_ls['alpha'], kwargs_ls['beta'], kwargs_ls['max_iters'], kwargs_ls['tol'],
                gene_active, cell_active, alpha_gene,
            )

            total_gene_updates += int(np.sum(gene_active))
            total_cell_updates += int(np.sum(cell_active))

            # Update convergence masks (G1, G2)
            if use_mask:
                if recheck_interval > 0 and (t + 1) % recheck_interval == 0:
                    # Periodically reactivate all entities to catch non-monotone moves
                    gene_active[:] = True
                    cell_active[:] = True
                else:
                    if tol_gene > 0:
                        delta_B_norm = np.linalg.norm(B - B_prev, axis=1)
                        gene_active = delta_B_norm > tol_gene
                    if tol_cell > 0:
                        delta_A_norm = np.linalg.norm(A[:, d:] - A_prev_latent, axis=1)
                        cell_active = delta_A_norm > tol_cell

            func_val = (func_val + np.sum(np.abs(B[:,d-a:d]) * weights)) / p
            hist.append(func_val)
            if not np.isfinite(func_val) or func_val>np.maximum(1e3*np.abs(func_val_pre),1e3):
                pprint.pprint('Encountered large or infinity values. Try to decrease the value of C for the norm constraints.')
                break
            elif es(func_val):
                pbar.set_postfix_str('Early stopped. ' + es.info)
                pbar.close()
                break
            else:
                func_val_pre = func_val
            pbar.set_postfix(nll='{:.02f}'.format(func_val))

    kwargs_glm['disp_glm'] = nuisance[0]
    res = {'n_iter':t, 'func_val':func_val, 'resid':func_val_pre - func_val,
           'hist':hist, 'kwargs_glm':kwargs_glm, 'kwargs_ls':kwargs_ls, 'kwargs_es':kwargs_es,
           'total_gene_updates': total_gene_updates, 'total_cell_updates': total_cell_updates}
    res['X_U'] = A; res['B_Gamma'] = B; res['U'] = A[:,d:]
    return res



