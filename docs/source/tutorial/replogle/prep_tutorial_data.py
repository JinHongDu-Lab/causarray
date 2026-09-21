"""Download and subset Replogle et al. (2022) K562 essential-gene Perturb-seq.

Source: scPerturb (Peidli et al. 2024), Zenodo record 13350497,
``ReplogleWeissman2022_K562_essential.h5ad`` (1.55 GB).

Steps (all streaming, via ``crispyx``; the full matrix is never loaded):

1. download the source file into ``data/`` (resumable; skipped if present),
2. select the ``N_TOP_PERT`` most abundant perturbations plus ``N_CTRL``
   randomly chosen control cells from ``obs`` alone,
3. stream the subset to ``replogle_subset.h5ad`` with
   :func:`crispyx.write_filtered_subset`,
4. verify that ``X`` holds raw integer UMI counts. causarray fits
   negative-binomial GLMs, so anything else is an error. If the matrix turns
   out to be ``log1p(normalize_total)`` values, the exact counts are restored
   from the per-cell scale implied by the smallest non-zero entry and checked
   against ``obs['ncounts']``; the file is rewritten with integer counts.

Run once from the project root (about 5 minutes after the download):

    python docs/source/tutorial/replogle/prep_tutorial_data.py

Background. Until 2026-09-20 the tutorial's ``replogle_subset.h5ad`` was
derived from a copy whose ``X`` had been ``log1p``-normalised to 10,000
counts per cell (non-integer entries, control variance 0.28x the mean), so
every negative-binomial fit in the tutorial ran on normalised values. Step 4
exists so that this cannot recur silently.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import requests

import crispyx

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
SOURCE_URL = ("https://zenodo.org/api/records/13350497/files/"
              "ReplogleWeissman2022_K562_essential.h5ad/content")
SOURCE_PATH = DATA_DIR / "ReplogleWeissman2022_K562_essential.h5ad"
OUT_PATH = HERE / "replogle_subset.h5ad"
BACKUP_PATH = HERE / "replogle_subset_lognorm_backup.h5ad"

PERT_COL_CANDIDATES = ("gene", "perturbation")
CTRL_LABEL_CANDIDATES = ("non-targeting", "control")
N_TOP_PERT = 200
N_CTRL = 2000
RANDOM_STATE = 0


def log(*args):
    print(time.strftime("[%H:%M:%S]"), *args, flush=True)


def download(url: str, path: Path, chunk: int = 1 << 22) -> None:
    """Resumable HTTP download."""
    path.parent.mkdir(parents=True, exist_ok=True)
    head = requests.head(url, allow_redirects=True, timeout=60)
    total = int(head.headers.get("content-length", 0))
    have = path.stat().st_size if path.exists() else 0
    if total and have == total:
        log(f"{path.name} already downloaded ({total / 1e9:.2f} GB)")
        return
    headers = {"Range": f"bytes={have}-"} if have else {}
    log(f"downloading {url} -> {path} ({have / 1e9:.2f}/{total / 1e9:.2f} GB present)")
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        mode = "ab" if have and r.status_code == 206 else "wb"
        with open(path, mode) as fh:
            for block in r.iter_content(chunk_size=chunk):
                fh.write(block)
    size = path.stat().st_size
    if total and size != total:
        raise RuntimeError(f"download incomplete: {size} of {total} bytes")
    log(f"download complete ({size / 1e9:.2f} GB)")


def pick_columns(obs: pd.DataFrame) -> tuple[str, str]:
    pert_col = next((c for c in PERT_COL_CANDIDATES if c in obs.columns), None)
    if pert_col is None:
        raise KeyError(f"none of {PERT_COL_CANDIDATES} in obs columns: {list(obs.columns)}")
    labels = set(obs[pert_col].astype(str).unique())
    ctrl = next((c for c in CTRL_LABEL_CANDIDATES if c in labels), None)
    if ctrl is None:
        raise KeyError(f"none of {CTRL_LABEL_CANDIDATES} found in obs[{pert_col!r}]")
    return pert_col, ctrl


def select_cells(obs: pd.DataFrame, pert_col: str, ctrl: str) -> np.ndarray:
    labels = obs[pert_col].astype(str)
    vc = labels.value_counts()
    top = vc[vc.index != ctrl].head(N_TOP_PERT).index.tolist()
    log(f"top {N_TOP_PERT} perturbations: {vc[top].min()}-{vc[top].max()} cells each")
    ctrl_idx = np.flatnonzero(labels.to_numpy() == ctrl)
    rng = np.random.default_rng(RANDOM_STATE)
    ctrl_sel = np.sort(rng.choice(ctrl_idx, size=min(N_CTRL, len(ctrl_idx)), replace=False))
    mask = labels.isin(top).to_numpy()
    mask[ctrl_sel] = True
    log(f"selected {int(labels.isin(top).sum()):,} perturbed + {len(ctrl_sel):,} control cells "
        f"of {len(obs):,}")
    return mask


def matrix_is_integer(X, n_check: int = 2000) -> bool:
    block = X[:n_check].toarray() if sp.issparse(X[:n_check]) else np.asarray(X[:n_check])
    return bool(np.allclose(block, np.round(block), atol=1e-6))


def restore_counts_from_lognorm(adata: ad.AnnData) -> np.ndarray:
    """Invert ``log1p(count * s_cell)``; ``s_cell`` is set by the smallest non-zero entry (count 1)."""
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    X = X.astype(np.float64)
    min_nz = np.where(X > 0, X, np.inf).min(axis=1)
    if not np.all(np.isfinite(min_nz)):
        raise ValueError("cells with no non-zero entries; cannot infer scale")
    scale = np.expm1(min_nz)
    counts = np.expm1(X) / scale[:, None]
    dev = np.abs(counts - np.round(counts)).max()
    if dev > 1e-2:
        raise ValueError(f"matrix is not log1p(normalised counts): max deviation {dev:.3f}")
    counts = np.round(counts)
    if "ncounts" in adata.obs:
        ratio = counts.sum(axis=1) / adata.obs["ncounts"].astype(float).to_numpy()
        if not np.allclose(ratio, 1.0, atol=1e-3):
            raise ValueError("recovered library sizes do not match obs['ncounts']")
    log(f"restored integer counts (max rounding deviation {dev:.1e})")
    return counts


def main() -> None:
    download(SOURCE_URL, SOURCE_PATH)

    obs = crispyx.load_obs(SOURCE_PATH)
    pert_col, ctrl = pick_columns(obs)
    log(f"source: {len(obs):,} cells; perturbation column {pert_col!r}, control label {ctrl!r}")
    cell_mask = select_cells(obs, pert_col, ctrl)
    n_vars = ad.read_h5ad(SOURCE_PATH, backed="r").n_vars
    gene_mask = np.ones(n_vars, dtype=bool)

    if OUT_PATH.exists() and not BACKUP_PATH.exists():
        OUT_PATH.rename(BACKUP_PATH)
        log(f"previous subset kept as {BACKUP_PATH.name}")
    log("streaming subset to disk with crispyx.write_filtered_subset ...")
    crispyx.write_filtered_subset(SOURCE_PATH, cell_mask=cell_mask, gene_mask=gene_mask,
                                  output_path=OUT_PATH)

    adata = ad.read_h5ad(OUT_PATH)
    if not matrix_is_integer(adata.X):
        log("X is not integer-valued; attempting to restore raw counts")
        adata.X = sp.csr_matrix(restore_counts_from_lognorm(adata).astype(np.float32))
    adata.uns["pert_col"] = pert_col
    adata.uns["ctrl_label"] = ctrl
    adata.write_h5ad(OUT_PATH)

    # ---- report ----
    X = adata.X
    labels = adata.obs[pert_col].astype(str).to_numpy()
    is_ctrl = labels == ctrl
    Xc = X[is_ctrl]
    mean = np.asarray(Xc.mean(axis=0)).ravel()
    sq = np.asarray(Xc.multiply(Xc).mean(axis=0)).ravel() if sp.issparse(Xc) else (Xc ** 2).mean(axis=0)
    var = sq - mean ** 2
    ok = mean > 0.5
    lib = np.asarray(X.sum(axis=1)).ravel()
    log(f"wrote {OUT_PATH.name}: {adata.n_obs:,} cells x {adata.n_vars:,} genes, "
        f"{len(set(labels)) - 1} perturbations, {int(is_ctrl.sum()):,} controls")
    log(f"integer counts: {matrix_is_integer(X)}; library size median {np.median(lib):,.0f} "
        f"(range {lib.min():,.0f}-{lib.max():,.0f})")
    log(f"control var/mean for genes with mean > 0.5: median {np.median(var[ok] / mean[ok]):.2f} "
        f"(>= 1 expected for counts)")


if __name__ == "__main__":
    main()
