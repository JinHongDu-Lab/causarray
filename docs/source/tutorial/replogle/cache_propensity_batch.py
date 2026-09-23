"""Cache one Replogle batch for propensity-score sensitivity analyses.

The final ``gcate_lfc_batch`` cache intentionally contains only result tables.
This script regenerates one focal batch, caches its nuisance predictions, and
then reuses the cached outcome predictions across a small propensity grid.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import scanpy as sc


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT))

from causarray import (  # noqa: E402
    LFC,
    estimate_propensity_scores,
    summarize_propensity_scores,
)
from causarray.gcate import fit_gcate  # noqa: E402
from causarray.gcate_glm import estimate_disp_auto  # noqa: E402
from causarray.utils import (  # noqa: E402
    comp_size_factor,
    prep_causarray_data,
    subsample_ctrl_cells,
    subsample_pert_cells,
)


RAW_PATH = HERE / "replogle_subset.h5ad"
R_PATH = (HERE / "replogle-r-0.0.10.csv" if (HERE / "replogle-r-0.0.10.csv").exists()
          else HERE / "replogle-r.csv")  # the raw-count JIC table, as the notebook uses
CACHE_PATH = HERE / "replogle_propensity_batch12.npz"
BASELINE_PATH = HERE / "replogle_propensity_batch12_baseline.csv.gz"
SUMMARY_PATH = HERE / "replogle_propensity_batch12_summary.csv"
TUNING_PATH = HERE / "replogle_propensity_batch12_tuning.csv"
SCORES_PATH = HERE / "replogle_propensity_batch12_selected_scores.npz"

BATCH_INDEX = 12
N_BATCHES = 14
N_CTRL = 2000
MAX_PERT_CELLS = 2000
RANDOM_STATE = 0
FDR = 0.05
FOCAL_PERTURBATIONS = ("SUPT5H", "SRRT", "SUPT6H", "TSR2")


def _prepare_batch():
    """Reproduce the cells and preprocessing used by batch 12."""
    adata = sc.read_h5ad(RAW_PATH, backed="r")
    label_column = str(adata.uns["pert_col"])
    control_label = str(adata.uns["ctrl_label"])
    labels = adata.obs[label_column].astype(str)
    A = pd.get_dummies(labels, drop_first=False).drop(columns=[control_label])
    A_array = A.to_numpy(dtype=float)

    chunks = [list(chunk) for chunk in np.array_split(range(A.shape[1]), N_BATCHES)]
    columns = chunks[BATCH_INDEX]
    perturbations = A.columns[columns].astype(str).tolist()
    missing = set(FOCAL_PERTURBATIONS).difference(perturbations)
    if missing:
        raise ValueError(f"Focal perturbations are absent from batch: {sorted(missing)}")

    control_rows = np.flatnonzero(A_array.sum(axis=1) == 0)
    control_rows = subsample_ctrl_cells(
        control_rows, n_ctrl=N_CTRL, random_state=RANDOM_STATE
    )
    perturbation_rows = np.flatnonzero(A_array[:, columns].sum(axis=1) > 0)
    perturbation_rows = subsample_pert_cells(
        perturbation_rows,
        max_cells=MAX_PERT_CELLS,
        random_state=RANDOM_STATE + BATCH_INDEX,
    )
    cell_rows = np.sort(np.concatenate([control_rows, perturbation_rows]))

    # Reproduce prep_causarray_data's global count cap and log-library scaling
    # without materializing the complete 79,865 x 8,563 count matrix. Two passes
    # are required because prep_causarray_data caps the counts *before* deriving
    # library sizes, and the cap itself depends on every cell.
    gene_maxima = np.zeros(adata.n_vars, dtype=float)
    for start in range(0, adata.n_obs, 5000):
        block = adata.X[start : start + 5000].toarray()
        gene_maxima = np.maximum(gene_maxima, block.max(axis=0))
    count_cap = float(np.round(np.quantile(gene_maxima, 0.999)))

    library_sizes = []
    for start in range(0, adata.n_obs, 5000):
        block = adata.X[start : start + 5000].toarray()
        library_sizes.append(np.minimum(block, count_cap).sum(axis=1))
    library_sizes = np.concatenate(library_sizes)
    log_library_size = np.log2(library_sizes)
    log_library_size = (
        (log_library_size - log_library_size.mean())
        / log_library_size.std(ddof=1)
    )

    Y = np.minimum(adata.X[cell_rows].toarray(), count_cap)
    X = np.ones((len(cell_rows), 1), dtype=float)
    A_batch = A_array[np.ix_(cell_rows, columns)]
    X_A = np.column_stack([np.ones(len(cell_rows)), log_library_size[cell_rows]])
    genes = adata.var_names.astype(str).to_numpy(dtype=str)
    adata.file.close()
    return {
        "Y": Y,
        "X": X,
        "A": A_batch,
        "X_A": X_A,
        "genes": genes,
        "perturbations": np.asarray(perturbations, dtype=str),
        "cell_idx": cell_rows,
        "ctrl_idx": control_rows,
        "pert_idx": perturbation_rows,
        "count_cap": count_cap,
    }


def _check_prep_causarray_data_contract():
    """Guard the inlined preprocessing against drift in ``prep_causarray_data``.

    ``_prepare_batch`` reimplements the count cap and log-library scaling to
    avoid materializing the full 79,865 x 8,563 matrix. If ``prep_causarray_data``
    ever changed, that shortcut would silently stop reproducing the batch the
    tutorial actually fit. Rather than compare on a data slice -- which cannot
    work, because the cap is a global quantile and a slice yields a different
    one -- assert the three properties ``_prepare_batch`` actually relies on, on
    a small synthetic matrix:

    1. counts are capped at ``round(quantile(per-gene max, 0.999))``;
    2. library sizes are derived from the *capped* counts, not the raw ones;
    3. ``X_A`` is ``[intercept, standardized log2 library size]``.
    """
    rng = np.random.default_rng(0)
    counts = rng.poisson(3.0, size=(200, 60)).astype(float)
    counts[0, 0] = 500.0  # an outlier the cap must flatten
    A = pd.DataFrame({"pert": np.r_[np.zeros(100), np.ones(100)]})

    Y_prep, _, _, X_A_prep = prep_causarray_data(pd.DataFrame(counts), A)
    Y_prep = np.asarray(Y_prep, dtype=float)
    X_A_prep = np.asarray(X_A_prep, dtype=float)

    cap = float(np.round(np.quantile(counts.max(axis=0), 0.999)))
    expected_Y = np.minimum(counts, cap)
    log_library = np.log2(expected_Y.sum(axis=1))
    expected_column = (
        (log_library - log_library.mean()) / log_library.std(ddof=1)
    )

    if not np.allclose(Y_prep, expected_Y):
        raise ValueError(
            "prep_causarray_data no longer applies the count cap that "
            "_prepare_batch reimplements; update _prepare_batch."
        )
    if X_A_prep.shape[1] != 2 or not np.allclose(X_A_prep[:, 0], 1.0):
        raise ValueError(
            "prep_causarray_data no longer returns X_A as "
            "[intercept, log-library size]; update _prepare_batch."
        )
    if not np.allclose(X_A_prep[:, 1], expected_column):
        raise ValueError(
            "prep_causarray_data's log-library size no longer matches the "
            "inlined formula (capped counts, log2, standardized with ddof=1); "
            "update _prepare_batch."
        )


def build_cache(force: bool = False):
    """Fit and cache the focal batch once."""
    if CACHE_PATH.exists() and BASELINE_PATH.exists() and not force:
        print(f"Using existing intermediate cache: {CACHE_PATH}")
        return

    _check_prep_causarray_data_contract()
    data = _prepare_batch()
    local_control = np.searchsorted(data["cell_idx"], data["ctrl_idx"])
    offset_control = np.log(comp_size_factor(data["Y"][local_control]))
    dispersion = estimate_disp_auto(
        data["Y"][local_control], data["X"][local_control], offset=offset_control
    )
    r_table = pd.read_csv(R_PATH)
    r = int(r_table.loc[r_table["JIC"].idxmin(), "r"])

    started = time.perf_counter()
    _, fitted = fit_gcate(
        data["Y"], data["X"], data["A"], r,
        family="nb", offset=True, disp_glm=dispersion, disp_family="poisson",
        kwargs_es_1={"rel_tol": 2e-4, "max_iters": 30},
        kwargs_es_2={"rel_tol": 2e-4, "max_iters": 30},
    )
    U = fitted["U"].copy()
    W = np.column_stack([data["X"], U])
    W_A = np.column_stack([data["X_A"], U])
    offset = np.log(fitted["kwargs_glm"]["size_factor"])
    result, estimation = LFC(
        pd.DataFrame(data["Y"], columns=data["genes"]),
        W,
        pd.DataFrame(data["A"], columns=data["perturbations"]),
        W_A,
        family="nb",
        offset=offset,
    )
    print(f"Focal batch fit completed in {(time.perf_counter() - started) / 60:.1f} min")

    np.savez_compressed(
        CACHE_PATH,
        batch=np.asarray([BATCH_INDEX]),
        perturbations=data["perturbations"],
        genes=data["genes"],
        cell_idx=data["cell_idx"],
        ctrl_idx=data["ctrl_idx"],
        pert_idx=data["pert_idx"],
        A=data["A"].astype(np.uint8),
        X_A=W_A.astype(np.float32),
        U=U.astype(np.float32),
        offset=offset.astype(np.float32),
        Y=data["Y"].astype(np.float32),
        Y_hat=estimation["Y_hat"].astype(np.float32),
        pi_hat=estimation["pi_hat"].astype(np.float32),
        pi_hat_raw=estimation["pi_hat_raw"].astype(np.float32),
        count_cap=np.asarray([data["count_cap"]]),
        random_state=np.asarray([RANDOM_STATE]),
        r=np.asarray([r]),
    )
    result.to_csv(BASELINE_PATH, index=False, compression="gzip")


def _validate_cache(cache):
    required = {
        "A", "X_A", "Y", "Y_hat", "offset", "genes", "perturbations",
        "pi_hat_raw",
    }
    missing = required.difference(cache.files)
    if missing:
        raise ValueError(f"Intermediate cache is missing: {sorted(missing)}")
    A = cache["A"]
    perturbations = cache["perturbations"].astype(str)
    genes = cache["genes"].astype(str)
    if A.shape[1] != len(perturbations) or cache["Y"].shape[1] != len(genes):
        raise ValueError("Intermediate cache labels and dimensions do not match")
    if set(FOCAL_PERTURBATIONS).difference(perturbations):
        raise ValueError("Intermediate cache does not contain every focal perturbation")


def run_sensitivities(force: bool = False):
    """Reuse cached outcomes across the prespecified propensity grid."""
    build_cache(force=force)
    # This cache is generated locally by ``build_cache`` and may contain
    # object-string arrays when produced by older NumPy/pandas combinations.
    cache = np.load(CACHE_PATH, allow_pickle=True)
    _validate_cache(cache)
    A = cache["A"].astype(float)
    W_A = cache["X_A"].astype(float)
    W = np.column_stack([W_A[:, 0], W_A[:, 2:]])
    perturbations = cache["perturbations"].astype(str)
    genes = cache["genes"].astype(str)
    Y = cache["Y"].astype(float)
    Y_hat = cache["Y_hat"].astype(float)
    offset = cache["offset"].astype(float)
    baseline = pd.read_csv(BASELINE_PATH)

    settings = [
        ("fitted balanced C=1", cache["pi_hat_raw"].astype(float), (0.01, 0.99), "fitted"),
    ]
    for label, class_weight, C in (
        ("OOF balanced C=1", "balanced", 1.0),
        ("OOF balanced C=0.1", "balanced", 0.1),
        ("OOF balanced C=0.01", "balanced", 0.01),
        ("OOF calibrated C=1", None, 1.0),
    ):
        scores = estimate_propensity_scores(
            A, W_A, K=5, random_state=RANDOM_STATE,
            class_weight=class_weight, C=C, clip=None,
        )
        settings.append((label, scores, (0.01, 0.99), "oof"))
    settings.append(
        ("OOF calibrated C=1, clip 0.05", settings[-1][1], (0.05, 0.95), "oof")
    )

    score_tables = []
    tuning_rows = []
    selected_scores = {"A": A.astype(np.uint8), "perturbations": perturbations}
    baseline_indexed = baseline.set_index(["trt", "gene_names"])
    for label, raw_scores, clip, score_type in settings:
        clipped_scores = np.clip(raw_scores, *clip)
        summary = summarize_propensity_scores(
            A, clipped_scores, treatment_names=perturbations, clip_bounds=clip
        )
        summary.insert(0, "model", label)
        summary.insert(1, "score_type", score_type)
        summary.insert(2, "clip", str(clip))
        score_tables.append(summary)
        for perturbation in FOCAL_PERTURBATIONS:
            j = list(perturbations).index(perturbation)
            selected_scores[f"{label}__{perturbation}"] = clipped_scores[:, j].astype(
                np.float32
            )

        result, _ = LFC(
            pd.DataFrame(Y, columns=genes),
            W,
            pd.DataFrame(A, columns=perturbations),
            W_A,
            family="nb",
            offset=offset,
            Y_hat=Y_hat,
            pi_hat=raw_scores,
            ps_clip=clip,
            usevar="unequal",
        )
        indexed = result.set_index(["trt", "gene_names"])
        for perturbation in perturbations:
            current = indexed.loc[perturbation]
            reference = baseline_indexed.loc[perturbation]
            aligned = current.join(
                reference[["tau", "std", "padj"]], rsuffix="_baseline"
            )
            finite = np.isfinite(aligned["tau"]) & np.isfinite(
                aligned["tau_baseline"]
            )
            standard_error_ratio = current["std"] / reference["std"].replace(0, np.nan)
            tuning_rows.append({
                "model": label,
                "score_type": score_type,
                "clip": str(clip),
                "treatment": perturbation,
                "discoveries": int((current["padj"] < FDR).sum()),
                "baseline_discoveries": int((reference["padj"] < FDR).sum()),
                "discovery_change": int(
                    (current["padj"] < FDR).sum() - (reference["padj"] < FDR).sum()
                ),
                "median_std": float(np.nanmedian(current["std"])),
                "median_std_ratio": float(np.nanmedian(standard_error_ratio)),
                "tau_pearson_vs_baseline": float(np.corrcoef(
                    aligned.loc[finite, "tau"], aligned.loc[finite, "tau_baseline"]
                )[0, 1]),
                "median_abs_tau_change": float(np.nanmedian(
                    np.abs(aligned["tau"] - aligned["tau_baseline"])
                )),
            })
        del result
        gc.collect()

    pd.concat(score_tables, ignore_index=True).to_csv(SUMMARY_PATH, index=False)
    pd.DataFrame(tuning_rows).to_csv(TUNING_PATH, index=False)
    np.savez_compressed(SCORES_PATH, **selected_scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="refit the focal batch even when the intermediate cache exists",
    )
    args = parser.parse_args()
    run_sensitivities(force=args.force)
    print(f"Wrote {SUMMARY_PATH.name}, {TUNING_PATH.name}, and {SCORES_PATH.name}")


if __name__ == "__main__":
    main()
