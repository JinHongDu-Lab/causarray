"""
Preprocessing script for the SEA-AD case-control tutorial.

Downloads excitatory-neuron data (MTG) from the CellxGene Census (SEA-AD collection),
subsamples cells per donor, pseudo-bulks to donor level, and saves the result as
`data/sea_ad_mtg_exneu_pb.h5ad` in the current directory.

Requirements
------------
    pip install cellxgene-census anndata

Usage
-----
    python 1_preprocess_sea_ad.py
"""

import re
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import cellxgene_census

# ── Dataset IDs for MTG excitatory-neuron subclasses in SEA-AD ─────────────────
# Retrieved from CellxGene Census (version 2025-11-08)
EXCITATORY_SUBCLASS_DATASET_IDS = {
    "L2/3 IT":   "b74100ea-1a1a-486a-9cad-70ae44150935",
    "L4 IT":     "ac0fee7e-0999-4319-b244-20278e1ff2fb",
    "L5 IT":     "605b89b1-c474-4180-8c0b-88afb5920991",
    "L5 ET":     "07428d73-fdea-4bd4-a801-94b00c4d961c",
    "L6 CT":     "c19275f5-739e-4796-ad5d-b0830b760db1",
    "L6b":       "8a8aedcb-5bb3-453d-a9f0-f37951ae1515",
    "L6 IT":     "de104f7e-14fa-4795-bd19-b5ee2c1563e0",
    "L6 IT Car3":"79d485a8-b8b1-49f2-85aa-c44e5206aa53",
    "L5/6 NP":   "75e6eee5-d0e3-4291-9360-f288ffe6c7c4",
}

# Obs columns to retain from Census for pseudo-bulk metadata
OBS_COLS = ["donor_id", "sex", "disease", "self_reported_ethnicity",
            "development_stage", "raw_sum"]

CENSUS_VERSION = "2025-11-08"
MAX_CELLS_PER_DONOR = 300  # cap applied to the COMBINED set (matching the paper)
RANDOM_SEED = 0
OUT_FILE = "data/sea_ad_mtg_exneu_pb.h5ad"

import os as _os
for _d in ("data", "results"):
    _os.makedirs(_d, exist_ok=True)



def _parse_age(dev_stage: str) -> float:
    """Extract numeric age from a development-stage string, e.g. '74-year-old stage' → 74."""
    m = re.search(r"(\d+)", str(dev_stage))
    return float(m.group(1)) if m else float("nan")


def download_subclass(census, dataset_id: str, subclass: str) -> ad.AnnData:
    """Download one excitatory-neuron subclass from CellxGene Census."""
    print(f"  Downloading {subclass} …", flush=True)
    adata = cellxgene_census.get_anndata(
        census,
        organism="Homo sapiens",
        obs_value_filter=f"dataset_id == '{dataset_id}'",
        obs_column_names=OBS_COLS,
    )
    adata.obs["subclass"] = subclass
    return adata


def subsample_per_donor(adata: ad.AnnData, max_cells: int, rng) -> ad.AnnData:
    """Keep at most `max_cells` randomly chosen cells per donor (vectorised, integer positions)."""
    donor_col = adata.obs["donor_id"].values
    positions  = np.arange(len(donor_col))
    # Group integer positions by donor
    df = pd.DataFrame({"pos": positions, "donor": donor_col})
    keep_pos = (
        df.groupby("donor", group_keys=False, observed=True)
          .apply(lambda g: g.sample(n=min(len(g), max_cells),
                                    random_state=rng.integers(2**31).item()),
                 include_groups=False)
          ["pos"]
          .values
    )
    keep_pos = np.sort(keep_pos)
    return adata[keep_pos].copy()


def pseudobulk(adata: ad.AnnData) -> ad.AnnData:
    """Sum raw integer counts per donor; collect one metadata row per donor."""
    # Raw counts from Census are stored as float32 integers
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.int32)

    donors = adata.obs["donor_id"].unique()
    X_list, meta_rows = [], []
    for donor in donors:
        mask = (adata.obs["donor_id"] == donor).values
        X_list.append(X[mask].sum(axis=0))
        row = adata.obs[mask].iloc[0][
            ["sex", "disease", "self_reported_ethnicity", "development_stage"]
        ].to_dict()
        row["donor_id"] = donor
        row["n_cells"] = int(mask.sum())
        # Total UMIs across all retained cells (used as library-size offset)
        row["library_size"] = int(adata.obs.loc[adata.obs["donor_id"] == donor, "raw_sum"].sum())
        meta_rows.append(row)

    X_pb = np.vstack(X_list)
    obs_pb = pd.DataFrame(meta_rows).set_index("donor_id")
    var_pb = adata.var[["feature_name"]].copy() if "feature_name" in adata.var.columns else adata.var[[]].copy()

    pb = ad.AnnData(X=sp.csr_matrix(X_pb), obs=obs_pb, var=var_pb)
    # Use gene symbols as var_names; make unique in case of duplicates
    pb.var_names = pb.var["feature_name"].tolist()
    pb.var_names_make_unique()
    return pb


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    print("Opening CellxGene Census …", flush=True)
    census = cellxgene_census.open_soma(census_version=CENSUS_VERSION)

    adatas = []
    for subclass, dataset_id in EXCITATORY_SUBCLASS_DATASET_IDS.items():
        adata = download_subclass(census, dataset_id, subclass)
        # No per-subclass subsampling here; final cap is applied after combining
        adatas.append(adata)
        print(f"    → {adata.n_obs} cells from {adata.obs['donor_id'].nunique()} donors", flush=True)

    census.close()

    print("Concatenating …", flush=True)
    combined = ad.concat(adatas, join="inner", merge="first")

    # Subsample 300 cells per donor from the COMBINED excitatory-neuron set,
    # matching the paper's approach (SEA-AD 1-preprocess.py, ROSMAP 1-preprocess.ipynb).
    print(f"Before subsampling: {combined.n_obs} cells, avg {combined.n_obs / combined.obs['donor_id'].nunique():.0f}/donor", flush=True)
    combined = subsample_per_donor(combined, MAX_CELLS_PER_DONOR, rng)
    print(f"After subsampling:  {combined.n_obs} cells, avg {combined.n_obs / combined.obs['donor_id'].nunique():.0f}/donor", flush=True)

    print("Pseudo-bulking …", flush=True)
    pb = pseudobulk(combined)

    # ── Exclude young 'Reference' donors (development_stage age < 60) ──────────
    pb.obs["age"] = pb.obs["development_stage"].apply(_parse_age)
    pb = pb[pb.obs["age"] >= 60].copy()

    # ── Treatment: dementia (AD/MCI) = 1, normal aging = 0 ─────────────────────
    pb.obs["trt"] = (pb.obs["disease"] != "normal").astype(int)

    # ── Binary sex covariate ────────────────────────────────────────────────────
    pb.obs["sex_bin"] = (pb.obs["sex"] == "male").astype(float)

    # ── Gene filtering: keep genes expressed in at least one donor ───────────────
    X_dense = pb.X.toarray()
    keep_genes = X_dense.max(axis=0) > 10
    pb = pb[:, keep_genes].copy()

    # ── Summary ─────────────────────────────────────────────────────────────────
    n_ad = int(pb.obs["trt"].sum())
    n_ctrl = int((pb.obs["trt"] == 0).sum())
    print(f"\nDonors: {pb.n_obs}  (AD/dementia={n_ad}, normal={n_ctrl})")
    print(f"Genes:  {pb.n_vars}")
    print(f"Writing {OUT_FILE} …", flush=True)
    pb.write_h5ad(OUT_FILE, compression="gzip")
    print("Done.")


if __name__ == "__main__":
    main()
