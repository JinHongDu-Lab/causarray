"""
Prepare the SCARF tutorial dataset for causarray.

Source: subset_adata_SCARF_10celltypes_10perturbations.h5ad
        (110,968 cells x 19,070 genes; 10 perturbations + Non_target controls)
        subset_adata_SCARF_10celltypes_48perturbations.h5ad
        (142,619 cells x 19,070 genes; 48 perturbations + Non_target controls)

Both files share the *same* 102,595 'Non_target' control cells (identical
barcodes) -- verified by set comparison. A naive concatenation of the two
files would therefore double-count every control cell. The correct combine
keeps one file whole and adds only the other file's perturbed cells:

    combined = file_10pert (whole, 110,968 cells)
             + file_48pert[gene_target != 'Non_target'] (40,024 cells)
             = 150,992 cells, 58 unique perturbations, one shared control pool

`crispyx` (as of 0.1.2) has no multi-file concat primitive -- only
single-file streaming filter/subset/write utilities (`write_filtered_subset`,
`subsample`, `filter_*_by_cell_count`). Merging two source files is done with
plain `anndata.experimental.concat_on_disk` (disk-to-disk, not scanpy); this
is the one step in the SCARF pipeline that isn't crispyx-native.

Output:
    scarf_combined.h5ad     -- 150,992 cells x 19,070 genes, 58 perturbations
                                (durable combined artifact; not fit in full --
                                that would be a ~5-10x heavier GCATE run than
                                the adamson tutorial)
    scarf_L45_subset.h5ad   -- 22,396 cells x 19,070 genes, 58 perturbations,
                                restricted to predicted_group == '005 L4-5 IT
                                CTX Glut' (the pptx's own pilot cell type).
                                This is the file the tutorial notebook fits
                                causarray/Wilcoxon on.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import anndata as ad
from anndata.experimental import concat_on_disk

import crispyx

HERE = Path(__file__).parent.resolve()
SRC_10PERT = HERE / "subset_adata_SCARF_10celltypes_10perturbations.h5ad"
SRC_48PERT = HERE / "subset_adata_SCARF_10celltypes_48perturbations.h5ad"

TMP_48PERT_UNIQUE = HERE / "scarf_48pert_unique.h5ad"
COMBINED = HERE / "scarf_combined.h5ad"
L45_SUBSET = HERE / "scarf_L45_subset.h5ad"

PERT_COL = "gene_target"
CTRL_LABEL = "Non_target"
TARGET_CELL_TYPE = "005 L4-5 IT CTX Glut"


def main() -> None:
    # ── Step A (crispyx): drop file B's duplicate control cells ─────────────
    if not TMP_48PERT_UNIQUE.exists():
        print(f"Loading obs metadata from {SRC_48PERT.name} ...")
        obs_48 = crispyx.load_obs(SRC_48PERT)
        cell_mask = (obs_48[PERT_COL].astype(str) != CTRL_LABEL).to_numpy()
        n_vars = ad.read_h5ad(SRC_48PERT, backed="r").n_vars
        gene_mask = np.ones(n_vars, dtype=bool)
        print(f"  Keeping {cell_mask.sum():,}/{len(cell_mask):,} perturbed cells "
              f"(dropping {(~cell_mask).sum():,} duplicate '{CTRL_LABEL}' cells)")
        crispyx.write_filtered_subset(
            SRC_48PERT, cell_mask=cell_mask, gene_mask=gene_mask,
            output_path=TMP_48PERT_UNIQUE,
        )
        print(f"  Saved -> {TMP_48PERT_UNIQUE}")
    else:
        print(f"{TMP_48PERT_UNIQUE.name} already exists, skipping Step A.")

    # ── Step B (anndata, documented crispyx-impossible step): disk concat ──
    if not COMBINED.exists():
        print(f"Concatenating {SRC_10PERT.name} + {TMP_48PERT_UNIQUE.name} "
              f"-> {COMBINED.name} (anndata.experimental.concat_on_disk) ...")
        concat_on_disk(
            [str(SRC_10PERT), str(TMP_48PERT_UNIQUE)],
            str(COMBINED),
            join="outer",
        )
        print(f"  Saved -> {COMBINED}")
    else:
        print(f"{COMBINED.name} already exists, skipping Step B.")

    # ── Step C (crispyx): restrict to the pptx pilot cell type ─────────────
    if not L45_SUBSET.exists():
        print(f"Loading obs metadata from {COMBINED.name} ...")
        obs_combined = crispyx.load_obs(COMBINED)
        cell_mask = (obs_combined["predicted_group"].astype(str) == TARGET_CELL_TYPE).to_numpy()
        n_vars = ad.read_h5ad(COMBINED, backed="r").n_vars
        gene_mask = np.ones(n_vars, dtype=bool)
        print(f"  Keeping {cell_mask.sum():,}/{len(cell_mask):,} cells "
              f"(predicted_group == '{TARGET_CELL_TYPE}')")
        crispyx.write_filtered_subset(
            COMBINED, cell_mask=cell_mask, gene_mask=gene_mask,
            output_path=L45_SUBSET,
        )
        # Tag control/perturbation convention, matching adamson/replogle tutorials.
        adata = ad.read_h5ad(L45_SUBSET)
        adata.uns["pert_col"] = PERT_COL
        adata.uns["ctrl_label"] = CTRL_LABEL
        adata.write_h5ad(L45_SUBSET)
        print(f"  Saved -> {L45_SUBSET}")
    else:
        print(f"{L45_SUBSET.name} already exists, skipping Step C.")

    # ── Summary ──────────────────────────────────────────────────────────────
    final = ad.read_h5ad(L45_SUBSET, backed="r")
    vc = final.obs[PERT_COL].astype(str).value_counts()
    n_perts = len(vc) - 1  # exclude control
    n_ctrl = int(vc.get(CTRL_LABEL, 0))
    n_pert_cells = int(final.n_obs - n_ctrl)
    print(f"\nFinal pilot subset: {final.n_obs:,} cells x {final.n_vars:,} genes")
    print(f"  {n_perts} perturbations, {n_pert_cells:,} pert cells, {n_ctrl:,} control cells")
    pert_counts = vc.drop(CTRL_LABEL)
    print(f"  Cells per pert: min={pert_counts.min():,}, "
          f"median={pert_counts.median():.0f}, max={pert_counts.max():,}")


if __name__ == "__main__":
    main()
