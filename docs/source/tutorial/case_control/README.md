# SEA-AD case-control tutorial

Layout: numbered scripts run first, then the notebook; `data/` holds prepared
inputs, `results/` holds everything generated.

    1_preprocess_sea_ad.py    CellxGene Census -> data/sea_ad_mtg_exneu_pb.h5ad
                              (excitatory-neuron pseudo-bulk, one row per donor)

    sea_ad_case_control.ipynb reads data/sea_ad_mtg_exneu_pb.h5ad,
                              results/sea_ad_r.csv (JIC table),
                              results/sea_ad_gcate.pkl (latent-factor fit),
                              results/sea_ad_lfc.csv

Everything here is tracked in git, including the fit caches, so the published
tutorial renders without a refit. Regenerate `results/sea_ad_gcate.pkl` by
deleting it and re-running the GCATE cell; the notebook writes it back.
