Generalized confounder estimation
=================================

The GCATE (Generalized Confounder Adjustment for Treatment Effects) functions
estimate latent factors that capture unmeasured confounders in the data.
Call :func:`estimate_r` first to select the number of factors *r* via the JIC
criterion, then call :func:`fit_gcate` (or :func:`fit_gcate_batch` for
large-scale screens) to obtain the latent factor matrix ``U``.  Append ``U``
to the observed covariate matrix before calling :func:`LFC`.

Case-control studies with few units
-----------------------------------

The first stage of :func:`fit_gcate` projects the factors orthogonal to the
treatment indicators, and with few units they can stay nearly so: on 85
SEA-AD donors, the R² of disease status on four factors was 0.0003. Such
factors cannot adjust for anything aligned with treatment, yet in
:func:`LFC` they shrink the standard errors. Check calibration with permuted
treatment labels before reading discovery counts, and consider
``usevar='unequal'`` in :func:`LFC`, which is more conservative in small
samples.

.. automodule:: causarray.gcate
   :members:
