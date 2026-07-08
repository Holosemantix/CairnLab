# Paper1 Diagnostic Region Validation

This directory is a paper-facing validation artifact for the ATR/SMPR sweep. It
does not rank threshold classifiers by F1, precision, recall, or interval IoU.
Instead, it checks whether full-sweep checkpoints form:

- a shared low-ATR/high-SMPR diagnostic region for recovered checkpoints,
- consistent directions within task--training-seed sweeps,
- separation between recovered and fragile checkpoints.

Regime labels are derived from normalized closed-loop recovery under
`pixels_std0.08`. Fragile means recovery <= 0.2;
robust means recovery >= 0.8; transition checkpoints are
kept visible rather than forced into either endpoint.
