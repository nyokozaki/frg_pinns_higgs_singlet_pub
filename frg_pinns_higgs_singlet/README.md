# frg_pinns_higgs_singlet

PINN for the full two-field ($\rho$, $\sigma$) Wetterich flow, plus the plotting
and $O(3)$ bounce-action scripts. This is the canonical solver of the paper.

Configuration:
- `perturbation/config_params.py` — benchmark point and temperature (`T_RAW`)
- `hyperparams.py` — all loss weights (read live during training)

See the repository-root `README.md` for the workflow, the figure→script map,
and how to load the trained-weight snapshots.
