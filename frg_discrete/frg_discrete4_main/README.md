# frg_discrete4_main

Grid-based benchmark for the same Wetterich flow equation solved by the PINN.
Rather than integrating forward in RG time from the UV boundary, scale and field
space are discretized together and the whole trajectory `U(t, rho, sigma)` is
solved as one global system (central differences in `(rho, sigma)`,
Crank-Nicolson in `t`). Used for the relaxation benchmark of Secs. 4-5 of the
paper.

Physics modules (`flow_equation.py`, `grid_fd.py`, `running_couplings.py`,
`seed_potential.py`, `perturbation/`) are shared with `../frg_higgs_only/` and
imported from `../frg_discrete/` via `_pathsetup.py`.

## Scripts

| script | what it does |
|---|---|
| `relax_tree.py` | tree-only limit (`rloop = eta = 0`): the system is linear, solved exactly by one block-bidiagonal LU factorization |
| `relax_full.py` | full flow (loop + `eta` + `coth` thresholds): nonlinear, solved with Jacobian-free Newton-Krylov (`scipy.optimize.newton_krylov`, LGMRES inner solver) |
| `relax_full_w.py` | first-order-form variant (auxiliary fields `R = u_rho`, `S = u_sigma`; no second-derivative stencil) |
| `coupling_prior.py` | soft perturbative-consistency forcing term added to the residual (grid counterpart of the PINN's `loss_extensions.py`) |
| `cw_thermal.py` | one-loop Coleman-Weinberg and Arnold-Espinosa ring-resummed thermal potentials, for the reference curves |
| `run_all_relax.sh` | the eight runs (two temperatures x {baseline, +prior, central4, central4+prior}) of the relaxation table |
| `plot_t200*.py`, `plot_domrestrict.py` | benchmark slice figures |

```bash
python relax_tree.py --n-rho 41 --n-sigma 41 --n-t 100
python relax_full.py --n-rho 21 --n-sigma 21 --n-t 40 --maxiter 100
bash run_all_relax.sh
```

The full two-field Newton-Krylov iteration does not reach `f_tol` for any
regularization choice we tried; see Sec. 5 of the paper.
