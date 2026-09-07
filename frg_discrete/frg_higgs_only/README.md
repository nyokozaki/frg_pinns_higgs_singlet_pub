# frg_higgs_only

Singlet-free (Higgs-only) reduction of `../frg_discrete4_main/`: Higgs plus the
same external gauge and top sector, one field direction `rho`, no `2x2` radial
mass matrix. Used for the cross-check of Sec. 4.3 of the paper (whether the
convergence failure of the full two-field solver originates in the
Higgs-singlet mass mixing).

Because it needs a different field content, this folder uses aliased local
modules `flow_equation_higgs_only.py`, `grid_fd_higgs_only.py`,
`running_couplings_higgs_only.py`, `seed_potential_higgs_only.py` instead of the
shared versions in `../frg_discrete/` (still imported via `_pathsetup.py` for
`perturbation/`).

## Scripts

| script | what it does |
|---|---|
| `relax_tree.py` | tree-only limit, exact linear solve |
| `relax_full.py` | full flow, Jacobian-free Newton-Krylov |
| `relax_full_w.py` | first-order-form variant |
| `solve_flow.py` | forward integration (`scipy.integrate.solve_ivp`, method of lines), an independent cross-check of `relax_full.py` |
| `coupling_prior.py` | soft perturbative-consistency forcing term |
| `cw_thermal.py` | one-loop CW and ring-resummed thermal reference potentials |
| `run_all_relax.sh`, `plot_t200_ring_compare.py`, `plot_*` | benchmark runs and figures (`plot_t200_ring_compare.py` produces the Higgs-only relaxation figure) |

```bash
python relax_full.py --n-rho 21 --n-t 40 --maxiter 100
python plot_t200_ring_compare.py
```

Here the iteration does reach `f_tol = 1e-8` at `T = 100 GeV` but converges to a
spurious solution; at `T = 200 GeV` it stalls. See Sec. 4.3.
