# FRG–PINN for the Higgs–singlet model

Code accompanying the paper

> *Nonperturbative functional renormalization group for Higgs–singlet models
> with physics-informed neural networks*, N. Yokozaki.

We solve the Wetterich flow equation for the scale-dependent effective potential
of the $Z_2$-symmetric real-singlet extension of the Standard Model, in the LPA′
truncation, at zero and finite temperature, **without a polynomial expansion of
the potential**. The same flow equation is attacked by two independent numerical
methods:

* **`frg_pinns_higgs_singlet/`** — physics-informed neural network (PyTorch): a
  hybrid tree-level + neural-network ansatz for `u(t, ρ, σ)`, trained by
  minimizing the PDE residual via automatic differentiation. Multi-block
  decomposition in RG scale and field space. Includes the `O(3)` bounce-action
  pipeline of Sec. 6.4.
* **`frg_discrete/frg_discrete4_main/`** — grid-based Jacobian-free
  Newton–Krylov relaxation solver for the same equation (SciPy), used as an
  independent benchmark.

Each method has a **singlet-free (Higgs-only) reduction** used for the
cross-check of Secs. 4.3 / 6.1:

* **`frg_pinns_higgs_only/`** — PINN, Higgs + external gauge/top only.
* **`frg_discrete/frg_higgs_only/`** — grid solver, Higgs-only.

`frg_discrete/frg_discrete/` holds the physics modules (`flow_equation.py`,
`grid_fd.py`, `running_couplings.py`, `seed_potential.py`, `perturbation/`)
shared by the two grid solvers via their `_pathsetup.py`.

## Install

```bash
pip install -r requirements.txt          # torch, numpy, scipy, matplotlib
pip install cosmoTransitions              # only for the bounce action (Sec. 6.4)
```

Python 3.11, PyTorch 2.1 (CUDA). `device = "cuda"` is the default in every
`hyperparams.py`; set it to `"cpu"` there if needed.

## Benchmark point

| | value |
|---|---|
| IR / UV scale | `k_IR = 150 GeV`, `Λ = k_UV = k_IR e^{2} ≈ 1108 GeV`, RG-time range `t ∈ [-2, 0]` |
| Higgs sector (fixed by data) | `m_h = 125.2 GeV`, `v = 246.2 GeV` → `λ_H^tree = m_h²/2v² ≈ 0.129`, `μ_H²^tree = -m_h²/2` |
| Singlet sector (benchmark) | `λ_S^tree = 0.27`, `λ_HS^tree = 1.0`, `μ_S²^tree = -(90 GeV)²` |
| Gauge / top at `k_IR` | `g1, g2, g3, y_t ≈ 0.358, 0.649, 1.176, 0.9` |
| Temperatures | `T = 100 GeV` and `T = 200 GeV` |

The one-loop Coleman–Weinberg matching at `k_IR` and the two-loop RGE running to
`Λ` are in `perturbation/` (`rges.py`, checked against PyR@TE3). The couplings
at `k_IR` and `Λ` are Table 1 of the paper.

## Trained weights

The trained network snapshots are attached to the GitHub **Release** of this
repository (they are too large to commit):

| file | contents |
|---|---|
| `T100data.zip` | 1D slices at `T = 100 GeV`: `data_UV/` (ρ) + `data_sigma/` (σ) + the `hyperparams.py` and `perturbation/config_params.py` used |
| `T200data.zip` | 1D slices at `T = 200 GeV`, same layout |
| `T1002Ddata.zip` | preliminary 2D network at `T = 100 GeV` (`data_UV_2D/`), input to the bounce/vacuum-line result; reuses the `T=100` config |

To use a snapshot, unzip it inside `frg_pinns_higgs_singlet/` — it drops the
`data_*` directories in place together with the exact `hyperparams.py` /
`perturbation/config_params.py` for that run (the temperature is set by
`config_params.py::T_RAW`). Then run the plot-only pipeline (below).

## Reproducing the results

### PINN (`frg_pinns_higgs_singlet/`)

```bash
# train (resumes from checkpoints; safe to re-run after interruption)
python run_pinn_pipeline.py                 # ρ + σ 1D slices, 10^4 epochs/block
python train_networks.py 2D --round 1 --epochs 10000   # full 2D grid of blocks

# plot only, from an unzipped snapshot
python run_pinn_pipeline.py --plot-only

# O(3) bounce on the 2D potential (Sec. 6.4)
python bounce_action.py --T 100
python plot_vacuum_line_bounce.py --T 100
```

`hyperparams.py` holds every loss weight; it is read live, so edits only affect
processes started afterwards. Architecture: `my_networks.py::BaseNet`
(stem MLP `3×160`, `u`/`ρ`/`σ` branches, mixed branch `2×320`, SiLU,
zero-initialized heads, ≈ `3.9e5` parameters/block). Appendix A of the paper
tabulates the full hyperparameter set.

### PINN Higgs-only (`frg_pinns_higgs_only/`)

```bash
python train_networks.py rho --epochs 10000                 # with the soft penalty
python train_networks.py rho --epochs 10000 --no-loss-ext    # without it
python compare_ext_noext.py                                  # Fig. pinn_higgs_only_ext_vs_noext
```

### Grid solver (`frg_discrete/frg_discrete4_main/`)

```bash
python relax_tree.py --n-rho 41 --n-sigma 41 --n-t 100       # tree-only: exact block solve
python relax_full.py --n-rho 21 --n-sigma 21 --n-t 40 --maxiter 100   # full flow: Newton–Krylov
bash run_all_relax.sh                                        # the 8 runs of Table (relax)
python plot_t200_full.py                                     # benchmark slice figure
```

### Grid solver Higgs-only (`frg_discrete/frg_higgs_only/`)

```bash
python relax_full.py --n-rho 21 --n-t 40 --maxiter 100
python plot_t200_ring_compare.py                             # Fig. higgs_only_discrete_T200
```

## Figure → code map

| paper figure | folder / entry point |
|---|---|
| `pinn_{rho,sigma}_slice_T{100,200}`, `pinn_training_history_rho_T200` | `frg_pinns_higgs_singlet/` — unzip snapshot, `run_pinn_pipeline.py --plot-only` (or `show_UV1D.py` / `show_1D_sigma.py`) |
| `pinn_higgs_only_ext_vs_noext` | `frg_pinns_higgs_only/compare_ext_noext.py` |
| `pinn_vacuum_line_T100`, `S_3(T)/T` | `frg_pinns_higgs_singlet/bounce_action.py` → `plot_vacuum_line_bounce.py` (2D snapshot) |
| `relax_benchmark_T{100,200}`, Table (relax) | `frg_discrete/frg_discrete4_main/` — `run_all_relax.sh`, then `plot_t200*.py` |
| `higgs_only_discrete_T200` | `frg_discrete/frg_higgs_only/plot_t200_ring_compare.py` |
| `mynet` | schematic of `BaseNet`, hand-drawn (no script) |

Note: some plot scripts and `run_all_relax.sh` refer to slightly different
`results/*.npz` filenames; adjust the `--out` names or the paths at the top of
the plot scripts to match. All of the physics is in the scripts as shipped.

## License

MIT — see `LICENSE`.
