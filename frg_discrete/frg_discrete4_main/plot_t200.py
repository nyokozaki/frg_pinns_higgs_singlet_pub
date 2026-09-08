#!/usr/bin/env python3
"""
Reproduce case_a_tree_slices_cw_thermal.png (default T_raw=100 GeV) at T_raw=200 GeV.

- numerical (relax_tree) / tree / RGE-run tree+CW: depend only on the tree
  structure and not on T_raw, so the existing results/case_a.npz (relax_tree.py,
  41x41, Nt=100, tree-only seed) is reused as-is.
- RGE-run tree+CW+thermal: overwrite T_raw with cw_thermal.set_T_raw(200), then
  re-evaluate u_thermal_finiteT.
- full flow eq. (rloop+eta) baseline / +coupling prior: use the existing
  results/full_n21_Traw200_baseline.npz / full_n21_Traw200_prior.npz
  results (T_raw=200, n_rho=n_sigma=21, n_t=40, disc_cut=0).
- full flow eq., central4 (+coupling prior): use
  results/full_n21_Traw200_central4_disccut1e-2(_prior015_006).npz, run anew at
  T_raw=200 with the same disc_cut=1e-2 as the T=100 version.

Readability first (keep the white background; make text and lines thick and large).
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import _pathsetup  # noqa: F401
from seed_potential import u_tree_exact
import cw_thermal

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10.5,
    "lines.linewidth": 2.4,
    "lines.markersize": 6,
    "axes.linewidth": 1.3,
    "font.weight": "medium",
})

T_RAW = 200.0
T_END = -2.0
CENTRAL4_BASE = "results/full_n21_Traw200_central4_disccut1e-2.npz"
CENTRAL4_PRIOR = "results/full_n21_Traw200_central4_disccut1e-2_prior015_006.npz"
have_central4 = os.path.exists(CENTRAL4_BASE) and os.path.exists(CENTRAL4_PRIOR)

cw_thermal.set_T_raw(T_RAW)

case_a = np.load("results/case_a.npz")
rho_a = case_a["rho"]
sigma_a = case_a["sigma"]
U_num = case_a["U_final"]  # (n_rho, n_sigma) at t_end (final stored state)

full_base = np.load("results/full_n21_Traw200_disccut1e-2_baseline.npz", allow_pickle=True)
full_prior = np.load("results/full_n21_Traw200_prior.npz", allow_pickle=True)
rho_f = full_base["rho"]
sigma_f = full_base["sigma"]
U_full_base = full_base["U_final"]
U_full_prior = full_prior["U_final"]

if have_central4:
    full_c4 = np.load(CENTRAL4_BASE, allow_pickle=True)
    full_c4_prior = np.load(CENTRAL4_PRIOR, allow_pickle=True)
    U_full_c4 = full_c4["U_final"]
    U_full_c4_prior = full_c4_prior["U_final"]

RHO_A, SIGMA_A = np.meshgrid(rho_a, sigma_a, indexing="ij")
U_tree_a = u_tree_exact(T_END, RHO_A, SIGMA_A)

RHO_F, SIGMA_F = np.meshgrid(rho_f, sigma_f, indexing="ij")
U_cw = cw_thermal.u_CW_zeroT(T_END, RHO_F, SIGMA_F)
U_cw_thermal = U_cw + cw_thermal.u_thermal_finiteT(T_END, RHO_F, SIGMA_F)
U_tree_rgerun = cw_thermal.u_tree_rgerun(T_END, RHO_F, SIGMA_F)
U_cw = U_cw + U_tree_rgerun
U_cw_thermal = U_cw_thermal + U_tree_rgerun

# align U=0 at the origin (rho=sigma=0), the same "shifted so U(origin)=0" convention as the original figure
i0_f = int(np.argmin(np.abs(rho_f)))
U_cw = U_cw - U_cw[i0_f, i0_f]
U_cw_thermal = U_cw_thermal - U_cw_thermal[i0_f, i0_f]

fig, axes = plt.subplots(1, 2, figsize=(16, 7.2))
subtitle = (
    "Tree/CW/thermal curves: relax_tree.py (T-independent), 41x41 grid, Nt=100  |  "
    "'full flow eq.' curves: relax_full.py, 21x21 grid, Nt=40"
)
if not have_central4:
    subtitle += "\n(central4 variants not yet computed at T=200)"
fig.suptitle(
    f"frg_discrete4, t_end={T_END:g}, T_raw={T_RAW:g} GeV (CW/thermal shifted so U(origin)=0)\n"
    + subtitle,
    fontsize=13, fontweight="bold",
)

i0_a = int(np.argmin(np.abs(rho_a)))
j0_f = int(np.argmin(np.abs(sigma_f)))


def draw(ax, x_a, y_num, y_tree, x_f, y_cw, y_cw_th, y_base, y_prior, y_c4, y_c4_prior, xlabel, ylabel):
    ax.plot(x_a, y_num, "o-", color="tab:blue", label="numerical (relax_tree)")
    ax.plot(x_a, y_tree, "k--", label="tree (UV-matched quartics, canonical scaling)")
    ax.plot(x_f, y_cw, "-.", color="tab:green", label="RGE-run tree + CW (scalar+W/Z+top)")
    ax.plot(x_f, y_cw_th, ":", color="tab:red", linewidth=3.4,
            label=f"RGE-run tree + CW + thermal (T={T_RAW:g} GeV)")
    ax.plot(x_f, y_base, "s-", color="tab:orange",
            label="full flow eq. (rloop+eta) relaxation [not fully converged]")
    ax.plot(x_f, y_prior, "^-", color="tab:purple",
            label="full flow eq. + coupling prior (0.015/0.006) [not fully converged]")
    if have_central4:
        ax.plot(x_f, y_c4, "D-", color="tab:brown",
                label="full flow eq., central4 (4th-order spatial) [not fully converged]")
        ax.plot(x_f, y_c4_prior, "P-", color="tab:pink",
                label="full flow eq., central4 + coupling prior (0.015/0.006) [not fully converged]")
    ax.axhline(0, color="gray", linewidth=1.2)
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.tick_params(width=1.3, length=6)
    ax.legend(loc="best", framealpha=0.95)


draw(
    axes[0],
    sigma_a, U_num[i0_a, :], U_tree_a[i0_a, :],
    sigma_f, U_cw[i0_f, :], U_cw_thermal[i0_f, :], U_full_base[i0_f, :], U_full_prior[i0_f, :],
    U_full_c4[i0_f, :] if have_central4 else None,
    U_full_c4_prior[i0_f, :] if have_central4 else None,
    "sigma", "U(rho=0, sigma, t_end)",
)
axes[0].set_title("rho = 0 fixed", fontweight="bold")

draw(
    axes[1],
    rho_a, U_num[:, i0_a], U_tree_a[:, i0_a],
    rho_f, U_cw[:, j0_f], U_cw_thermal[:, j0_f], U_full_base[:, j0_f], U_full_prior[:, j0_f],
    U_full_c4[:, j0_f] if have_central4 else None,
    U_full_c4_prior[:, j0_f] if have_central4 else None,
    "rho", "U(rho, sigma=0, t_end)",
)
axes[1].set_title("sigma = 0 fixed", fontweight="bold")

fig.tight_layout(rect=[0, 0, 1, 0.86])
out_path = "plots/case_a_tree_slices_cw_thermal_T200.png"
fig.savefig(out_path, dpi=150)
print("saved:", out_path, "| central4 included:", have_central4)
