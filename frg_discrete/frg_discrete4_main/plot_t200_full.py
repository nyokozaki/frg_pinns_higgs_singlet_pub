#!/usr/bin/env python3
"""
figures/case_a_tree_slices_cw_thermal.png (T_raw=100 GeV, 8曲線版) を
T_raw=200 GeV で再現する (原稿 frg_discrete4_relaxation_and_prior.tex の
Figure \\ref{fig:slices} 差し替え用)。

8曲線の対応 (原稿の enumerate (1)-(8) と同じ):
  (1) numerical (relax_tree)              -> results/case_a.npz (T非依存)
  (2) tree (UV-matched quartics)          -> u_tree_exact (T非依存)
  (3) RGE-run tree + CW                   -> cw_thermal.u_CW_zeroT (T非依存)
  (4) RGE-run tree + CW + thermal         -> cw_thermal.u_thermal_finiteT (T_raw=200で再評価)
  (5) full flow eq. relaxation            -> results/full_n21_Traw200_disccut1e-2.npz
  (6) full flow eq. + coupling prior      -> results/full_n21_Traw200_prior_w015_006.npz
  (7) full flow eq., central4             -> results/full_n21_Traw200_disccut1e-2_central4.npz
  (8) full flow eq., central4 + prior     -> results/full_n21_Traw200_disccut1e-2_central4_prior015_006.npz
"""

import numpy as np
import matplotlib.pyplot as plt

import _pathsetup  # noqa: F401
from seed_potential import u_tree_exact
import cw_thermal

T_RAW = 200.0
T_END = -2.0

cw_thermal.set_T_raw(T_RAW)

case_a = np.load("results/case_a.npz")
rho_a = case_a["rho"]
sigma_a = case_a["sigma"]
U_num = case_a["U_final"]

full_base = np.load("results/full_n21_Traw200_disccut1e-2.npz", allow_pickle=True)
full_prior = np.load("results/full_n21_Traw200_prior_w015_006.npz", allow_pickle=True)
full_c4 = np.load("results/full_n21_Traw200_disccut1e-2_central4.npz", allow_pickle=True)
full_c4_prior = np.load("results/full_n21_Traw200_disccut1e-2_central4_prior015_006.npz", allow_pickle=True)

rho_f = full_base["rho"]
sigma_f = full_base["sigma"]

RHO_A, SIGMA_A = np.meshgrid(rho_a, sigma_a, indexing="ij")
U_tree_a = u_tree_exact(T_END, RHO_A, SIGMA_A)

RHO_F, SIGMA_F = np.meshgrid(rho_f, sigma_f, indexing="ij")
U_cw = cw_thermal.u_CW_zeroT(T_END, RHO_F, SIGMA_F)
U_cw_thermal = U_cw + cw_thermal.u_thermal_finiteT(T_END, RHO_F, SIGMA_F)
U_tree_rgerun = cw_thermal.u_tree_rgerun(T_END, RHO_F, SIGMA_F)
U_cw = U_cw + U_tree_rgerun
U_cw_thermal = U_cw_thermal + U_tree_rgerun

i0_f = int(np.argmin(np.abs(rho_f)))
U_cw = U_cw - U_cw[i0_f, i0_f]
U_cw_thermal = U_cw_thermal - U_cw_thermal[i0_f, i0_f]

i0_a = int(np.argmin(np.abs(rho_a)))
j0_f = int(np.argmin(np.abs(sigma_f)))

fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
fig.suptitle(
    f"frg_discrete4, t_end={T_END:g}, T_raw={T_RAW:g} GeV (CW/thermal shifted so U(origin)=0)\n"
    "(1),(2): relax_tree.py 41x41 grid, Nt=100 (T-independent)  |  (3),(4): analytic  |  "
    "(5)-(8): relax_full.py 21x21 grid, Nt=40",
    fontsize=10,
)

series = [
    dict(x=sigma_a, y=U_num[i0_a, :], xr=rho_a, yr=U_num[:, i0_a],
         style="o-", color="tab:blue", label="numerical (relax_tree)", ms=4),
    dict(x=sigma_a, y=U_tree_a[i0_a, :], xr=rho_a, yr=U_tree_a[:, i0_a],
         style="k--", label="tree (UV-matched quartics, canonical scaling)"),
    dict(x=sigma_f, y=U_cw[i0_f, :], xr=rho_f, yr=U_cw[:, j0_f],
         style="-.", color="tab:green", label="RGE-run tree + CW (scalar+W/Z+top)"),
    dict(x=sigma_f, y=U_cw_thermal[i0_f, :], xr=rho_f, yr=U_cw_thermal[:, j0_f],
         style=":", color="tab:red", lw=2.5, label=f"RGE-run tree + CW + thermal (T={T_RAW:g} GeV)"),
    dict(x=sigma_f, y=full_base["U_final"][i0_f, :], xr=rho_f, yr=full_base["U_final"][:, j0_f],
         style="s-", color="tab:orange", label="full flow eq. (rloop+eta) relaxation [not fully converged]"),
    dict(x=sigma_f, y=full_prior["U_final"][i0_f, :], xr=rho_f, yr=full_prior["U_final"][:, j0_f],
         style="^-", color="tab:purple", label="full flow eq. + coupling prior (0.015/0.006) [not fully converged]"),
    dict(x=sigma_f, y=full_c4["U_final"][i0_f, :], xr=rho_f, yr=full_c4["U_final"][:, j0_f],
         style="d-", color="tab:brown", label="full flow eq., central4 (4th-order spatial) [not fully converged]"),
    dict(x=sigma_f, y=full_c4_prior["U_final"][i0_f, :], xr=rho_f, yr=full_c4_prior["U_final"][:, j0_f],
         style="v-", color="tab:pink", label="full flow eq., central4 + coupling prior [not fully converged]"),
]

ax = axes[0]
ax.set_title("rho = 0 fixed")
for s in series:
    kwargs = dict(label=s["label"])
    if "color" in s:
        kwargs["color"] = s["color"]
    if "ms" in s:
        kwargs["markersize"] = s["ms"]
    if "lw" in s:
        kwargs["linewidth"] = s["lw"]
    ax.plot(s["x"], s["y"], s["style"], **kwargs)
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_xlabel("sigma")
ax.set_ylabel("U(rho=0, sigma, t_end)")
ax.legend(fontsize=7.5, loc="best")

ax = axes[1]
ax.set_title("sigma = 0 fixed")
for s in series:
    kwargs = dict(label=s["label"])
    if "color" in s:
        kwargs["color"] = s["color"]
    if "ms" in s:
        kwargs["markersize"] = s["ms"]
    if "lw" in s:
        kwargs["linewidth"] = s["lw"]
    ax.plot(s["xr"], s["yr"], s["style"], **kwargs)
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_xlabel("rho")
ax.set_ylabel("U(rho, sigma=0, t_end)")
ax.legend(fontsize=7.5, loc="best")

fig.tight_layout(rect=[0, 0, 1, 0.90])
out_path = "plots/case_a_tree_slices_cw_thermal_T200.png"
fig.savefig(out_path, dpi=150)
print("saved:", out_path)

for name, d in [("baseline(5)", full_base), ("prior(6)", full_prior),
                 ("central4(7)", full_c4), ("central4+prior(8)", full_c4_prior)]:
    print(f"{name}: final_residual_norm={float(d['final_residual_norm']):.4f}")
