#!/usr/bin/env python3
"""
Comparison figure of the 4 full-flow-eq. variants (baseline / +coupling prior /
central4 / central4+prior) using a domain restriction (rho,sigma >= 1e-3)
instead of disc_cut, for T_raw=100 and 200 GeV.

Slices at rho=1e-3 fixed / sigma=1e-3 fixed (the old rho=0/sigma=0 cannot be
used because those are regions excluded by the domain restriction).

The 8 curves:
  (1) numerical (relax_tree)              -> results/case_a.npz (T-independent, no domain restriction)
  (2) tree (UV-matched quartics)          -> u_tree_exact (T-independent)
  (3) RGE-run tree + CW                   -> cw_thermal.u_CW_zeroT (T-independent)
  (4) RGE-run tree + CW + thermal         -> cw_thermal.u_thermal_finiteT (re-evaluated at T_raw)
  (5) full flow eq. relaxation            -> results/full_n21_domrestrict_T{T}_baseline.npz
  (6) full flow eq. + coupling prior      -> results/full_n21_domrestrict_T{T}_prior.npz
  (7) full flow eq., central4             -> results/full_n21_domrestrict_T{T}_central4.npz
  (8) full flow eq., central4 + prior     -> results/full_n21_domrestrict_T{T}_central4_prior.npz
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

import _pathsetup  # noqa: F401
from seed_potential import u_tree_exact
import cw_thermal

T_RAW = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
T_END = -2.0
RHO_MIN = 1e-3
SIGMA_MIN = 1e-3

cw_thermal.set_T_raw(T_RAW)

case_a = np.load("results/case_a.npz")
rho_a = case_a["rho"]
sigma_a = case_a["sigma"]
U_num = case_a["U_final"]

tag = f"full_n21_domrestrict_T{T_RAW:g}"
full_base = np.load(f"results/{tag}_baseline.npz", allow_pickle=True)
full_prior = np.load(f"results/{tag}_prior.npz", allow_pickle=True)
full_c4 = np.load(f"results/{tag}_central4.npz", allow_pickle=True)
full_c4_prior = np.load(f"results/{tag}_central4_prior.npz", allow_pickle=True)

rho_f = full_base["rho"]
sigma_f = full_base["sigma"]
assert abs(rho_f[0] - RHO_MIN) < 1e-12 and abs(sigma_f[0] - SIGMA_MIN) < 1e-12

RHO_A, SIGMA_A = np.meshgrid(rho_a, sigma_a, indexing="ij")
U_tree_a = u_tree_exact(T_END, RHO_A, SIGMA_A)

RHO_F, SIGMA_F = np.meshgrid(rho_f, sigma_f, indexing="ij")
U_cw = cw_thermal.u_CW_zeroT(T_END, RHO_F, SIGMA_F)
U_cw_thermal = U_cw + cw_thermal.u_thermal_finiteT_ring(T_END, RHO_F, SIGMA_F)
U_tree_rgerun = cw_thermal.u_tree_rgerun(T_END, RHO_F, SIGMA_F)
U_cw = U_cw + U_tree_rgerun
U_cw_thermal = U_cw_thermal + U_tree_rgerun

# origin for shifting reference curves: (rho_min, sigma_min), i.e. grid index 0,
# matching how full_*['U_final'] is already normalized (origin = U_final[0,0]).
i0_f = 0
U_cw = U_cw - U_cw[i0_f, i0_f]
U_cw_thermal = U_cw_thermal - U_cw_thermal[i0_f, i0_f]

i0_a = int(np.argmin(np.abs(rho_a - RHO_MIN)))
j0_f = 0

fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
fig.suptitle(
    f"frg_discrete4, t_end={T_END:g}, T_raw={T_RAW:g} GeV "
    f"(domain restriction rho,sigma>={RHO_MIN:g}, disc_cut=0; CW/thermal shifted so U(rho_min,sigma_min)=0)\n"
    "(1),(2): relax_tree.py 41x41 grid, Nt=100 (T-independent, no domain restriction)  |  (3),(4): analytic  |  "
    "(5)-(8): relax_full.py 21x21 grid, Nt=40",
    fontsize=10,
)

series = [
    dict(x=sigma_a, y=U_num[i0_a, :], xr=rho_a, yr=U_num[:, i0_a],
         style="o-", color="tab:blue", label="numerical (relax_tree)", ms=4),
    dict(x=sigma_a, y=U_tree_a[i0_a, :], xr=rho_a, yr=U_tree_a[:, i0_a],
         style="k--", label="tree (UV-matched quartics, canonical scaling)"),
    dict(x=sigma_f, y=U_cw[i0_f, :], xr=rho_f, yr=U_cw[:, j0_f],
         style="-.", color="0.45", label="RGE-run tree + CW (T=0)"),
    dict(x=sigma_f, y=U_cw_thermal[i0_f, :], xr=rho_f, yr=U_cw_thermal[:, j0_f],
         style=":", color="tab:green", lw=2.5,
         label="RGE-run tree + CW + thermal + ring (Arnold-Espinosa) -- physical reference"),
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
ax.set_title(f"rho = {RHO_MIN:g} fixed")
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
ax.set_ylabel(f"U(rho={RHO_MIN:g}, sigma, t_end)")
ax.legend(fontsize=7.5, loc="best")

ax = axes[1]
ax.set_title(f"sigma = {SIGMA_MIN:g} fixed")
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
ax.set_ylabel(f"U(rho, sigma={SIGMA_MIN:g}, t_end)")
ax.legend(fontsize=7.5, loc="best")

fig.tight_layout(rect=[0, 0, 1, 0.90])
out_path = f"plots/case_a_tree_slices_cw_thermal_domrestrict_T{T_RAW:g}.png"
fig.savefig(out_path, dpi=150)
print("saved:", out_path)

for name, d in [("baseline", full_base), ("prior", full_prior),
                 ("central4", full_c4), ("central4+prior", full_c4_prior)]:
    print(f"{name}: final_residual_norm={float(d['final_residual_norm']):.4f}  elapsed={float(d['elapsed']):.0f}s")
    print(f"  u(rho={RHO_MIN:g},sigma_max)={d['U_final'][i0_f,-1]:.4f}   "
          f"u(rho_max,sigma={SIGMA_MIN:g})={d['U_final'][-1,j0_f]:.4f}")

print(f"reference u(rho={RHO_MIN:g},sigma_max): tree={U_tree_a[i0_a,-1]:.4f} "
      f"CW={U_cw[i0_f,-1]:.4f} CW+thermal={U_cw_thermal[i0_f,-1]:.4f}")
print(f"reference u(rho_max,sigma={SIGMA_MIN:g}): tree={U_tree_a[-1,i0_a]:.4f} "
      f"CW={U_cw[-1,j0_f]:.4f} CW+thermal={U_cw_thermal[-1,j0_f]:.4f}")
