#!/usr/bin/env python3
"""
T_raw=200 GeV: full flow eq. (rloop+eta, Newton-Krylov) の2つの解
(baseline / coupling-prior c_lower=0.5) を、
"tree level potential with running coupling + CW + one-loop thermal + ring summation"
(Arnold-Espinosa完全版, cw_thermal.u_thermal_finiteT_ring) と比較する。
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import _pathsetup  # noqa: F401
from seed_potential_higgs_only import u_tree_exact
import cw_thermal

T_END = -2.0
T_RAW = 200.0

cw_thermal.set_T_raw(T_RAW)
import flow_equation_higgs_only as _fe
_fe.tau_uv = cw_thermal.tau_uv

baseline_path = "results/relax_full_T200_baseline_fresh.npz"
clower_path = "results/relax_full_T200_clower05_rhocwcut.npz"

d_base = np.load(baseline_path, allow_pickle=True)
d_prior = np.load(clower_path, allow_pickle=True)

rho = d_base["rho"]
U_base = d_base["U_final"]
U_prior = d_prior["U_final"]
res_base = float(d_base["final_residual_norm"])
res_prior = float(d_prior["final_residual_norm"])

U_tree = u_tree_exact(T_END, rho)

U_cw = cw_thermal.u_CW_zeroT(T_END, rho)
U_tree_rgerun = cw_thermal.u_tree_rgerun(T_END, rho)
U_cw = U_cw + U_tree_rgerun

U_ring = U_cw + cw_thermal.u_thermal_finiteT_ring(T_END, rho)

# 原点で U=0 に揃える (relax_full側もU(rho_min)=0で出力されている)
U_cw = U_cw - U_cw[0]
U_ring = U_ring - U_ring[0]

fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle(
    f"frg_higgs_only, T_raw={T_RAW:g} GeV, t_end={T_END:g}\n"
    f"baseline final residual={res_base:.2e} / coupling-prior(c_lower=0.5) final residual={res_prior:.2e}",
    fontsize=11,
)

ax.plot(rho, U_tree, "k--", label="tree (UV-matched quartics)")
ax.plot(rho, U_cw, "-.", color="0.45", label="RGE-run tree + CW (T=0)")
ax.plot(rho, U_ring, ":", color="tab:green", linewidth=2.5,
        label="RGE-run tree + CW + thermal + ring (Arnold-Espinosa) -- physical reference")
ax.plot(rho, U_base, "s-", color="tab:orange", markersize=4,
        label=f"full flow eq., Newton-Krylov, baseline (resid={res_base:.1e})")
ax.plot(rho, U_prior, "^-", color="tab:purple", markersize=4,
        label=f"full flow eq. + coupling prior (resid={res_prior:.1e})")

ax.axhline(0, color="gray", linewidth=1.0)
ax.set_xlabel("rho")
ax.set_ylabel("U(rho, t_end)")
ax.legend(loc="best", fontsize=8)
ax.grid(alpha=0.3)

os.makedirs("plots", exist_ok=True)
out_path = "plots/higgs_only_T200_ring_compare.png"
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(out_path, dpi=150)
print("saved:", out_path)
