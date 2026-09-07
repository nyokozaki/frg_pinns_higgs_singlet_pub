#!/usr/bin/env python3
"""
Higgs-only (1次元 rho) 版の縮約結果を可視化する。

- tree (UV-matched quartics, canonical scaling)
- RGE-run tree + CW (T=0)
- RGE-run tree + CW + thermal (T_raw=100 GeV)
- numerical (relax_tree, tree-only リラクゼーション)
- full flow eq. (rloop+eta, relax_full.py の Newton-Krylov解)

singlet版 (frg_discrete4_main) の disc->0 分岐点非平滑性がここでは構造的に
存在しないため、full flow eq. が f_tol まで収束することを示すのが主眼。
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import _pathsetup  # noqa: F401
from seed_potential_higgs_only import u_tree_exact
import cw_thermal

T_END = -2.0

tree_path = "results/relax_tree_test.npz"
full_path = "results/relax_full_test.npz"

case_tree = np.load(tree_path)
rho_a = case_tree["rho"]
U_num_tree = case_tree["U_final"]

full = np.load(full_path, allow_pickle=True)
rho_f = full["rho"]
U_full = full["U_final"]
final_res = float(full["final_residual_norm"])

U_tree_a = u_tree_exact(T_END, rho_a)

U_cw = cw_thermal.u_CW_zeroT(T_END, rho_f)
# ring/daisy (Arnold-Espinosa) 補正込みを物理的参照として使う。1-loopのみ
# (u_thermal_finiteT) 比で全rhoにわたって一律-13〜-15%ほど下がる(cw_thermal.py
# 参照) — 定性的な形は変わらないが無視できる大きさではないため、こちらを採用。
U_cw_thermal = U_cw + cw_thermal.u_thermal_finiteT_ring(T_END, rho_f)
U_tree_rgerun = cw_thermal.u_tree_rgerun(T_END, rho_f)
U_cw = U_cw + U_tree_rgerun
U_cw_thermal = U_cw_thermal + U_tree_rgerun

# 原点 (rho=rho_min) で U=0 に揃える
U_cw = U_cw - U_cw[0]
U_cw_thermal = U_cw_thermal - U_cw_thermal[0]

fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle(
    f"frg_higgs_only (Higgs+Yukawa+gauge, no singlet), t_end={T_END:g}\n"
    f"full flow eq. Newton-Krylov: final residual norm = {final_res:.2e} "
    f"(singlet-version baseline stalled at 0.175)",
    fontsize=11,
)

ax.plot(rho_a, U_num_tree, "o-", color="tab:blue", markersize=4, label="numerical (relax_tree, tree-only)")
ax.plot(rho_a, U_tree_a, "k--", label="tree (UV-matched quartics)")
ax.plot(rho_f, U_cw, "-.", color="tab:green", label="RGE-run tree + CW (T=0)")
ax.plot(rho_f, U_cw_thermal, ":", color="tab:red", linewidth=2.5, label="RGE-run tree + CW + thermal + ring (T=100 GeV)")
ax.plot(rho_f, U_full, "s-", color="tab:orange", label="full flow eq. (rloop+eta), Newton-Krylov [converged]")

ax.axhline(0, color="gray", linewidth=1.0)
ax.set_xlabel("rho")
ax.set_ylabel("U(rho, t_end)")
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3)

os.makedirs("plots", exist_ok=True)
out_path = "plots/higgs_only_full_vs_references.png"
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(out_path, dpi=150)
print("saved:", out_path)
