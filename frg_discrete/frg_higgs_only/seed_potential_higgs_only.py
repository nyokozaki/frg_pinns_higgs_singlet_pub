"""
frg_discrete/seed_potential.py の singlet除去版(Higgs単一チャンネル)。
FRGフローの初期条件 (UV端, t_phys=0) を与える種ポテンシャル。
"""

import numpy as np

from perturbation.config_params import tau_uv, k_IR, t_range, finite_T
from perturbation.jbjf import J_Bnp as J_B, J_Fnp as J_F
from running_couplings_higgs_only import tree_params, get_running_couplings, get_running_quartics, get_running_masses

aH, lamH = tree_params

NGS = 3.0
_EPS = 1e-10


def u_tree_exact(t_phys, rho_phys):
    muH2_t = aH * np.exp(-2.0 * t_phys)
    return muH2_t * rho_phys + lamH * rho_phys ** 2


def _scalar_masses_bare(rho_phys, t_phys):
    kt = k_IR * np.exp(-t_range + t_phys)

    g1, g2, yt = get_running_couplings(t_phys)
    lamH_t = get_running_quartics(t_phys)
    mhsq_t = get_running_masses(t_phys)

    muH2_t = mhsq_t / (kt * kt)

    u_rho = muH2_t + 2.0 * lamH_t * rho_phys
    u_rhorho = 2.0 * lamH_t

    mG2_b = u_rho
    mH2_b = u_rho + 2.0 * rho_phys * u_rhorho

    return mG2_b, mH2_b, g1, g2, yt


def u_thermal_finiteT(t_phys, rho_phys):
    """
    有限温度1-loop熱ポテンシャル u_th(t, rho) (Arnold-Espinosa style, ring項なし)。
    seed_potential.u_thermal_finiteT と同じ内容(bare質量のみ使用)。
    """
    mG2_b, mH2_b, g1, g2, yt = _scalar_masses_bare(rho_phys, t_phys)

    mW_T2 = 0.5 * g2 ** 2 * rho_phys
    mZ_T2 = 0.5 * (g2 ** 2 + g1 ** 2) * rho_phys
    mA_T2 = np.zeros_like(rho_phys, dtype=float) * 1.0
    mt2 = yt ** 2 * rho_phys

    tau_t = tau_uv * np.exp(-t_phys)
    tau2 = np.clip(tau_t ** 2, 1e-12, None)
    tau4 = tau2 * tau2

    JB_G = J_B(mG2_b / tau2)
    JB_H = J_B(mH2_b / tau2)

    JB_W = J_B(mW_T2 / tau2)
    JB_Z = J_B(mZ_T2 / tau2)
    JB_A = J_B(mA_T2 / tau2)

    JF_t = J_F(mt2 / tau2)

    u_th_scalar = (tau4 / (2.0 * np.pi ** 2)) * (NGS * JB_G + JB_H)
    u_th_gauge = (tau4 / (2.0 * np.pi ** 2)) * (6.0 * JB_W + 3.0 * JB_Z + 1.0 * JB_A)
    u_th_top = -(12.0 * tau4 / (2.0 * np.pi ** 2)) * JF_t

    return u_th_scalar + u_th_gauge + u_th_top


def u_seed(t_phys, rho_phys):
    """FRGフローの初期条件 (t_phys=0, config_params.t_range に対応するUV端)。"""
    u = u_tree_exact(t_phys, rho_phys)
    if finite_T:
        u = u + u_thermal_finiteT(t_phys, rho_phys)
    return u
